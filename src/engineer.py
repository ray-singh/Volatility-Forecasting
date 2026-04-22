"""
Feature engineering for volatility forecasting.

Builds microstructure features from LOB snapshots including realized volatility,
order flow imbalance, queue imbalance, depth ratios, HAR features, and
multi-horizon forward targets.
"""
from __future__ import annotations

import polars as pl
import numpy as np


def build_features(
    df: pl.DataFrame,
    rv_windows: list[int] = None,
    ofi_window: int = 20,
    har_lags: list[int] = None,
    horizons: list[int] = None,
    ticks_per_second: float = 2.0,
    lob_levels: int = 10,
) -> pl.DataFrame:
    """
    Build all microstructure features from LOB snapshots.

    Parameters
    ----------
    df : pl.DataFrame
        LOB snapshot data with columns: mid_price, spread, bid_p*, bid_q*, ask_p*, ask_q*
    rv_windows : list[int]
        Windows for rolling realized volatility (default [50, 300])
    ofi_window : int
        Window for order flow imbalance (default 20)
    har_lags : list[int]
        HAR lags in ticks (default [50])
    horizons : list[int]
        Forecast horizons in seconds (default [1, 5, 30]). Converted to ticks
        via ticks_per_second.
    ticks_per_second : float
        Tick rate for horizon conversion. At 0.5s poll interval, 2.0 ticks/s.
    lob_levels : int
        Number of LOB levels available (used for depth imbalance features).

    Returns
    -------
    pl.DataFrame with original columns plus feature and target columns
    """
    if rv_windows is None:
        rv_windows = [50, 300]
    if har_lags is None:
        har_lags = [50]
    if horizons is None:
        horizons = [1, 5, 30]

    # ── Log returns ────────────────────────────────────────────────────────────
    df = df.with_columns(
        (pl.col("mid_price") / pl.col("mid_price").shift(1)).log().alias("log_return")
    )

    # ── Realized volatility ────────────────────────────────────────────────────
    # RV_t = sqrt(sum_{i=t-w+1}^{t} r_i^2)  — Andersen & Bollerslev (1998)
    # Clamp to 0 before sqrt: floating-point cancellation can produce tiny
    # negative sums (~-1e-24) when all squared returns in the window are near
    # zero, which would make sqrt() return NaN.
    for window in rv_windows:
        df = df.with_columns(
            (pl.col("log_return") ** 2)
            .rolling_sum(window_size=window)
            .clip(lower_bound=0.0)
            .sqrt()
            .alias(f"rv_{window}")
        )

    # ── Bid-ask spread features ────────────────────────────────────────────────
    df = df.with_columns(
        pl.col("spread").rolling_mean(window_size=50).alias("spread_ma_50"),
        (
            (pl.col("spread") - pl.col("spread").rolling_mean(50))
            / (pl.col("spread").rolling_std(50).fill_null(1e-9).clip(lower_bound=1e-9))
        ).alias("spread_zscore_50"),
        (pl.col("spread") / pl.col("mid_price").clip(lower_bound=1e-9)).alias("relative_spread"),
    )

    # ── Microprice & price pressure ────────────────────────────────────────────
    # microprice = volume-weighted average of best bid/ask prices
    # price_pressure = microprice - mid_price (signed order-flow pressure)
    if "bid_q0" in df.columns and "ask_q0" in df.columns:
        df = df.with_columns(
            (
                (pl.col("ask_p0") * pl.col("bid_q0") + pl.col("bid_p0") * pl.col("ask_q0"))
                / (pl.col("bid_q0") + pl.col("ask_q0") + 1e-9)
            ).alias("microprice"),
        )
        df = df.with_columns(
            (pl.col("microprice") - pl.col("mid_price")).alias("price_pressure"),
        )

    # ── Queue imbalance (top level) ────────────────────────────────────────────
    df = df.with_columns(
        (pl.col("bid_q0") / (pl.col("ask_q0") + 1e-9)).alias("queue_imbalance")
    )

    # ── Per-level depth imbalance (top 5 levels) ───────────────────────────────
    for level in range(min(5, lob_levels)):
        bid_col = f"bid_q{level}"
        ask_col = f"ask_q{level}"
        if bid_col in df.columns and ask_col in df.columns:
            df = df.with_columns(
                (
                    (pl.col(bid_col) - pl.col(ask_col))
                    / (pl.col(bid_col) + pl.col(ask_col) + 1e-9)
                ).alias(f"depth_imb_{level}")
            )

    # ── Total bid/ask depth & depth near-touch vs far ─────────────────────────
    bid_q_cols = [f"bid_q{i}" for i in range(lob_levels) if f"bid_q{i}" in df.columns]
    ask_q_cols = [f"ask_q{i}" for i in range(lob_levels) if f"ask_q{i}" in df.columns]
    if bid_q_cols and ask_q_cols:
        df = df.with_columns(
            pl.sum_horizontal(bid_q_cols).alias("total_bid_depth"),
            pl.sum_horizontal(ask_q_cols).alias("total_ask_depth"),
        )
        # Near-touch (levels 0–1) vs far (levels 3–4) depth ratio
        near_bid = [f"bid_q{i}" for i in range(2) if f"bid_q{i}" in df.columns]
        far_bid  = [f"bid_q{i}" for i in range(3, 5) if f"bid_q{i}" in df.columns]
        near_ask = [f"ask_q{i}" for i in range(2) if f"ask_q{i}" in df.columns]
        far_ask  = [f"ask_q{i}" for i in range(3, 5) if f"ask_q{i}" in df.columns]
        if near_bid and far_bid:
            df = df.with_columns(
                (
                    pl.sum_horizontal(near_bid) /
                    (pl.sum_horizontal(far_bid) + 1e-9)
                ).alias("depth_near_far_bid"),
                (
                    pl.sum_horizontal(near_ask) /
                    (pl.sum_horizontal(far_ask) + 1e-9)
                ).alias("depth_near_far_ask"),
            )
        # Level concentration: max queue / total depth
        if bid_q_cols:
            df = df.with_columns(
                (
                    pl.max_horizontal(bid_q_cols) /
                    (pl.sum_horizontal(bid_q_cols) + 1e-9)
                ).alias("bid_level_concentration"),
            )

    # ── Book shape: bid/ask slope & price gaps ─────────────────────────────────
    # Slope: (q0 - q4) / 5 captures how steeply depth falls away from touch
    if "bid_q0" in df.columns and f"bid_q{min(4, lob_levels-1)}" in df.columns:
        far_lvl = min(4, lob_levels - 1)
        df = df.with_columns(
            ((pl.col("bid_q0") - pl.col(f"bid_q{far_lvl}")) / (far_lvl + 1)).alias("bid_slope"),
            ((pl.col("ask_q0") - pl.col(f"ask_q{far_lvl}")) / (far_lvl + 1)).alias("ask_slope"),
        )
    # Price gaps: distance between consecutive best levels
    if "bid_p0" in df.columns and "bid_p1" in df.columns:
        df = df.with_columns(
            (pl.col("bid_p0") - pl.col("bid_p1")).alias("bid_price_gap"),
            (pl.col("ask_p1") - pl.col("ask_p0")).alias("ask_price_gap"),
        )

    # ── Depth ratio rolling mean (baseline for depletion shock) ───────────────
    if "depth_ratio" in df.columns:
        df = df.with_columns(
            pl.col("depth_ratio").rolling_mean(window_size=20).alias("depth_ratio_ma_20")
        )

    # ── Order flow imbalance (Cont, Kukanov & Stoikov 2014) ────────────────────
    # OFI_t = dV^B_t - dV^A_t with sign-adjusted deltas
    # Proper implementation: when best price rises, entire old queue is consumed.
    # Store per-tick raw OFI first so multiple windows can be derived from it.
    df = df.with_columns([
        pl.col("bid_p0").diff().alias("_dbid_p"),
        pl.col("ask_p0").diff().alias("_dask_p"),
        pl.col("bid_q0").diff().alias("_dbid_q"),
        pl.col("ask_q0").diff().alias("_dask_q"),
        pl.col("bid_q0").shift(1).alias("_bid_q0_prev"),
        pl.col("ask_q0").shift(1).alias("_ask_q0_prev"),
    ])
    df = df.with_columns(
        (
            pl.when(pl.col("_dbid_p") > 0)
              .then(pl.col("_bid_q0_prev"))        # price rose: old queue consumed
              .when(pl.col("_dbid_p") == 0)
              .then(pl.col("_dbid_q"))             # price same: delta
              .otherwise(-pl.col("_bid_q0_prev"))  # price fell: queue depleted
            - pl.when(pl.col("_dask_p") < 0)
              .then(pl.col("_ask_q0_prev"))        # price fell: old queue consumed
              .when(pl.col("_dask_p") == 0)
              .then(pl.col("_dask_q"))             # price same: delta
              .otherwise(pl.col("_ask_q0_prev"))   # price rose: queue depleted
        ).alias("_ofi_tick")
    )
    df = df.drop(["_dbid_p", "_dask_p", "_dbid_q", "_dask_q", "_bid_q0_prev", "_ask_q0_prev"])

    # Primary OFI window (configurable, default 20 ticks)
    df = df.with_columns(
        pl.col("_ofi_tick").rolling_mean(window_size=ofi_window).alias(f"ofi_{ofi_window}")
    )

    # ── Multi-window OFI (1s, 3s, 5s) ─────────────────────────────────────────
    # Short-window OFI captures faster order-flow dynamics for shock prediction.
    for w_s in [1, 3, 5]:
        w_ticks = max(1, int(round(w_s * ticks_per_second)))
        alias = f"ofi_{w_s}s"
        if w_ticks != ofi_window:
            df = df.with_columns(
                pl.col("_ofi_tick").rolling_mean(window_size=w_ticks).alias(alias)
            )
        else:
            # ofi_window happens to equal this horizon's tick count — alias it
            df = df.with_columns(pl.col(f"ofi_{ofi_window}").alias(alias))

    df = df.drop("_ofi_tick")

    # ── Rolling stats for imbalance & OFI (Section G) ─────────────────────────
    # std/min/max capture dispersion and extremes that means miss
    for window in [10, 30]:
        df = df.with_columns(
            pl.col("depth_imb_0").rolling_std(window_size=window).alias(f"imb_std_{window}"),
            pl.col("depth_imb_0").rolling_min(window_size=window).alias(f"imb_min_{window}"),
            pl.col("depth_imb_0").rolling_max(window_size=window).alias(f"imb_max_{window}"),
        )
    df = df.with_columns(
        pl.col(f"ofi_{ofi_window}").rolling_std(window_size=20).alias("ofi_std_20"),
        pl.col(f"ofi_{ofi_window}").rolling_min(window_size=20).alias("ofi_min_20"),
        pl.col(f"ofi_{ofi_window}").rolling_max(window_size=20).alias("ofi_max_20"),
    )

    # ── Trade intensity ────────────────────────────────────────────────────────
    df = df.with_columns(
        pl.col("log_return").rolling_sum(window_size=50).abs().alias("trade_intensity_50")
    )

    # ── HAR features ──────────────────────────────────────────────────────────
    # Use the longest RV window as the "daily" HAR component
    base_rv_col = f"rv_{max(rv_windows)}"
    for lag in har_lags:
        df = df.with_columns(
            pl.col(base_rv_col).shift(lag).alias(f"har_rv_{lag}")
        )

    # ── Squared return (GARCH-like lagged feature) ─────────────────────────────
    df = df.with_columns(
        (pl.col("log_return") ** 2).alias("squared_return")
    )

    # ── Temporal features (for TCN/Transformer) ───────────────────────────────────────
    # These capture temporal dynamics and trends that RNNs can exploit.

    # Lagged returns: captures momentum at different timescales
    for lag in [1, 5, 10, 20]:
        df = df.with_columns(
            pl.col("log_return").shift(lag).alias(f"lag_return_{lag}")
        )

    # Rolling return mean (drift): short and long-term trends
    for window in [10, 30, 50]:
        df = df.with_columns(
            pl.col("log_return").rolling_mean(window_size=window).alias(f"return_mean_{window}")
        )

    # Rolling spread trend: how spread is evolving
    for window in [10, 20, 50]:
        df = df.with_columns(
            pl.col("spread").rolling_mean(window_size=window).alias(f"spread_trend_{window}")
        )

    # RV momentum: how volatility is changing
    df = df.with_columns(
        (pl.col("rv_50") - pl.col("rv_50").shift(10)).alias("rv_momentum_10"),
        (pl.col("rv_50") - pl.col("rv_50").shift(20)).alias("rv_momentum_20"),
    )

    # Price velocity: rate of price change
    for window in [5, 10, 20]:
        df = df.with_columns(
            (pl.col("mid_price") / pl.col("mid_price").shift(window) - 1).alias(f"price_velocity_{window}")
        )

    # Multi-horizon forward targets ──────────────────────────────────────────
    # NOTE: Thresholds are computed on the FULL series here. In the training pipeline,
    # these MUST be recomputed on the training slice only to avoid lookahead bias.
    # See dataset.py:make_splits for the correct implementation.
    spread_threshold = float(df["spread"].quantile(0.75))
    depth_threshold  = (
        float(df["depth_ratio"].quantile(0.25))
        if "depth_ratio" in df.columns else None
    )

    for h_s in horizons:
        h_ticks = max(1, int(round(h_s * ticks_per_second)))

        # RV target: forward realized vol over next h_ticks
        df = df.with_columns(
            (pl.col("log_return") ** 2)
            .rolling_sum(window_size=h_ticks)
            .clip(lower_bound=0.0)
            .sqrt()
            .shift(-h_ticks)
            .alias(f"target_rv_{h_s}s")
        )

        # Spread shock: spread exceeds Q75 at any point in the forward window.
        # Forward rolling max: shift by -(h_ticks-1) then rolling_max(h_ticks).
        df = df.with_columns(
            (
                pl.col("spread")
                .shift(-(h_ticks - 1))
                .rolling_max(window_size=h_ticks)
                > spread_threshold
            ).cast(pl.Int32).alias(f"target_shock_spread_{h_s}s")
        )

        # Depth depletion shock: depth_ratio drops below Q25 in the forward window.
        if "depth_ratio" in df.columns and depth_threshold is not None:
            df = df.with_columns(
                (
                    pl.col("depth_ratio")
                    .shift(-(h_ticks - 1))
                    .rolling_min(window_size=h_ticks)
                    < depth_threshold
                ).cast(pl.Int32).alias(f"target_shock_depth_{h_s}s")
            )

    # ── Legacy single-horizon targets (backward compat) ───────────────────────
    # These use the 5s horizon if available, else 50-tick fallback.
    if "target_rv_5s" in df.columns:
        df = df.with_columns(pl.col("target_rv_5s").alias("target_rv"))
        df = df.with_columns(pl.col("target_shock_spread_5s").alias("target_shock"))
    else:
        df = df.with_columns(
            (pl.col("log_return") ** 2)
            .rolling_sum(window_size=50)
            .clip(lower_bound=0.0)
            .sqrt()
            .shift(-50)
            .alias("target_rv")
        )
        spread_threshold_legacy = float(df["spread"].quantile(0.75))
        df = df.with_columns(
            (pl.col("spread") > spread_threshold_legacy)
            .cast(pl.Int32)
            .shift(-50)
            .alias("target_shock")
        )

    return df


def feature_cols(
    df: pl.DataFrame,
    har_lags: list[int] | None = None,
    lob_levels: int = 10,
) -> list[str]:
    """
    Dynamically return list of feature columns present in the dataframe.

    Parameters
    ----------
    df : pl.DataFrame
    har_lags : list[int]
        Expected HAR lags (for checking availability)
    lob_levels : int
        Number of LOB levels (for depth imbalance columns)

    Returns
    -------
    list[str] of feature column names
    """
    if har_lags is None:
        har_lags = [50]

    candidates = [
        # Core microstructure
        "queue_imbalance",
        "ofi_20",
        "ofi_1s",
        "ofi_3s",
        "ofi_5s",
        "rv_50",
        "rv_300",
        *[f"har_rv_{lag}" for lag in har_lags],
        # Spread
        "spread",
        "spread_ma_50",
        "spread_zscore_50",
        "relative_spread",
        # Spread rolling trends
        "spread_trend_10",
        "spread_trend_20",
        "spread_trend_50",
        # Microprice
        "microprice",
        "price_pressure",
        # Depth & book shape
        "depth_ratio",
        "depth_ratio_ma_20",
        "total_bid_depth",
        "total_ask_depth",
        "depth_near_far_bid",
        "depth_near_far_ask",
        "bid_level_concentration",
        "bid_slope",
        "ask_slope",
        "bid_price_gap",
        "ask_price_gap",
        *[f"depth_imb_{i}" for i in range(min(5, lob_levels))],
        # Imbalance rolling stats
        "imb_std_10",
        "imb_min_10",
        "imb_max_10",
        "imb_std_30",
        "imb_min_30",
        "imb_max_30",
        # OFI rolling stats
        "ofi_std_20",
        "ofi_min_20",
        "ofi_max_20",
        # Intensity & volatility proxies
        "trade_intensity_50",
        "squared_return",
        "rv_momentum_10",
        "rv_momentum_20",
        # Lag returns
        "lag_return_1",
        "lag_return_5",
        "lag_return_10",
        "lag_return_20",
        # Return rolling means
        "return_mean_10",
        "return_mean_30",
        "return_mean_50",
        # Price velocity
        "price_velocity_5",
        "price_velocity_10",
        "price_velocity_20",
    ]
    return [c for c in candidates if c in df.columns]


def clean(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """
    Clean features by replacing infinities with None and dropping null rows.

    Parameters
    ----------
    df : pl.DataFrame
    cols : list[str]
        Columns to clean (infinities → None, then rows with any null dropped)

    Returns
    -------
    pl.DataFrame
    """
    float_cols = [c for c in cols if c in df.columns and df[c].dtype in (pl.Float32, pl.Float64)]
    other_cols  = [c for c in cols if c in df.columns and df[c].dtype not in (pl.Float32, pl.Float64)]

    # For float columns: replace both inf and NaN with null
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
          .then(None)
          .otherwise(pl.col(c))
          .alias(c)
        for c in float_cols
    ])
    # For non-float columns: replace inf only (NaN not applicable)
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in other_cols
        if hasattr(pl.col(c), "is_infinite")
    ])
    return df.drop_nulls(subset=[c for c in cols if c in df.columns])
