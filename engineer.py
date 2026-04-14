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
    for window in rv_windows:
        df = df.with_columns(
            (pl.col("log_return") ** 2)
            .rolling_sum(window_size=window)
            .sqrt()
            .alias(f"rv_{window}")
        )

    # ── Bid-ask spread features ────────────────────────────────────────────────
    df = df.with_columns(
        pl.col("spread").rolling_mean(window_size=50).alias("spread_ma_50"),
        (
            (pl.col("spread") - pl.col("spread").rolling_mean(50))
            / pl.col("spread").rolling_std(50).fill_null(1e-9)
        ).alias("spread_zscore_50"),
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

    # ── Depth ratio rolling mean (baseline for depletion shock) ───────────────
    if "depth_ratio" in df.columns:
        df = df.with_columns(
            pl.col("depth_ratio").rolling_mean(window_size=20).alias("depth_ratio_ma_20")
        )

    # ── Order flow imbalance (Cont, Kukanov & Stoikov 2014) ────────────────────
    # OFI_t = dV^B_t - dV^A_t with sign-adjusted deltas
    df = df.with_columns([
        pl.col("bid_p0").diff().alias("_dbid_p"),
        pl.col("ask_p0").diff().alias("_dask_p"),
        pl.col("bid_q0").diff().alias("_dbid_q"),
        pl.col("ask_q0").diff().alias("_dask_q"),
    ])
    df = df.with_columns(
        (
            pl.when(pl.col("_dbid_p") >= 0)
              .then(pl.col("_dbid_q"))
              .otherwise(-pl.col("bid_q0"))
            - pl.when(pl.col("_dask_p") <= 0)
              .then(pl.col("_dask_q"))
              .otherwise(pl.col("ask_q0"))
        )
        .rolling_mean(window_size=ofi_window)
        .alias(f"ofi_{ofi_window}")
    )
    df = df.drop(["_dbid_p", "_dask_p", "_dbid_q", "_dask_q"])

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

    # ── Multi-horizon forward targets ──────────────────────────────────────────
    # Compute spread and depth thresholds once (Q75 / Q25 on full series)
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
        "queue_imbalance",
        f"ofi_20",
        "rv_50",
        "rv_300",
        *[f"har_rv_{lag}" for lag in har_lags],
        "spread",
        "spread_ma_50",
        "spread_zscore_50",
        "trade_intensity_50",
        "depth_ratio",
        *[f"depth_imb_{i}" for i in range(min(5, lob_levels))],
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
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in cols
        if c in df.columns
    ])
    return df.drop_nulls(subset=cols)
