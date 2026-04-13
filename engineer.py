"""
Feature engineering for volatility forecasting.

Builds microstructure features from LOB snapshots including realized volatility,
order flow imbalance, queue imbalance, depth ratios, and HAR features.
"""
from __future__ import annotations

import polars as pl
import numpy as np
from arch import arch_model


def build_features(
    df: pl.DataFrame,
    rv_windows: list[int] = None,
    ofi_window: int = 20,
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

    Returns
    -------
    pl.DataFrame with original columns plus feature columns
    """
    if rv_windows is None:
        rv_windows = [50, 300]

    # Log returns (more appropriate than pct_change for RV)
    df = df.with_columns(
        (pl.col("mid_price") / pl.col("mid_price").shift(1)).log().alias("log_return")
    )

    # Realized volatility: sqrt of sum of squared log returns over window
    # RV_t = sqrt(sum_{i=t-w+1}^{t} r_i^2)  -- Andersen & Bollerslev (1998)
    # rolling_std is wrong here: it demeans and uses (n-1), biasing RV downward
    for window in rv_windows:
        df = df.with_columns(
            (pl.col("log_return") ** 2)
            .rolling_sum(window_size=window)
            .sqrt()
            .alias(f"rv_{window}")
        )

    # Bid-ask spread features
    df = df.with_columns(
        pl.col("spread").rolling_mean(window_size=50).alias("spread_ma_50"),
        (
            (pl.col("spread") - pl.col("spread").rolling_mean(50))
            / pl.col("spread").rolling_std(50).fill_null(1e-9)
        ).alias("spread_zscore_50"),
    )

    # Queue imbalance: normalized ratio of bid vs ask volume at top level
    df = df.with_columns(
        (pl.col("bid_q0") / (pl.col("ask_q0") + 1e-9)).alias("queue_imbalance")
    )

    # Order flow imbalance (Cont, Kukanov & Stoikov 2014)
    # OFI_t = dV^B_t - dV^A_t where deltas are sign-adjusted for price moves:
    #   bid contribution: +delta_bid_q if bid price unchanged/higher, else -bid_q
    #   ask contribution: +delta_ask_q if ask price unchanged/lower, else -ask_q
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

    # Trade intensity: mid-price changes per window
    df = df.with_columns(
        pl.col("log_return").rolling_sum(window_size=50).abs().alias("trade_intensity_50")
    )

    # HAR (Heterogeneous Autoregressive) features
    # HAR-RV: one lag of RV across multiple horizons
    for lag in [50]:  # Daily (at 1-sec freq, 50 ticks ≈ ~50 sec)
        df = df.with_columns(
            pl.col(f"rv_300").shift(lag).alias(f"har_rv_{lag}")
        )

    # Volatility regime (GARCH-like): lagged squared returns
    df = df.with_columns(
        (pl.col("log_return") ** 2).alias("squared_return")
    )

    # Target: realized volatility (next 50 ticks)
    # Shift -50 so the target is the RV of the *next* 50 ticks
    df = df.with_columns(
        (pl.col("log_return") ** 2)
        .rolling_sum(window_size=50)
        .sqrt()
        .shift(-50)
        .alias("target_rv")
    )

    # Target: liquidity shock (spread expansion beyond threshold)
    spread_threshold = df["spread"].quantile(0.75)
    df = df.with_columns(
        (pl.col("spread") > spread_threshold).cast(pl.Int32).shift(-50).alias("target_shock")
    )

    return df


def feature_cols(df: pl.DataFrame, har_lags: list[int] | None = None) -> list[str]:
    """
    Dynamically return list of feature columns present in dataframe.

    Parameters
    ----------
    df : pl.DataFrame
    har_lags : list[int]
        Expected HAR lags (for checking availability)

    Returns
    -------
    list[str] of feature column names
    """
    if har_lags is None:
        har_lags = [50]

    candidates = [
        "queue_imbalance",
        "ofi_20",
        "rv_50",
        "rv_300",
        *[f"har_rv_{lag}" for lag in har_lags],
        "spread",
        "spread_ma_50",
        "spread_zscore_50",
        "trade_intensity_50",
        "depth_ratio",
    ]
    return [c for c in candidates if c in df.columns]


def clean(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """
    Clean features by removing infinite and null values.

    Parameters
    ----------
    df : pl.DataFrame
    cols : list[str]
        Columns to clean

    Returns
    -------
    pl.DataFrame with infinities replaced by None in specified columns, then nulls dropped
    """
    # Replace infinities with None
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in cols
        if c in df.columns
    ])
    # Drop rows with any nulls in target columns
    return df.drop_nulls(subset=cols)
