"""
Dataset creation and splitting for train/validation/test.

Implements chronological splits (no shuffling) to prevent lookahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import polars as pl


@dataclass
class Splits:
    """Container for train/val/test splits with X and y arrays."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_rv_train: np.ndarray
    y_rv_val: np.ndarray
    y_rv_test: np.ndarray
    y_shock_train: np.ndarray
    y_shock_val: np.ndarray
    y_shock_test: np.ndarray


def make_splits(
    feat_df: pl.DataFrame,
    feature_cols: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Splits:
    """
    Create chronological train/val/test splits.

    Parameters
    ----------
    feat_df : pl.DataFrame
        Feature dataframe with target columns
    feature_cols : list[str]
        List of feature column names
    train_frac : float
        Fraction of data for training (default 0.70)
    val_frac : float
        Fraction of data for validation (default 0.15)

    Returns
    -------
    Splits object with X and y arrays
    """
    n = len(feat_df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # Extract features and targets
    X = feat_df.select(feature_cols).to_numpy()
    y_rv = feat_df.select("target_rv").to_numpy().ravel()
    y_shock = feat_df.select("target_shock").to_numpy().ravel()

    # Chronological split
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_rv_train, y_rv_val, y_rv_test = y_rv[:train_end], y_rv[train_end:val_end], y_rv[val_end:]
    y_shock_train, y_shock_val, y_shock_test = y_shock[:train_end], y_shock[train_end:val_end], y_shock[val_end:]

    return Splits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_rv_train=y_rv_train,
        y_rv_val=y_rv_val,
        y_rv_test=y_rv_test,
        y_shock_train=y_shock_train,
        y_shock_val=y_shock_val,
        y_shock_test=y_shock_test,
    )
