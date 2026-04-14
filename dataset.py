"""
Dataset creation and splitting for train/validation/test.

Implements chronological splits (no shuffling) to prevent lookahead bias.
Supports multiple forecast horizons with per-horizon RV, spread shock,
and depth depletion shock targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import polars as pl


@dataclass
class Splits:
    """
    Container for chronological train/val/test splits.

    Multi-horizon targets are stored as nested dicts:
        y_rv["{h}s"]["train" | "val" | "test"]
        y_shock_spread["{h}s"]["train" | "val" | "test"]
        y_shock_depth["{h}s"]["train" | "val" | "test"]

    Legacy single-horizon arrays (y_rv_train etc.) are kept for backward
    compatibility and point to the 5s horizon when available, else 1s.
    """
    X_train: np.ndarray
    X_val:   np.ndarray
    X_test:  np.ndarray

    # Multi-horizon targets
    y_rv:           dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    y_shock_spread: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    y_shock_depth:  dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    # Feature column names (stored for ablation)
    feature_cols: list[str] = field(default_factory=list)

    # Index boundaries for recovering the test Polars frame
    test_slice: tuple[int, int] = (0, 0)   # (val_end, n)

    # ── Legacy convenience properties ─────────────────────────────────────────
    @property
    def _primary_horizon(self) -> str:
        """Return the 5s key if present, else the first available key."""
        if "5s" in self.y_rv:
            return "5s"
        keys = list(self.y_rv.keys())
        return keys[0] if keys else "5s"

    @property
    def y_rv_train(self) -> np.ndarray:
        return self.y_rv[self._primary_horizon]["train"]

    @property
    def y_rv_val(self) -> np.ndarray:
        return self.y_rv[self._primary_horizon]["val"]

    @property
    def y_rv_test(self) -> np.ndarray:
        return self.y_rv[self._primary_horizon]["test"]

    @property
    def y_shock_train(self) -> np.ndarray:
        return self.y_shock_spread[self._primary_horizon]["train"]

    @property
    def y_shock_val(self) -> np.ndarray:
        return self.y_shock_spread[self._primary_horizon]["val"]

    @property
    def y_shock_test(self) -> np.ndarray:
        return self.y_shock_spread[self._primary_horizon]["test"]


def make_splits(
    feat_df: pl.DataFrame,
    feature_cols: list[str],
    horizons: list[int] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Splits:
    """
    Create chronological train/val/test splits for multiple forecast horizons.

    Parameters
    ----------
    feat_df : pl.DataFrame
        Feature dataframe with target columns for each horizon.
    feature_cols : list[str]
        List of feature column names.
    horizons : list[int]
        Forecast horizons in seconds (e.g. [1, 5, 30]). Expects columns
        target_rv_{h}s, target_shock_spread_{h}s, target_shock_depth_{h}s.
    train_frac : float
        Fraction for training (default 0.70).
    val_frac : float
        Fraction for validation (default 0.15).

    Returns
    -------
    Splits object with X arrays and per-horizon y dicts.
    """
    if horizons is None:
        horizons = [1, 5, 30]

    n = len(feat_df)
    train_end = int(n * train_frac)
    val_end   = train_end + int(n * val_frac)

    # Feature matrix — contiguous for stride_tricks in sequence building
    X = np.ascontiguousarray(feat_df.select(feature_cols).to_numpy())
    X_train = X[:train_end].copy()
    X_val   = X[train_end:val_end].copy()
    X_test  = X[val_end:].copy()

    y_rv:           dict[str, dict[str, np.ndarray]] = {}
    y_shock_spread: dict[str, dict[str, np.ndarray]] = {}
    y_shock_depth:  dict[str, dict[str, np.ndarray]] = {}

    def _split(arr: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "train": arr[:train_end].copy(),
            "val":   arr[train_end:val_end].copy(),
            "test":  arr[val_end:].copy(),
        }

    for h_s in horizons:
        key = f"{h_s}s"

        rv_col = f"target_rv_{h_s}s"
        if rv_col in feat_df.columns:
            y_rv[key] = _split(
                np.ascontiguousarray(feat_df[rv_col].to_numpy().ravel())
            )

        ss_col = f"target_shock_spread_{h_s}s"
        if ss_col in feat_df.columns:
            y_shock_spread[key] = _split(
                np.ascontiguousarray(
                    feat_df[ss_col].to_numpy().ravel().astype(np.int64)
                )
            )

        sd_col = f"target_shock_depth_{h_s}s"
        if sd_col in feat_df.columns:
            y_shock_depth[key] = _split(
                np.ascontiguousarray(
                    feat_df[sd_col].to_numpy().ravel().astype(np.int64)
                )
            )

    return Splits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_rv=y_rv,
        y_shock_spread=y_shock_spread,
        y_shock_depth=y_shock_depth,
        feature_cols=feature_cols,
        test_slice=(val_end, n),
    )
