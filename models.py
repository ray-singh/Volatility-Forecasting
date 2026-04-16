"""
Model implementations for volatility forecasting.

Includes:
- DataLoader: multi-source data loading (parquet, Kraken)
- HARRVModel: HAR-RV baseline (OLS on rolling RV lags)
- GARCHModel: GARCH(p,q) conditional volatility baseline
- LGBMDualModel: LightGBM regression + classification
- diebold_mariano: forecast comparison test

Deep sequence models (TCN, Transformer) live in deep_models.py.
They are re-exported here for backward compatibility.
"""
from __future__ import annotations

import numpy as np
# torch must be imported before lightgbm — both ship libomp and the one
# loaded first wins; importing lgb first causes a segfault during torch
# backward() on macOS.
import torch
import polars as pl
from statsmodels.regression.linear_model import OLS
import lightgbm as lgb

# Re-export deep models so existing imports from models.py continue to work
from deep_models import (  # noqa: F401
    TCNModel,
    TransformerModel,
)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════


class DataLoader:
    """Multi-source data loader with validation."""

    def __init__(self, source: str, **kwargs):
        self.source = source
        self.kwargs = kwargs

    def load(self, n_rows: int | None = None) -> pl.DataFrame:
        """
        Load data from source with validation.

        Parameters
        ----------
        n_rows : int | None
            Max rows to load. None (default) loads all available rows.

        Returns
        -------
        pl.DataFrame with LOB snapshot schema (validated)
        """
        if self.source == "parquet":
            path = self.kwargs.get("path")
            if not path:
                raise ValueError("'path' kwarg required for parquet source")
            df = pl.read_parquet(path)
            if n_rows is not None:
                df = df.head(n_rows)
        elif self.source == "csv":
            from pathlib import Path as _Path
            csv_dir = _Path(self.kwargs.get("csv_dir", "data/csv"))
            csv_files = sorted(csv_dir.glob("order-book-*.csv"))
            if not csv_files:
                raise ValueError(f"No order-book-*.csv files found in {csv_dir}")
            frames = [pl.read_csv(f, try_parse_dates=True) for f in csv_files]
            df = pl.concat(frames, how="vertical")
            # Normalise column names to match the rest of the pipeline
            if "mid" in df.columns and "mid_price" not in df.columns:
                df = df.rename({"mid": "mid_price"})
            # Sort chronologically
            if "ts" in df.columns:
                df = df.sort("ts")
            if n_rows is not None:
                df = df.head(n_rows)
        elif self.source == "kraken":
            from kraken_feed import collect_kraken_snapshots
            df = collect_kraken_snapshots(
                pair=self.kwargs.get("pair", "XBTUSD"),
                n_snapshots=self.kwargs.get("n_snapshots", n_rows),
                levels=self.kwargs.get("levels", 10),
                poll_interval_s=self.kwargs.get("poll_interval_s", 0.5),
            )
        else:
            raise ValueError(f"Unknown source '{self.source}'. Choose: parquet | csv | kraken")

        # Data validation: check for crossed books, invalid prices/quantities
        df = self._validate_data(df)
        return df

    def _validate_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Validate LOB data: drop rows with crossed books, NaN prices, or invalid quantities.

        Checks:
          - Spread > 0 (bid < ask)
          - Prices are finite and positive
          - Quantities are non-negative
        """
        rows_before = len(df)

        # Check spread > 0 (no crossed books)
        if "bid_p0" in df.columns and "ask_p0" in df.columns:
            df = df.filter(pl.col("spread") > 0)

        # Check only best-level prices are finite and positive (deeper levels may be NaN)
        for col in ["bid_p0", "ask_p0"]:
            if col in df.columns:
                df = df.filter(pl.col(col).is_finite() & (pl.col(col) > 0))

        # Check best-level quantities are non-negative and finite
        for col in ["bid_q0", "ask_q0"]:
            if col in df.columns:
                df = df.filter(pl.col(col).is_finite() & (pl.col(col) >= 0))

        rows_after = len(df)
        if rows_before > rows_after:
            dropped = rows_before - rows_after
            print(f"[DataLoader] Dropped {dropped} invalid rows "
                  f"({dropped / rows_before * 100:.1f}%)")

        return df


# ══════════════════════════════════════════════════════════════════════════════
# Baseline models
# ══════════════════════════════════════════════════════════════════════════════

class HARRVModel:
    """
    HAR-RV baseline: OLS regression on rolling realized volatility lags.

    Corsi (2009) heterogeneous autoregressive model of realized volatility.
    """

    def __init__(self):
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> HARRVModel:
        if X_train.size == 0 or X_train.shape[1] == 0:
            print("[HAR-RV] Insufficient features, skipping")
            return self
        self.model = OLS(y_train, X_train).fit()
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X_test))
        return self.model.predict(X_test)


class GARCHModel:
    """
    GARCH(p,q) conditional volatility baseline using the arch package.

    Fits on training data only. For test-period conditional volatility, uses
    the filter() method with fixed training parameters — no re-estimation on
    test data.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        vol: str = "Garch",
        dist: str = "normal",
    ):
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self._res = None   # fitted arch result (train only)

    def fit(self, log_returns: np.ndarray) -> GARCHModel:
        """
        Fit GARCH on training log returns (scaled to % returns for optimizer stability).

        Parameters
        ----------
        log_returns : np.ndarray, shape (n,)
        """
        from arch import arch_model
        scaled = log_returns * 100
        am = arch_model(scaled, vol=self.vol, p=self.p, q=self.q, dist=self.dist)
        self._res = am.fit(disp="off", show_warning=False)
        return self

    def forecast_test(
        self,
        train_returns: np.ndarray,
        test_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Return conditional volatility for the test period using filter().

        Applies the fitted GARCH parameters (from training) to the test data
        without re-estimation, avoiding lookahead bias.

        Parameters
        ----------
        train_returns : np.ndarray
        test_returns  : np.ndarray

        Returns
        -------
        np.ndarray, shape (len(test_returns),)  — conditional sigma
        """
        if self._res is None:
            raise RuntimeError("GARCH model not fitted. Call fit() first.")

        # Use fix() to apply trained parameters to full series without re-estimation
        from arch import arch_model as _arch_model
        all_returns = np.concatenate([train_returns, test_returns]) * 100
        am2 = _arch_model(all_returns, vol=self.vol, p=self.p, q=self.q, dist=self.dist)
        fixed = am2.fix(self._res.params)
        cond_vol = fixed.conditional_volatility / 100  # unscale
        return cond_vol[len(train_returns):]


# ══════════════════════════════════════════════════════════════════════════════
# LightGBM dual model
# ══════════════════════════════════════════════════════════════════════════════

class LGBMDualModel:
    """
    LightGBM dual-head model: separate regression (RV) and classification (shock) heads.
    """

    def __init__(self, lgbm_params: dict | None = None):
        self.lgbm_params = lgbm_params or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
        }
        self.rv_model = None
        self.shock_model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_rv_train: np.ndarray,
        y_shock_train: np.ndarray,
        X_val: np.ndarray = None,
        y_rv_val: np.ndarray = None,
        y_shock_val: np.ndarray = None,
    ) -> LGBMDualModel:
        # Defensive cleaning: remove rows with NaNs in targets before fitting
        y_rv = np.asarray(y_rv_train, dtype=float)
        y_sh = np.asarray(y_shock_train, dtype=float)
        mask_train = np.isfinite(y_rv) & np.isfinite(y_sh)
        if not mask_train.all():
            n_bad = int(np.count_nonzero(~mask_train))
            print(f"[LGBM] Dropping {n_bad} training rows with NaN targets")
            X_train = X_train[mask_train]
            y_rv = y_rv[mask_train]
            y_sh = y_sh[mask_train]

        X_val_used = None
        y_rv_val_used = None
        y_sh_val_used = None
        if X_val is not None and y_rv_val is not None and y_shock_val is not None:
            yv_rv = np.asarray(y_rv_val, dtype=float)
            yv_sh = np.asarray(y_shock_val, dtype=float)
            mask_val = np.isfinite(yv_rv) & np.isfinite(yv_sh)
            if not mask_val.all():
                n_bad = int(np.count_nonzero(~mask_val))
                print(f"[LGBM] Dropping {n_bad} validation rows with NaN targets")
            # Keep only valid validation rows; if none remain, disable early stopping
            if mask_val.any():
                X_val_used = X_val[mask_val]
                y_rv_val_used = yv_rv[mask_val]
                y_sh_val_used = yv_sh[mask_val]

        if X_train.shape[0] == 0:
            raise ValueError("No training rows remain after dropping NaN targets")

        self.rv_model = lgb.LGBMRegressor(**self.lgbm_params)
        self.rv_model.fit(
            X_train, y_rv,
            eval_set=[(X_val_used, y_rv_val_used)] if X_val_used is not None else None,
            callbacks=[lgb.early_stopping(50, verbose=False)] if X_val_used is not None else [],
        )

        self.shock_model = lgb.LGBMClassifier(**self.lgbm_params)
        self.shock_model.fit(
            X_train, y_sh.astype(int),
            eval_set=[(X_val_used, y_sh_val_used)] if X_val_used is not None else None,
            callbacks=[lgb.early_stopping(50, verbose=False)] if X_val_used is not None else [],
        )
        return self

    def predict_rv(self, X_test: np.ndarray) -> np.ndarray:
        if self.rv_model is None:
            return np.zeros(len(X_test))
        return self.rv_model.predict(X_test)

    def predict_shock(self, X_test: np.ndarray) -> np.ndarray:
        if self.shock_model is None:
            return np.zeros(len(X_test), dtype=int)
        return self.shock_model.predict(X_test)

    def predict_shock_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.shock_model is None:
            return np.column_stack([np.ones(len(X_test)), np.zeros(len(X_test))])
        return self.shock_model.predict_proba(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════════════

def diebold_mariano(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
) -> dict:
    """
    Diebold-Mariano test: compare forecast accuracy of two models.

    H0: equal forecast accuracy.  H1: different accuracy.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred1, y_pred2 : np.ndarray

    Returns
    -------
    dict with statistic, p_value, model1_better, interpretation
    """
    from scipy import stats

    e1 = (y_true - y_pred1) ** 2
    e2 = (y_true - y_pred2) ** 2
    d  = e1 - e2

    mean_d = np.mean(d)
    var_d  = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "statistic":    float(dm_stat),
        "p_value":      float(p_value),
        "model1_better": bool(mean_d < 0),
        "interpretation": (
            "Model 1 significantly better" if p_value < 0.05 and mean_d < 0
            else "Model 2 significantly better" if p_value < 0.05
            else "No significant difference"
        ),
    }
