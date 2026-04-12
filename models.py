"""
Model implementations for volatility forecasting.

Includes:
- DataLoader: multi-source data loading (parquet, Kraken)
- HARRVModel: HAR-RV baseline (OLS on rolling RV lags)
- LGBMDualModel: LightGBM regression + classification
- diebold_mariano: forecast comparison test
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc
import polars as pl
from statsmodels.regression.linear_model import OLS
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset


class DataLoader:
    """Multi-source data loader."""

    def __init__(self, source: str, **kwargs):
        self.source = source
        self.kwargs = kwargs

    def load(self, n_rows: int = 5_000) -> pl.DataFrame:
        """
        Load data from source.

        Parameters
        ----------
        n_rows : int
            Max rows to load

        Returns
        -------
        pl.DataFrame with LOB snapshot schema
        """
        if self.source == "parquet":
            path = self.kwargs.get("path")
            if not path:
                raise ValueError("'path' kwarg required for parquet source")
            df = pl.read_parquet(path)
            return df.head(n_rows)

        if self.source == "kraken":
            from kraken_feed import collect_kraken_snapshots
            return collect_kraken_snapshots(
                pair=self.kwargs.get("pair", "XBTUSD"),
                n_snapshots=self.kwargs.get("n_snapshots", n_rows),
                levels=self.kwargs.get("levels", 10),
                poll_interval_s=self.kwargs.get("poll_interval_s", 0.5),
            )

        raise ValueError(f"Unknown source '{self.source}'. Choose: parquet | kraken")


class HARRVModel:
    """
    HAR-RV baseline: OLS regression on rolling realized volatility lags.

    Corsi (2009) heterogeneous autoregressive model of realized volatility.
    """

    def __init__(self):
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> HARRVModel:
        """
        Fit HAR model using OLS.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            HAR features (usually just lagged RV)
        y_train : np.ndarray, shape (n_samples,)
            Target RV

        Returns
        -------
        self
        """
        if X_train.size == 0 or X_train.shape[1] == 0:
            print("[HAR-RV] Insufficient features for training, skipping")
            return self

        # OLS fit
        self.model = OLS(y_train, X_train).fit()
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict RV."""
        if self.model is None:
            return np.zeros(len(X_test))
        return self.model.predict(X_test)


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
        """
        Fit RV regressor and shock classifier.

        Parameters
        ----------
        X_train : np.ndarray
        y_rv_train : np.ndarray
            RV targets (continuous)
        y_shock_train : np.ndarray
            Shock targets (binary)
        X_val, y_rv_val, y_shock_val : optional
            Validation data for early stopping

        Returns
        -------
        self
        """
        # RV regressor
        self.rv_model = lgb.LGBMRegressor(**self.lgbm_params)
        self.rv_model.fit(
            X_train, y_rv_train,
            eval_set=[(X_val, y_rv_val)] if X_val is not None else None,
            callbacks=[lgb.early_stopping(50)] if X_val is not None else []
        )

        # Shock classifier
        self.shock_model = lgb.LGBMClassifier(**self.lgbm_params)
        self.shock_model.fit(
            X_train, y_shock_train,
            eval_set=[(X_val, y_shock_val)] if X_val is not None else None,
            callbacks=[lgb.early_stopping(50)] if X_val is not None else []
        )

        return self

    def predict_rv(self, X_test: np.ndarray) -> np.ndarray:
        """Predict RV."""
        if self.rv_model is None:
            return np.zeros(len(X_test))
        return self.rv_model.predict(X_test)

    def predict_shock(self, X_test: np.ndarray) -> np.ndarray:
        """Predict shock class."""
        if self.shock_model is None:
            return np.zeros(len(X_test), dtype=int)
        return self.shock_model.predict(X_test)

    def predict_shock_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict shock probability."""
        if self.shock_model is None:
            return np.column_stack([np.ones(len(X_test)), np.zeros(len(X_test))])
        return self.shock_model.predict_proba(X_test)


class DualHeadLSTM(nn.Module):
    """Dual-head LSTM: shared encoder with separate RV and shock heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.rv_head = nn.Linear(hidden_size, 1)
        self.shock_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        rv_pred = self.rv_head(last_hidden)
        shock_pred = self.shock_head(last_hidden)
        return rv_pred, shock_pred


def diebold_mariano(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    horizon: int = 1,
) -> dict:
    """
    Diebold-Mariano test: compare forecast accuracy of two models.

    H0: Model 1 and Model 2 have equal forecast accuracy.
    H1: Models have different accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred1, y_pred2 : np.ndarray
        Predictions from Model 1 and 2
    horizon : int
        Forecast horizon (for loss differential scaling)

    Returns
    -------
    dict with test statistic, p-value, and interpretation
    """
    from scipy import stats

    # Loss differential
    e1 = (y_true - y_pred1) ** 2
    e2 = (y_true - y_pred2) ** 2
    d = e1 - e2

    # Diebold-Mariano statistic
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d))

    # Two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "statistic": float(dm_stat),
        "p_value": float(p_value),
        "model1_better": bool(mean_d < 0),
        "interpretation": "Model 1 significantly better" if p_value < 0.05 and mean_d < 0
                         else "Model 2 significantly better" if p_value < 0.05
                         else "No significant difference"
    }
