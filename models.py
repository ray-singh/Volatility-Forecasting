"""
Model implementations for volatility forecasting.

Includes:
- DataLoader: multi-source data loading (parquet, Kraken)
- HARRVModel: HAR-RV baseline (OLS on rolling RV lags)
- LGBMDualModel: LightGBM regression + classification
- DualHeadLSTM: shared LSTM encoder with RV regression + shock classification heads
- diebold_mariano: forecast comparison test
"""
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
# torch must be imported before lightgbm — both ship libomp and the one
# loaded first wins; importing lgb first causes a segfault during torch
# backward() on macOS.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import polars as pl
from statsmodels.regression.linear_model import OLS
import lightgbm as lgb


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


def _make_sequences(
    X: np.ndarray,
    y_rv: np.ndarray,
    y_shock: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window of length seq_len over X/y arrays using stride tricks —
    zero-copy view, no Python loop.

    Returns
    -------
    X_seq : (N - seq_len, seq_len, F)
    y_rv_seq : (N - seq_len,)  -- target at the last timestep of each window
    y_shock_seq : (N - seq_len,)
    """
    n, f = X.shape
    # as_strided view: shape (n-seq_len, seq_len, f), no data copied
    shape = (n - seq_len, seq_len, f)
    strides = (X.strides[0], X.strides[0], X.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides).copy()
    indices = np.arange(seq_len, n)
    print(f"Built {len(X_seq)} sequences of length {seq_len}", flush=True)
    print(f"X_seq shape: {X_seq.shape}, y_rv_seq shape: {y_rv[indices].shape}, "
          f"y_shock_seq shape: {y_shock[indices].shape}", flush=True)
    return X_seq, np.ascontiguousarray(y_rv[indices]), np.ascontiguousarray(y_shock[indices])


class _LSTMNet(nn.Module):
    """Internal PyTorch module — use DualHeadLSTM for the sklearn-style API."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.rv_head = nn.Linear(hidden_size, 1)
        self.shock_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]          # (B, hidden)
        rv_pred = self.rv_head(last_hidden)        # (B, 1)
        shock_pred = self.shock_head(last_hidden)  # (B, 2)
        return rv_pred, shock_pred


class DualHeadLSTM:
    """
    Dual-head LSTM: shared LSTM encoder with separate RV regression and
    shock classification heads.

    Follows an sklearn-style fit/predict API. Internally handles:
    - Feature standardisation (StandardScaler fitted on train only)
    - Sliding-window sequence construction
    - Combined MSE + cross-entropy loss training loop
    - Validation-based early stopping
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 50,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-3,
        rv_loss_weight: float = 1.0,
        shock_loss_weight: float = 1.0,
        patience: int = 5,
        device: str | None = None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.rv_loss_weight = rv_loss_weight
        self.shock_loss_weight = shock_loss_weight
        self.patience = patience
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # MPS has known deadlocks with LSTM on some PyTorch versions
            self.device = torch.device("cpu")
        self.scaler = StandardScaler()
        self.net: _LSTMNet | None = None
        self.train_history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_rv_train: np.ndarray,
        y_shock_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_rv_val: np.ndarray | None = None,
        y_shock_val: np.ndarray | None = None,
    ) -> DualHeadLSTM:
        """
        Fit the LSTM.

        Parameters
        ----------
        X_train : (N, F)
        y_rv_train : (N,)  continuous RV targets
        y_shock_train : (N,)  binary shock targets
        X_val, y_rv_val, y_shock_val : optional validation arrays for early stopping

        Returns
        -------
        self
        """
        # Scale features — fit only on train
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        print("Finished scaling features", flush=True)

        # Build sequences
        X_seq, y_rv_seq, y_shock_seq = _make_sequences(
            X_train_s, y_rv_train, y_shock_train, self.seq_len
        )
        if X_seq.shape[0] == 0:
            print("[LSTM] Not enough data to build sequences, skipping", flush=True)
            return self
        else:
            print("Finished building sequences", flush=True)

        input_size = X_seq.shape[2]
        self.net = _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
        print(f"[LSTM] Moving model to {self.device} (MPS first-run compiles Metal shaders, may take ~30s)...")
        self.net.to(self.device)
        print(f"[LSTM] Model ready on {self.device}. Starting training...")

        # Build val sequences if provided
        val_loader = None
        if X_val_s is not None and y_rv_val is not None and y_shock_val is not None:
            Xv_seq, yv_rv_seq, yv_shock_seq = _make_sequences(
                X_val_s, y_rv_val, y_shock_val, self.seq_len
            )
            if len(Xv_seq) > 0:
                val_loader = self._make_loader(Xv_seq, yv_rv_seq, yv_shock_seq, shuffle=False)

        train_loader = self._make_loader(X_seq, y_rv_seq, y_shock_seq, shuffle=True)
        print("Finished creating data loaders", flush=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        rv_criterion = nn.MSELoss()
        shock_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            print(f"[LSTM] Starting epoch {epoch}/{self.epochs}...", flush=True)
            train_loss = self._run_epoch(
                train_loader, optimizer, rv_criterion, shock_criterion, train=True
            )
            log = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_loss = self._run_epoch(
                    val_loader, None, rv_criterion, shock_criterion, train=False
                )
                log["val_loss"] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                print(f"[LSTM] Epoch {epoch}/{self.epochs} — "
                      f"train={train_loss:.4f}  val={val_loss:.4f}")

                if epochs_no_improve >= self.patience:
                    print(f"[LSTM] Early stopping at epoch {epoch}")
                    break
            else:
                print(f"[LSTM] Epoch {epoch}/{self.epochs} — train={train_loss:.4f}")

            self.train_history.append(log)

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        return self

    def predict_rv(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RV for each row of X.

        The first seq_len - 1 rows cannot form a full sequence and are filled
        with the prediction of the first valid window.
        """
        if self.net is None:
            return np.zeros(len(X))
        X_s = self.scaler.transform(X)
        rv_preds, _ = self._predict_raw(X_s)
        return rv_preds

    def predict_shock_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict shock probabilities, shape (N, 2).

        Rows with insufficient history are filled with the first valid prediction.
        """
        if self.net is None:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        X_s = self.scaler.transform(X)
        _, shock_proba = self._predict_raw(X_s)
        return shock_proba

    def predict_shock(self, X: np.ndarray) -> np.ndarray:
        """Predict shock class (0 or 1)."""
        return self.predict_shock_proba(X)[:, 1] > 0.5

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        X_seq: np.ndarray,
        y_rv: np.ndarray,
        y_shock: np.ndarray,
        shuffle: bool,
    ) -> TorchDataLoader:
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_rv, dtype=torch.float32),
            torch.tensor(y_shock, dtype=torch.int64),
        )
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def _run_epoch(
        self,
        loader: TorchDataLoader,
        optimizer,
        rv_criterion,
        shock_criterion,
        train: bool,
    ) -> float:
        self.net.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for X_batch, y_rv_batch, y_shock_batch in loader:
                X_batch = X_batch.to(self.device)
                y_rv_batch = y_rv_batch.to(self.device)
                y_shock_batch = y_shock_batch.to(self.device)

                rv_pred, shock_pred = self.net(X_batch)
                rv_loss = rv_criterion(rv_pred.squeeze(1), y_rv_batch)
                shock_loss = shock_criterion(shock_pred, y_shock_batch)
                loss = self.rv_loss_weight * rv_loss + self.shock_loss_weight * shock_loss

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    def _predict_raw(
        self, X_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference over scaled X_s, returning rv_preds (N,) and
        shock_proba (N, 2). Rows with insufficient history are padded
        with the first valid prediction.
        """
        n = len(X_s)
        # Build sequences for all valid positions
        dummy_y = np.zeros(n)
        X_seq, _, _ = _make_sequences(X_s, dummy_y, dummy_y, self.seq_len)

        self.net.eval()
        rv_out = []
        shock_out = []
        with torch.no_grad():
            for start in range(0, len(X_seq), self.batch_size):
                batch = torch.from_numpy(X_seq[start:start + self.batch_size]).float().to(self.device)
                rv_pred, shock_pred = self.net(batch)
                rv_out.append(rv_pred.squeeze(1).cpu().numpy())
                shock_out.append(torch.softmax(shock_pred, dim=1).cpu().numpy())

        rv_preds_valid = np.concatenate(rv_out)       # (n - seq_len,)
        shock_proba_valid = np.concatenate(shock_out)  # (n - seq_len, 2)

        # Pad the first seq_len rows with the first valid prediction
        rv_preds = np.empty(n)
        rv_preds[:self.seq_len] = rv_preds_valid[0]
        rv_preds[self.seq_len:] = rv_preds_valid

        shock_proba = np.empty((n, 2))
        shock_proba[:self.seq_len] = shock_proba_valid[0]
        shock_proba[self.seq_len:] = shock_proba_valid

        return rv_preds, shock_proba


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
