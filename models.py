"""
Model implementations for volatility forecasting.

Includes:
- DataLoader: multi-source data loading (parquet, Kraken)
- HARRVModel: HAR-RV baseline (OLS on rolling RV lags)
- GARCHModel: GARCH(p,q) conditional volatility baseline
- LGBMDualModel: LightGBM regression + classification
- _SequenceModel: shared base for sequence models (LSTM, TCN)
- DualHeadLSTM: shared LSTM encoder with RV regression + shock classification heads
- TCNModel: Temporal Convolutional Network with dual heads
- diebold_mariano: forecast comparison test
"""
from __future__ import annotations

import numpy as np
# torch must be imported before lightgbm — both ship libomp and the one
# loaded first wins; importing lgb first causes a segfault during torch
# backward() on macOS.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import polars as pl
from statsmodels.regression.linear_model import OLS
import lightgbm as lgb


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

class _PositiveOutput(nn.Module):
    """Ensures positive output using scaled Softplus: preserves gradients, smooth, no hard floor."""
    def __init__(self, min_value: float = 1e-5):
        super().__init__()
        self.min_value = min_value
        self.softplus = nn.Softplus(beta=2.0)  # beta controls smoothness
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softplus: log(1 + exp(beta*x)) / beta
        # With beta=2, the transition is smooth and gradients are well-behaved
        return self.softplus(x) + self.min_value


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

    Fit on training log returns, then produce a conditional vol series for
    the test period via an in-sample filter (train+test concatenated, test
    slice extracted). This avoids the prohibitive cost of rolling re-fitting
    while still being informative for comparison.

    Note: This approach involves a minor lookahead for the test parameters;
    document this limitation clearly in results.
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
        self._params = None   # fitted parameter array (train only)

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
        res = am.fit(disp="off", show_warning=False)
        self._params = res.params.values
        return self

    def forecast_test(
        self,
        train_returns: np.ndarray,
        test_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Return conditional volatility for the test period.

        Fits arch_model on the concatenated series and slices the test portion.
        Volatility is unscaled to match the log-return scale of target_rv.

        Parameters
        ----------
        train_returns : np.ndarray
        test_returns  : np.ndarray

        Returns
        -------
        np.ndarray, shape (len(test_returns),)  — conditional sigma
        """
        from arch import arch_model
        all_returns = np.concatenate([train_returns, test_returns]) * 100
        am = arch_model(all_returns, vol=self.vol, p=self.p, q=self.q, dist=self.dist)
        starting = self._params if self._params is not None else None
        res = am.fit(
            starting_values=starting,
            disp="off",
            show_warning=False,
        )
        cond_vol = res.conditional_volatility / 100   # unscale
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
        if X_val is not None and y_rv_val is not None and y_sh_val is not None:
            yv_rv = np.asarray(y_rv_val, dtype=float)
            yv_sh = np.asarray(y_sh_val, dtype=float)
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
# Shared sequence model infrastructure
# ══════════════════════════════════════════════════════════════════════════════

def _get_device(device_str: str | None = None) -> torch.device:
    """
    Select the best available device: explicit > CUDA > MPS > CPU.
    
    On Apple Silicon, MPS (Metal Performance Shaders) provides GPU acceleration.
    On NVIDIA, CUDA is used. Otherwise falls back to CPU.
    """
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_sequences(
    X: np.ndarray,
    seq_len: int,
    *y_arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Slide a window of length seq_len over X using stride tricks (zero-copy).

    Parameters
    ----------
    X : (N, F)
    seq_len : int
    *y_arrays : any number of (N,) arrays

    Returns
    -------
    (X_seq, *y_seq_arrays) where X_seq is (N-seq_len, seq_len, F) and each
    y_seq is (N-seq_len,) aligned to the last timestep of each window.
    """
    n, f = X.shape
    shape   = (n - seq_len, seq_len, f)
    strides = (X.strides[0], X.strides[0], X.strides[1])
    X_seq   = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides).copy()
    indices = np.arange(seq_len, n)
    y_seqs  = tuple(np.ascontiguousarray(y[indices]) for y in y_arrays)
    return (X_seq, *y_seqs)


class _SequenceModel:
    """
    Shared infrastructure for sequence models (LSTM, TCN).

    Subclasses must set:
        self.net      : nn.Module with forward(x) -> (rv_pred, shock_pred)
        self.seq_len  : int
        self.batch_size : int
        self.device   : torch.device
        self.scaler   : StandardScaler
    """

    def _make_loader(
        self,
        X_seq: np.ndarray,
        y_rv: np.ndarray,
        y_shock: np.ndarray,
        shuffle: bool,
    ) -> TorchDataLoader:
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_rv,    dtype=torch.float32),
            torch.tensor(y_shock, dtype=torch.int64),
        )
        return TorchDataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
        )

    def _run_epoch(
        self,
        loader: TorchDataLoader,
        optimizer,
        rv_criterion,
        shock_criterion,
        train: bool,
        rv_loss_weight: float = 1.0,
        shock_loss_weight: float = 1.0,
    ) -> float:
        self.net.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for X_batch, y_rv_batch, y_shock_batch in loader:
                X_batch       = X_batch.to(self.device)
                y_rv_batch    = y_rv_batch.to(self.device)
                y_shock_batch = y_shock_batch.to(self.device)

                rv_pred, shock_pred = self.net(X_batch)
                rv_loss    = rv_criterion(rv_pred.squeeze(1), y_rv_batch)
                shock_loss = shock_criterion(shock_pred, y_shock_batch)
                loss = rv_loss_weight * rv_loss + shock_loss_weight * shock_loss

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    def _predict_raw(self, X_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference over scaled X_s.

        Returns rv_preds (N,) and shock_proba (N, 2).
        Rows with insufficient history are padded with the first valid prediction.
        """
        n = len(X_s)
        dummy_y = np.zeros(n)
        X_seq, _, _ = _make_sequences(X_s, self.seq_len, dummy_y, dummy_y)

        self.net.eval()
        rv_out    = []
        shock_out = []
        with torch.no_grad():
            for start in range(0, len(X_seq), self.batch_size):
                batch = torch.from_numpy(
                    X_seq[start:start + self.batch_size]
                ).float().to(self.device)
                rv_pred, shock_pred = self.net(batch)
                rv_out.append(rv_pred.squeeze(1).cpu().numpy())
                shock_out.append(torch.softmax(shock_pred, dim=1).cpu().numpy())

        rv_preds_valid    = np.concatenate(rv_out)
        shock_proba_valid = np.concatenate(shock_out)

        rv_preds = np.empty(n)
        rv_preds[:self.seq_len]  = rv_preds_valid[0]
        rv_preds[self.seq_len:]  = rv_preds_valid

        shock_proba = np.empty((n, 2))
        shock_proba[:self.seq_len] = shock_proba_valid[0]
        shock_proba[self.seq_len:] = shock_proba_valid

        return rv_preds, shock_proba

    def _fit_loop(
        self,
        X_train: np.ndarray,
        y_rv_train: np.ndarray,
        y_shock_train: np.ndarray,
        X_val: np.ndarray | None,
        y_rv_val: np.ndarray | None,
        y_shock_val: np.ndarray | None,
        epochs: int,
        lr: float,
        patience: int,
        rv_loss_weight: float,
        shock_loss_weight: float,
        model_tag: str = "Model",
    ):
        """Shared training loop with early stopping."""
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val) if X_val is not None else None

        X_seq, y_rv_seq, y_shock_seq = _make_sequences(
            X_train_s, self.seq_len, y_rv_train, y_shock_train
        )
        if X_seq.shape[0] == 0:
            print(f"[{model_tag}] Not enough data to build sequences, skipping")
            return

        self.net.to(self.device)
        print(f"[{model_tag}] Model on {self.device}. Starting training...")

        val_loader = None
        if X_val_s is not None and y_rv_val is not None and y_shock_val is not None:
            Xv_seq, yv_rv, yv_shock = _make_sequences(
                X_val_s, self.seq_len, y_rv_val, y_shock_val
            )
            if len(Xv_seq) > 0:
                val_loader = self._make_loader(Xv_seq, yv_rv, yv_shock, shuffle=False)

        train_loader  = self._make_loader(X_seq, y_rv_seq, y_shock_seq, shuffle=True)
        optimizer     = torch.optim.Adam(self.net.parameters(), lr=lr)
        rv_criterion  = nn.MSELoss()
        shock_criterion = nn.CrossEntropyLoss()

        best_val_loss  = float("inf")
        epochs_no_imp  = 0
        best_state     = None
        self.train_history = []

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(
                train_loader, optimizer, rv_criterion, shock_criterion,
                train=True,
                rv_loss_weight=rv_loss_weight,
                shock_loss_weight=shock_loss_weight,
            )
            log = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_loss = self._run_epoch(
                    val_loader, None, rv_criterion, shock_criterion,
                    train=False,
                    rv_loss_weight=rv_loss_weight,
                    shock_loss_weight=shock_loss_weight,
                )
                log["val_loss"] = val_loss
                print(f"[{model_tag}] Epoch {epoch}/{epochs} — "
                      f"train={train_loss:.4f}  val={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    epochs_no_imp = 0
                else:
                    epochs_no_imp += 1

                if epochs_no_imp >= patience:
                    print(f"[{model_tag}] Early stopping at epoch {epoch}")
                    break
            else:
                print(f"[{model_tag}] Epoch {epoch}/{epochs} — train={train_loss:.4f}")

            self.train_history.append(log)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()


# ══════════════════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════════════════

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.rv_head = nn.Linear(hidden_size, 1)
        self.shock_head = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.rv_head(last), self.shock_head(last)


class DualHeadLSTM(_SequenceModel):
    """
    Dual-head LSTM: shared LSTM encoder with separate RV regression and
    shock classification heads.

    Follows an sklearn-style fit/predict API.
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
        rv_loss_weight: float = 2.0,
        shock_loss_weight: float = 1.0,
        patience: int = 10,
        device: str | None = None,
    ) -> None:
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.dropout         = dropout
        self.seq_len         = seq_len
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.rv_loss_weight  = rv_loss_weight
        self.shock_loss_weight = shock_loss_weight
        self.patience        = patience
        self.device = _get_device(device)
        self.scaler       = StandardScaler()
        self.net: _LSTMNet | None = None
        self.train_history: list[dict] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_rv_train: np.ndarray,
        y_shock_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_rv_val: np.ndarray | None = None,
        y_shock_val: np.ndarray | None = None,
    ) -> DualHeadLSTM:
        input_size = X_train.shape[1]
        self.net = _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
        self._fit_loop(
            X_train, y_rv_train, y_shock_train,
            X_val, y_rv_val, y_shock_val,
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            rv_loss_weight=self.rv_loss_weight,
            shock_loss_weight=self.shock_loss_weight,
            model_tag="LSTM",
        )
        return self

    def predict_rv(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.zeros(len(X))
        raw_preds = self._predict_raw(self.scaler.transform(X))[0]
        # Clip to valid range: ensure non-negative predictions
        return np.clip(raw_preds, 0, None)

    def predict_shock_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        return self._predict_raw(self.scaler.transform(X))[1]

    def predict_shock(self, X: np.ndarray) -> np.ndarray:
        return self.predict_shock_proba(X)[:, 1] > 0.5

    def get_training_loss_summary(self) -> dict | None:
        """
        Return summary statistics of training history.

        Useful for diagnosing early stopping and convergence issues.
        
        Returns
        -------
        dict with keys: epochs_trained, train_loss_initial, train_loss_final,
        train_loss_improved (%), and optionally val_loss_initial, val_loss_final,
        val_loss_improved (%) if validation was used.
        """
        if not self.train_history:
            return None
        
        train_losses = [entry["train_loss"] for entry in self.train_history]
        val_losses = [entry.get("val_loss") for entry in self.train_history if "val_loss" in entry]
        
        summary = {
            "epochs_trained": len(self.train_history),
            "train_loss_initial": train_losses[0] if train_losses else None,
            "train_loss_final": train_losses[-1] if train_losses else None,
            "train_loss_improved": (train_losses[0] - train_losses[-1]) / train_losses[0] * 100 if train_losses and train_losses[0] > 0 else None,
        }
        
        if val_losses:
            summary.update({
                "val_loss_initial": val_losses[0],
                "val_loss_final": val_losses[-1],
                "val_loss_improved": (val_losses[0] - val_losses[-1]) / val_losses[0] * 100,
            })
        
        return summary


# ══════════════════════════════════════════════════════════════════════════════
# TCN
# ══════════════════════════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """Single dilated causal conv block with residual connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self._pad  = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal padding: trim the right side introduced by padding
        out = self.relu(self.conv1(x)[..., :-self._pad] if self._pad else self.conv1(x))
        out = self.drop(out)
        out = self.relu(self.conv2(out)[..., :-self._pad] if self._pad else self.conv2(out))
        out = self.drop(out)
        res = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + res)


class _TCNNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        channels: list[int],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network   = nn.Sequential(*layers)
        self.rv_head   = nn.Linear(channels[-1], 1)
        self.shock_head = nn.Sequential(
            nn.Linear(channels[-1], 16), nn.ReLU(), nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F) — transpose for Conv1d
        out  = self.network(x.transpose(1, 2))   # (B, C, T)
        last = out[:, :, -1]                      # (B, C)
        return self.rv_head(last), self.shock_head(last)


class TCNModel(_SequenceModel):
    """
    Temporal Convolutional Network with dual heads (RV regression + shock classification).

    Same sklearn-style API as DualHeadLSTM; uses dilated causal convolutions
    instead of recurrence, which can train faster on CPU.
    """

    def __init__(
        self,
        channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        seq_len: int = 50,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        rv_loss_weight: float = 1.0,
        shock_loss_weight: float = 1.0,
        patience: int = 10,
        device: str | None = None,
    ):
        self.channels      = channels or [32, 64, 64]
        self.kernel_size   = kernel_size
        self.dropout       = dropout
        self.seq_len       = seq_len
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.rv_loss_weight  = rv_loss_weight
        self.shock_loss_weight = shock_loss_weight
        self.patience      = patience
        self.device = _get_device(device)
        self.scaler        = StandardScaler()
        self.net: _TCNNet | None = None
        self.train_history: list[dict] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_rv_train: np.ndarray,
        y_shock_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_rv_val: np.ndarray | None = None,
        y_shock_val: np.ndarray | None = None,
    ) -> TCNModel:
        input_size = X_train.shape[1]
        self.net = _TCNNet(input_size, self.channels, self.kernel_size, self.dropout)
        self._fit_loop(
            X_train, y_rv_train, y_shock_train,
            X_val, y_rv_val, y_shock_val,
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            rv_loss_weight=self.rv_loss_weight,
            shock_loss_weight=self.shock_loss_weight,
            model_tag="TCN",
        )
        return self

    def predict_rv(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.zeros(len(X))
        raw_preds = self._predict_raw(self.scaler.transform(X))[0]
        # Clip to valid range: ensure non-negative predictions
        return np.clip(raw_preds, 0, None)

    def predict_shock_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        return self._predict_raw(self.scaler.transform(X))[1]

    def predict_shock(self, X: np.ndarray) -> np.ndarray:
        return self.predict_shock_proba(X)[:, 1] > 0.5


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
