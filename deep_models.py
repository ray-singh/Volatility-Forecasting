"""
Deep sequence models for volatility forecasting.

Implements:
- _SequenceDataset      : lazy sliding-window dataset
- _SequenceModel        : shared training loop (TCN, Transformer)
- _get_device           : best-available device selection
- TCNModel              : dilated causal TCN with expanded receptive field
- TransformerModel      : encoder-only Transformer with RV + shock heads
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _get_device(device_str: str | None = None) -> torch.device:
    """Select best available device: explicit > CUDA > MPS > CPU."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



# ══════════════════════════════════════════════════════════════════════════════
# Shared dataset
# ══════════════════════════════════════════════════════════════════════════════

class _SequenceDataset(torch.utils.data.Dataset):
    """
    Lazy sliding-window dataset. Sequences are sliced on-the-fly so the full
    (N, seq_len, F) tensor is never materialized in RAM.

    Parameters
    ----------
    X       : (N, F) float32 array (pre-scaled)
    y_rv    : (N,) float32
    y_shock : (N,) int64
    seq_len : int
    """

    def __init__(self, X: np.ndarray, y_rv: np.ndarray, y_shock: np.ndarray, seq_len: int):
        self.X       = X
        self.y_rv    = y_rv
        self.y_shock = y_shock
        self.seq_len = seq_len
        self.n       = len(X) - seq_len

    def __len__(self) -> int:
        return max(self.n, 0)

    def __getitem__(self, idx: int):
        x_seq   = torch.from_numpy(
            np.ascontiguousarray(self.X[idx: idx + self.seq_len])
        ).float()
        y_rv    = torch.tensor(self.y_rv[idx + self.seq_len],    dtype=torch.float32)
        y_shock = torch.tensor(self.y_shock[idx + self.seq_len], dtype=torch.int64)
        return x_seq, y_rv, y_shock


# ══════════════════════════════════════════════════════════════════════════════
# Shared training loop
# ══════════════════════════════════════════════════════════════════════════════

class _SequenceModel:
    """
    Shared infrastructure for sequence models (TCN, Transformer).

    Subclasses must set:
        self.net        : nn.Module with forward(x) -> (rv_pred, shock_logits)
        self.seq_len    : int
        self.batch_size : int
        self.device     : torch.device
        self.scaler     : StandardScaler
    """

    def _make_loader(
        self,
        X_s: np.ndarray,
        y_rv: np.ndarray,
        y_shock: np.ndarray,
        shuffle: bool,
    ) -> TorchDataLoader:
        dataset = _SequenceDataset(X_s, y_rv, y_shock, self.seq_len)
        return TorchDataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=False,
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
        The first seq_len rows (no full history) are filled with the first
        valid prediction.
        """
        n = len(X_s)
        dummy_y = np.zeros(n)
        dataset = _SequenceDataset(X_s, dummy_y, dummy_y, self.seq_len)
        loader  = TorchDataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        self.net.eval()
        rv_out    = []
        shock_out = []
        with torch.no_grad():
            for X_batch, _, _ in loader:
                X_batch = X_batch.to(self.device)
                rv_pred, shock_pred = self.net(X_batch)
                rv_out.append(rv_pred.squeeze(1).cpu().numpy())
                shock_out.append(torch.softmax(shock_pred, dim=1).cpu().numpy())

        rv_preds_valid    = np.concatenate(rv_out)
        shock_proba_valid = np.concatenate(shock_out)

        # Invert log-scaling applied during training
        rv_log_shift = getattr(self, "_rv_log_shift", 0.0)
        rv_preds_valid = np.exp(rv_preds_valid + rv_log_shift)

        rv_preds = np.empty(n)
        rv_preds[:self.seq_len] = rv_preds_valid[0]
        rv_preds[self.seq_len:] = rv_preds_valid

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
        """Shared training loop with early stopping and best-weights restore."""
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val) if X_val is not None else None

        # Log-scale RV targets so MSE loss operates on ~O(1) values instead of ~1e-10.
        # Store the scale shift so predict_rv can invert it.
        rv_eps = 1e-8
        self._rv_log_shift = float(np.log(np.clip(y_rv_train, rv_eps, None)).mean())
        y_rv_train_s = np.log(np.clip(y_rv_train, rv_eps, None)) - self._rv_log_shift
        y_rv_val_s   = (np.log(np.clip(y_rv_val,   rv_eps, None)) - self._rv_log_shift
                        if y_rv_val is not None else None)

        if len(X_train_s) <= self.seq_len:
            print(f"[{model_tag}] Not enough data to build sequences, skipping")
            return

        self.net.to(self.device)
        print(f"[{model_tag}] Device={self.device}  params="
              f"{sum(p.numel() for p in self.net.parameters()):,}  Starting training...")

        val_loader = None
        if X_val_s is not None and y_rv_val_s is not None and y_shock_val is not None:
            if len(X_val_s) > self.seq_len:
                val_loader = self._make_loader(X_val_s, y_rv_val_s, y_shock_val, shuffle=False)

        train_loader    = self._make_loader(X_train_s, y_rv_train_s, y_shock_train, shuffle=True)
        optimizer       = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        rv_criterion    = nn.MSELoss()
        shock_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        epochs_no_imp = 0
        best_state    = None
        self.train_history: list[dict] = []

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
                print(f"[{model_tag}] Epoch {epoch:>3}/{epochs} — "
                      f"train={train_loss:.5f}  val={val_loss:.5f}")

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
                print(f"[{model_tag}] Epoch {epoch:>3}/{epochs} — train={train_loss:.5f}")

            self.train_history.append(log)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()


# ══════════════════════════════════════════════════════════════════════════════
# TCN  (expanded receptive field via deeper dilation stack)
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
        out = self.conv1(x)
        out = out[..., :-self._pad] if self._pad > 0 else out
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = out[..., :-self._pad] if self._pad > 0 else out
        out = self.relu(out)
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
        self.network        = nn.Sequential(*layers)
        self.rv_head_linear = nn.Linear(channels[-1], 1)
        self.shock_head     = nn.Sequential(
            nn.Linear(channels[-1], 16), nn.ReLU(), nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out  = self.network(x.transpose(1, 2))   # (B, T, F) → (B, C, T)
        last = out[:, :, -1]                      # (B, C)
        rv_pred = self.rv_head_linear(last)                       # log-RV (B, 1)
        return rv_pred, self.shock_head(last)

    @staticmethod
    def receptive_field(channels: list[int], kernel_size: int) -> int:
        """Compute the receptive field in ticks for logging."""
        return 1 + 2 * (kernel_size - 1) * sum(2 ** i for i in range(len(channels)))


class TCNModel(_SequenceModel):
    """
    Temporal Convolutional Network with dual heads (RV regression + shock
    classification).

    Default channels [32, 64, 128, 128, 128] with kernel_size=3 gives a
    receptive field of 1 + 2*2*(1+2+4+8+16) = 125 ticks (~62s at 2 ticks/s),
    covering the full 30s prediction horizon.

    Same sklearn-style API as TransformerModel.
    """

    def __init__(
        self,
        channels: list[int] | None = None,
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
        # 5-layer dilation stack: receptive field = 125 ticks ≈ 62s at 2 ticks/s
        self.channels          = channels or [32, 64, 128, 128, 128]
        self.kernel_size       = kernel_size
        self.dropout           = dropout
        self.seq_len           = seq_len
        self.epochs            = epochs
        self.batch_size        = batch_size
        self.lr                = lr
        self.rv_loss_weight    = rv_loss_weight
        self.shock_loss_weight = shock_loss_weight
        self.patience          = patience
        self.device            = _get_device(device)
        self.scaler            = StandardScaler()
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
        self.net   = _TCNNet(input_size, self.channels, self.kernel_size, self.dropout)
        rf = _TCNNet.receptive_field(self.channels, self.kernel_size)
        print(f"[TCN] channels={self.channels}  kernel={self.kernel_size}  "
              f"receptive_field={rf} ticks")
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
        return np.clip(self._predict_raw(self.scaler.transform(X))[0], 0, None)

    def predict_shock_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        return self._predict_raw(self.scaler.transform(X))[1]

    def predict_shock(self, X: np.ndarray) -> np.ndarray:
        return self.predict_shock_proba(X)[:, 1] > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Transformer
# ══════════════════════════════════════════════════════════════════════════════

class _TransformerNet(nn.Module):
    """
    Encoder-only Transformer for sequence-to-point prediction.

    Architecture:
      - Linear input projection (F → d_model)
      - Learnable positional encoding
      - N × TransformerEncoderLayer (pre-norm, causal mask)
      - CLS-token aggregation (first position)
      - RV regression head + shock classification head

    Causal masking ensures no future information leaks into the prediction.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        seq_len: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        # Learnable positional encoding — more flexible than sinusoidal for
        # non-uniform tick rates and short sequences
        self.pos_emb    = nn.Embedding(seq_len + 1, d_model)  # +1 for CLS token
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # RV head predicts log-RV (targets are log-scaled in _fit_loop); no
        # positive activation needed since _predict_raw applies exp() to invert.
        self.rv_head_linear = nn.Linear(d_model, 1)
        self.shock_head     = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2)
        )

        self._seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        B, T, _ = x.shape

        # Project input features to model dimension
        x = self.input_proj(x)                                    # (B, T, d_model)

        # Add positional embeddings to input sequence (positions 0..T-1)
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_emb(positions)                           # (B, T, d_model)

        # Append CLS token at the end so it attends to the full sequence
        # Position T is reserved for CLS in pos_emb (Embedding has seq_len+1 rows)
        cls_pos = torch.tensor([T], device=x.device)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_emb(cls_pos)  # (B, 1, d_model)
        x   = torch.cat([x, cls], dim=1)                          # (B, T+1, d_model)

        # Causal mask: position i cannot attend to position j > i.
        # CLS is at T, so it attends to all T prior positions (correct).
        mask = torch.triu(
            torch.ones(T + 1, T + 1, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Encode
        out = self.encoder(x, mask=mask, is_causal=True)          # (B, T+1, d_model)

        # Read off CLS token (last position) for prediction
        cls_out = out[:, -1, :]                                    # (B, d_model)

        rv_pred = self.rv_head_linear(cls_out)                    # log-RV (B, 1)
        return rv_pred, self.shock_head(cls_out)


class TransformerModel(_SequenceModel):
    """
    Encoder-only Transformer with dual heads (RV regression + shock
    classification).

    Captures long-range dependencies via self-attention. Causal masking
    prevents lookahead. Attention complexity is O(T²) but T is small here (≤200).

    Sklearn-style fit/predict API, identical to TCNModel.

    Parameters
    ----------
    d_model : int
        Attention dimension (must be divisible by nhead). Default 64.
    nhead : int
        Number of attention heads. Default 4.
    num_layers : int
        Number of TransformerEncoderLayer blocks. Default 2.
    dim_feedforward : int
        FFN hidden dimension inside each block. Default 256.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 100,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 5e-4,
        rv_loss_weight: float = 1.0,
        shock_loss_weight: float = 1.0,
        patience: int = 10,
        device: str | None = None,
    ):
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model           = d_model
        self.nhead             = nhead
        self.num_layers        = num_layers
        self.dim_feedforward   = dim_feedforward
        self.dropout           = dropout
        self.seq_len           = seq_len
        self.epochs            = epochs
        self.batch_size        = batch_size
        self.lr                = lr
        self.rv_loss_weight    = rv_loss_weight
        self.shock_loss_weight = shock_loss_weight
        self.patience          = patience
        self.device            = _get_device(device)
        self.scaler            = StandardScaler()
        self.net: _TransformerNet | None = None
        self.train_history: list[dict] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_rv_train: np.ndarray,
        y_shock_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_rv_val: np.ndarray | None = None,
        y_shock_val: np.ndarray | None = None,
    ) -> TransformerModel:
        input_size = X_train.shape[1]
        self.net = _TransformerNet(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            seq_len=self.seq_len,
        )
        self._fit_loop(
            X_train, y_rv_train, y_shock_train,
            X_val, y_rv_val, y_shock_val,
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            rv_loss_weight=self.rv_loss_weight,
            shock_loss_weight=self.shock_loss_weight,
            model_tag="Transformer",
        )
        return self

    def predict_rv(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.zeros(len(X))
        return np.clip(self._predict_raw(self.scaler.transform(X))[0], 0, None)

    def predict_shock_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        return self._predict_raw(self.scaler.transform(X))[1]

    def predict_shock(self, X: np.ndarray) -> np.ndarray:
        return self.predict_shock_proba(X)[:, 1] > 0.5
