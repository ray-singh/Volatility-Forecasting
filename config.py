"""
Configuration dataclasses for the volatility forecasting pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data source and I/O configuration."""
    source: str = "kraken"  # "kraken" or "parquet"
    symbol: str = "XBTUSD"
    lob_levels: int = 10
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    train_frac: float = 0.70
    val_frac: float = 0.15


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    rv_windows: list[int] = field(default_factory=lambda: [50, 300])
    har_lags: list[int] = field(default_factory=lambda: [50])
    ofi_window: int = 20
    # Forecast horizons in seconds; at 0.5s poll interval, 1s ≈ 2 ticks
    horizons: list[int] = field(default_factory=lambda: [1, 5, 30])
    ticks_per_second: float = 2.0
    # Feature groups for ablation analysis
    feature_groups: dict = field(default_factory=lambda: {
        "rv":        ["rv_50", "rv_300"],
        "ofi":       ["ofi_20"],
        "spread":    ["spread", "spread_ma_50", "spread_zscore_50"],
        "depth":     ["queue_imbalance", "depth_ratio",
                      "depth_imb_0", "depth_imb_1", "depth_imb_2",
                      "depth_imb_3", "depth_imb_4"],
        "har":       ["har_rv_50"],
        "intensity": ["trade_intensity_50"],
    })


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
    })
    # LSTM
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_seq_len: int = 50
    lstm_epochs: int = 15
    lstm_batch: int = 256
    lstm_rv_loss_weight: float = 1.0
    lstm_patience: int = 10
    # GARCH
    garch_p: int = 1
    garch_q: int = 1
    garch_vol: str = "Garch"    # arch_model vol= argument
    garch_dist: str = "normal"
    # TCN
    tcn_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    tcn_seq_len: int = 50
    tcn_epochs: int = 15
    tcn_batch: int = 256


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
