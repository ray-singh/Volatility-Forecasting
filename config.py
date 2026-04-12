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
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_seq_len: int = 50
    lstm_epochs: int = 30
    lstm_batch: int = 128


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
