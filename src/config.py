"""
Configuration dataclasses for the volatility forecasting pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data source and I/O configuration."""
    source: str = "kraken"  # "kraken", "csv", or "parquet"
    symbol: str = "XBTUSD"
    lob_levels: int = 10
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    csv_dir: Path = field(default_factory=lambda: Path("data/csv"))
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
        "rv":        ["rv_50", "rv_300", "rv_momentum_10", "rv_momentum_20"],
        "ofi":       ["ofi_20", "ofi_1s", "ofi_3s", "ofi_5s",
                      "ofi_std_20", "ofi_min_20", "ofi_max_20"],
        "spread":    ["spread", "spread_ma_50", "spread_zscore_50", "relative_spread",
                      "spread_trend_10", "spread_trend_20", "spread_trend_50"],
        "depth":     ["queue_imbalance", "depth_ratio", "depth_ratio_ma_20",
                      "total_bid_depth", "total_ask_depth",
                      "depth_near_far_bid", "depth_near_far_ask",
                      "bid_level_concentration",
                      "depth_imb_0", "depth_imb_1", "depth_imb_2",
                      "depth_imb_3", "depth_imb_4",
                      "imb_std_10", "imb_min_10", "imb_max_10",
                      "imb_std_30", "imb_min_30", "imb_max_30"],
        "book_shape": ["bid_slope", "ask_slope", "bid_price_gap", "ask_price_gap"],
        "microprice": ["microprice", "price_pressure"],
        "har":        ["har_rv_50"],
        "intensity":  ["trade_intensity_50", "squared_return"],
        "lags":       ["lag_return_1", "lag_return_5", "lag_return_10", "lag_return_20",
                       "return_mean_10", "return_mean_30", "return_mean_50",
                       "price_velocity_5", "price_velocity_10", "price_velocity_20"],
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
    # GARCH
    garch_p: int = 1
    garch_q: int = 1
    garch_vol: str = "Garch"    # arch_model vol= argument
    garch_dist: str = "normal"
    # TCN (expanded dilation stack: receptive field ≈ 125 ticks at kernel=3)
    tcn_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 128, 128])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    tcn_seq_len: int = 200
    tcn_epochs: int = 15
    tcn_batch: int = 256
    # Transformer
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    transformer_seq_len: int = 100
    transformer_epochs: int = 20
    transformer_batch: int = 256


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
