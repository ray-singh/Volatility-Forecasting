# Volatility & Liquidity Forecasting from Limit Order Book Data

An end-to-end market microstructure research pipeline for forecasting short-horizon realized volatility and liquidity shocks using Level-2 (L2) limit order book data. Built on 5.5M rows of BTC/USD tick data with 5 LOB levels.

## What it does

- **Regression**: forecasts forward realized volatility at 1s, 5s, and 30s horizons
- **Classification**: detects liquidity shocks — bid-ask spread expansions and depth depletions
- **Live feed**: streams real-time LOB snapshots from Kraken/Bybit WebSocket APIs
- **Dashboard**: Streamlit app with feature analysis, model comparison, and live inference

## Models

| Model | Type | Notes |
|---|---|---|
| HAR-RV | Econometric baseline | Heterogeneous autoregressive realized volatility |
| GARCH | Econometric baseline | Volatility clustering baseline |
| LightGBM | ML | Regression (RV) + classification (shock) |
| TCN | Deep learning | Temporal Convolutional Network, PyTorch |
| Transformer | Deep learning | Self-attention sequence model, PyTorch |

## Results (5s horizon)

| Model | RMSE | AUROC |
|---|---|---|
| HAR-RV | 6.99e-05 | — |
| GARCH | 7.80e-05 | — |
| LightGBM | 6.70e-05 | 0.843 |

*TCN and Transformer metrics populated after running `evaluate_deep.py`.*

## Project structure

```
├── config.py           # PipelineConfig (horizons, fractions, LOB levels)
├── engineer.py         # Feature engineering (RV, OFI, spread, depth, HAR lags)
├── dataset.py          # Train/val/test splits (70/15/15 chronological)
├── models.py           # LightGBM wrapper + DataLoader
├── deep_models.py      # TCN and Transformer (PyTorch)
├── train.py            # Full training pipeline + MLflow logging
├── evaluate_deep.py    # Post-hoc evaluation of saved deep model pkl files
├── app.py              # Streamlit dashboard
├── kraken_feed.py      # Kraken WebSocket live feed
├── bybit.py            # Bybit WebSocket live feed
├── make_parquet.py     # Convert raw CSVs to parquet
├── convert_data.py     # Data format utilities
├── models/             # Saved model pkl files
├── data/orderbook/     # Training data (btcusd_full.parquet, 5.5M rows)
└── results.json        # Evaluation metrics (patched by train.py / evaluate_deep.py)
```

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train all models
python train.py --source parquet --path data/orderbook/btcusd_full.parquet

# Evaluate saved deep models without retraining
python evaluate_deep.py

# Launch dashboard
streamlit run app.py
```

## Nightly retraining

A launchd plist (`com.volcast.retrain.plist`) schedules nightly retraining at 03:00:

```bash
cp com.volcast.retrain.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.volcast.retrain.plist
# Logs: tail -f /tmp/volcast_retrain.log
```

## Stack

- **Data**: Polars, PyArrow
- **ML**: LightGBM, PyTorch (MPS/CUDA/CPU)
- **Tracking**: MLflow
- **Dashboard**: Streamlit
- **Live data**: Kraken & Bybit WebSocket APIs
