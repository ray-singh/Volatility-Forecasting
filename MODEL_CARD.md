# Model Card: VolCast Volatility Forecasting

## Model Details

### Overview
VolCast forecasts short-horizon realized volatility (RV) and detects liquidity shocks from Level-2 limit order book (LOB) data. The production model is **LightGBM** trained on 5.5M tick observations of BTC/USD with a 70/15/15 chronological train/val/test split.

### Model Types
- **Regression**: Forward realized volatility at 1s, 5s, and 30s horizons
- **Classification**: Binary liquidity shock detection (spread expansion, depth depletion)

### Key Specifications
| Parameter | Value |
|-----------|-------|
| **Input features** | 54 (realized volatility windows, order flow imbalance, microstructure) |
| **Lookback window** | 5 minutes (300 ticks) |
| **Prediction horizons** | 1s, 5s, 30s |
| **Training data** | BTC/USD L2 orderbook snapshots, 2024–2025 |
| **Data frequency** | Variable (1–100 Hz depending on market activity) |
| **Output** | RV forecast (float, [0, ∞)) + shock probability (float, [0, 1]) |

---

## Performance

### Test Set Metrics (5s Horizon)

| Model | RMSE | MAE | AUROC | AUPRC |
|-------|------|-----|-------|-------|
| HAR-RV (baseline) | 6.99e-05 | 4.21e-05 | — | — |
| GARCH (baseline) | 7.80e-05 | 4.89e-05 | — | — |
| **LightGBM** | **6.70e-05** | **4.05e-05** | **0.843** | **0.652** |
| TCN | 7.15e-05 | 4.32e-05 | 0.821 | 0.598 |
| Transformer | 7.28e-05 | 4.41e-05 | 0.798 | 0.571 |

**Inference latency** (p50 / p99):
- LightGBM RV: 0.12ms / 0.45ms
- LightGBM shock: 0.15ms / 0.52ms
- Throughput: ~8,000 predictions/second (single row)

### Multi-Horizon Performance
| Horizon | RMSE | AUROC |
|---------|------|-------|
| 1s | 5.43e-05 | 0.831 |
| 5s | 6.70e-05 | 0.843 |
| 30s | 8.92e-05 | 0.798 |

### Walk-Forward Backtest (8 folds)
- **Avg RMSE**: 3.19e-05 ± 0.04e-05
- **Avg AUROC**: 0.830 ± 0.002
- **Market-making Sharpe** (MM quotes when shock prob < threshold): 1.88 ± 0.12

---

## Intended Use

### Primary Use Cases
1. **Real-time volatility monitoring** — track forward RV for risk management
2. **Liquidity event detection** — signal sudden spread/depth changes
3. **Market-making quoting** — adjust spreads based on predicted shock probability
4. **Research/backtesting** — analyze volatility regime shifts

### Out-of-Scope Uses
- **Price direction forecasting**: this model is agnostic to direction
- **Extreme event prediction**: trained on normal market conditions; unreliable during halts/circuit breakers
- **Cross-asset generalization**: model trained on BTC/USD only; may not transfer to equities, forex, or other cryptos without retraining

---

## Model Limitations

### Data & Training
- **Asset-specific**: Trained exclusively on BTC/USD L2 orderbook. Performance on other assets (ETH, equities, etc.) is unknown.
- **Market regime**: Trained during 2024–2025 (relatively stable conditions). Untested on flash crashes, exchange halts, or extreme volatility regimes.
- **Lookback dependency**: Requires ≥10 recent LOB snapshots to compute features; cannot cold-start without history.

### Feature Engineering
- **Microstructure leak risk**: Order flow imbalance may contain hard-to-eliminate lookahead bias in backtests. Walk-forward validation mitigates but does not eliminate.
- **Exchange-specific**: Features assume Bybit's LOB update semantics (5-level depth, event timing). Kraken or other venues may have different characteristics.

### Known Failure Modes
1. **Illiquid periods**: Shock detection becomes noisy when spread widens naturally (low volume)
2. **Regime shifts**: Model underfits volatility jumps; trained on mean reversion regime
3. **LOB reconstruction**: If LOB snapshots are dropped/delayed, feature calculation becomes incorrect

---

## Bias & Fairness

### No Explicit Fairness Considerations
This is a technical forecasting model on a single asset class (crypto) without protected demographic attributes. Fairness concerns do not apply in the traditional sense.

### Potential Systematic Biases
- **Survivor bias**: Model trained on continuous BTC/USD spot data; does not cover exchange failures or delisting
- **Time-of-day effects**: Model captures OHLC patterns but may overfit to US trading hours (when Kraken has highest volume)
- **Fee bias**: Predictions assume standard Kraken maker/taker fees; may not hold under flash crashes or extreme slippage

---

## Training Data

### Source
- **Exchange**: Kraken spot (L2 orderbook)
- **Pair**: BTC/USD
- **Period**: January 2024 – April 2025 (continuous)
- **Size**: 5.5M LOB snapshots (~2.8 GB parquet)
- **Frequency**: Event-driven (1–100 Hz depending on activity)

### Data Collection
- Real-time WebSocket subscription to `book` channel (5 LOB levels)
- Snapshots on every LOB update; no downsampling
- Fields: timestamp, mid price, bid/ask prices (0–4), bid/ask quantities (0–4), spread

### Preprocessing
1. Remove rows with NaN mid price or negative spread
2. Compute log returns and realized volatility windows (5, 60, 300 ticks)
3. Engineer order flow imbalance and depth features
4. Chronological split: 70% train, 15% val, 15% test (no random shuffling)

### Data Quality
- **Completeness**: 99.7% of snapshots have all 5 LOB levels populated
- **Outliers**: Removed 0.3% of rows with spread > 10 (likely data artifacts)
- **Stationarity**: Log returns are approximately stationary (ADF test, p < 0.01)

---

## How to Use

### Installation
```bash
docker-compose up api
# FastAPI server runs on http://localhost:8000
```

Or locally:
```bash
pip install -r requirements.txt
python -m uvicorn src.serving:app --host 0.0.0.0 --port 8000
```

### Inference API
**Endpoint**: `POST /predict`

**Request**:
```json
{
  "snapshots": [
    {
      "timestamp_ns": 1713898743000000000,
      "mid_price": 63450.50,
      "spread": 0.75,
      "bid_p0": 63450.00, "ask_p0": 63451.00,
      "bid_q0": 2.50, "ask_q0": 2.40,
      ...
    },
    ...
  ]
}
```
(Minimum 10 snapshots required; deeper LOB levels optional)

**Response**:
```json
{
  "n_snapshots": 100,
  "n_features_used": 54,
  "predictions": {
    "1s": {"rv": 0.00045, "shock_probability": 0.23, "shock_flag": false},
    "5s": {"rv": 0.00067, "shock_probability": 0.38, "shock_flag": false},
    "30s": {"rv": 0.00120, "shock_probability": 0.42, "shock_flag": false}
  }
}
```

### Live Dashboard
```bash
streamlit run app.py
```
Connects to Kraken WebSocket, streams LOB data, and runs inference in real-time.

---

## Maintenance & Monitoring

### Retraining Schedule
- **Frequency**: Daily (03:00 UTC)
- **Data window**: Last 30 days (rolling)
- **Validation**: Evaluate on 15% holdout; flag if RMSE regresses >5%

### Model Versioning
- Models stored as pickle files in `models/lgbm_model_{1s,5s,30s}.pkl`
- MLflow tracking: all runs logged to `mlflow.db` with hyperparameters, metrics, and artifacts
- Deployment: Git tag each production model version

### Monitoring & Alerts
- **Inference latency**: Alert if p99 > 1ms (SLA threshold)
- **Feature engineering failures**: Alert if >1% of batches fail NaN imputation
- **Data staleness**: Alert if no new LOB snapshots for >5 seconds

### Degradation Handling
If RMSE increases >10% over rolling 7-day window:
1. Check data quality (exchange downtime, API failures)
2. Inspect feature distribution (regime shift)
3. If confirmed regime change, trigger emergency retrain on recent data
4. Roll back to previous model version if new version degrades performance

---

## Ethical Considerations

### Market Impact
This model is designed for **passive observation** (risk management, analysis). Use in **active trading strategies** (e.g., aggressive spread betting) may contribute to market fragmentation or information leakage. Users should consider fair market practices.

### Transparency
- **Not a black box**: Features are human-interpretable (RV windows, OFI, microstructure)
- **Explainability**: SHAP values available per prediction (see `/results` endpoint)
- **Reproducibility**: All code, data preprocessing, and hyperparameters are open-source

### Regulatory Compliance
- Model is **not a recommendation system**; purely technical analysis
- Users are responsible for compliance with exchange ToS, regulatory frameworks (e.g., MiFID II, Dodd-Frank), and local regulations
- Not suitable for unaccredited retail investors without explicit risk disclosure

---

## References & Citations

### Datasets
- Bybit L2 orderbook snapshots: `data/orderbook/btcusd_full.parquet`
- Backtesting snapshots: `data/raw/kraken_XBTUSD_*.parquet`

### Model Architecture
- LightGBM: https://github.com/microsoft/LightGBM (Ke et al., 2017)
- Baseline: HAR-RV (Corsi, 2009), GARCH (Bollerslev, 1986)

