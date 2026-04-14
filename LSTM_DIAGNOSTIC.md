# LSTM Hyperparameter Diagnostic

## Purpose
This script tests four LSTM configurations to identify whether your LSTM underperformance is due to hyperparameters (seq_len, hidden_size, layers) or architectural limitations.

## Configurations Tested

| Config | seq_len | hidden | layers | epochs | batch | Note |
|--------|---------|--------|--------|--------|-------|------|
| **Baseline** | 50 | 64 | 2 | 15 | 256 | Factory defaults |
| **Large-SeqLen** | 100 | 128 | 2 | 20 | 128 | Doubled seq + hidden |
| **XLarge** | 150 | 256 | 2 | 25 | 64 | Max seq, largest hidden |
| **Deep** | 100 | 256 | 3 | 20 | 128 | 3-layer LSTM |

## Who Wins?

**If LightGBM still wins decisively:**
- Your raw features are rich enough that LSTM's sequential modeling adds overhead
- Volatility patterns may be memoryless (Efficient Market Hypothesis)
- Consider LightGBM as primary model

**If larger configs beat Baseline but lose to LightGBM:**
- Sequence length matters, but LSTM struggles with dual-task learning
- Consider: weight the RV loss higher (rv_loss_weight > 1.0)
- Or: train separate LSTM models (one for RV, one for shocks)

**If XLarge or Deep wins:**
- Hyperparameters were the bottleneck
- Recommend updating config.py with new defaults

## Running

```bash
python test_lstm_configs.py [--data-path data/raw/your_data.parquet]
```

Results are saved to `lstm_comparison.json` for further analysis.

## Key Metrics to Watch

- **RMSE**: Lower is better (vs LightGBM baseline)
- **Epochs Trained**: Indicates how much early stopping kicked in
- **Train Loss Improvement %**: High values = good convergence
- **Val Loss Pattern**: Diverging train/val = overfitting

## Quick Interpretation

```json
{
  "lgbm_baseline": {
    "rv_metrics": {"rmse": 0.123456, "mae": 0.098765},
    "shock_metrics": {"auroc": 0.8234, "auprc": 0.7654}
  },
  "lstm_results": {
    "Baseline": {...},
    "Large-SeqLen": {...}
  }
}
```

Check if any LSTM config has:
1. ✅ Lower RMSE than LGBM → LSTM can work, tune further
2. ✅ Higher AUROC than LGBM → LSTM better at shock detection
3. ✅ Fast convergence (few epochs, high loss improvement) → Stable training
