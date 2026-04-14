#!/usr/bin/env python
"""
Hyperparameter sweep: test LSTM with different seq_len and hidden_size values.

Compares:
  - Baseline: seq_len=50, hidden=64
  - Config A: seq_len=100, hidden=128
  - Config B: seq_len=150, hidden=256
  - Config C: seq_len=100, hidden=256

Run with:
    python test_lstm_configs.py [--data-path path/to/data.parquet]

Results are printed to stdout and saved to lstm_comparison.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import polars as pl

from config import PipelineConfig
from engineer import build_features, feature_cols, clean
from dataset import make_splits
from models import DataLoader, LGBMDualModel, DualHeadLSTM


def evaluate_rv_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate RV forecast with RMSE, MAE, and QLIKE loss."""
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    
    # Clip predictions to avoid log(0) warnings
    y_pred_clipped = np.clip(y_pred, 1e-6, None)
    y_true_clipped = np.clip(y_true, 1e-6, None)
    
    # QLIKE: robust calculation with safeguards
    ratio = y_true_clipped / y_pred_clipped
    qlike = float(np.mean(ratio - np.log(ratio) - 1))
    
    return {"rmse": rmse, "mae": mae, "qlike": qlike}


def evaluate_shock_forecast(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Evaluate shock classification with AUROC, AUPRC, and Brier score."""
    auroc = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    auprc = float(auc(recall, precision))
    brier = float(np.mean((y_pred_proba[:, 1] - y_true) ** 2))
    return {"auroc": auroc, "auprc": auprc, "brier": brier}


def main(data_path: str | None = None):
    cfg = PipelineConfig()
    horizons = cfg.features.horizons
    tps = cfg.features.ticks_per_second
    lstm_horizon_s = 5

    print("\n" + "=" * 80)
    print("  LSTM Hyperparameter Sweep")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # 1. Load and prepare data
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    loader = DataLoader(
        source='parquet',
        path='data/raw/kraken_XBTUSD_20260413_230215.parquet',
        pair=cfg.data.symbol,
        levels=cfg.data.lob_levels,
        poll_interval_s=0.5,
    )
    raw_df = loader.load(n_rows=5_000)
    print(f"  Rows loaded: {len(raw_df):,}")

    if len(raw_df) < 200:
        raise RuntimeError(f"Only {len(raw_df)} snapshots; need >= 200")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Feature engineering
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2] Feature engineering...")
    feat_df = build_features(
        raw_df,
        rv_windows=cfg.features.rv_windows,
        ofi_window=cfg.features.ofi_window,
        har_lags=cfg.features.har_lags,
        horizons=horizons,
        ticks_per_second=tps,
        lob_levels=cfg.data.lob_levels,
    )
    fcols = feature_cols(feat_df, har_lags=cfg.features.har_lags,
                        lob_levels=cfg.data.lob_levels)
    print(f"  Features: {len(fcols)}")

    # ─────────────────────────────────────────────────────────────────────
    # 3. Clean
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3] Cleaning...")
    target_cols = (
        [f"target_rv_{h}s" for h in horizons] +
        [f"target_shock_spread_{h}s" for h in horizons]
    )
    all_cols = fcols + target_cols
    rows_before = len(feat_df)
    feat_df = clean(feat_df, all_cols)
    rows_after = len(feat_df)
    print(f"  Rows: {rows_before:,} → {rows_after:,} (dropped {rows_before - rows_after})")

    if rows_after < 100:
        raise RuntimeError(f"After cleaning, only {rows_after} rows; need >= 100")

    # ─────────────────────────────────────────────────────────────────────
    # 4. Train / Val / Test splits
    # ─────────────────────────────────────────────────────────────────────
    print("\n[4] Creating splits...")
    splits = make_splits(
        feat_df,
        feature_cols=fcols,
        horizons=horizons,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
    )
    print(f"  Train: {splits.X_train.shape[0]:,}  "
          f"Val: {splits.X_val.shape[0]:,}  "
          f"Test: {splits.X_test.shape[0]:,}")

    # ─────────────────────────────────────────────────────────────────────
    # 5. LightGBM baseline for comparison
    # ─────────────────────────────────────────────────────────────────────
    print("\n[5] LightGBM baseline (for comparison)...")
    lstm_key = f"{lstm_horizon_s}s"
    if lstm_key not in splits.y_rv:
        print(f"  Warning: {lstm_key} not available, using {horizons[0]}s instead")
        lstm_key = f"{horizons[0]}s"

    lgbm = LGBMDualModel(lgbm_params=cfg.models.lgbm_params)
    lgbm.fit(
        splits.X_train,
        splits.y_rv[lstm_key]["train"],
        splits.y_shock_spread[lstm_key]["train"],
        X_val=splits.X_val,
        y_rv_val=splits.y_rv[lstm_key]["val"],
        y_shock_val=splits.y_shock_spread[lstm_key]["val"],
    )
    lgbm_rv_pred = lgbm.predict_rv(splits.X_test)
    lgbm_shock_proba = lgbm.predict_shock_proba(splits.X_test)
    lgbm_rv_metrics = evaluate_rv_forecast(splits.y_rv[lstm_key]["test"], lgbm_rv_pred)
    lgbm_shock_metrics = evaluate_shock_forecast(
        splits.y_shock_spread[lstm_key]["test"], lgbm_shock_proba
    )
    print(f"  LGBM RV:    RMSE={lgbm_rv_metrics['rmse']:.6f}  "
          f"MAE={lgbm_rv_metrics['mae']:.6f}")
    print(f"  LGBM Shock: AUROC={lgbm_shock_metrics['auroc']:.4f}  "
          f"AUPRC={lgbm_shock_metrics['auprc']:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # 6. LSTM configurations
    # ─────────────────────────────────────────────────────────────────────
    lstm_configs = [
        {
            "name": "Baseline",
            "seq_len": 50,
            "hidden_size": 64,
            "num_layers": 2,
            "epochs": 15,
            "batch_size": 256,
            "rv_loss_weight": 1.0,
            "patience": 10,
            "lr": 1e-3,
        },
        {
            "name": "Large-SeqLen",
            "seq_len": 100,
            "hidden_size": 128,
            "num_layers": 2,
            "epochs": 20,
            "batch_size": 128,
            "rv_loss_weight": 1.0,
            "patience": 12,
            "lr": 1e-3,
        },
        {
            "name": "XLarge",
            "seq_len": 150,
            "hidden_size": 256,
            "num_layers": 2,
            "epochs": 25,
            "batch_size": 64,
            "rv_loss_weight": 1.0,
            "patience": 15,
            "lr": 5e-4,
        },
        {
            "name": "Deep",
            "seq_len": 100,
            "hidden_size": 256,
            "num_layers": 3,
            "epochs": 20,
            "batch_size": 128,
            "rv_loss_weight": 1.0,
            "patience": 12,
            "lr": 1e-3,
        },
    ]

    results = {}

    for config in lstm_configs:
        name = config["name"]
        print(f"\n[LSTM] {name}: seq_len={config['seq_len']}, "
              f"hidden={config['hidden_size']}, layers={config['num_layers']}, "
              f"epochs={config['epochs']}, batch={config['batch_size']}, "
              f"rv_weight={config['rv_loss_weight']}, patience={config['patience']}, "
              f"lr={config['lr']}")

        t0 = time.perf_counter()
        lstm = DualHeadLSTM(
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            seq_len=config["seq_len"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            rv_loss_weight=config["rv_loss_weight"],
            patience=config["patience"],
            lr=config["lr"],
        )
        print(f"  Device: {lstm.device}")
        print(f"  Training...")

        lstm.fit(
            splits.X_train,
            splits.y_rv[lstm_key]["train"],
            splits.y_shock_spread[lstm_key]["train"],
            X_val=splits.X_val,
            y_rv_val=splits.y_rv[lstm_key]["val"],
            y_shock_val=splits.y_shock_spread[lstm_key]["val"],
        )

        lstm_time = time.perf_counter() - t0
        lstm_rv_pred = lstm.predict_rv(splits.X_test)
        lstm_shock_proba = lstm.predict_shock_proba(splits.X_test)

        lstm_rv_metrics = evaluate_rv_forecast(
            splits.y_rv[lstm_key]["test"], lstm_rv_pred
        )
        lstm_shock_metrics = evaluate_shock_forecast(
            splits.y_shock_spread[lstm_key]["test"], lstm_shock_proba
        )

        # Get training diagnostics
        loss_summary = lstm.get_training_loss_summary()

        print(f"  Time: {lstm_time:.2f}s")
        print(f"  RV:    RMSE={lstm_rv_metrics['rmse']:.6f}  "
              f"MAE={lstm_rv_metrics['mae']:.6f}")
        print(f"  Shock: AUROC={lstm_shock_metrics['auroc']:.4f}  "
              f"AUPRC={lstm_shock_metrics['auprc']:.4f}")
        
        # Prediction diagnostics
        y_te = splits.y_rv[lstm_key]["test"]
        print(f"  Target stats:  min={y_te.min():.2e}  mean={y_te.mean():.2e}  max={y_te.max():.2e}")
        print(f"  Pred   stats:  min={lstm_rv_pred.min():.2e}  mean={lstm_rv_pred.mean():.2e}  max={lstm_rv_pred.max():.2e}")
        
        if loss_summary:
            print(f"  Training: {loss_summary['epochs_trained']} epochs  "
                  f"train loss improved {loss_summary['train_loss_improved']:.1f}%", end="")
            if "val_loss_improved" in loss_summary:
                print(f"  val loss improved {loss_summary['val_loss_improved']:.1f}%")
            else:
                print()

        # Compute improvement vs LGBM
        rmse_improvement = (lgbm_rv_metrics["rmse"] - lstm_rv_metrics["rmse"]) / lgbm_rv_metrics["rmse"] * 100
        auroc_improvement = (lstm_shock_metrics["auroc"] - lgbm_shock_metrics["auroc"]) * 100
        print(f"  vs LGBM: RMSE {rmse_improvement:+.1f}%, AUROC {auroc_improvement:+.1f}%")

        results[name] = {
            "config": config,
            "rv_metrics": lstm_rv_metrics,
            "shock_metrics": lstm_shock_metrics,
            "fit_time": lstm_time,
            "loss_summary": loss_summary,
            "vs_lgbm": {
                "rmse_improvement_pct": rmse_improvement,
                "auroc_improvement_pct": auroc_improvement,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # 7. Summary table
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Summary Comparison")
    print("=" * 80)
    print(f"\n  LGBM Baseline:")
    print(f"    RV RMSE:      {lgbm_rv_metrics['rmse']:.6f}")
    print(f"    RV MAE:       {lgbm_rv_metrics['mae']:.6f}")
    print(f"    Shock AUROC:  {lgbm_shock_metrics['auroc']:.4f}")
    print(f"    Shock AUPRC:  {lgbm_shock_metrics['auprc']:.4f}\n")

    print("  Config            | RMSE      | RMSE Δ%  | AUROC     | AUROC Δ%  | Time")
    print("  " + "-" * 76)
    for name, res in results.items():
        rmse = res["rv_metrics"]["rmse"]
        rmse_delta = res["vs_lgbm"]["rmse_improvement_pct"]
        auroc = res["shock_metrics"]["auroc"]
        auroc_delta = res["vs_lgbm"]["auroc_improvement_pct"]
        fit_time = res["fit_time"]
        print(f"  {name:17} | {rmse:.6f}  | {rmse_delta:+7.1f}% | {auroc:.4f}  | "
              f"{auroc_delta:+7.1f}% | {fit_time:6.2f}s")

    # ─────────────────────────────────────────────────────────────────────
    # 8. Save results
    # ─────────────────────────────────────────────────────────────────────
    output_file = Path("lstm_comparison.json")
    output_data = {
        "lgbm_baseline": {
            "rv_metrics": lgbm_rv_metrics,
            "shock_metrics": lgbm_shock_metrics,
        },
        "lstm_results": results,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved to {output_file}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LSTM hyperparameter configurations"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to parquet file (if using parquet source)",
    )
    args = parser.parse_args()
    main(args.data_path)
