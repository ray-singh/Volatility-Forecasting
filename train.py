"""
Training pipeline: data loading → feature engineering → model training → evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import json

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc

import polars as pl

from config import PipelineConfig
from engineer import build_features, feature_cols, clean
from dataset import make_splits
from models import DataLoader, HARRVModel, LGBMDualModel, diebold_mariano


def evaluate_rv_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate RV forecast."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": rmse, "mae": mae}


def evaluate_shock_forecast(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Evaluate shock classification."""
    auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    auprc = auc(recall, precision)
    return {"auroc": auroc, "auprc": auprc}


def train(cfg: PipelineConfig, data_path: str | None = None) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    cfg : PipelineConfig
    data_path : str | None
        For parquet source, the file path to load

    Returns
    -------
    dict with results
    """
    print("[train] Starting training pipeline...")

    # =========================================================================
    # 1. Load raw data
    # =========================================================================
    print(f"[train] Loading data from source='{cfg.data.source}'...")
    loader = DataLoader(
        source=cfg.data.source,
        path=data_path,
        pair=cfg.data.symbol,
        levels=cfg.data.lob_levels,
        poll_interval_s=0.5,
    )
    raw_df = loader.load(n_rows=5_000)
    print(f"[train] Loaded {len(raw_df)} snapshots")

    if len(raw_df) < 200:
        raise RuntimeError(f"Only {len(raw_df)} snapshots; need >= 200 for pipeline")

    # =========================================================================
    # 2. Feature engineering
    # =========================================================================
    print("[train] Engineering features...")
    feat_df = build_features(
        raw_df,
        rv_windows=cfg.features.rv_windows,
        ofi_window=cfg.features.ofi_window,
    )
    fcols = feature_cols(feat_df, har_lags=cfg.features.har_lags)
    print(f"[train] Generated {len(fcols)} features: {fcols}")

    # =========================================================================
    # 3. Clean features
    # =========================================================================
    all_cols = fcols + ["target_rv", "target_shock"]
    print(f"[train] Cleaning {len(all_cols)} columns...")
    feat_df = clean(feat_df, all_cols)
    print(f"[train] After cleaning: {len(feat_df)} usable rows")

    if len(feat_df) < 100:
        raise RuntimeError(f"After cleaning, only {len(feat_df)} rows; need >= 100")

    # =========================================================================
    # 4. Create splits
    # =========================================================================
    print("[train] Creating train/val/test splits...")
    splits = make_splits(
        feat_df,
        feature_cols=fcols,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
    )
    print(f"[train] Train: {splits.X_train.shape[0]}, "
          f"Val: {splits.X_val.shape[0]}, Test: {splits.X_test.shape[0]}")

    # =========================================================================
    # 5. HAR-RV baseline
    # =========================================================================
    print("[train] Training HAR-RV baseline...")
    har_cols = [c for c in fcols if c.startswith("har_rv_")]
    if har_cols:
        har_idx = [fcols.index(c) for c in har_cols]
        X_har_train = splits.X_train[:, har_idx]
        X_har_val = splits.X_val[:, har_idx]
        X_har_test = splits.X_test[:, har_idx]
    else:
        X_har_train = np.ones((splits.X_train.shape[0], 1))
        X_har_val = np.ones((splits.X_val.shape[0], 1))
        X_har_test = np.ones((splits.X_test.shape[0], 1))

    har = HARRVModel()
    har.fit(X_har_train, splits.y_rv_train)
    har_pred_val = har.predict(X_har_val)
    har_pred_test = har.predict(X_har_test)
    har_metrics = evaluate_rv_forecast(splits.y_rv_test, har_pred_test)
    print(f"[train] HAR-RV RMSE (test): {har_metrics['rmse']:.4f}")

    # =========================================================================
    # 6. LightGBM dual model
    # =========================================================================
    print("[train] Training LightGBM dual model...")
    lgbm = LGBMDualModel(lgbm_params=cfg.models.lgbm_params)
    lgbm.fit(
        splits.X_train,
        splits.y_rv_train,
        splits.y_shock_train,
        X_val=splits.X_val,
        y_rv_val=splits.y_rv_val,
        y_shock_val=splits.y_shock_val,
    )

    lgbm_rv_pred_test = lgbm.predict_rv(splits.X_test)
    lgbm_shock_proba_test = lgbm.predict_shock_proba(splits.X_test)

    lgbm_rv_metrics = evaluate_rv_forecast(splits.y_rv_test, lgbm_rv_pred_test)
    lgbm_shock_metrics = evaluate_shock_forecast(splits.y_shock_test, lgbm_shock_proba_test)

    print(f"[train] LightGBM RV RMSE (test): {lgbm_rv_metrics['rmse']:.4f}")
    print(f"[train] LightGBM Shock AUROC (test): {lgbm_shock_metrics['auroc']:.3f}")

    # =========================================================================
    # 7. Model comparison (Diebold-Mariano)
    # =========================================================================
    print("[train] Running Diebold-Mariano test...")
    dm_result = diebold_mariano(splits.y_rv_test, har_pred_test, lgbm_rv_pred_test)
    print(f"[train] DM test: {dm_result['interpretation']}")

    # =========================================================================
    # 8. Save models and results
    # =========================================================================
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / "har_model.pkl", "wb") as f:
        pickle.dump(har, f)
    with open(models_dir / "lgbm_model.pkl", "wb") as f:
        pickle.dump(lgbm, f)

    print(f"[train] Models saved to {models_dir}/")

    # Aggregate results
    results = {
        "har_rmse": float(har_metrics["rmse"]),
        "lgbm_rmse": float(lgbm_rv_metrics["rmse"]),
        "rmse": float(lgbm_rv_metrics["rmse"]),  # Main metric
        "auroc": float(lgbm_shock_metrics["auroc"]),
        "auprc": float(lgbm_shock_metrics["auprc"]),
        "diebold_mariano": dm_result,
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[train] Results saved to results.json")
    print(f"[train] Final metrics: RMSE={results['rmse']:.4f}, "
          f"AUROC={results['auroc']:.3f}, AUPRC={results['auprc']:.3f}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train volatility forecasting models")
    parser.add_argument("--source", required=True, choices=["parquet", "kraken"],
                       help="Data source")
    parser.add_argument("--path", type=str, default=None,
                       help="Path to parquet file (for parquet source)")
    parser.add_argument("--pair", type=str, default="XBTUSD",
                       help="Kraken pair (for kraken source)")
    args = parser.parse_args()

    cfg = PipelineConfig()

    try:
        results = train(cfg, data_path=args.path)
        print("\n✓ Training completed successfully")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
