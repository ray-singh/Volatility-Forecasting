"""
Evaluate saved TCN and Transformer pkl files against the test split
and patch results.json with their metrics.

Run from the project root:
    venv/bin/python scripts/eval_deep_models.py
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc

# ── make src importable ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # bare 'deep_models' alias for pkl compat

from src.config import PipelineConfig
from src.engineer import build_features, feature_cols, clean
from src.dataset import make_splits


def evaluate_rv_forecast(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    y_true_var = np.square(y_true)
    y_pred_var = np.clip(np.square(y_pred), 1e-8, None)
    ratio = np.clip(y_true_var, 1e-8, None) / y_pred_var
    qlike = float(np.mean(ratio - np.log(ratio) - 1))
    return {"rmse": rmse, "mae": mae, "qlike": qlike}


def evaluate_shock_forecast(y_true, y_pred_proba):
    auroc = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    auprc = float(sk_auc(recall, precision))
    brier = float(np.mean((y_pred_proba[:, 1] - y_true) ** 2))
    return {"auroc": auroc, "auprc": auprc, "brier": brier}

# ── CPU-safe unpickler ────────────────────────────────────────────────────────

class _CPUUnpickler(pickle.Unpickler):
    """Redirect any CUDA storage to CPU on machines without CUDA."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pkl_cpu(path: Path):
    with open(path, "rb") as f:
        obj = _CPUUnpickler(f).load()
    # Remap to MPS if available, else CPU
    target = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if hasattr(obj, "net") and obj.net is not None:
        obj.net = obj.net.to(target)
    if hasattr(obj, "device"):
        obj.device = target
    return obj


# ── data pipeline ─────────────────────────────────────────────────────────────

CACHE_PATH = Path("data/eval_split_cache.npz")


def build_test_splits(force_rebuild: bool = False):
    cfg = PipelineConfig()

    if CACHE_PATH.exists() and not force_rebuild:
        print(f"Loading cached test split from {CACHE_PATH} ...")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        from src.dataset import Splits
        splits = Splits(
            X_train=cache["X_train"],
            X_val=cache["X_val"],
            X_test=cache["X_test"],
            y_rv={k: v for k, v in zip(cache["y_rv_keys"], cache["y_rv_vals"])},
            y_shock_spread={k: v for k, v in zip(cache["y_shock_keys"], cache["y_shock_vals"])},
        )
        return splits, cfg

    data_path = Path("data/orderbook/btcusd_full.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    import polars as pl
    print("Reading parquet...")
    raw_df = pl.read_parquet(data_path)

    print("Building features (this takes a few minutes)...")
    feat_df = build_features(
        raw_df,
        rv_windows=cfg.features.rv_windows,
        ofi_window=cfg.features.ofi_window,
        har_lags=cfg.features.har_lags,
        horizons=cfg.features.horizons,
        ticks_per_second=cfg.features.ticks_per_second,
        lob_levels=cfg.data.lob_levels,
    )

    fcols = feature_cols(feat_df, har_lags=cfg.features.har_lags,
                         lob_levels=cfg.data.lob_levels)

    horizons = cfg.features.horizons
    target_cols = (
        [f"target_rv_{h}s"           for h in horizons] +
        [f"target_shock_spread_{h}s" for h in horizons]
    )
    feat_df = clean(feat_df, fcols + target_cols)

    splits = make_splits(
        feat_df, fcols,
        horizons=horizons,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
    )

    print(f"Saving test split cache to {CACHE_PATH} ...")
    np.savez(
        CACHE_PATH,
        X_train=splits.X_train,
        X_val=splits.X_val,
        X_test=splits.X_test,
        y_rv_keys=np.array(list(splits.y_rv.keys())),
        y_rv_vals=np.array([splits.y_rv[k] for k in splits.y_rv], dtype=object),
        y_shock_keys=np.array(list(splits.y_shock_spread.keys())),
        y_shock_vals=np.array([splits.y_shock_spread[k] for k in splits.y_shock_spread], dtype=object),
    )

    return splits, cfg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of test rows to evaluate (0 < sample <= 1.0)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of feature matrix even if cache exists")
    args = parser.parse_args()

    splits, cfg = build_test_splits(force_rebuild=args.rebuild)

    results_path = Path("results.json")
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for model_name, pkl_path, ph_rv_key, ph_shock_key in [
        ("TCN",         Path("models/tcn_model.pkl"),         "tcn_rv",         "tcn_shock"),
        ("Transformer", Path("models/transformer_model.pkl"), "transformer_rv",  "transformer_shock"),
    ]:
        if not pkl_path.exists():
            print(f"  {model_name}: pkl not found at {pkl_path}, skipping.")
            continue

        print(f"\nLoading {model_name} from {pkl_path} (CPU mode)...")
        try:
            model = load_pkl_cpu(pkl_path)
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            continue

        horizon_key = "5s"  # default seq_horizon_s used at training time
        if horizon_key not in splits.y_rv:
            print(f"  Horizon {horizon_key} not in splits, skipping.")
            continue

        X_test  = splits.X_test
        y_rv    = splits.y_rv[horizon_key]["test"]
        y_shock = splits.y_shock_spread[horizon_key]["test"]

        if args.sample < 1.0:
            n = max(1, int(len(X_test) * args.sample))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_test), size=n, replace=False)
            idx.sort()
            X_test, y_rv, y_shock = X_test[idx], y_rv[idx], y_shock[idx]
            print(f"  Sampling {args.sample:.0%} → {len(X_test):,} rows")

        print(f"  Running predict_rv on {len(X_test):,} test rows...")
        rv_pred    = model.predict_rv(X_test)
        shock_prob = model.predict_shock_proba(X_test)

        rv_metrics    = evaluate_rv_forecast(y_rv, rv_pred)
        shock_metrics = evaluate_shock_forecast(y_shock, shock_prob)

        print(f"  {model_name} RV   — RMSE={rv_metrics['rmse']:.6f}  MAE={rv_metrics['mae']:.6f}  QLIKE={rv_metrics['qlike']:.4f}")
        print(f"  {model_name} Shock— AUROC={shock_metrics['auroc']:.4f}  AUPRC={shock_metrics['auprc']:.4f}  Brier={shock_metrics['brier']:.4f}")

        # Patch results.json
        per_h = results.setdefault("per_horizon", {})
        bucket = per_h.setdefault(horizon_key, {})
        bucket[ph_rv_key]    = rv_metrics
        bucket[ph_shock_key] = shock_metrics

        # Also write flat top-level keys for backward-compat
        prefix = "tcn" if model_name == "TCN" else "transformer"
        results[f"{prefix}_rmse"]  = rv_metrics["rmse"]
        results[f"{prefix}_mae"]   = rv_metrics["mae"]
        results[f"{prefix}_auroc"] = shock_metrics["auroc"]
        results[f"{prefix}_auprc"] = shock_metrics["auprc"]
        results[f"{prefix}_brier"] = shock_metrics["brier"]

    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nresults.json updated at {results_path}")


if __name__ == "__main__":
    main()
