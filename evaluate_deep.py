"""
Evaluate saved TCN / Transformer pkl models against the test split and
patch the metrics into results.json — without retraining.

Usage:
    python evaluate_deep.py                          # uses data/csv/ (default)
    python evaluate_deep.py --source parquet --path data/raw/kraken_XBTUSD_*.parquet
    python evaluate_deep.py --no-tcn                 # skip TCN
    python evaluate_deep.py --no-transformer         # skip Transformer
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path

import torch

from config import PipelineConfig
from dataset import make_splits
from engineer import build_features, clean, feature_cols
from models import DataLoader
from train import evaluate_rv_forecast, evaluate_shock_forecast, calibrate_shock_probabilities


class _CPUUnpickler(pickle.Unpickler):
    """Remap CUDA tensors to CPU when loading on a MPS-only machine."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="mps", weights_only=False)
        return super().find_class(module, name)


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        model = _CPUUnpickler(f).load()
    # Model was saved on CUDA; remap everything to CPU
    model.device = torch.device("mps")
    if hasattr(model, "net") and model.net is not None:
        model.net.to("mps")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="parquet", choices=["csv", "parquet"])
    parser.add_argument("--path",   default="data/orderbook/btcusd_full.parquet",
                        help="Parquet file path")
    parser.add_argument("--lob-levels", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Truncate data to first N rows (useful when memory is tight)")
    parser.add_argument("--no-tcn",         dest="tcn",         action="store_false")
    parser.add_argument("--no-transformer", dest="transformer", action="store_false")
    args = parser.parse_args()

    cfg = PipelineConfig()
    lob_levels = args.lob_levels or cfg.data.lob_levels
    horizons   = cfg.features.horizons   # e.g. [1, 5, 30]
    tps        = cfg.features.ticks_per_second

    # ── 1. Load data (same as train.py) ───────────────────────────────────────
    print("Loading data…")
    loader = DataLoader(
        source=args.source,
        path=args.path,
        csv_dir=str(cfg.data.csv_dir),
        pair=cfg.data.symbol,
        levels=lob_levels,
    )
    raw_df = loader.load()
    if args.max_rows:
        raw_df = raw_df.head(args.max_rows)
    print(f"  {len(raw_df):,} rows")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("Engineering features…")
    feat_df = build_features(
        raw_df,
        rv_windows=cfg.features.rv_windows,
        ofi_window=cfg.features.ofi_window,
        har_lags=cfg.features.har_lags,
        horizons=horizons,
        ticks_per_second=tps,
        lob_levels=lob_levels,
    )
    fcols = feature_cols(feat_df, har_lags=cfg.features.har_lags, lob_levels=lob_levels)

    target_cols = (
        [f"target_rv_{h}s"           for h in horizons] +
        [f"target_shock_spread_{h}s" for h in horizons] +
        [f"target_shock_depth_{h}s"  for h in horizons
         if f"target_shock_depth_{h}s" in feat_df.columns]
    )
    feat_df = clean(feat_df, fcols + target_cols)
    print(f"  {len(feat_df):,} rows after cleaning, {len(fcols)} features")

    # ── 3. Splits (same fractions as training) ────────────────────────────────
    splits = make_splits(
        feat_df,
        feature_cols=fcols,
        horizons=horizons,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
    )
    print(f"  test set: {splits.X_test.shape[0]:,} rows")

    # ── 4. Load results.json ──────────────────────────────────────────────────
    results_path = Path("results.json")
    results: dict = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    per_horizon: dict = results.get("per_horizon", {})

    # ── 5. Evaluate each deep model ───────────────────────────────────────────
    model_specs = []
    if args.tcn:
        model_specs.append(("TCN",         Path("models/tcn_model.pkl"),         "tcn_rv",          "tcn_shock"))
    if args.transformer:
        model_specs.append(("Transformer", Path("models/transformer_model.pkl"), "transformer_rv",  "transformer_shock"))

    for name, pkl_path, rv_key, sh_key in model_specs:
        if not pkl_path.exists():
            print(f"  [{name}] {pkl_path} not found — skipping")
            continue

        print(f"\nEvaluating {name}…")
        model = _load_pkl(pkl_path)

        # Deep models are trained on a single seq_horizon (default 5s); try all horizons
        # but only the one they were trained on will produce valid predictions.
        # We infer the horizon from which y_rv keys exist and produce non-trivial output.
        # Simpler: evaluate against all horizons that have test labels.
        for h in horizons:
            key = f"{h}s"
            if key not in splits.y_rv:
                continue

            y_rv_test    = splits.y_rv[key]["test"]
            y_shock_test = splits.y_shock_spread[key]["test"]
            y_shock_val  = splits.y_shock_spread[key]["val"]

            try:
                rv_pred        = model.predict_rv(splits.X_test)
                shock_proba    = model.predict_shock_proba(splits.X_test)
                shock_proba_val = model.predict_shock_proba(splits.X_val)
            except Exception as e:
                print(f"  [{name} {key}] predict failed: {e} — skipping")
                continue

            rv_metrics    = evaluate_rv_forecast(y_rv_test, rv_pred)
            shock_metrics = evaluate_shock_forecast(y_shock_test, shock_proba)
            shock_calib   = calibrate_shock_probabilities(
                y_shock_val, shock_proba_val, y_shock_test, shock_proba
            )
            shock_metrics_calib = evaluate_shock_forecast(y_shock_test, shock_calib)

            print(f"  [{name} {key}] RMSE={rv_metrics['rmse']:.5f}  "
                  f"AUROC={shock_metrics['auroc']:.4f}  "
                  f"AUROC_calib={shock_metrics_calib['auroc']:.4f}")

            # Patch into per_horizon
            if key not in per_horizon:
                per_horizon[key] = {}
            per_horizon[key][rv_key] = rv_metrics
            per_horizon[key][sh_key] = shock_metrics

        # Flat legacy keys (use 5s if available, else first horizon)
        primary = "5s" if "5s" in splits.y_rv else f"{horizons[0]}s"
        if primary in per_horizon and rv_key in per_horizon[primary]:
            flat_prefix = name.lower().replace(" ", "_")  # "tcn" or "transformer"
            results[f"{flat_prefix}_rmse"]  = float(per_horizon[primary][rv_key]["rmse"])
            results[f"{flat_prefix}_auroc"] = float(per_horizon[primary][sh_key]["auroc"])
            results[f"{flat_prefix}_auprc"] = float(per_horizon[primary][sh_key]["auprc"])
            results[f"{flat_prefix}_brier"] = float(per_horizon[primary][sh_key]["brier"])

    # ── 6. Write results.json ─────────────────────────────────────────────────
    results["per_horizon"] = per_horizon
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPatched {results_path}")


if __name__ == "__main__":
    main()
