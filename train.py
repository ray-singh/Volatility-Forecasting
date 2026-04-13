"""
Training pipeline: data loading → feature engineering → model training → evaluation.

MLflow tracking is enabled by default. Each call to train() creates a new run
under the experiment "volatility-forecasting". View results with:

    mlflow ui
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import pickle
import json

import os
os.environ["MLFLOW_DISABLE_ENV_CREATION"] = "true"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc
import mlflow

import polars as pl

from config import PipelineConfig
from engineer import build_features, feature_cols, clean
from dataset import make_splits
from models import DataLoader, HARRVModel, LGBMDualModel, DualHeadLSTM, diebold_mariano

EXPERIMENT_NAME = "volatility-forecasting"


def _section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _log(msg: str) -> None:
    print(f"  {msg}")


def evaluate_rv_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate RV forecast."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    # QLIKE loss: common in vol forecasting literature
    eps = 1e-10
    qlike = np.mean(y_true / (y_pred + eps) - np.log(y_true / (y_pred + eps)) - 1)
    return {"rmse": rmse, "mae": mae, "qlike": qlike}


def evaluate_shock_forecast(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Evaluate shock classification."""
    auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    auprc = auc(recall, precision)
    # Brier score: calibration of probabilities
    brier = np.mean((y_pred_proba[:, 1] - y_true) ** 2)
    return {"auroc": auroc, "auprc": auprc, "brier": brier}


def _print_rv_metrics(label: str, m: dict) -> None:
    _log(f"{label:<20}  RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  QLIKE={m['qlike']:.4f}")


def _print_shock_metrics(label: str, m: dict) -> None:
    _log(f"{label:<20}  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")


def _print_dm(label: str, dm: dict) -> None:
    sign = "Model1 < Model2" if dm["model1_better"] else "Model1 > Model2"
    _log(f"{label:<30}  stat={dm['statistic']:+.3f}  p={dm['p_value']:.4f}  "
         f"({sign})  → {dm['interpretation']}")


def train(
    cfg: PipelineConfig,
    data_path: str | None = None,
    train_lstm: bool = True,
) -> dict:
    """
    Full training pipeline with MLflow tracking.

    Parameters
    ----------
    cfg : PipelineConfig
    data_path : str | None
        For parquet source, the file path to load
    train_lstm : bool
        Whether to train the LSTM (can be slow on CPU)

    Returns
    -------
    dict with results
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n{'#' * 72}")
        print(f"  MLflow run: {run_id}")
        print(f"  Experiment: {EXPERIMENT_NAME}")
        print(f"{'#' * 72}")

        # Log config as params
        mlflow.log_params({
            "source": cfg.data.source,
            "symbol": cfg.data.symbol,
            "train_frac": cfg.data.train_frac,
            "val_frac": cfg.data.val_frac,
            "rv_windows": str(cfg.features.rv_windows),
            "ofi_window": cfg.features.ofi_window,
            "har_lags": str(cfg.features.har_lags),
            "lgbm_n_estimators": cfg.models.lgbm_params.get("n_estimators"),
            "lgbm_lr": cfg.models.lgbm_params.get("learning_rate"),
            "lgbm_num_leaves": cfg.models.lgbm_params.get("num_leaves"),
            "lstm_enabled": train_lstm,
            "lstm_hidden": cfg.models.lstm_hidden,
            "lstm_layers": cfg.models.lstm_layers,
            "lstm_seq_len": cfg.models.lstm_seq_len,
            "lstm_epochs": cfg.models.lstm_epochs,
            "lstm_batch": cfg.models.lstm_batch,
        })

        # =====================================================================
        # 1. Load raw data
        # =====================================================================
        _section("1. Data Loading")
        t0 = time.perf_counter()
        loader = DataLoader(
            source=cfg.data.source,
            path=data_path,
            pair=cfg.data.symbol,
            levels=cfg.data.lob_levels,
            poll_interval_s=0.5,
        )
        raw_df = loader.load(n_rows=5_000)
        load_time = time.perf_counter() - t0

        _log(f"Source:       {cfg.data.source}")
        _log(f"Rows loaded:  {len(raw_df):,}")
        _log(f"Columns:      {raw_df.columns}")
        _log(f"Load time:    {load_time:.2f}s")

        mlflow.log_metric("data_rows_raw", len(raw_df))

        if len(raw_df) < 200:
            raise RuntimeError(f"Only {len(raw_df)} snapshots; need >= 200 for pipeline")

        # =====================================================================
        # 2. Feature engineering
        # =====================================================================
        _section("2. Feature Engineering")
        t0 = time.perf_counter()
        feat_df = build_features(
            raw_df,
            rv_windows=cfg.features.rv_windows,
            ofi_window=cfg.features.ofi_window,
        )
        fcols = feature_cols(feat_df, har_lags=cfg.features.har_lags)
        eng_time = time.perf_counter() - t0

        _log(f"Feature count:  {len(fcols)}")
        for col in fcols:
            series = feat_df[col].drop_nulls()
            _log(f"  {col:<30}  mean={series.mean():.6f}  std={series.std():.6f}  "
                 f"null%={feat_df[col].null_count() / len(feat_df) * 100:.1f}%")
        _log(f"Engineering time: {eng_time:.2f}s")

        mlflow.log_param("n_features", len(fcols))
        mlflow.log_param("feature_names", str(fcols))

        # =====================================================================
        # 3. Clean
        # =====================================================================
        _section("3. Cleaning")
        all_cols = fcols + ["target_rv", "target_shock"]
        rows_before = len(feat_df)
        feat_df = clean(feat_df, all_cols)
        rows_after = len(feat_df)
        dropped = rows_before - rows_after

        _log(f"Rows before:  {rows_before:,}")
        _log(f"Rows after:   {rows_after:,}  ({dropped:,} dropped, "
             f"{dropped / rows_before * 100:.1f}%)")

        target_rv_series = feat_df["target_rv"]
        target_shock_series = feat_df["target_shock"]
        shock_rate = target_shock_series.mean()
        _log(f"Target RV     mean={target_rv_series.mean():.6f}  "
             f"std={target_rv_series.std():.6f}  "
             f"min={target_rv_series.min():.6f}  max={target_rv_series.max():.6f}")
        _log(f"Shock rate:   {shock_rate:.3f}  ({shock_rate * 100:.1f}% positive)")

        mlflow.log_metrics({
            "data_rows_clean": rows_after,
            "data_rows_dropped": dropped,
            "shock_rate": float(shock_rate),
            "target_rv_mean": float(target_rv_series.mean()),
            "target_rv_std": float(target_rv_series.std()),
        })

        if rows_after < 100:
            raise RuntimeError(f"After cleaning, only {rows_after} rows; need >= 100")

        # =====================================================================
        # 4. Splits
        # =====================================================================
        _section("4. Train / Val / Test Split")
        splits = make_splits(
            feat_df,
            feature_cols=fcols,
            train_frac=cfg.data.train_frac,
            val_frac=cfg.data.val_frac,
        )
        n_train, n_val, n_test = (
            splits.X_train.shape[0], splits.X_val.shape[0], splits.X_test.shape[0]
        )
        _log(f"Train:  {n_train:,} rows  ({n_train / rows_after * 100:.1f}%)")
        _log(f"Val:    {n_val:,} rows  ({n_val / rows_after * 100:.1f}%)")
        _log(f"Test:   {n_test:,} rows  ({n_test / rows_after * 100:.1f}%)")
        _log(f"Features: {splits.X_train.shape[1]}")

        mlflow.log_metrics({"n_train": n_train, "n_val": n_val, "n_test": n_test})

        # =====================================================================
        # 5. HAR-RV baseline
        # =====================================================================
        _section("5. HAR-RV Baseline")
        t0 = time.perf_counter()
        har_cols = [c for c in fcols if c.startswith("har_rv_")]
        if har_cols:
            har_idx = [fcols.index(c) for c in har_cols]
            X_har_train = splits.X_train[:, har_idx]
            X_har_test  = splits.X_test[:, har_idx]
        else:
            X_har_train = np.ones((n_train, 1))
            X_har_test  = np.ones((n_test, 1))

        _log(f"HAR features used: {har_cols or ['intercept-only']}")
        har = HARRVModel()
        har.fit(X_har_train, splits.y_rv_train)
        har_pred_test = har.predict(X_har_test)
        har_metrics   = evaluate_rv_forecast(splits.y_rv_test, har_pred_test)
        har_time = time.perf_counter() - t0

        _print_rv_metrics("HAR-RV (test)", har_metrics)
        _log(f"Fit time: {har_time:.2f}s")

        mlflow.log_metrics({
            "har_rmse": har_metrics["rmse"],
            "har_mae":  har_metrics["mae"],
            "har_qlike": har_metrics["qlike"],
            "har_fit_time": har_time,
        })

        # =====================================================================
        # 6. LightGBM dual model
        # =====================================================================
        _section("6. LightGBM Dual Model")
        _log(f"Params: {cfg.models.lgbm_params}")
        t0 = time.perf_counter()

        lgbm = LGBMDualModel(lgbm_params=cfg.models.lgbm_params)
        lgbm.fit(
            splits.X_train, splits.y_rv_train, splits.y_shock_train,
            X_val=splits.X_val, y_rv_val=splits.y_rv_val, y_shock_val=splits.y_shock_val,
        )
        lgbm_time = time.perf_counter() - t0

        lgbm_rv_pred_test    = lgbm.predict_rv(splits.X_test)
        lgbm_shock_proba_test = lgbm.predict_shock_proba(splits.X_test)
        lgbm_rv_metrics      = evaluate_rv_forecast(splits.y_rv_test, lgbm_rv_pred_test)
        lgbm_shock_metrics   = evaluate_shock_forecast(splits.y_shock_test, lgbm_shock_proba_test)

        _print_rv_metrics("LGBM RV (test)",    lgbm_rv_metrics)
        _print_shock_metrics("LGBM Shock (test)", lgbm_shock_metrics)
        _log(f"Fit time: {lgbm_time:.2f}s")

        # Feature importances (top 10)
        if hasattr(lgbm.rv_model, "feature_importances_"):
            importances = lgbm.rv_model.feature_importances_
            top = sorted(zip(fcols, importances), key=lambda x: -x[1])[:10]
            _log("Top RV feature importances:")
            for fname, imp in top:
                _log(f"  {fname:<30} {imp:>8.1f}")

        mlflow.log_metrics({
            "lgbm_rv_rmse":  lgbm_rv_metrics["rmse"],
            "lgbm_rv_mae":   lgbm_rv_metrics["mae"],
            "lgbm_rv_qlike": lgbm_rv_metrics["qlike"],
            "lgbm_shock_auroc":  lgbm_shock_metrics["auroc"],
            "lgbm_shock_auprc":  lgbm_shock_metrics["auprc"],
            "lgbm_shock_brier":  lgbm_shock_metrics["brier"],
            "lgbm_fit_time": lgbm_time,
        })

        # =====================================================================
        # 7. LSTM dual-head model (optional)
        # MLflow's background threads interfere with PyTorch on macOS, so we
        # train outside the run context and only log metrics inside it.
        # =====================================================================
        lstm = None
        lstm_rv_pred_test  = None
        lstm_rv_metrics    = None
        lstm_shock_metrics = None
        lstm_time          = None

        if train_lstm:
            _section("7. LSTM Dual-Head Model")
            _log(f"hidden={cfg.models.lstm_hidden}  layers={cfg.models.lstm_layers}  "
                 f"seq_len={cfg.models.lstm_seq_len}  epochs={cfg.models.lstm_epochs}  "
                 f"batch={cfg.models.lstm_batch}")
            t0 = time.perf_counter()

            lstm = DualHeadLSTM(
                hidden_size=cfg.models.lstm_hidden,
                num_layers=cfg.models.lstm_layers,
                seq_len=cfg.models.lstm_seq_len,
                epochs=cfg.models.lstm_epochs,
                batch_size=cfg.models.lstm_batch,
            )
            _log(f"Device: {lstm.device}")

            # Train outside the mlflow run to avoid signal-handler conflicts
            mlflow.end_run()
            lstm.fit(
                splits.X_train, splits.y_rv_train, splits.y_shock_train,
                X_val=splits.X_val, y_rv_val=splits.y_rv_val, y_shock_val=splits.y_shock_val,
            )
            mlflow.start_run(run_id=run_id)

            lstm_time = time.perf_counter() - t0
            lstm_rv_pred_test     = lstm.predict_rv(splits.X_test)
            lstm_shock_proba_test = lstm.predict_shock_proba(splits.X_test)
            lstm_rv_metrics       = evaluate_rv_forecast(splits.y_rv_test, lstm_rv_pred_test)
            lstm_shock_metrics    = evaluate_shock_forecast(splits.y_shock_test, lstm_shock_proba_test)

            _print_rv_metrics("LSTM RV (test)",    lstm_rv_metrics)
            _print_shock_metrics("LSTM Shock (test)", lstm_shock_metrics)
            _log(f"Fit time: {lstm_time:.2f}s")

            if lstm.train_history:
                _log("Epoch history:")
                for entry in lstm.train_history:
                    val_str = f"  val={entry['val_loss']:.4f}" if "val_loss" in entry else ""
                    _log(f"  epoch {entry['epoch']:>3}  train={entry['train_loss']:.4f}{val_str}")
                    mlflow.log_metrics(
                        {"lstm_train_loss": entry["train_loss"],
                         **( {"lstm_val_loss": entry["val_loss"]} if "val_loss" in entry else {})},
                        step=entry["epoch"],
                    )

            mlflow.log_metrics({
                "lstm_rv_rmse":      lstm_rv_metrics["rmse"],
                "lstm_rv_mae":       lstm_rv_metrics["mae"],
                "lstm_rv_qlike":     lstm_rv_metrics["qlike"],
                "lstm_shock_auroc":  lstm_shock_metrics["auroc"],
                "lstm_shock_auprc":  lstm_shock_metrics["auprc"],
                "lstm_shock_brier":  lstm_shock_metrics["brier"],
                "lstm_fit_time":     lstm_time,
            })
        else:
            _section("7. LSTM")
            _log("Skipped (train_lstm=False)")

        # =====================================================================
        # 8. Model comparison (Diebold-Mariano)
        # =====================================================================
        _section("8. Diebold-Mariano Tests")
        _log("H0: equal forecast accuracy.  p < 0.05 → reject H0.")
        _log("")

        dm_lgbm_vs_har = diebold_mariano(splits.y_rv_test, har_pred_test, lgbm_rv_pred_test)
        _print_dm("LGBM vs HAR", dm_lgbm_vs_har)
        dm_results = {"lgbm_vs_har": dm_lgbm_vs_har}

        mlflow.log_metrics({
            "dm_lgbm_vs_har_stat":    dm_lgbm_vs_har["statistic"],
            "dm_lgbm_vs_har_pvalue":  dm_lgbm_vs_har["p_value"],
        })

        if lstm_rv_pred_test is not None:
            dm_lstm_vs_har  = diebold_mariano(splits.y_rv_test, har_pred_test, lstm_rv_pred_test)
            dm_lstm_vs_lgbm = diebold_mariano(splits.y_rv_test, lgbm_rv_pred_test, lstm_rv_pred_test)
            _print_dm("LSTM vs HAR",  dm_lstm_vs_har)
            _print_dm("LSTM vs LGBM", dm_lstm_vs_lgbm)
            dm_results["lstm_vs_har"]  = dm_lstm_vs_har
            dm_results["lstm_vs_lgbm"] = dm_lstm_vs_lgbm
            mlflow.log_metrics({
                "dm_lstm_vs_har_stat":     dm_lstm_vs_har["statistic"],
                "dm_lstm_vs_har_pvalue":   dm_lstm_vs_har["p_value"],
                "dm_lstm_vs_lgbm_stat":    dm_lstm_vs_lgbm["statistic"],
                "dm_lstm_vs_lgbm_pvalue":  dm_lstm_vs_lgbm["p_value"],
            })

        # =====================================================================
        # 9. Save models and results
        # =====================================================================
        _section("9. Saving")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        for name, obj in [("har_model", har), ("lgbm_model", lgbm)]:
            path = models_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            mlflow.log_artifact(str(path))
            _log(f"Saved {path}")

        if lstm is not None:
            path = models_dir / "lstm_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(lstm, f)
            mlflow.log_artifact(str(path))
            _log(f"Saved {path}")

        results = {
            "har_rmse":  float(har_metrics["rmse"]),
            "har_mae":   float(har_metrics["mae"]),
            "lgbm_rmse": float(lgbm_rv_metrics["rmse"]),
            "lgbm_mae":  float(lgbm_rv_metrics["mae"]),
            "rmse":      float(lgbm_rv_metrics["rmse"]),  # primary metric
            "auroc":     float(lgbm_shock_metrics["auroc"]),
            "auprc":     float(lgbm_shock_metrics["auprc"]),
            "brier":     float(lgbm_shock_metrics["brier"]),
            "diebold_mariano": dm_results,
        }
        if lstm_rv_metrics is not None:
            results["lstm_rmse"]  = float(lstm_rv_metrics["rmse"])
            results["lstm_auroc"] = float(lstm_shock_metrics["auroc"])
            results["lstm_auprc"] = float(lstm_shock_metrics["auprc"])
            results["lstm_brier"] = float(lstm_shock_metrics["brier"])

        results_path = Path("results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(results_path))
        _log(f"Saved {results_path}")

        # =====================================================================
        # Summary
        # =====================================================================
        _section("Summary")
        _log(f"{'Model':<12}  {'RMSE':>10}  {'MAE':>10}  {'QLIKE':>8}  "
             f"{'AUROC':>8}  {'AUPRC':>8}  {'Brier':>8}")
        _log("-" * 70)
        _log(f"{'HAR-RV':<12}  {har_metrics['rmse']:>10.6f}  "
             f"{har_metrics['mae']:>10.6f}  {har_metrics['qlike']:>8.4f}  "
             f"{'—':>8}  {'—':>8}  {'—':>8}")
        _log(f"{'LGBM':<12}  {lgbm_rv_metrics['rmse']:>10.6f}  "
             f"{lgbm_rv_metrics['mae']:>10.6f}  {lgbm_rv_metrics['qlike']:>8.4f}  "
             f"{lgbm_shock_metrics['auroc']:>8.4f}  "
             f"{lgbm_shock_metrics['auprc']:>8.4f}  "
             f"{lgbm_shock_metrics['brier']:>8.4f}")
        if lstm_rv_metrics is not None:
            _log(f"{'LSTM':<12}  {lstm_rv_metrics['rmse']:>10.6f}  "
                 f"{lstm_rv_metrics['mae']:>10.6f}  {lstm_rv_metrics['qlike']:>8.4f}  "
                 f"{lstm_shock_metrics['auroc']:>8.4f}  "
                 f"{lstm_shock_metrics['auprc']:>8.4f}  "
                 f"{lstm_shock_metrics['brier']:>8.4f}")
        _log("")
        _log(f"MLflow run ID: {run_id}")
        _log("View UI:       mlflow ui")

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train volatility forecasting models")
    parser.add_argument("--source", required=True, choices=["parquet", "kraken"])
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--pair", type=str, default="XBTUSD")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.data.source = args.source
    if args.source == "kraken":
        cfg.data.symbol = args.pair

    try:
        results = train(cfg, data_path=args.path, train_lstm=not args.no_lstm)
        print("\n✓ Training completed successfully")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
