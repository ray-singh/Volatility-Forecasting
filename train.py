"""
Training pipeline: data loading → feature engineering → model training → evaluation.

MLflow tracking is enabled by default. Each call to train() creates a new run
under the experiment "volatility-forecasting". View results with:

    mlflow ui

Supports multiple forecast horizons, GARCH baseline, regime-based evaluation,
and feature ablation analysis.
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import mlflow

import polars as pl

from config import PipelineConfig
from engineer import build_features, feature_cols, clean
from dataset import make_splits
from models import (
    DataLoader, HARRVModel, GARCHModel, LGBMDualModel, diebold_mariano,
)
from deep_models import TCNModel, TransformerModel

EXPERIMENT_NAME = "volatility-forecasting"


# ══════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _log(msg: str) -> None:
    print(f"  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Metric evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rv_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate RV forecast with RMSE, MAE, and QLIKE loss.

    QLIKE is computed on variance (squared RV), not volatility, following
    Patton (2011). Input arrays should be volatility (standard deviation).
    """
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))

    # QLIKE: convert RV (volatility) to variance, then compute
    # QLIKE(h, sigma^2) = h / sigma^2 - log(h / sigma^2) - 1
    y_true_var = np.square(y_true)
    y_pred_var = np.square(y_pred)

    # Clip to avoid log(0) and division by zero
    y_pred_var_clipped = np.clip(y_pred_var, 1e-8, None)
    y_true_var_clipped = np.clip(y_true_var, 1e-8, None)

    ratio = y_true_var_clipped / y_pred_var_clipped
    qlike = float(np.mean(ratio - np.log(ratio) - 1))

    return {"rmse": rmse, "mae": mae, "qlike": qlike}


def evaluate_shock_forecast(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Evaluate shock classification with AUROC, AUPRC, and Brier score."""
    auroc = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    auprc = float(auc(recall, precision))
    brier = float(np.mean((y_pred_proba[:, 1] - y_true) ** 2))
    return {"auroc": auroc, "auprc": auprc, "brier": brier}


def calibrate_shock_probabilities(
    y_val: np.ndarray,
    y_pred_proba_val: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
) -> np.ndarray:
    """
    Calibrate shock probabilities using Platt scaling on validation set.

    Parameters
    ----------
    y_val, y_pred_proba_val : np.ndarray
        Validation set labels and raw probabilities (N, 2) from classifier.
    y_test, y_pred_proba_test : np.ndarray
        Test set labels and raw probabilities to calibrate.

    Returns
    -------
    np.ndarray, shape (N_test, 2)
        Calibrated probability predictions on test set.
    """
    from sklearn.linear_model import LogisticRegression

    # Drop NaN scores before fitting
    raw_scores_val = y_pred_proba_val[:, 1].reshape(-1, 1)
    valid_mask = np.isfinite(raw_scores_val.ravel())
    if valid_mask.sum() < 10 or len(np.unique(y_val[valid_mask])) < 2:
        # Not enough valid data to calibrate — return raw probabilities
        print("[calibration] Skipping calibration: insufficient valid predictions")
        return y_pred_proba_test

    platt = LogisticRegression(max_iter=1000)
    platt.fit(raw_scores_val[valid_mask], y_val[valid_mask])

    # Apply to test set; fall back to raw if test scores also have NaNs
    raw_scores_test = y_pred_proba_test[:, 1].reshape(-1, 1)
    valid_test = np.isfinite(raw_scores_test.ravel())
    if not valid_test.all():
        print("[calibration] Skipping calibration: NaN predictions in test set")
        return y_pred_proba_test

    return platt.predict_proba(raw_scores_test)


def _print_rv_metrics(label: str, m: dict) -> None:
    _log(f"{label:<25}  RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  QLIKE={m['qlike']:.4f}")


def _print_shock_metrics(label: str, m: dict) -> None:
    _log(f"{label:<25}  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")


def _print_dm(label: str, dm: dict) -> None:
    sign = "M1 < M2" if dm["model1_better"] else "M1 > M2"
    _log(f"{label:<35}  stat={dm['statistic']:+.3f}  p={dm['p_value']:.4f}"
         f"  ({sign})  → {dm['interpretation']}")


# ══════════════════════════════════════════════════════════════════════════════
# Regime labelling
# ══════════════════════════════════════════════════════════════════════════════

def label_regimes(feat_df_test: pl.DataFrame) -> pl.DataFrame:
    """
    Add vol_regime and spread_regime columns to the test slice.

    Regime is "high" if the value exceeds the test-set median, "low" otherwise.
    Using the test-set median avoids train→test leakage of threshold values.
    """
    rv_median  = float(feat_df_test["rv_50"].median())
    spd_median = float(feat_df_test["spread"].median())
    return feat_df_test.with_columns([
        pl.when(pl.col("rv_50") > rv_median)
          .then(pl.lit("high")).otherwise(pl.lit("low"))
          .alias("vol_regime"),
        pl.when(pl.col("spread") > spd_median)
          .then(pl.lit("high")).otherwise(pl.lit("low"))
          .alias("spread_regime"),
    ])


def evaluate_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels: np.ndarray,
    task: str = "rv",
) -> dict:
    """
    Return per-regime metrics dict.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        For task="shock", y_pred must be proba array (N, 2).
    regime_labels : np.ndarray of str, shape (N,)
    task : "rv" | "shock"
    """
    results = {}
    for regime in sorted(np.unique(regime_labels)):
        mask = regime_labels == regime
        if task == "rv":
            results[regime] = evaluate_rv_forecast(y_true[mask], y_pred[mask])
        else:
            results[regime] = evaluate_shock_forecast(y_true[mask], y_pred[mask])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Feature ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_ablation(
    splits,
    fcols: list[str],
    feature_groups: dict,
    lgbm_params: dict,
    horizon_key: str = "5s",
) -> dict:
    """
    Train LightGBM with each feature group dropped in turn.

    Returns dict keyed by group name (and "_baseline") with RV metrics
    and delta_rmse vs. baseline.
    """
    y_rv_tr  = splits.y_rv[horizon_key]["train"]
    y_rv_va  = splits.y_rv[horizon_key]["val"]
    y_rv_te  = splits.y_rv[horizon_key]["test"]
    y_sh_tr  = splits.y_shock_spread[horizon_key]["train"]
    y_sh_va  = splits.y_shock_spread[horizon_key]["val"]

    # Baseline with all features
    base = LGBMDualModel(lgbm_params=lgbm_params)
    base.fit(
        splits.X_train, y_rv_tr, y_sh_tr,
        X_val=splits.X_val, y_rv_val=y_rv_va, y_shock_val=y_sh_va,
    )
    base_pred    = base.predict_rv(splits.X_test)
    base_metrics = evaluate_rv_forecast(y_rv_te, base_pred)

    ablation = {"_baseline": base_metrics}

    for group_name, group_cols in feature_groups.items():
        drop_idx = [i for i, c in enumerate(fcols) if c in group_cols]
        keep_idx = [i for i in range(len(fcols)) if i not in drop_idx]
        if not keep_idx:
            continue

        X_tr = splits.X_train[:, keep_idx]
        X_va = splits.X_val[:, keep_idx]
        X_te = splits.X_test[:, keep_idx]

        m = LGBMDualModel(lgbm_params=lgbm_params)
        m.fit(X_tr, y_rv_tr, y_sh_tr, X_val=X_va, y_rv_val=y_rv_va, y_shock_val=y_sh_va)
        pred = m.predict_rv(X_te)
        met  = evaluate_rv_forecast(y_rv_te, pred)
        ablation[group_name] = {
            **met,
            "delta_rmse": met["rmse"] - base_metrics["rmse"],
        }
        _log(f"  drop '{group_name}': RMSE={met['rmse']:.6f}  "
             f"ΔRMSE={met['rmse'] - base_metrics['rmse']:+.6f}")

    return ablation


# ══════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train(
    cfg: PipelineConfig,
    data_path: str | None = None,
    train_tcn: bool = True,
    train_transformer: bool = False,
    train_garch: bool = True,
    seq_horizon_s: int = 5,
    seed: int = 42,
    max_rows: int | None = None,
) -> dict:
    """
    Full training pipeline with MLflow tracking.

    Parameters
    ----------
    cfg : PipelineConfig
    data_path : str | None
        For parquet source, the file path to load.
    train_tcn : bool
        Whether to train the TCN.
    train_garch : bool
        Whether to fit the GARCH baseline.
    seq_horizon_s : int
        Which horizon (seconds) to train the TCN/Transformer on.
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict with results
    """
    # Set seeds for reproducibility
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg.models.lgbm_params["seed"] = seed

    horizons = cfg.features.horizons
    tps      = cfg.features.ticks_per_second

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n{'#' * 72}")
        print(f"  MLflow run: {run_id}")
        print(f"  Experiment: {EXPERIMENT_NAME}")
        print(f"{'#' * 72}")

        mlflow.log_params({
            "seed":            seed,
            "source":          cfg.data.source,
            "symbol":          cfg.data.symbol,
            "train_frac":      cfg.data.train_frac,
            "val_frac":        cfg.data.val_frac,
            "horizons":        str(horizons),
            "ticks_per_second": tps,
            "rv_windows":      str(cfg.features.rv_windows),
            "ofi_window":      cfg.features.ofi_window,
            "har_lags":        str(cfg.features.har_lags),
            "lgbm_n_estimators": cfg.models.lgbm_params.get("n_estimators"),
            "lgbm_lr":         cfg.models.lgbm_params.get("learning_rate"),
            "lgbm_num_leaves": cfg.models.lgbm_params.get("num_leaves"),
            "tcn_enabled":     train_tcn,
            "garch_enabled":   train_garch,
        })

        # ─────────────────────────────────────────────────────────────────────
        # 1. Load raw data
        # ─────────────────────────────────────────────────────────────────────
        _section("1. Data Loading")
        t0 = time.perf_counter()
        loader = DataLoader(
            source=cfg.data.source,
            path=data_path,
            csv_dir=str(cfg.data.csv_dir),
            pair=cfg.data.symbol,
            levels=cfg.data.lob_levels,
            poll_interval_s=0.5,
        )
        raw_df    = loader.load(n_rows=max_rows)
        load_time = time.perf_counter() - t0

        _log(f"Source:      {cfg.data.source}")
        _log(f"Rows loaded: {len(raw_df):,}")
        _log(f"Columns:     {raw_df.columns}")
        _log(f"Load time:   {load_time:.2f}s")
        mlflow.log_metric("data_rows_raw", len(raw_df))

        if len(raw_df) < 200:
            raise RuntimeError(f"Only {len(raw_df)} snapshots; need >= 200")

        # ─────────────────────────────────────────────────────────────────────
        # 2. Feature engineering
        # ─────────────────────────────────────────────────────────────────────
        _section("2. Feature Engineering")
        t0 = time.perf_counter()
        feat_df = build_features(
            raw_df,
            rv_windows=cfg.features.rv_windows,
            ofi_window=cfg.features.ofi_window,
            har_lags=cfg.features.har_lags,
            horizons=horizons,
            ticks_per_second=tps,
            lob_levels=cfg.data.lob_levels,
        )
        fcols    = feature_cols(feat_df, har_lags=cfg.features.har_lags,
                                lob_levels=cfg.data.lob_levels)
        eng_time = time.perf_counter() - t0

        _log(f"Feature count:    {len(fcols)}")
        for col in fcols:
            s = feat_df[col].drop_nulls()
            _log(f"  {col:<32}  mean={s.mean():.6f}  std={s.std():.6f}  "
                 f"null%={feat_df[col].null_count() / len(feat_df) * 100:.1f}%")
        _log(f"Engineering time: {eng_time:.2f}s")

        mlflow.log_param("n_features",    len(fcols))
        mlflow.log_param("feature_names", str(fcols))

        # ─────────────────────────────────────────────────────────────────────
        # 3. Clean
        # ─────────────────────────────────────────────────────────────────────
        _section("3. Cleaning")
        target_cols = (
            [f"target_rv_{h}s"            for h in horizons] +
            [f"target_shock_spread_{h}s"  for h in horizons] +
            [f"target_shock_depth_{h}s"   for h in horizons
             if f"target_shock_depth_{h}s" in feat_df.columns]
        )
        all_cols    = fcols + target_cols
        rows_before = len(feat_df)
        feat_df     = clean(feat_df, all_cols)
        rows_after  = len(feat_df)
        dropped     = rows_before - rows_after

        _log(f"Rows before: {rows_before:,}")
        _log(f"Rows after:  {rows_after:,}  ({dropped:,} dropped, "
             f"{dropped / rows_before * 100:.1f}%)")

        for h in horizons:
            rv_s = feat_df[f"target_rv_{h}s"]
            sk_s = feat_df[f"target_shock_spread_{h}s"]
            _log(f"  target_rv_{h}s      mean={rv_s.mean():.6f}  std={rv_s.std():.6f}")
            _log(f"  shock_spread_{h}s   rate={sk_s.mean():.3f}")

        mlflow.log_metrics({
            "data_rows_clean":   rows_after,
            "data_rows_dropped": dropped,
        })

        if rows_after < 100:
            raise RuntimeError(f"After cleaning, only {rows_after} rows; need >= 100")

        # ─────────────────────────────────────────────────────────────────────
        # 4. Train/Val/Test splits
        # ─────────────────────────────────────────────────────────────────────
        _section("4. Train / Val / Test Split")
        splits = make_splits(
            feat_df,
            feature_cols=fcols,
            horizons=horizons,
            train_frac=cfg.data.train_frac,
            val_frac=cfg.data.val_frac,
        )
        n_train, n_val, n_test = (
            splits.X_train.shape[0],
            splits.X_val.shape[0],
            splits.X_test.shape[0],
        )
        _log(f"Train:    {n_train:,} rows  ({n_train / rows_after * 100:.1f}%)")
        _log(f"Val:      {n_val:,} rows  ({n_val / rows_after * 100:.1f}%)")
        _log(f"Test:     {n_test:,} rows  ({n_test / rows_after * 100:.1f}%)")
        _log(f"Features: {splits.X_train.shape[1]}")
        mlflow.log_metrics({"n_train": n_train, "n_val": n_val, "n_test": n_test})

        # Recover test slice for regime labelling
        val_end, n = splits.test_slice
        feat_df_test = feat_df.slice(val_end, n - val_end)

        # ─────────────────────────────────────────────────────────────────────
        # 5. HAR-RV baseline
        # ─────────────────────────────────────────────────────────────────────
        _section("5. HAR-RV Baseline")
        t0 = time.perf_counter()
        har_cols = [c for c in fcols if c.startswith("har_rv_")]
        if har_cols:
            har_idx     = [fcols.index(c) for c in har_cols]
            X_har_train = splits.X_train[:, har_idx]
            X_har_test  = splits.X_test[:, har_idx]
        else:
            X_har_train = np.ones((n_train, 1))
            X_har_test  = np.ones((n_test, 1))

        _log(f"HAR features: {har_cols or ['intercept-only']}")
        har = HARRVModel()
        har.fit(X_har_train, splits.y_rv_train)
        har_pred_test = har.predict(X_har_test)

        har_metrics: dict[str, dict] = {}
        for h in horizons:
            key = f"{h}s"
            har_metrics[key] = evaluate_rv_forecast(splits.y_rv[key]["test"], har_pred_test)
            _print_rv_metrics(f"HAR-RV ({key})", har_metrics[key])

        har_time = time.perf_counter() - t0
        _log(f"Fit time: {har_time:.2f}s")
        mlflow.log_metric("har_fit_time", har_time)
        for h in horizons:
            key = f"{h}s"
            mlflow.log_metrics({
                f"har_{key}_rmse":  har_metrics[key]["rmse"],
                f"har_{key}_mae":   har_metrics[key]["mae"],
                f"har_{key}_qlike": har_metrics[key]["qlike"],
            })

        # ─────────────────────────────────────────────────────────────────────
        # 6. GARCH baseline
        # ─────────────────────────────────────────────────────────────────────
        garch_metrics: dict[str, dict] = {}
        garch_pred_test = None

        if train_garch:
            _section("6. GARCH Baseline")
            t0 = time.perf_counter()

            # Extract log returns from feat_df (not in feature matrix)
            log_ret_train = feat_df.slice(0, int(rows_after * cfg.data.train_frac))[
                "log_return"
            ].to_numpy()
            log_ret_test = feat_df.slice(val_end, n - val_end)["log_return"].to_numpy()

            garch = GARCHModel(
                p=cfg.models.garch_p,
                q=cfg.models.garch_q,
                vol=cfg.models.garch_vol,
                dist=cfg.models.garch_dist,
            )
            try:
                garch.fit(log_ret_train)
                garch_cond_vol = garch.forecast_test(log_ret_train, log_ret_test)
                garch_pred_test = garch_cond_vol

                for h in horizons:
                    key = f"{h}s"
                    y_te = splits.y_rv[key]["test"]
                    pred = garch_pred_test[:len(y_te)]
                    garch_metrics[key] = evaluate_rv_forecast(y_te, pred)
                    _print_rv_metrics(f"GARCH ({key})", garch_metrics[key])
                    mlflow.log_metrics({
                        f"garch_{key}_rmse":  garch_metrics[key]["rmse"],
                        f"garch_{key}_mae":   garch_metrics[key]["mae"],
                        f"garch_{key}_qlike": garch_metrics[key]["qlike"],
                    })

                garch_time = time.perf_counter() - t0
                _log(f"Fit time: {garch_time:.2f}s")
                mlflow.log_metric("garch_fit_time", garch_time)

            except Exception as exc:
                _log(f"GARCH failed: {exc} — skipping")
                garch_pred_test = None
        else:
            _section("6. GARCH Baseline")
            _log("Skipped (train_garch=False)")

        # ─────────────────────────────────────────────────────────────────────
        # 7. LightGBM — per horizon
        # ─────────────────────────────────────────────────────────────────────
        _section("7. LightGBM Dual Model (per horizon)")
        _log("Skipped due to scikit-learn compatibility issue")

        lgbm_models:        dict[str, LGBMDualModel] = {}
        lgbm_rv_metrics:    dict[str, dict] = {}
        lgbm_shock_metrics: dict[str, dict] = {}
        lgbm_shock_metrics_calib: dict[str, dict] = {}
        lgbm_rv_preds:      dict[str, np.ndarray] = {}
        primary_key = f"{seq_horizon_s}s" if f"{seq_horizon_s}s" in splits.y_rv else f"{horizons[0]}s"

        # ─────────────────────────────────────────────────────────────────────
        # 8. TCN (optional, one horizon)
        # ─────────────────────────────────────────────────────────────────────
        tcn = None
        tcn_rv_metrics    = None
        tcn_shock_metrics = None
        tcn_key           = f"{seq_horizon_s}s"

        if train_tcn and tcn_key in splits.y_rv:
            _section(f"8. TCN Model ({tcn_key} horizon)")
            t0 = time.perf_counter()

            tcn = TCNModel(
                channels=cfg.models.tcn_channels,
                kernel_size=cfg.models.tcn_kernel_size,
                dropout=cfg.models.tcn_dropout,
                seq_len=cfg.models.tcn_seq_len,
                epochs=cfg.models.tcn_epochs,
                batch_size=cfg.models.tcn_batch,
            )
            _log(f"Device: {tcn.device}")

            mlflow.end_run()
            tcn.fit(
                splits.X_train, splits.y_rv[tcn_key]["train"],
                splits.y_shock_spread[tcn_key]["train"],
                X_val=splits.X_val,
                y_rv_val=splits.y_rv[tcn_key]["val"],
                y_shock_val=splits.y_shock_spread[tcn_key]["val"],
            )
            mlflow.start_run(run_id=run_id)

            tcn_time           = time.perf_counter() - t0
            tcn_rv_pred_test   = tcn.predict_rv(splits.X_test)
            tcn_shock_proba    = tcn.predict_shock_proba(splits.X_test)
            tcn_shock_proba_val = tcn.predict_shock_proba(splits.X_val)

            # Calibrate shock probabilities using validation set
            tcn_shock_proba_calib = calibrate_shock_probabilities(
                splits.y_shock_spread[tcn_key]["val"],
                tcn_shock_proba_val,
                splits.y_shock_spread[tcn_key]["test"],
                tcn_shock_proba
            )

            tcn_rv_metrics     = evaluate_rv_forecast(
                splits.y_rv[tcn_key]["test"], tcn_rv_pred_test
            )
            tcn_shock_metrics  = evaluate_shock_forecast(
                splits.y_shock_spread[tcn_key]["test"], tcn_shock_proba
            )
            tcn_shock_metrics_calib = evaluate_shock_forecast(
                splits.y_shock_spread[tcn_key]["test"], tcn_shock_proba_calib
            )

            _print_rv_metrics(f"TCN RV ({tcn_key})",    tcn_rv_metrics)
            _print_shock_metrics(f"TCN Shock ({tcn_key})", tcn_shock_metrics)
            _print_shock_metrics(f"TCN Shock Calib ({tcn_key})", tcn_shock_metrics_calib)
            _log(f"Fit time: {tcn_time:.2f}s")
            mlflow.log_metrics({
                f"tcn_{tcn_key}_rv_rmse":     tcn_rv_metrics["rmse"],
                f"tcn_{tcn_key}_rv_qlike":    tcn_rv_metrics["qlike"],
                f"tcn_{tcn_key}_shock_auroc": tcn_shock_metrics["auroc"],
                f"tcn_{tcn_key}_shock_auprc": tcn_shock_metrics["auprc"],
                f"tcn_{tcn_key}_shock_brier": tcn_shock_metrics["brier"],
                f"tcn_{tcn_key}_shock_auroc_calib": tcn_shock_metrics_calib["auroc"],
                f"tcn_{tcn_key}_shock_auprc_calib": tcn_shock_metrics_calib["auprc"],
                f"tcn_{tcn_key}_shock_brier_calib": tcn_shock_metrics_calib["brier"],
                "tcn_fit_time": tcn_time,
            })
        else:
            _section("8. TCN")
            _log("Skipped (train_tcn=False or horizon not available)")

        # ─────────────────────────────────────────────────────────────────────
        # 9. Transformer (optional, one horizon)
        # ─────────────────────────────────────────────────────────────────────
        transformer = None
        transformer_rv_metrics    = None
        transformer_shock_metrics = None
        transformer_key           = tcn_key

        if train_transformer and transformer_key in splits.y_rv:
            _section(f"9. Transformer Model ({transformer_key} horizon)")
            t0 = time.perf_counter()

            transformer = TransformerModel(
                d_model=cfg.models.transformer_d_model,
                nhead=cfg.models.transformer_nhead,
                num_layers=cfg.models.transformer_num_layers,
                dim_feedforward=cfg.models.transformer_dim_feedforward,
                dropout=cfg.models.transformer_dropout,
                seq_len=cfg.models.transformer_seq_len,
                epochs=cfg.models.transformer_epochs,
                batch_size=cfg.models.transformer_batch,
            )
            _log(f"Device: {transformer.device}")

            mlflow.end_run()
            transformer.fit(
                splits.X_train, splits.y_rv[transformer_key]["train"],
                splits.y_shock_spread[transformer_key]["train"],
                X_val=splits.X_val,
                y_rv_val=splits.y_rv[transformer_key]["val"],
                y_shock_val=splits.y_shock_spread[transformer_key]["val"],
            )
            mlflow.start_run(run_id=run_id)

            transformer_time           = time.perf_counter() - t0
            transformer_rv_pred_test   = transformer.predict_rv(splits.X_test)
            transformer_shock_proba    = transformer.predict_shock_proba(splits.X_test)
            transformer_shock_proba_val = transformer.predict_shock_proba(splits.X_val)

            transformer_shock_proba_calib = calibrate_shock_probabilities(
                splits.y_shock_spread[transformer_key]["val"],
                transformer_shock_proba_val,
                splits.y_shock_spread[transformer_key]["test"],
                transformer_shock_proba,
            )

            transformer_rv_metrics    = evaluate_rv_forecast(
                splits.y_rv[transformer_key]["test"], transformer_rv_pred_test
            )
            transformer_shock_metrics = evaluate_shock_forecast(
                splits.y_shock_spread[transformer_key]["test"], transformer_shock_proba
            )
            transformer_shock_metrics_calib = evaluate_shock_forecast(
                splits.y_shock_spread[transformer_key]["test"], transformer_shock_proba_calib
            )

            _print_rv_metrics(f"Transformer RV ({transformer_key})",    transformer_rv_metrics)
            _print_shock_metrics(f"Transformer Shock ({transformer_key})", transformer_shock_metrics)
            _print_shock_metrics(f"Transformer Shock Calib ({transformer_key})", transformer_shock_metrics_calib)
            _log(f"Fit time: {transformer_time:.2f}s")
            mlflow.log_metrics({
                f"transformer_{transformer_key}_rv_rmse":          transformer_rv_metrics["rmse"],
                f"transformer_{transformer_key}_rv_qlike":         transformer_rv_metrics["qlike"],
                f"transformer_{transformer_key}_shock_auroc":      transformer_shock_metrics["auroc"],
                f"transformer_{transformer_key}_shock_auprc":      transformer_shock_metrics["auprc"],
                f"transformer_{transformer_key}_shock_brier":      transformer_shock_metrics["brier"],
                f"transformer_{transformer_key}_shock_auroc_calib": transformer_shock_metrics_calib["auroc"],
                f"transformer_{transformer_key}_shock_auprc_calib": transformer_shock_metrics_calib["auprc"],
                f"transformer_{transformer_key}_shock_brier_calib": transformer_shock_metrics_calib["brier"],
                "transformer_fit_time": transformer_time,
            })

            if transformer is not None:
                path = Path("models") / "transformer_model.pkl"
                import pickle
                with open(path, "wb") as f:
                    pickle.dump(transformer, f)
                mlflow.log_artifact(str(path))
                _log(f"Saved {path}")
        else:
            _section("9. Transformer")
            _log("Skipped (train_transformer=False or horizon not available)")

        # ─────────────────────────────────────────────────────────────────────
        # 10. Regime-based evaluation
        # ─────────────────────────────────────────────────────────────────────
        _section("10. Regime-Based Evaluation")
        regime_results: dict = {}

        if lgbm_rv_preds and "rv_50" in feat_df_test.columns and "spread" in feat_df_test.columns:
            feat_df_test = label_regimes(feat_df_test)
            vol_labels    = feat_df_test["vol_regime"].to_numpy()
            spread_labels = feat_df_test["spread_regime"].to_numpy()

            # Evaluate LGBM on primary horizon by regime
            lgbm_rv_pred_primary = lgbm_rv_preds[primary_key]
            y_rv_te_primary      = splits.y_rv[primary_key]["test"]
            n_common = min(len(y_rv_te_primary), len(vol_labels),
                          len(lgbm_rv_pred_primary))

            regime_results["vol_regime"] = evaluate_by_regime(
                y_rv_te_primary[:n_common],
                lgbm_rv_pred_primary[:n_common],
                vol_labels[:n_common],
                task="rv",
            )
            regime_results["spread_regime"] = evaluate_by_regime(
                y_rv_te_primary[:n_common],
                lgbm_rv_pred_primary[:n_common],
                spread_labels[:n_common],
                task="rv",
            )

            for regime_type, regime_data in regime_results.items():
                _log(f"\n  {regime_type}:")
                for regime_val, metrics in regime_data.items():
                    _log(f"    {regime_val:>5}  RMSE={metrics['rmse']:.6f}  "
                         f"MAE={metrics['mae']:.6f}")
                    mlflow.log_metrics({
                        f"regime_{regime_type}_{regime_val}_rmse":
                            metrics["rmse"],
                        f"regime_{regime_type}_{regime_val}_mae":
                            metrics["mae"],
                    })
        else:
            _log("Skipped (rv_50 or spread not in test slice)")

        # ─────────────────────────────────────────────────────────────────────
        # 11. Feature ablation (LGBM, primary horizon)
        # ─────────────────────────────────────────────────────────────────────
        _section(f"11. Feature Ablation (LGBM, {primary_key} horizon)")
        ablation_results: dict = {}

        if primary_key in splits.y_rv:
            ablation_results = run_feature_ablation(
                splits, fcols, cfg.features.feature_groups,
                cfg.models.lgbm_params, horizon_key=primary_key,
            )
            _log(f"  Baseline RMSE: {ablation_results['_baseline']['rmse']:.6f}")
        else:
            _log("Skipped (primary horizon not in splits)")

        # ─────────────────────────────────────────────────────────────────────
        # 13. Diebold-Mariano tests (primary horizon)
        # ─────────────────────────────────────────────────────────────────────
        _section("12. Diebold-Mariano Tests")
        dm_results: dict = {}

        if lgbm_rv_preds and primary_key in lgbm_rv_preds:
            _log(f"Horizon: {primary_key}   H0: equal forecast accuracy   p<0.05 → reject")
            _log("")

            y_rv_te = splits.y_rv[primary_key]["test"]

            dm_lgbm_vs_har = diebold_mariano(
                y_rv_te, har_pred_test[:len(y_rv_te)], lgbm_rv_preds[primary_key]
            )
            _print_dm("LGBM vs HAR", dm_lgbm_vs_har)
            dm_results["lgbm_vs_har"] = dm_lgbm_vs_har
            mlflow.log_metrics({
                "dm_lgbm_vs_har_stat":   dm_lgbm_vs_har["statistic"],
                "dm_lgbm_vs_har_pvalue": dm_lgbm_vs_har["p_value"],
            })

            if garch_pred_test is not None:
                garch_pred_prim = garch_pred_test[:len(y_rv_te)]
                dm_lgbm_vs_garch = diebold_mariano(y_rv_te, garch_pred_prim,
                                                   lgbm_rv_preds[primary_key])
                _print_dm("LGBM vs GARCH", dm_lgbm_vs_garch)
                dm_results["lgbm_vs_garch"] = dm_lgbm_vs_garch

            if tcn is not None and tcn_key == primary_key:
                tcn_rv_pred_test_prim = tcn.predict_rv(splits.X_test)
                dm_tcn_vs_har  = diebold_mariano(y_rv_te, har_pred_test[:len(y_rv_te)],
                                                 tcn_rv_pred_test_prim[:len(y_rv_te)])
                dm_tcn_vs_lgbm = diebold_mariano(y_rv_te, lgbm_rv_preds[primary_key],
                                                 tcn_rv_pred_test_prim[:len(y_rv_te)])
                _print_dm("TCN vs HAR",  dm_tcn_vs_har)
                _print_dm("TCN vs LGBM", dm_tcn_vs_lgbm)
                dm_results["tcn_vs_har"]  = dm_tcn_vs_har
                dm_results["tcn_vs_lgbm"] = dm_tcn_vs_lgbm
        else:
            _log("Skipped (LGBM not trained)")

        # ─────────────────────────────────────────────────────────────────────
        # 13. Save models
        # ─────────────────────────────────────────────────────────────────────
        _section("13. Saving")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        for name, obj in [("har_model", har)]:
            path = models_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            mlflow.log_artifact(str(path))
            _log(f"Saved {path}")

        # Save one LGBM per horizon
        for key, m in lgbm_models.items():
            path = models_dir / f"lgbm_model_{key}.pkl"
            with open(path, "wb") as f:
                pickle.dump(m, f)
            mlflow.log_artifact(str(path))
            _log(f"Saved {path}")

        if lgbm_models and primary_key in lgbm_models:
            path_legacy = models_dir / "lgbm_model.pkl"
            with open(path_legacy, "wb") as f:
                pickle.dump(lgbm_models[primary_key], f)

        if tcn is not None:
            path = models_dir / "tcn_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(tcn, f)
            mlflow.log_artifact(str(path))
            _log(f"Saved {path}")

        # ─────────────────────────────────────────────────────────────────────
        # Build results dict
        # ─────────────────────────────────────────────────────────────────────
        per_horizon: dict = {}
        for h in horizons:
            key = f"{h}s"
            per_horizon[key] = {
                "har":        har_metrics.get(key, {}),
                "garch":      garch_metrics.get(key, {}),
                "lgbm_rv":    lgbm_rv_metrics.get(key, {}),
                "lgbm_shock": lgbm_shock_metrics.get(key, {}),
            }
            if tcn_rv_metrics is not None and key == tcn_key:
                per_horizon[key]["tcn_rv"]    = tcn_rv_metrics
                per_horizon[key]["tcn_shock"] = tcn_shock_metrics

        results = {
            "horizons":    horizons,
            "per_horizon": per_horizon,
            "regime_eval": {
                k: {r: {m: float(v) for m, v in metrics.items()}
                    for r, metrics in v.items()}
                for k, v in regime_results.items()
            },
            "feature_ablation": {
                k: {m: float(v) for m, v in metrics.items()}
                for k, metrics in ablation_results.items()
            },
            "diebold_mariano": dm_results,
            # ── Legacy flat keys (primary horizon) for backward compat ──
            "har_rmse":  float(har_metrics[primary_key]["rmse"]),
            "har_mae":   float(har_metrics[primary_key]["mae"]),
            **({"lgbm_rmse": float(lgbm_rv_metrics[primary_key]["rmse"]),
                "lgbm_mae":  float(lgbm_rv_metrics[primary_key]["mae"]),
                "rmse":      float(lgbm_rv_metrics[primary_key]["rmse"]),
                "auroc":     float(lgbm_shock_metrics[primary_key]["auroc"]),
                "auprc":     float(lgbm_shock_metrics[primary_key]["auprc"]),
                "brier":     float(lgbm_shock_metrics[primary_key]["brier"]),
               } if lgbm_rv_metrics and primary_key in lgbm_rv_metrics else {}),
        }
        if tcn_rv_metrics is not None:
            results["tcn_rmse"]  = float(tcn_rv_metrics["rmse"])
            results["tcn_auroc"] = float(tcn_shock_metrics["auroc"])
            results["tcn_auprc"] = float(tcn_shock_metrics["auprc"])
            results["tcn_brier"] = float(tcn_shock_metrics["brier"])

        results_path = Path("results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(results_path))
        _log(f"Saved {results_path}")

        # ─────────────────────────────────────────────────────────────────────
        # Summary table
        # ─────────────────────────────────────────────────────────────────────
        _section("Summary")
        _log(f"{'Horizon':<8} {'Model':<12}  {'RMSE':>10}  {'MAE':>10}  "
             f"{'QLIKE':>8}  {'AUROC':>8}  {'AUPRC':>8}")
        _log("-" * 72)
        for h in horizons:
            key = f"{h}s"
            ph  = per_horizon[key]
            for mname, mkey in [("HAR-RV", "har"), ("GARCH", "garch"), ("LGBM", "lgbm_rv")]:
                m = ph.get(mkey, {})
                if not m:
                    continue
                sh = ph.get("lgbm_shock", {}) if mname == "LGBM" else {}
                _log(f"{key:<8} {mname:<12}  "
                     f"{m.get('rmse', float('nan')):>10.6f}  "
                     f"{m.get('mae',  float('nan')):>10.6f}  "
                     f"{m.get('qlike',float('nan')):>8.4f}  "
                     f"{sh.get('auroc', float('nan')):>8.4f}  "
                     f"{sh.get('auprc', float('nan')):>8.4f}")
        _log("")
        _log(f"MLflow run ID: {run_id}")
        _log("View UI:       mlflow ui")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train volatility forecasting models")
    parser.add_argument("--source",         required=True, choices=["parquet", "csv", "kraken"])
    parser.add_argument("--path",           type=str,   default=None)
    parser.add_argument("--pair",           type=str,   default="XBTUSD")
    parser.add_argument("--tcn",            action="store_true", help="Train TCN model")
    parser.add_argument("--transformer",    action="store_true", help="Train Transformer model")
    parser.add_argument("--no-garch",       action="store_true", help="Skip GARCH baseline")
    parser.add_argument("--horizons",       nargs="+", type=int, default=[1, 5, 30],
                        help="Forecast horizons in seconds (default: 1 5 30)")
    parser.add_argument("--seq-horizon",    type=int,   default=5,
                        help="Horizon (s) for TCN/Transformer (default: 5)")
    parser.add_argument("--max-rows",       type=int,   default=None,
                        help="Max rows to load (default: all)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.data.source          = args.source
    cfg.features.horizons    = args.horizons
    if args.source == "kraken":
        cfg.data.symbol = args.pair

    try:
        results = train(
            cfg,
            data_path=args.path,
            train_tcn=args.tcn,
            train_transformer=args.transformer,
            train_garch=not args.no_garch,
            seq_horizon_s=args.seq_horizon,
            max_rows=args.max_rows,
        )
        print("\n✓ Training completed successfully")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
