"""
Walk-forward backtest for volatility forecasting models.

Implements expanding-window cross-validation: the initial training window
grows by one fold at each step. Each fold retrains LightGBM from scratch
on all available history, then predicts the next held-out block.

Also computes a simple market-making PnL simulation: quote normally when
predicted shock probability is low; widen the spread (or sit out) when
it is high. This converts AUROC/RMSE into a tangible trading signal.

Usage:
    python -m src.backtest --source csv
    python -m src.backtest --source parquet --path data/orderbook/btcusd_full.parquet
    python -m src.backtest --n-folds 8 --horizon 5
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from .config import PipelineConfig
from .engineer import build_features, feature_cols, clean
from .models import DataLoader, LGBMDualModel
from .train import evaluate_rv_forecast, evaluate_shock_forecast


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward fold generator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Fold:
    idx: int
    train_slice: tuple[int, int]   # [start, end)
    test_slice: tuple[int, int]   # [start, end)


def walk_forward_folds(
    n: int,
    n_folds: int = 5,
    min_train_frac: float = 0.40,
) -> Iterator[Fold]:
    """
    Generate expanding-window fold boundaries.

    The first fold uses `min_train_frac` of data for training. Each
    subsequent fold expands the training window by one fold-width.
    The test window is always the next fold-width block of unseen data.

    Parameters
    ----------
    n : int
        Total number of samples.
    n_folds : int
        Number of folds (default 5). With min_train_frac=0.40 and 5 folds,
        training starts at 40% and each test block is ~12% of the data.
    min_train_frac : float
        Minimum fraction of data used for the first training window.
    """
    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = remaining // (n_folds + 1)

    if fold_size < 50:
        raise ValueError(
            f"Fold size too small ({fold_size} rows). "
            "Use more data or fewer folds."
        )

    for i in range(n_folds):
        train_end = min_train + i * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            break
        yield Fold(
            idx=i,
            train_slice=(0, train_end),
            test_slice=(test_start, test_end),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Market-making PnL simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_mm_pnl(
    spread: np.ndarray,
    shock_proba: np.ndarray,
    rv_pred: np.ndarray,
    rv_true: np.ndarray,
    shock_threshold: float = 0.50,
    half_spread_base: float | None = None,
) -> dict:
    """
    Simplified market-making simulation.

    Strategy:
    - At each tick, the market maker quotes a half-spread equal to the
      predicted RV (scaled to a fair spread). This is a proxy for the
      optimal Avellaneda-Stoikov spread adjustment.
    - When predicted shock probability > threshold: DO NOT quote (sit out).
      This avoids adverse selection during predicted toxic periods.
    - When quoting: earn the spread if the position closes flat, lose to
      adverse selection proportional to realized RV.

    Parameters
    ----------
    spread : np.ndarray
        Raw observed bid-ask spread at each tick (used for scaling).
    shock_proba : np.ndarray
        Predicted probability of a liquidity shock at each tick.
    rv_pred : np.ndarray
        Predicted realized volatility.
    rv_true : np.ndarray
        Actual realized volatility (the adverse-selection cost).
    shock_threshold : float
        Probability above which the MM sits out (default 0.50).
    half_spread_base : float | None
        Override for base half-spread. Defaults to median spread / 2.

    Returns
    -------
    dict with per-tick pnl array and summary statistics.
    """
    n = len(spread)
    if half_spread_base is None:
        half_spread_base = float(np.nanmedian(spread)) / 2.0

    pnl = np.zeros(n)
    quoted_mask = shock_proba < shock_threshold

    # When quoting: earn half_spread_quoted, pay rv_true (adverse selection)
    # half_spread_quoted scales with rv_pred to represent dynamic spread widening
    scale = half_spread_base / (float(np.nanmedian(rv_pred)) + 1e-12)
    half_spread_quoted = np.clip(rv_pred * scale, half_spread_base * 0.5, half_spread_base * 3.0)

    # PnL per tick: spread earned minus adverse-selection cost
    # The 0.5 factor reflects that only ~50% of quotes get filled (simplified)
    pnl[quoted_mask] = (
        half_spread_quoted[quoted_mask] * 0.5
        - rv_true[quoted_mask]
    )
    # Sitting out: zero revenue, zero risk
    pnl[~quoted_mask] = 0.0

    cumulative = np.cumsum(pnl)
    total_ticks = int(n)
    quoted_ticks = int(quoted_mask.sum())
    participation = quoted_ticks / total_ticks if total_ticks > 0 else 0.0

    # Sharpe-like: mean / std of per-tick PnL (annualised at 2 ticks/s)
    pnl_quoted = pnl[quoted_mask]
    sharpe = (
        float(pnl_quoted.mean() / (pnl_quoted.std() + 1e-12)) * np.sqrt(2 * 3600 * 24)
        if len(pnl_quoted) > 1 else 0.0
    )
    max_drawdown = float(np.min(cumulative - np.maximum.accumulate(cumulative)))

    return {
        "total_pnl":      float(cumulative[-1]),
        "sharpe":         float(sharpe),
        "max_drawdown":   max_drawdown,
        "participation":  float(participation),
        "quoted_ticks":   quoted_ticks,
        "total_ticks":    total_ticks,
        "pnl_per_tick":   float(pnl[quoted_mask].mean()) if quoted_ticks > 0 else 0.0,
        "pnl_series":     pnl.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward backtest runner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    fold_idx: int
    n_train: int
    n_test: int
    rv_metrics: dict
    shock_metrics: dict
    pnl: dict
    fit_time_s: float


@dataclass
class BacktestResult:
    horizon: str
    n_folds: int
    folds: list[FoldResult] = field(default_factory=list)

    # Aggregates (populated by summarise())
    rmse_mean: float = 0.0
    rmse_std: float = 0.0
    mae_mean: float = 0.0
    auroc_mean: float = 0.0
    auroc_std: float = 0.0
    sharpe_mean: float = 0.0
    total_pnl: float = 0.0

    def summarise(self) -> BacktestResult:
        rmses = [f.rv_metrics["rmse"] for f in self.folds]
        maes = [f.rv_metrics["mae"] for f in self.folds]
        aurocs = [f.shock_metrics["auroc"] for f in self.folds]
        sharpes = [f.pnl["sharpe"]        for f in self.folds]

        self.rmse_mean = float(np.mean(rmses))
        self.rmse_std = float(np.std(rmses, ddof=1))
        self.mae_mean = float(np.mean(maes))
        self.auroc_mean = float(np.mean(aurocs))
        self.auroc_std = float(np.std(aurocs, ddof=1))
        self.sharpe_mean = float(np.mean(sharpes))
        self.total_pnl = float(sum(f.pnl["total_pnl"] for f in self.folds))
        return self

    def to_dict(self) -> dict:
        return {
            "horizon":    self.horizon,
            "n_folds":    self.n_folds,
            "rmse_mean":  self.rmse_mean,
            "rmse_std":   self.rmse_std,
            "mae_mean":   self.mae_mean,
            "auroc_mean": self.auroc_mean,
            "auroc_std":  self.auroc_std,
            "sharpe_mean": self.sharpe_mean,
            "total_pnl":  self.total_pnl,
            "folds": [
                {
                    "fold":         f.fold_idx,
                    "n_train":      f.n_train,
                    "n_test":       f.n_test,
                    "rv_rmse":      f.rv_metrics["rmse"],
                    "rv_mae":       f.rv_metrics["mae"],
                    "rv_qlike":     f.rv_metrics["qlike"],
                    "shock_auroc":  f.shock_metrics["auroc"],
                    "shock_auprc":  f.shock_metrics["auprc"],
                    "shock_brier":  f.shock_metrics["brier"],
                    "pnl_total":    f.pnl["total_pnl"],
                    "pnl_sharpe":   f.pnl["sharpe"],
                    "pnl_drawdown": f.pnl["max_drawdown"],
                    "participation": f.pnl["participation"],
                    "fit_time_s":   f.fit_time_s,
                }
                for f in self.folds
            ],
        }


def run_backtest(
    feat_df: pl.DataFrame,
    fcols: list[str],
    horizon_s: int = 5,
    n_folds: int = 5,
    min_train_frac: float = 0.40,
    lgbm_params: dict | None = None,
    shock_threshold: float = 0.50,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run walk-forward backtest on LightGBM for a single forecast horizon.

    Parameters
    ----------
    feat_df : pl.DataFrame
        Cleaned feature dataframe (output of engineer.clean).
    fcols : list[str]
        Feature column names.
    horizon_s : int
        Forecast horizon in seconds (default 5).
    n_folds : int
        Number of walk-forward folds (default 5).
    min_train_frac : float
        Fraction of data for first training window (default 0.40).
    lgbm_params : dict | None
        LightGBM hyperparameters (defaults to standard params).
    shock_threshold : float
        Shock probability above which the MM sits out (default 0.50).
    verbose : bool
        Print per-fold results (default True).

    Returns
    -------
    BacktestResult with per-fold and aggregate metrics.
    """
    horizon_key = f"{horizon_s}s"
    rv_col = f"target_rv_{horizon_s}s"
    shock_col = f"target_shock_spread_{horizon_s}s"

    if rv_col not in feat_df.columns:
        raise ValueError(f"Target column '{rv_col}' not in dataframe. "
                         f"Available: {[c for c in feat_df.columns if c.startswith('target_')]}")

    if lgbm_params is None:
        lgbm_params = {
            "n_estimators":    300,
            "learning_rate":   0.05,
            "num_leaves":      31,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "verbosity":       -1,
        }

    X = np.ascontiguousarray(feat_df.select(fcols).to_numpy())
    y_rv = feat_df[rv_col].to_numpy().ravel()
    y_sh = feat_df[shock_col].to_numpy().ravel() if shock_col in feat_df.columns else np.zeros(len(X))
    spread_arr = feat_df["spread"].to_numpy().ravel() if "spread" in feat_df.columns else np.ones(len(X))

    result = BacktestResult(horizon=horizon_key, n_folds=n_folds)
    n = len(X)

    if verbose:
        width = 72
        print(f"\n{'=' * width}")
        print(f"  Walk-Forward Backtest  |  horizon={horizon_key}  |  folds={n_folds}  |  n={n:,}")
        print(f"{'=' * width}")
        print(f"  {'Fold':<5} {'Train':>8} {'Test':>7} {'RMSE':>10} {'AUROC':>8} "
              f"{'Sharpe':>8} {'PnL':>10} {'Time':>7}")
        print(f"  {'-' * 66}")

    for fold in walk_forward_folds(n, n_folds=n_folds, min_train_frac=min_train_frac):
        tr_s, tr_e = fold.train_slice
        te_s, te_e = fold.test_slice

        X_train = X[tr_s:tr_e]
        X_test = X[te_s:te_e]
        y_rv_train = y_rv[tr_s:tr_e]
        y_rv_test = y_rv[te_s:te_e]
        y_sh_train = y_sh[tr_s:tr_e]
        y_sh_test = y_sh[te_s:te_e]
        spread_test = spread_arr[te_s:te_e]

        # Use the last 15% of training as a validation set for early stopping
        val_split = int(len(X_train) * 0.85)
        X_val_fold = X_train[val_split:]
        y_rv_val_f = y_rv_train[val_split:]
        y_sh_val_f = y_sh_train[val_split:]
        X_train_f = X_train[:val_split]
        y_rv_train_f = y_rv_train[:val_split]
        y_sh_train_f = y_sh_train[:val_split]

        t0 = time.perf_counter()
        model = LGBMDualModel(lgbm_params=lgbm_params)
        model.fit(
            X_train_f, y_rv_train_f, y_sh_train_f,
            X_val=X_val_fold, y_rv_val=y_rv_val_f, y_shock_val=y_sh_val_f,
        )
        fit_time = time.perf_counter() - t0

        rv_pred = model.predict_rv(X_test)
        sh_proba = model.predict_shock_proba(X_test)

        # Mask NaNs in targets
        rv_mask = np.isfinite(y_rv_test) & np.isfinite(rv_pred)
        sh_mask = np.isfinite(y_sh_test) & rv_mask

        rv_metrics = evaluate_rv_forecast(y_rv_test[rv_mask], rv_pred[rv_mask])
        shock_metrics = (
            evaluate_shock_forecast(y_sh_test[sh_mask].astype(int), sh_proba[sh_mask])
            if sh_mask.sum() > 10 and len(np.unique(y_sh_test[sh_mask])) == 2
            else {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
        )

        pnl = simulate_mm_pnl(
            spread=spread_test[rv_mask],
            shock_proba=sh_proba[rv_mask, 1],
            rv_pred=rv_pred[rv_mask],
            rv_true=y_rv_test[rv_mask],
            shock_threshold=shock_threshold,
        )

        fold_result = FoldResult(
            fold_idx=fold.idx,
            n_train=len(X_train),
            n_test=len(X_test),
            rv_metrics=rv_metrics,
            shock_metrics=shock_metrics,
            pnl=pnl,
            fit_time_s=fit_time,
        )
        result.folds.append(fold_result)

        if verbose:
            print(
                f"  {fold.idx:<5} {len(X_train):>8,} {len(X_test):>7,} "
                f"{rv_metrics['rmse']:>10.2e} "
                f"{shock_metrics['auroc']:>8.4f} "
                f"{pnl['sharpe']:>8.2f} "
                f"{pnl['total_pnl']:>10.4f} "
                f"{fit_time:>6.1f}s"
            )

    result.summarise()

    if verbose:
        print(f"  {'-' * 66}")
        print(f"  {'MEAN':<5} {'':>8} {'':>7} "
              f"{result.rmse_mean:>10.2e} "
              f"{result.auroc_mean:>8.4f} "
              f"{result.sharpe_mean:>8.2f} "
              f"{result.total_pnl:>10.4f}")
        print(f"  {'STD':<5} {'':>8} {'':>7} "
              f"{result.rmse_std:>10.2e} "
              f"{result.auroc_std:>8.4f}")
        print(f"\n  RMSE:  {result.rmse_mean:.4e} ± {result.rmse_std:.4e}")
        print(f"  AUROC: {result.auroc_mean:.4f} ± {result.auroc_std:.4f}")
        print(f"  Sharpe (MM sim): {result.sharpe_mean:.2f}")
        print(f"  Total PnL (MM sim): {result.total_pnl:.4f}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for volatility forecasting"
    )
    parser.add_argument("--source",   required=True, choices=["parquet", "csv", "kraken"])
    parser.add_argument("--path",     type=str, default=None)
    parser.add_argument("--horizon",  type=int, default=5,
                        help="Forecast horizon in seconds (default: 5)")
    parser.add_argument("--n-folds",  type=int, default=5,
                        help="Number of walk-forward folds (default: 5)")
    parser.add_argument("--min-train-frac", type=float, default=0.40,
                        help="Min fraction of data for first train window (default: 0.40)")
    parser.add_argument("--shock-threshold", type=float, default=0.50,
                        help="Shock probability threshold for MM sit-out (default: 0.50)")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output",   type=str, default="backtest_results.json",
                        help="Path to write results JSON (default: backtest_results.json)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.data.source = args.source

    loader = DataLoader(
        source=cfg.data.source,
        path=args.path,
        csv_dir=str(cfg.data.csv_dir),
    )
    raw_df = loader.load(n_rows=args.max_rows)

    feat_df = build_features(
        raw_df,
        rv_windows=cfg.features.rv_windows,
        ofi_window=cfg.features.ofi_window,
        har_lags=cfg.features.har_lags,
        horizons=[args.horizon],
        ticks_per_second=cfg.features.ticks_per_second,
        lob_levels=cfg.data.lob_levels,
    )
    fcols = feature_cols(feat_df, har_lags=cfg.features.har_lags,
                         lob_levels=cfg.data.lob_levels)

    rv_col = f"target_rv_{args.horizon}s"
    shock_col = f"target_shock_spread_{args.horizon}s"
    all_cols = fcols + [c for c in [rv_col, shock_col, "spread"] if c in feat_df.columns]
    feat_df = clean(feat_df, all_cols)

    result = run_backtest(
        feat_df,
        fcols,
        horizon_s=args.horizon,
        n_folds=args.n_folds,
        min_train_frac=args.min_train_frac,
        lgbm_params=cfg.models.lgbm_params,
        shock_threshold=args.shock_threshold,
    )

    out = result.to_dict()
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()