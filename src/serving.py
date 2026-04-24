"""
FastAPI serving layer for the volatility forecasting pipeline.

Endpoints:
    GET  /health              — liveness check + loaded model inventory
    GET  /results             — serve results.json (benchmark metrics)
    POST /predict             — run RV + shock inference on a LOB snapshot batch

Usage:
    uvicorn src.serving:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
    MODEL_DIR   — directory containing *.pkl model files (default: models/)
    LOB_LEVELS  — number of LOB price levels in incoming snapshots (default: 5)
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import FeatureConfig
from .engineer import build_features, clean, feature_cols

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR  = Path(os.getenv("MODEL_DIR",  "models"))
LOB_LEVELS = int(os.getenv("LOB_LEVELS",  "5"))
RESULTS_PATH = Path("results.json")

_feat_cfg = FeatureConfig()

# ── Model registry ────────────────────────────────────────────────────────────

_models: dict[str, Any] = {}


class _ModuleRemapUnpickler(pickle.Unpickler):
    """Remap bare module names saved before the src/ restructure."""
    def find_class(self, module, name):
        if module in ("models", "deep_models", "config", "engineer", "dataset"):
            module = f"src.{module}"
        return super().find_class(module, name)


def _load_models() -> None:
    for h in ("1s", "5s", "30s"):
        path = MODEL_DIR / f"lgbm_model_{h}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _models[h] = _ModuleRemapUnpickler(f).load()
    if not _models:
        print("[serving] Warning: no LGBM models found in", MODEL_DIR)
    else:
        print(f"[serving] Loaded models for horizons: {list(_models)}")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VolCast API",
    description="Real-time volatility and shock probability forecasting from LOB snapshots.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    _load_models()


# ── Schemas ───────────────────────────────────────────────────────────────────

class LOBSnapshot(BaseModel):
    """
    A single LOB snapshot row. Fields match the schema produced by kraken_feed.py.
    At minimum, bid_p0/ask_p0/bid_q0/ask_q0/mid_price/spread/timestamp_ns are required.
    Deeper levels (bid_p1..bid_p4, etc.) are optional but improve feature quality.
    """
    timestamp_ns: int
    mid_price: float
    spread: float
    log_return: float = 0.0
    bid_p0: float
    ask_p0: float
    bid_q0: float
    ask_q0: float
    bid_p1: float | None = None
    ask_p1: float | None = None
    bid_q1: float | None = None
    ask_q1: float | None = None
    bid_p2: float | None = None
    ask_p2: float | None = None
    bid_q2: float | None = None
    ask_q2: float | None = None
    bid_p3: float | None = None
    ask_p3: float | None = None
    bid_q3: float | None = None
    ask_q3: float | None = None
    bid_p4: float | None = None
    ask_p4: float | None = None
    bid_q4: float | None = None
    ask_q4: float | None = None


class PredictRequest(BaseModel):
    snapshots: list[LOBSnapshot]


class HorizonPrediction(BaseModel):
    rv: float
    shock_probability: float
    shock_flag: bool


class PredictResponse(BaseModel):
    n_snapshots: int
    n_features_used: int
    predictions: dict[str, HorizonPrediction]  # keyed by "1s", "5s", "30s"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _snapshots_to_features(snapshots: list[LOBSnapshot]):
    """
    Convert a list of LOBSnapshot objects into a feature matrix.
    Returns (feat_df, fcols) or raises HTTPException on failure.
    """
    rows = [s.model_dump() for s in snapshots]

    # Fill None LOB levels with NaN so Polars schema is consistent
    for row in rows:
        for i in range(1, 5):
            for side in ("bid", "ask"):
                for kind in ("p", "q"):
                    key = f"{side}_{kind}{i}"
                    if row.get(key) is None:
                        row[key] = float("nan")

    try:
        df = pl.DataFrame(rows)
        df = df.with_columns(pl.col("timestamp_ns").cast(pl.Int64))

        feat_df = build_features(
            df,
            rv_windows=_feat_cfg.rv_windows,
            ofi_window=_feat_cfg.ofi_window,
            har_lags=_feat_cfg.har_lags,
            horizons=_feat_cfg.horizons,
            ticks_per_second=_feat_cfg.ticks_per_second,
            lob_levels=LOB_LEVELS,
        )
        fcols = feature_cols(feat_df, har_lags=_feat_cfg.har_lags, lob_levels=LOB_LEVELS)
        if not fcols:
            raise ValueError("Feature engineering produced no columns")
        feat_df = clean(feat_df, fcols)
        if len(feat_df) == 0:
            raise ValueError("All rows dropped after cleaning")
        return feat_df, fcols
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": list(_models.keys()),
        "lob_levels": LOB_LEVELS,
    }


@app.get("/results")
def results() -> dict:
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="results.json not found — run train.py first")
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if len(req.snapshots) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 10 snapshots for stable features, got {len(req.snapshots)}",
        )
    if not _models:
        raise HTTPException(status_code=503, detail="No models loaded — check MODEL_DIR")

    feat_df, fcols = _snapshots_to_features(req.snapshots)
    last_row = feat_df.tail(1).select(fcols).to_numpy()

    predictions: dict[str, HorizonPrediction] = {}
    for horizon, model in _models.items():
        try:
            rv_val = float(np.clip(model.predict_rv(last_row)[0], 0, None))
            proba  = model.predict_shock_proba(last_row)[0]
            shock_p = float(proba[1]) if len(proba) > 1 else float(proba[0])
            predictions[horizon] = HorizonPrediction(
                rv=rv_val,
                shock_probability=round(shock_p, 6),
                shock_flag=shock_p > 0.5,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed for {horizon}: {exc}")

    return PredictResponse(
        n_snapshots=len(req.snapshots),
        n_features_used=len(fcols),
        predictions=predictions,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving:app", host="0.0.0.0", port=8000, reload=True)
