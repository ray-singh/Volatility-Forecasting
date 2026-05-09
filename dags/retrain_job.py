"""
Cloud Run Job: rolling-window retraining for LGBM + HAR models.

Loads all parquet files currently in GCS under orderbook/ (up to the last
14 days — GCS lifecycle deletes files older than 7 days, so in practice
1–2 files are available), concatenates them, and retrains LGBM + HAR.

New models + results.json are uploaded to GCS. TCN/Transformer are skipped
(no GPU in Cloud Run; retrain those manually on Kaggle when needed).

Environment variables:
  GCS_BUCKET   — GCS bucket name (same as ingest job)
  SUPABASE_URL — Supabase project URL  (for retrain_log)
  SUPABASE_KEY — Supabase service role key
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Make src importable when running as Cloud Run Job
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.cloud import storage
from supabase import create_client

from src.config import PipelineConfig
from src.train import train

GCS_BUCKET = os.environ["GCS_BUCKET"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

MIN_DAYS = 2  # abort if fewer than this many days of data are in GCS


def list_orderbook_blobs(gcs_client) -> list[str]:
    bucket = gcs_client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix="orderbook/"))
    return sorted(
        b.name for b in blobs
        if b.name.endswith(".parquet") and b.name != "orderbook/"
    )


def download_parquet(gcs_client, blob_name: str) -> bytes:
    bucket = gcs_client.bucket(GCS_BUCKET)
    return bucket.blob(blob_name).download_as_bytes()


def log_retrain(supabase_client, status: str, days_used: int,
                lgbm_rmse: float | None, error: str = "") -> None:
    supabase_client.table("retrain_log").insert({
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "days_used": days_used,
        "lgbm_rmse": lgbm_rmse,
        "status": status,
        "error": error,
    }).execute()


def main() -> None:
    gcs_client = storage.Client()
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Listing orderbook parquet files in GCS...")
    blob_names = list_orderbook_blobs(gcs_client)
    print(f"Found {len(blob_names)} file(s): {blob_names}")

    if len(blob_names) < MIN_DAYS:
        msg = (f"Only {len(blob_names)} day(s) of data in GCS "
               f"(need >= {MIN_DAYS}). Skipping retrain.")
        print(msg)
        log_retrain(supabase, "skipped", len(blob_names), None, msg)
        return

    # Download and concatenate all available parquet files into one temp file
    import polars as pl

    frames: list[pl.DataFrame] = []
    for blob_name in blob_names:
        print(f"  Downloading {blob_name}...")
        data = download_parquet(gcs_client, blob_name)
        df = pl.read_parquet(io.BytesIO(data))
        frames.append(df)
        print(f"    {len(df):,} rows")

    combined = pl.concat(frames, how="diagonal")
    print(f"Combined dataset: {len(combined):,} rows across {len(frames)} day(s)")

    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "combined.parquet"
        combined.write_parquet(data_path)
        del combined, frames  # free memory before training

        cfg = PipelineConfig()
        cfg.data.source = "parquet"
        cfg.data.lob_levels = 5  # Bybit inverse feed uses 5 levels

        print("\nStarting retraining (LGBM + HAR, no TCN)...")
        try:
            results = train(
                cfg,
                data_path=str(data_path),
                train_tcn=False,
                train_transformer=False,
                train_garch=False,  # GARCH fit is slow on CPU; skip for weekly retrain
                max_rows=None,      # use all available data
            )
        except Exception as exc:
            print(f"Training failed: {exc}")
            log_retrain(supabase, "error", len(blob_names), None, str(exc))
            raise

    lgbm_rmse = results.get("lgbm_rmse") or results.get("rmse")
    print(f"\nRetrain complete. LGBM RMSE: {lgbm_rmse}")
    log_retrain(supabase, "success", len(blob_names), lgbm_rmse)
    print("Done.")


if __name__ == "__main__":
    main()
