"""
Cloud Run Job: weekly order book ingestion pipeline.

Steps:
  1. Scrape latest .zip from quote-saver.bycsi.com
  2. Extract .data file, convert to parquet via convert_data.py logic
  3. Upload parquet to GCS
  4. Insert metadata row into Supabase (date, gcs_path, row_count, status)

Environment variables required:
  GCS_BUCKET          - GCS bucket name (e.g. volcast-data)
  SUPABASE_URL        - Supabase project URL
  SUPABASE_KEY        - Supabase service role key
  GOOGLE_CLOUD_PROJECT - GCP project ID (set automatically on Cloud Run)
"""

from __future__ import annotations
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from supabase import create_client

# ── config ────────────────────────────────────────────────────────────────────

BASE_URL    = "https://quote-saver.bycsi.com/orderbook/inverse/BTCUSD/"
GCS_BUCKET  = os.environ["GCS_BUCKET"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TOP_N = 5


# ── step 1: scrape ────────────────────────────────────────────────────────────

def get_all_zip_urls() -> list[str]:
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    from urllib.parse import urljoin
    return [
        urljoin(BASE_URL, a["href"])
        for a in soup.find_all("a")
        if a.get("href", "").endswith(".zip")
    ]


def already_ingested(supabase_client, date_str: str) -> bool:
    result = (
        supabase_client.table("ingest_log")
        .select("date")
        .eq("date", date_str)
        .eq("status", "success")
        .execute()
    )
    return len(result.data) > 0


def date_from_zip_url(url: str) -> str | None:
    """Extract YYYY-MM-DD from URLs like .../2026-03-26_BTCUSD_ob200.zip"""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    return match.group(1) if match else None


def download_zip(url: str, dest: Path) -> Path:
    print(f"  Downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    return dest


def extract_data_file(zip_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    for p in extract_dir.rglob("*.data"):
        return p
    raise FileNotFoundError(f"No .data file found in {zip_path}")


# ── step 2: convert ───────────────────────────────────────────────────────────

def parse_data_file(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data = obj.get("data", {})
            records.append({
                "ts_ms": obj.get("ts"),
                "type":  obj.get("type"),
                "bids":  data.get("b", []),
                "asks":  data.get("a", []),
            })
    raw = pd.DataFrame(records)
    raw["ts"] = pd.to_datetime(raw["ts_ms"], unit="ms", utc=True)
    return raw


def reconstruct_book(records: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows = []

    for rec in records.itertuples():
        for p, q in ((float(v[0]), float(v[1])) for v in rec.bids):
            if q == 0:
                bids.pop(p, None)
            else:
                bids[p] = q
        for p, q in ((float(v[0]), float(v[1])) for v in rec.asks):
            if q == 0:
                asks.pop(p, None)
            else:
                asks[p] = q

        if not bids or not asks:
            continue

        sorted_bids = sorted(bids.items(), reverse=True)[:top_n]
        sorted_asks = sorted(asks.items())[:top_n]
        best_bid, best_ask = sorted_bids[0][0], sorted_asks[0][0]

        row = {
            "ts":     rec.ts,
            "type":   rec.type,
            "mid":    (best_bid + best_ask) / 2,
            "spread": best_ask - best_bid,
        }
        for i, (pr, q) in enumerate(sorted_bids):
            row[f"bid_p{i}"] = pr
            row[f"bid_q{i}"] = q
        for i, (pr, q) in enumerate(sorted_asks):
            row[f"ask_p{i}"] = pr
            row[f"ask_q{i}"] = q
        rows.append(row)

    return pd.DataFrame(rows)


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


# ── step 3: upload to GCS ─────────────────────────────────────────────────────

def upload_to_gcs(data: bytes, date_str: str) -> str:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob_name = f"orderbook/{date_str}.parquet"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="application/octet-stream")
    gcs_path = f"gs://{GCS_BUCKET}/{blob_name}"
    print(f"  Uploaded {len(data)/1e6:.1f}MB → {gcs_path}")
    return gcs_path


# ── step 4: log to Supabase ───────────────────────────────────────────────────

def log_to_supabase(client, date_str: str, gcs_path: str, row_count: int, status: str, error: str = ""):
    client.table("ingest_log").upsert({
        "date":      date_str,
        "gcs_path":  gcs_path,
        "row_count": row_count,
        "status":    status,
        "error":     error,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }, on_conflict="date").execute()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Fetching ZIP listing...")
    zip_urls = get_all_zip_urls()
    print(f"Found {len(zip_urls)} ZIPs on source")

    new_count = 0
    for url in zip_urls:
        date_str = date_from_zip_url(url)
        if date_str is None:
            print(f"  SKIP (no date): {url}")
            continue

        if already_ingested(supabase, date_str):
            print(f"  SKIP (already ingested): {date_str}")
            continue

        print(f"\nProcessing {date_str}...")
        gcs_path = ""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                zip_path  = download_zip(url, tmp / f"{date_str}.zip")
                data_file = extract_data_file(zip_path, tmp / "extracted")

                print("  Parsing + reconstructing order book...")
                raw  = parse_data_file(data_file)
                book = reconstruct_book(raw, top_n=TOP_N)
                print(f"  {len(book):,} rows")

                parquet_bytes = to_parquet_bytes(book)
                gcs_path = upload_to_gcs(parquet_bytes, date_str)

            log_to_supabase(supabase, date_str, gcs_path, len(book), "success")
            new_count += 1
            print(f"  ✓ {date_str} done")

        except Exception as exc:
            print(f"  ERROR {date_str}: {exc}", file=sys.stderr)
            log_to_supabase(supabase, date_str, gcs_path, 0, "error", str(exc))

    print(f"\nDone. {new_count} new date(s) ingested.")


if __name__ == "__main__":
    main()
