"""
Cloud Run Job: weekly order book ingestion pipeline.

Steps:
  1. Scrape all unprocessed .zip files from quote-saver.bycsi.com
  2. Extract .data file, reconstruct order book (via src/convert_data.py)
  3. Upload parquet to GCS
  4. Log metadata row to Supabase (date, gcs_path, row_count, status)

Environment variables required:
  GCS_BUCKET    - GCS bucket name (e.g. volcast-ray-volcast-prod)
  SUPABASE_URL  - Supabase project URL
  SUPABASE_KEY  - Supabase service role key
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from supabase import create_client

# reuse existing src modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.convert_data import parse_file, reconstruct_book

# ── config ────────────────────────────────────────────────────────────────────

BASE_URL = "https://quote-saver.bycsi.com/orderbook/inverse/BTCUSD/"
GCS_BUCKET = os.environ["GCS_BUCKET"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TOP_N = 5


# ── step 1: scrape ────────────────────────────────────────────────────────────

def get_all_zip_urls() -> list[str]:
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return [
        urljoin(BASE_URL, a["href"])
        for a in soup.find_all("a")
        if a.get("href", "").endswith(".zip")
    ]


def date_from_zip_url(url: str) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    return match.group(1) if match else None


def already_ingested(supabase_client, date_str: str) -> bool:
    result = (
        supabase_client.table("ingest_log")
        .select("date")
        .eq("date", date_str)
        .eq("status", "success")
        .execute()
    )
    return len(result.data) > 0


def download_and_extract(url: str, dest_dir: Path) -> Path:
    """Download zip, extract, return path to the .data file."""
    print(f"  Downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        raw = r.content

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile as e:
        raise ValueError(f"Bad ZIP from {url}") from e

    for p in dest_dir.rglob("*.data"):
        return p
    raise FileNotFoundError(f"No .data file found in ZIP from {url}")


# ── step 2: convert (delegates to src/convert_data.py) ───────────────────────

def data_file_to_parquet_bytes(data_path: Path) -> tuple[bytes, int]:
    raw  = parse_file(data_path)
    book = reconstruct_book(raw, top_n=TOP_N)
    buf  = io.BytesIO()
    book.to_parquet(buf, index=False)
    return buf.getvalue(), len(book)


# ── step 3: upload to GCS ─────────────────────────────────────────────────────

def upload_to_gcs(data: bytes, date_str: str) -> str:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob_name = f"orderbook/{date_str}.parquet"
    bucket.blob(blob_name).upload_from_string(data, content_type="application/octet-stream")
    gcs_path = f"gs://{GCS_BUCKET}/{blob_name}"
    print(f"  Uploaded {len(data)/1e6:.1f} MB → {gcs_path}")
    return gcs_path


# ── step 4: log to Supabase ───────────────────────────────────────────────────

def log_to_supabase(client, date_str: str, gcs_path: str, row_count: int, status: str, error: str = ""):
    client.table("ingest_log").upsert({
        "date":        date_str,
        "gcs_path":    gcs_path,
        "row_count":   row_count,
        "status":      status,
        "error":       error,
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
            print(f"  SKIP (no date in URL): {url}")
            continue

        if already_ingested(supabase, date_str):
            print(f"  SKIP (already ingested): {date_str}")
            continue

        print(f"\nProcessing {date_str}...")
        gcs_path = ""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data_file              = download_and_extract(url, Path(tmp))
                parquet_bytes, n_rows = data_file_to_parquet_bytes(data_file)

            print(f"  {n_rows:,} rows reconstructed")
            gcs_path = upload_to_gcs(parquet_bytes, date_str)
            log_to_supabase(supabase, date_str, gcs_path, n_rows, "success")
            new_count += 1
            print(f"  ✓ {date_str} done")

        except Exception as exc:
            print(f"  ERROR {date_str}: {exc}", file=sys.stderr)
            log_to_supabase(supabase, date_str, gcs_path, 0, "error", str(exc))

    print(f"\nDone. {new_count} new date(s) ingested.")


if __name__ == "__main__":
    main()
