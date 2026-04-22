"""
Download 1 week of Bybit BTCUSD (inverse perpetual) L2 order book data.

Source: quote-saver.bycsi.com  (Bybit's public historical data mirror)
No API key or registration required.

Output: One Parquet (or CSV) file per day in ./data/orderbook/BTCUSD/

Dependencies:
    pip install requests pandas pyarrow tqdm
"""

import gzip
import io
import json
import os
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOL      = "BTCUSD"          # Bybit inverse perpetual (USD-settled)
# For USDT-settled linear perpetual use "BTCUSDT"

OUTPUT_FMT  = "parquet"         # "parquet" or "csv"
OUTPUT_DIR  = f"./data/orderbook/{SYMBOL}"

# Data mirror (quote-saver.bycsi.com) lags real-time by several months.
# Default to a known-good window from 2025; override via CLI --start/--end.
END_DATE    = date(2025, 8, 20)
START_DATE  = END_DATE - timedelta(days=6)   # 7 days inclusive

# Bybit public data base URL (inverse / linear perpetuals)
BASE_URL = "https://quote-saver.bycsi.com/orderbook/inverse/{symbol}/{date}_{symbol}_ob500.data.zip"

# ── Helpers ───────────────────────────────────────────────────────────────────

def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_zip(url: str) -> bytes | None:
    """Download a ZIP archive; return raw bytes or None on failure."""
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            return r.content
        print(f"  HTTP {r.status_code} — skipping {url}")
        return None
    except requests.RequestException as e:
        print(f"  Request error: {e}")
        return None


def parse_ob_file(raw_bytes: bytes) -> pd.DataFrame:
    """
    Parse a Bybit orderbook .data file (newline-delimited JSON snapshots).
    Each line is a JSON object with fields:
        ts   – server timestamp (ms)
        cts  – client timestamp (ms)
        type – 'snapshot' | 'delta'
        data.b – list of [price, qty] bids
        data.a – list of [price, qty] asks
        data.u – update id
        data.seq – sequence number
    Returns a flat DataFrame with one row per snapshot/delta.
    """
    records = []
    for line in raw_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        data = obj.get("data", {})
        records.append({
            "ts":    obj.get("ts"),
            "cts":   obj.get("cts"),
            "type":  obj.get("type"),
            "u":     data.get("u"),
            "seq":   data.get("seq"),
            # Store bids/asks as compact JSON strings to keep tabular shape.
            # Use pd.read_json / json.loads downstream to reconstruct the book.
            "bids":  json.dumps(data.get("b", [])),
            "asks":  json.dumps(data.get("a", [])),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["ts"]  = pd.to_datetime(df["ts"],  unit="ms", utc=True)
    df["cts"] = pd.to_datetime(df["cts"], unit="ms", utc=True)
    return df


def save(df: pd.DataFrame, path: str, fmt: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False, compression="zstd")
    else:
        df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    symbol: str = SYMBOL,
    start: date = START_DATE,
    end: date = END_DATE,
    fmt: str = OUTPUT_FMT,
):
    output_dir = f"./data/orderbook/{symbol}"
    base_url = "https://quote-saver.bycsi.com/orderbook/inverse/{symbol}/{date}_{symbol}_ob500.data.zip"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {symbol} L2 data  {start} → {end}")
    print(f"Output format : {fmt.upper()}   →   {output_dir}\n")

    for day in tqdm(list(date_range(start, end)), unit="day"):
        ds = day.strftime("%Y-%m-%d")
        out_ext  = "parquet" if fmt == "parquet" else "csv"
        out_path = os.path.join(output_dir, f"{ds}_{symbol}_ob500.{out_ext}")

        if os.path.exists(out_path):
            print(f"[{ds}] Already exists — skipping.")
            continue

        url = base_url.format(symbol=symbol, date=ds)
        print(f"[{ds}] Downloading {url} …")
        raw_zip = download_zip(url)
        if raw_zip is None:
            continue

        # The ZIP contains one .data file (newline-delimited JSON)
        try:
            with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
                data_file = next(
                    (n for n in zf.namelist() if n.endswith(".data")), None
                )
                if data_file is None:
                    print(f"  No .data file found in ZIP — skipping.")
                    continue
                raw_data = zf.read(data_file)
        except zipfile.BadZipFile:
            print("  Bad ZIP — skipping.")
            continue

        print(f"  Parsing …")
        df = parse_ob_file(raw_data)
        if df.empty:
            print("  Empty result — skipping.")
            continue

        save(df, out_path, fmt)

    print("\nDone. Files written to:", os.path.abspath(output_dir))
    print()
    print("Schema of each output file:")
    print("  ts    – server timestamp (UTC, datetime64[ns, UTC])")
    print("  cts   – client timestamp (UTC, datetime64[ns, UTC])")
    print("  type  – 'snapshot' or 'delta'")
    print("  u     – update id (int)")
    print("  seq   – sequence number (int)")
    print("  bids  – JSON string: [[price, qty], ...]  sorted descending")
    print("  asks  – JSON string: [[price, qty], ...]  sorted ascending")
    print()
    print("Tip: to reconstruct the full book at a point in time,")
    print("  start from the latest snapshot before that time,")
    print("  then apply all deltas in order.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Bybit L2 orderbook data")
    parser.add_argument("--start",  default=None, help="Start date YYYY-MM-DD (default: 7 days before --end)")
    parser.add_argument("--end",    default="2025-08-20", help="End date YYYY-MM-DD (default: 2025-08-20, last available)")
    parser.add_argument("--symbol", default=SYMBOL, help="Bybit symbol (default: BTCUSD)")
    parser.add_argument("--fmt",    default=OUTPUT_FMT, choices=["parquet", "csv"])
    args = parser.parse_args()

    end_date   = date.fromisoformat(args.end)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=6)

    main(symbol=args.symbol, start=start_date, end=end_date, fmt=args.fmt)