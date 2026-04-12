"""
Kraken L2 order-book data ingestion.

Polls the Kraken REST endpoint  GET /0/public/Depth  at a configurable
interval and emits one LOB snapshot row per poll, normalised to the same
schema produced by generate_synthetic_lob() so the feature pipeline works
unchanged.

Schema (one row = one snapshot):
    timestamp_ns : int64   — poll time in nanoseconds
    mid_price    : float64 — (best_bid + best_ask) / 2
    spread       : float64 — best_ask - best_bid
    bid_p{i}     : float64 — bid price at level i  (i = 0 … levels-1)
    bid_q{i}     : float64 — bid quantity at level i
    ask_p{i}     : float64 — ask price at level i
    ask_q{i}     : float64 — ask quantity at level i
    last_trade_qty: float64 — None (Kraken /Depth does not include trades)

Kraken API reference:
    https://docs.kraken.com/api/docs/rest-api/get-order-book
"""
from __future__ import annotations

import time
import urllib.request
import json
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Kraken REST helpers
# ---------------------------------------------------------------------------

_KRAKEN_BASE = "https://api.kraken.com"


def _get_depth(pair: str, count: int = 10) -> dict[str, Any]:
    """
    Call  GET /0/public/Depth  and return the parsed JSON.

    Parameters
    ----------
    pair  : Kraken asset pair string, e.g. "XBTUSD" or "XXBTZUSD"
    count : number of price levels to request (1–500)
    """
    url = f"{_KRAKEN_BASE}/0/public/Depth?pair={pair}&count={count}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")

    return data["result"]


def _parse_snapshot(result: dict, pair: str, levels: int) -> dict[str, Any]:
    """
    Parse one /Depth response into a flat dict matching the LOB schema.
    Kraken returns bids/asks sorted best-first (highest bid, lowest ask).
    """
    # Kraken may key the result under the canonical pair name, not the
    # query pair — grab whichever key is present.
    book = result.get(pair) or next(iter(result.values()))

    bids: list[list] = book["bids"]  # [[price_str, qty_str, ts], ...]
    asks: list[list] = book["asks"]

    row: dict[str, Any] = {"timestamp_ns": time.time_ns()}

    for i in range(levels):
        if i < len(bids):
            row[f"bid_p{i}"] = float(bids[i][0])
            row[f"bid_q{i}"] = float(bids[i][1])
        else:
            row[f"bid_p{i}"] = float("nan")
            row[f"bid_q{i}"] = 0.0

        if i < len(asks):
            row[f"ask_p{i}"] = float(asks[i][0])
            row[f"ask_q{i}"] = float(asks[i][1])
        else:
            row[f"ask_p{i}"] = float("nan")
            row[f"ask_q{i}"] = 0.0

    best_bid = row["bid_p0"]
    best_ask = row["ask_p0"]
    row["mid_price"] = (best_bid + best_ask) / 2.0
    row["spread"] = best_ask - best_bid
    row["last_trade_qty"] = None  # not available from /Depth

    return row


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def collect_kraken_snapshots(
    pair: str = "XBTUSD",
    n_snapshots: int = 500,
    levels: int = 10,
    poll_interval_s: float = 0.5,
) -> pl.DataFrame:
    """
    Poll Kraken /Depth and collect `n_snapshots` LOB snapshots.

    Parameters
    ----------
    pair            : Kraken pair string (e.g. "XBTUSD", "ETHUSD")
    n_snapshots     : total number of snapshots to collect
    levels          : number of book levels to retain per side (max 500)
    poll_interval_s : seconds to wait between polls (Kraken rate-limits
                      to ~1 req/s per IP on public endpoints)

    Returns
    -------
    pl.DataFrame in the standard LOB snapshot schema
    """
    rows: list[dict] = []
    print(f"[kraken] collecting {n_snapshots} snapshots for {pair} "
          f"({levels} levels, {poll_interval_s}s interval) …")

    for i in range(n_snapshots):
        try:
            result = _get_depth(pair=pair, count=levels)
            row = _parse_snapshot(result, pair=pair, levels=levels)
            rows.append(row)
        except Exception as exc:
            print(f"[kraken] poll {i} failed: {exc} — skipping")

        if i < n_snapshots - 1:
            time.sleep(poll_interval_s)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[kraken]   {i + 1}/{n_snapshots} snapshots collected")

    if not rows:
        raise RuntimeError("No snapshots collected from Kraken.")

    df = pl.DataFrame(rows)

    # Cast timestamp to Int64 explicitly (time_ns() returns Python int)
    df = df.with_columns(pl.col("timestamp_ns").cast(pl.Int64))

    # Ensure depth_ratio column expected by FEATURE_COLS
    bid_depth = sum(pl.col(f"bid_q{i}") for i in range(levels))
    ask_depth = sum(pl.col(f"ask_q{i}") for i in range(levels))
    df = df.with_columns(
        (bid_depth / (ask_depth + 1e-9)).alias("depth_ratio")
    )

    print(f"[kraken] done — {len(df):,} snapshots, "
          f"mid_price range [{df['mid_price'].min():.2f}, {df['mid_price'].max():.2f}]")
    return df


# ---------------------------------------------------------------------------
# CLI smoke-test:  python kraken_feed.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Kraken L2 snapshots")
    parser.add_argument("--pair",      default="XBTUSD", help="Kraken pair (e.g. XBTUSD)")
    parser.add_argument("--snapshots", type=int, default=20, help="Number of snapshots")
    parser.add_argument("--levels",    type=int, default=10, help="Book depth levels")
    parser.add_argument("--interval",  type=float, default=1.0, help="Poll interval (s)")
    parser.add_argument("--out",       default=None, help="Save to parquet path")
    args = parser.parse_args()

    df = collect_kraken_snapshots(
        pair=args.pair,
        n_snapshots=args.snapshots,
        levels=args.levels,
        poll_interval_s=args.interval,
    )
    print(df.head())

    if args.out:
        df.write_parquet(args.out)
        print(f"[kraken] saved → {args.out}")
