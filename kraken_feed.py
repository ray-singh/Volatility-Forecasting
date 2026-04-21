"""
Kraken L2 order-book data ingestion.

Two collection modes:
  1. REST polling  — collect_kraken_snapshots()
     Polls GET /0/public/Depth at a configurable interval.

  2. WebSocket streaming — collect_kraken_ws_snapshots()
     Subscribes to the Kraken WebSocket v2 `book` channel for real-time
     order-book updates. Each snapshot is taken whenever the book changes,
     up to `n_snapshots` total, or after `timeout_s` seconds.

Both emit the same LOB schema so the feature pipeline works unchanged:
    timestamp_ns : int64   — event time in nanoseconds
    mid_price    : float64 — (best_bid + best_ask) / 2
    spread       : float64 — best_ask - best_bid
    bid_p{i}     : float64 — bid price at level i  (i = 0 … levels-1)
    bid_q{i}     : float64 — bid quantity at level i
    ask_p{i}     : float64 — ask price at level i
    ask_q{i}     : float64 — ask quantity at level i
    last_trade_qty: float64 — None (not available from /Depth or book channel)

Kraken API references:
    REST:  https://docs.kraken.com/api/docs/rest-api/get-order-book
    WS v2: https://docs.kraken.com/api/docs/websocket-v2/book
"""
from __future__ import annotations

import asyncio
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
    bid_qty_cols = [f"bid_q{i}" for i in range(levels)]
    ask_qty_cols = [f"ask_q{i}" for i in range(levels)]
    bid_depth = pl.sum_horizontal(bid_qty_cols)
    ask_depth = pl.sum_horizontal(ask_qty_cols)
    df = df.with_columns(
        (bid_depth / (ask_depth + 1e-9)).alias("depth_ratio")
    )

    print(f"[kraken] done — {len(df):,} snapshots, "
          f"mid_price range [{df['mid_price'].min():.2f}, {df['mid_price'].max():.2f}]")
    return df


# ---------------------------------------------------------------------------
# WebSocket streaming collector
# ---------------------------------------------------------------------------

_KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

# Kraken WS v2 uses different symbol format: BTC/USD instead of XBTUSD
_WS_SYMBOL_MAP = {
    "XBTUSD":  "BTC/USD",
    "ETHUSD":  "ETH/USD",
    "DOGEUSD": "DOGE/USD",
    "SOLUSD":  "SOL/USD",
}


def _apply_book_delta(
    bids: dict[float, float],
    asks: dict[float, float],
    delta: dict,
) -> None:
    """
    Apply one WS book update (snapshot or delta) in-place to the local
    bid/ask dicts, keyed by price → quantity. Zero quantity = remove level.

    Kraken WS v2 sends levels as {"price": float, "qty": float} dicts.
    """
    for level in delta.get("bids", []):
        price, qty = float(level["price"]), float(level["qty"])
        if qty == 0.0:
            bids.pop(price, None)
        else:
            bids[price] = qty

    for level in delta.get("asks", []):
        price, qty = float(level["price"]), float(level["qty"])
        if qty == 0.0:
            asks.pop(price, None)
        else:
            asks[price] = qty


def _book_to_row(bids: dict, asks: dict, levels: int) -> dict[str, Any] | None:
    """Convert current best-N bid/ask dicts into a LOB snapshot row."""
    sorted_bids = sorted(bids.items(), key=lambda x: -x[0])[:levels]
    sorted_asks = sorted(asks.items(), key=lambda x:  x[0])[:levels]

    if not sorted_bids or not sorted_asks:
        return None

    row: dict[str, Any] = {"timestamp_ns": time.time_ns()}
    for i in range(levels):
        if i < len(sorted_bids):
            row[f"bid_p{i}"], row[f"bid_q{i}"] = sorted_bids[i]
        else:
            row[f"bid_p{i}"], row[f"bid_q{i}"] = float("nan"), 0.0
        if i < len(sorted_asks):
            row[f"ask_p{i}"], row[f"ask_q{i}"] = sorted_asks[i]
        else:
            row[f"ask_p{i}"], row[f"ask_q{i}"] = float("nan"), 0.0

    row["mid_price"] = (sorted_bids[0][0] + sorted_asks[0][0]) / 2.0
    row["spread"]    = sorted_asks[0][0] - sorted_bids[0][0]
    row["last_trade_qty"] = None
    return row


async def _ws_collect(
    symbol: str,
    n_snapshots: int,
    levels: int,
    timeout_s: float,
    on_snapshot=None,
) -> list[dict]:
    """
    Core async coroutine: subscribe to the Kraken WS v2 book channel,
    maintain a local order book, and record a snapshot on every update.
    """
    import websockets

    rows: list[dict] = []
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}

    subscribe_msg = json.dumps({
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": [symbol],
            "depth":  max(levels, 10),  # Kraken minimum depth is 10
        },
    })

    deadline = time.monotonic() + timeout_s

    async with websockets.connect(_KRAKEN_WS_URL) as ws:
        await ws.send(subscribe_msg)
        print(f"[kraken-ws] subscribed to book/{symbol} (depth={max(levels,10)})")

        while len(rows) < n_snapshots and time.monotonic() < deadline:
            try:
                remaining = deadline - time.monotonic()
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                break

            msg = json.loads(raw)

            # Only process book channel messages
            if msg.get("channel") != "book":
                continue

            msg_type = msg.get("type")  # "snapshot" | "update"
            for entry in msg.get("data", []):
                _apply_book_delta(bids, asks, entry)

            if msg_type in ("snapshot", "update"):
                row = _book_to_row(bids, asks, levels)
                if row is not None:
                    rows.append(row)
                    if on_snapshot is not None:
                        on_snapshot(len(rows), row)
                    if len(rows) % 50 == 0 or len(rows) == 1:
                        print(f"[kraken-ws]   {len(rows)}/{n_snapshots} snapshots "
                              f"  mid={row['mid_price']:.2f}")

    print(f"[kraken-ws] done — {len(rows)} snapshots collected")
    return rows


def _finalise_df(rows: list[dict], levels: int) -> pl.DataFrame:
    """Convert row list to a polars DataFrame with depth_ratio column."""
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("timestamp_ns").cast(pl.Int64))
    bid_depth = sum(pl.col(f"bid_q{i}") for i in range(levels))
    ask_depth = sum(pl.col(f"ask_q{i}") for i in range(levels))
    df = df.with_columns(
        (bid_depth / (ask_depth + 1e-9)).alias("depth_ratio")
    )
    return df


def collect_kraken_ws_snapshots(
    pair: str = "XBTUSD",
    n_snapshots: int = 500,
    levels: int = 10,
    timeout_s: float = 300.0,
    on_snapshot=None,
) -> pl.DataFrame:
    """
    Stream Kraken order-book updates via WebSocket v2 and collect snapshots.

    A snapshot is recorded on every book update message received from Kraken,
    so the effective rate tracks real market activity (typically many per second
    for liquid pairs). Collection stops when `n_snapshots` are reached or
    `timeout_s` seconds elapse, whichever comes first.

    Parameters
    ----------
    pair        : Kraken pair string (e.g. "XBTUSD", "ETHUSD").
    n_snapshots : maximum number of snapshots to collect.
    levels      : number of book levels to retain per side.
    timeout_s   : hard time-limit in seconds (default 5 min).
    on_snapshot : optional callback(count: int, row: dict) called per snapshot.

    Returns
    -------
    pl.DataFrame in the standard LOB snapshot schema (same as REST collector).
    """
    symbol = _WS_SYMBOL_MAP.get(pair, pair.replace("USD", "/USD"))
    print(f"[kraken-ws] collecting up to {n_snapshots} snapshots for {symbol} "
          f"({levels} levels, timeout={timeout_s}s) …")

    rows = asyncio.run(_ws_collect(symbol, n_snapshots, levels, timeout_s, on_snapshot))

    if not rows:
        raise RuntimeError("No snapshots collected from Kraken WebSocket.")

    df = _finalise_df(rows, levels)
    print(f"[kraken-ws] mid_price range "
          f"[{df['mid_price'].min():.2f}, {df['mid_price'].max():.2f}]")
    return df


# ---------------------------------------------------------------------------
# Continuous background stream (for Streamlit live page)
# ---------------------------------------------------------------------------

class KrakenWSStream:
    """
    Long-lived WebSocket stream that appends LOB rows to a deque.

    Designed for Streamlit: call start() once (idempotent), then read
    .snapshot_deque from the UI thread at any time.

    Usage
    -----
    stream = KrakenWSStream(pair="XBTUSD", levels=5)
    stream.start()
    rows = list(stream.snapshot_deque)   # latest up to maxlen rows
    stream.stop()
    """

    def __init__(
        self,
        pair: str = "XBTUSD",
        levels: int = 5,
        maxlen: int = 600,
        persist_path: str | None = "data/live_buffer.parquet",
        flush_every: int = 100,
    ):
        from collections import deque
        import threading
        self.pair   = pair
        self.levels = levels
        self.symbol = _WS_SYMBOL_MAP.get(pair, pair.replace("USD", "/USD"))
        self.snapshot_deque: deque = deque(maxlen=maxlen)
        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None
        self._persist_path    = persist_path
        self._flush_every     = flush_every
        self._unflushed: list[dict] = []
        self._flush_lock      = threading.Lock()
        self._rows_persisted  = 0

    def start(self):
        """Start background thread (idempotent — safe to call multiple times)."""
        import threading
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="kraken-ws-stream")
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def rows_persisted(self) -> int:
        """Total rows written to the parquet buffer so far."""
        return self._rows_persisted

    def _flush(self):
        """Append unflushed rows to the parquet buffer (thread-safe)."""
        import pathlib
        with self._flush_lock:
            if not self._unflushed:
                return
            batch = self._unflushed[:]
            self._unflushed.clear()

        new_df = pl.DataFrame(batch).with_columns(pl.col("timestamp_ns").cast(pl.Int64))
        path = pathlib.Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pl.read_parquet(path)
            combined = pl.concat([existing, new_df], how="diagonal")
        else:
            combined = new_df

        combined.write_parquet(path)
        self._rows_persisted = len(combined)
        print(f"[KrakenWSStream] flushed {len(batch)} rows → {path} "
              f"(total {self._rows_persisted:,})")

    def _run(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._stream())
        finally:
            loop.close()

    async def _stream(self):
        import websockets
        subscribe_msg = json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": [self.symbol],
                "depth":  max(self.levels, 10),
            },
        })
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(_KRAKEN_WS_URL, ping_interval=20) as ws:
                    await ws.send(subscribe_msg)
                    while not self._stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(raw)
                        if msg.get("channel") != "book":
                            continue
                        for entry in msg.get("data", []):
                            _apply_book_delta(bids, asks, entry)
                        if msg.get("type") in ("snapshot", "update"):
                            row = _book_to_row(bids, asks, self.levels)
                            if row is not None:
                                self.snapshot_deque.append(row)
                                if self._persist_path:
                                    with self._flush_lock:
                                        self._unflushed.append(row)
                                    if len(self._unflushed) >= self._flush_every:
                                        self._flush()
            except Exception as exc:
                if not self._stop_event.is_set():
                    print(f"[KrakenWSStream] reconnecting after error: {exc}")
                    await asyncio.sleep(2.0)
        # flush any remaining rows on clean stop
        if self._persist_path and self._unflushed:
            self._flush()


# ---------------------------------------------------------------------------
# CLI smoke-test:  python kraken_feed.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Kraken L2 snapshots")
    parser.add_argument("--pair",      default="XBTUSD", help="Kraken pair (e.g. XBTUSD)")
    parser.add_argument("--snapshots", type=int, default=20, help="Number of snapshots")
    parser.add_argument("--levels",    type=int, default=10, help="Book depth levels")
    parser.add_argument("--interval",  type=float, default=1.0, help="Poll interval (s) — REST only")
    parser.add_argument("--timeout",   type=float, default=300.0, help="Timeout (s) — WS only")
    parser.add_argument("--mode",      default="rest", choices=["rest", "ws"],
                        help="Collection mode: rest (poll) or ws (WebSocket stream)")
    parser.add_argument("--out",       default=None, help="Save to parquet path")
    args = parser.parse_args()

    if args.mode == "ws":
        df = collect_kraken_ws_snapshots(
            pair=args.pair,
            n_snapshots=args.snapshots,
            levels=args.levels,
            timeout_s=args.timeout,
        )
    else:
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
