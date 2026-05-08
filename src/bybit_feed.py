"""
Bybit L2 order-book WebSocket stream (inverse-perpetual, v5 API).

Emits the same LOB schema as kraken_feed so the feature pipeline is unchanged:
    timestamp_ns : int64   — exchange event time in nanoseconds (from msg["cts"] ms field)
    mid_price    : float64 — (best_bid + best_ask) / 2
    spread       : float64 — best_ask - best_bid
    bid_p{i}     : float64 — bid price at level i  (i = 0 … levels-1)
    bid_q{i}     : float64 — bid quantity at level i  (USD, coin-margined)
    ask_p{i}     : float64 — ask price at level i
    ask_q{i}     : float64 — ask quantity at level i
    depth_ratio  : float64 — total_bid_depth / total_ask_depth

Raw deltas arrive at ~20ms intervals (~50/s). The stream resamples to a fixed
interval (default 500ms) so downstream feature windows have consistent time
semantics matching the training data (2 ticks/s).

Bybit API references:
    WS v5 inverse: wss://stream.bybit.com/v5/public/inverse
    Book channel:  https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook

Valid depths for inverse endpoint (confirmed live): 1, 50, 200, 500.
depth=25 is NOT supported and silently fails with "handler not found".
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import polars as pl

_BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/inverse"

# Bybit inverse-perpetual symbols (coin-margined, quantities in USD)
_BYBIT_PAIRS = {
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    "SOLUSD": "SOLUSD",
}


def _apply_book_delta(
    bids: dict[float, float],
    asks: dict[float, float],
    data: dict,
) -> None:
    """
    Apply one Bybit book message (snapshot or delta) in-place.
    Bybit sends levels as ["price_str", "qty_str"] pairs.
    Zero quantity means remove the level.
    """
    for price_s, qty_s in data.get("b", []):
        price, qty = float(price_s), float(qty_s)
        if qty == 0.0:
            bids.pop(price, None)
        else:
            bids[price] = qty

    for price_s, qty_s in data.get("a", []):
        price, qty = float(price_s), float(qty_s)
        if qty == 0.0:
            asks.pop(price, None)
        else:
            asks[price] = qty


def _book_to_row(
    bids: dict, asks: dict, levels: int, exchange_ts_ms: int | None = None
) -> dict[str, Any] | None:
    """Convert current best-N bid/ask dicts into a LOB snapshot row.

    exchange_ts_ms: Bybit cts field (confirmed-trade-time, milliseconds).
    Falls back to local clock only when not provided.
    """
    sorted_bids = sorted(bids.items(), key=lambda x: -x[0])[:levels]
    sorted_asks = sorted(asks.items(), key=lambda x:  x[0])[:levels]

    if not sorted_bids or not sorted_asks:
        return None

    ts_ns = exchange_ts_ms * 1_000_000 if exchange_ts_ms is not None else time.time_ns()
    row: dict[str, Any] = {"timestamp_ns": ts_ns}
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
    row["spread"] = sorted_asks[0][0] - sorted_bids[0][0]

    bid_depth = sum(q for _, q in sorted_bids)
    ask_depth = sum(q for _, q in sorted_asks)
    row["depth_ratio"] = bid_depth / (ask_depth + 1e-9)

    return row


# ---------------------------------------------------------------------------
# Continuous background stream (for Streamlit live page)
# ---------------------------------------------------------------------------

class BybitWSStream:
    """
    Long-lived Bybit inverse-perpetual WebSocket stream.

    Identical interface to KrakenWSStream — drop-in replacement.
    Appends LOB snapshot rows to `snapshot_deque` on every book update.

    Usage
    -----
    stream = BybitWSStream(symbol="BTCUSD", levels=5)
    stream.start()
    rows = list(stream.snapshot_deque)
    stream.stop()
    """

    def __init__(
        self,
        symbol: str = "BTCUSD",
        levels: int = 5,
        maxlen: int = 600,
        persist_path: str | None = "data/live_buffer.parquet",
        flush_every: int = 100,
        resample_ms: int = 500,
    ):
        from collections import deque
        import threading

        self.symbol = _BYBIT_PAIRS.get(symbol, symbol)
        self.levels = levels
        # Valid depths for inverse endpoint: 1, 50, 200, 500.
        # depth=25 does NOT exist and silently fails — confirmed by live inspection.
        self._depth = next(d for d in (1, 50, 200, 500) if d >= levels)
        self.snapshot_deque: deque = deque(maxlen=maxlen)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._persist_path = persist_path
        self._flush_every = flush_every
        self._unflushed: list[dict] = []
        self._flush_lock = threading.Lock()
        self._rows_persisted = 0
        # Resample raw ~20ms deltas to a fixed interval matching training cadence.
        # 500ms → 2 ticks/s, consistent with engineer.py ticks_per_second=2.0.
        self._resample_ms = resample_ms

    # -- public interface (mirrors KrakenWSStream) ----------------------------

    def start(self):
        """Start background thread (idempotent)."""
        import threading
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bybit-ws-stream")
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def rows_persisted(self) -> int:
        return self._rows_persisted

    # -- internals ------------------------------------------------------------

    def _flush(self):
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
        print(f"[BybitWSStream] flushed {len(batch)} rows → {path} "
              f"(total {self._rows_persisted:,})")

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._stream())
        finally:
            loop.close()

    async def _stream(self):
        import websockets

        topic = f"orderbook.{self._depth}.{self.symbol}"
        subscribe_msg = json.dumps({"op": "subscribe", "args": [topic]})

        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        # Resampler state: last time we emitted a row (exchange ms).
        _last_emit_ms: int = 0

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(_BYBIT_WS_URL, ping_interval=20) as ws:
                    await ws.send(subscribe_msg)
                    print(f"[BybitWSStream] subscribed to {topic}")

                    while not self._stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"op": "ping"}))
                            continue

                        msg = json.loads(raw)

                        # Skip op-response frames (subscribe ack, ping/pong)
                        if "topic" not in msg:
                            continue
                        if msg.get("topic") != topic:
                            continue

                        msg_type = msg.get("type")  # "snapshot" | "delta"
                        data = msg.get("data", {})
                        # cts = confirmed-trade-time (ms); more precise than ts.
                        cts_ms: int | None = msg.get("cts")

                        if msg_type == "snapshot":
                            bids.clear()
                            asks.clear()
                            # Reset resampler on reconnect so we don't skip the
                            # first row after the new snapshot arrives.
                            _last_emit_ms = 0

                        _apply_book_delta(bids, asks, data)

                        # Resample: only emit once per resample_ms window.
                        event_ms = cts_ms if cts_ms is not None else time.time_ns() // 1_000_000
                        if event_ms - _last_emit_ms < self._resample_ms:
                            continue
                        _last_emit_ms = event_ms

                        row = _book_to_row(bids, asks, self.levels, exchange_ts_ms=cts_ms)
                        if row is None:
                            continue

                        self.snapshot_deque.append(row)
                        if self._persist_path:
                            with self._flush_lock:
                                self._unflushed.append(row)
                            if len(self._unflushed) >= self._flush_every:
                                self._flush()

            except Exception as exc:
                if not self._stop_event.is_set():
                    print(f"[BybitWSStream] reconnecting after error: {exc}")
                    await asyncio.sleep(2.0)

        if self._persist_path and self._unflushed:
            self._flush()
