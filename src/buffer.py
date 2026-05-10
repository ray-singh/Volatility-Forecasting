"""
Stateful ring buffer for incremental LOB feature computation.

Maintains the last `capacity` LOB snapshots and updates all rolling features
in O(1) per tick using running sums and deques — no full DataFrame recompute.

Usage:
    buf = LOBBuffer(capacity=300, lob_levels=5)
    buf.push(snapshot_dict)
    if buf.ready:
        features = buf.features()   # numpy array, shape (n_features,)
"""
from __future__ import annotations

from collections import deque

import numpy as np


# Minimum ticks before features are valid (largest rolling window)
_MIN_TICKS = 300


class _RollingSum:
    """O(1) rolling sum over a fixed window using a deque."""

    def __init__(self, window: int):
        self.window = window
        self._buf: deque[float] = deque()
        self._total = 0.0

    def push(self, value: float) -> float:
        self._buf.append(value)
        self._total += value
        if len(self._buf) > self.window:
            self._total -= self._buf.popleft()
        return self._total

    @property
    def value(self) -> float:
        return self._total

    @property
    def count(self) -> int:
        return len(self._buf)


class _WelfordStats:
    """Online mean and variance (Welford's algorithm)."""

    def __init__(self, window: int):
        self.window = window
        self._buf: deque[float] = deque()
        self._mean = 0.0
        self._m2 = 0.0

    def push(self, value: float):
        if len(self._buf) == self.window:
            old = self._buf.popleft()
            old_mean = self._mean
            self._mean += (old - old_mean) / len(self._buf) if self._buf else 0.0
            self._m2 -= (old - old_mean) * (old - self._mean)
        self._buf.append(value)
        n = len(self._buf)
        delta = value - self._mean
        self._mean += delta / n
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        n = len(self._buf)
        return np.sqrt(self._m2 / n) if n > 1 else 0.0

    @property
    def count(self) -> int:
        return len(self._buf)


class LOBBuffer:
    """
    Incremental feature buffer for one LOB stream.

    Parameters
    ----------
    capacity : int
        Ring buffer size — must be >= largest rolling window (300).
    lob_levels : int
        Number of LOB price levels in incoming snapshots.
    ofi_window : int
        Window for primary OFI rolling mean.
    ticks_per_second : float
        Used to convert second-based OFI windows to tick counts.
    """

    def __init__(
        self,
        capacity: int = 300,
        lob_levels: int = 5,
        ofi_window: int = 20,
        ticks_per_second: float = 10.0,
    ):
        self.capacity = max(capacity, _MIN_TICKS)
        self.lob_levels = lob_levels
        self.ofi_window = ofi_window
        self.ticks_per_second = ticks_per_second

        # Raw snapshot ring buffer (numpy structured or plain list)
        self._snaps: deque[dict] = deque(maxlen=self.capacity)

        # Scalar state carried tick-to-tick
        self._prev_mid: float | None = None
        self._prev_bid_p0: float | None = None
        self._prev_ask_p0: float | None = None
        self._prev_bid_q0: float | None = None
        self._prev_ask_q0: float | None = None

        # Rolling sums for RV windows
        self._rv_sq_50 = _RollingSum(50)
        self._rv_sq_300 = _RollingSum(300)

        # OFI rolling sums
        self._ofi_tick_buf: deque[float] = deque(maxlen=300)
        self._ofi_primary = _RollingSum(ofi_window)
        self._ofi_1s = _RollingSum(max(1, int(round(1 * ticks_per_second))))
        self._ofi_3s = _RollingSum(max(1, int(round(3 * ticks_per_second))))
        self._ofi_5s = _RollingSum(max(1, int(round(5 * ticks_per_second))))

        # Spread stats
        self._spread_welford = _WelfordStats(50)
        self._spread_10 = _RollingSum(10)
        self._spread_20 = _RollingSum(20)
        self._spread_50 = _RollingSum(50)
        self._spread_cnt_10 = 0
        self._spread_cnt_20 = 0
        self._spread_cnt_50 = 0

        # Depth imbalance rolling stats (level 0)
        self._imb_std_10 = _WelfordStats(10)
        self._imb_std_30 = _WelfordStats(30)
        self._imb_min_10: deque[float] = deque(maxlen=10)
        self._imb_max_10: deque[float] = deque(maxlen=10)
        self._imb_min_30: deque[float] = deque(maxlen=30)
        self._imb_max_30: deque[float] = deque(maxlen=30)

        # OFI rolling stats
        self._ofi_welford = _WelfordStats(20)
        self._ofi_min_20: deque[float] = deque(maxlen=20)
        self._ofi_max_20: deque[float] = deque(maxlen=20)

        # Trade intensity (|log_return| rolling sum)
        self._intensity_50 = _RollingSum(50)

        # Log return history for lag features
        self._log_returns: deque[float] = deque(maxlen=self.capacity)

        # RV_50 history for momentum and vol-of-vol
        self._rv50_hist: deque[float] = deque(maxlen=50)
        self._vov_welford = _WelfordStats(50)

        # HAR: rv_300 shifted by 50 ticks
        self._rv300_hist: deque[float] = deque(maxlen=self.capacity)

        # Mid-price history for price velocity
        self._mid_hist: deque[float] = deque(maxlen=self.capacity)

        self._n_ticks = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, snap: dict) -> None:
        """Ingest one LOB snapshot dict and update all incremental state."""
        mid = float(snap.get("mid_price", 0.0))
        spread = float(snap.get("spread", 0.0))
        bid_p0 = float(snap.get("bid_p0", 0.0))
        ask_p0 = float(snap.get("ask_p0", 0.0))
        bid_q0 = float(snap.get("bid_q0", 0.0))
        ask_q0 = float(snap.get("ask_q0", 0.0))

        # Bybit quantity normalisation (same heuristic as engineer.py)
        if bid_q0 > 10 and mid > 0:
            bid_q0 /= mid
            ask_q0 /= mid
            for i in range(1, self.lob_levels):
                k_b = f"bid_q{i}"
                k_a = f"ask_q{i}"
                if k_b in snap:
                    snap[k_b] = float(snap[k_b]) / mid
                if k_a in snap:
                    snap[k_a] = float(snap[k_a]) / mid

        # Log return
        log_ret = np.log(mid / self._prev_mid) if self._prev_mid and self._prev_mid > 0 else 0.0
        self._log_returns.append(log_ret)
        self._mid_hist.append(mid)

        # RV rolling sums
        sq_ret = log_ret ** 2
        self._rv_sq_50.push(sq_ret)
        self._rv_sq_300.push(sq_ret)
        rv50 = np.sqrt(max(0.0, self._rv_sq_50.value))
        rv300 = np.sqrt(max(0.0, self._rv_sq_300.value))
        self._rv50_hist.append(rv50)
        self._rv300_hist.append(rv300)
        self._vov_welford.push(rv50)

        # OFI tick value
        ofi_tick = self._compute_ofi_tick(bid_p0, ask_p0, bid_q0, ask_q0)
        self._ofi_tick_buf.append(ofi_tick)
        ofi_primary = self._ofi_primary.push(ofi_tick) / max(1, self._ofi_primary.count)
        ofi_1s = self._ofi_1s.push(ofi_tick) / max(1, self._ofi_1s.count)
        ofi_3s = self._ofi_3s.push(ofi_tick) / max(1, self._ofi_3s.count)
        ofi_5s = self._ofi_5s.push(ofi_tick) / max(1, self._ofi_5s.count)
        self._ofi_welford.push(ofi_primary)
        self._ofi_min_20.append(ofi_primary)
        self._ofi_max_20.append(ofi_primary)

        # Spread stats
        self._spread_welford.push(spread)
        self._spread_10.push(spread)
        self._spread_20.push(spread)
        self._spread_50.push(spread)

        # Depth imbalance level 0
        depth_imb_0 = (bid_q0 - ask_q0) / (bid_q0 + ask_q0 + 1e-9)
        self._imb_std_10.push(depth_imb_0)
        self._imb_std_30.push(depth_imb_0)
        self._imb_min_10.append(depth_imb_0)
        self._imb_max_10.append(depth_imb_0)
        self._imb_min_30.append(depth_imb_0)
        self._imb_max_30.append(depth_imb_0)

        # Trade intensity
        self._intensity_50.push(abs(log_ret))

        # Store snapshot and update prev state
        self._snaps.append(snap)
        self._prev_mid = mid
        self._prev_bid_p0 = bid_p0
        self._prev_ask_p0 = ask_p0
        self._prev_bid_q0 = bid_q0
        self._prev_ask_q0 = ask_q0
        self._n_ticks += 1

    @property
    def ready(self) -> bool:
        """True once enough ticks have been seen for all rolling windows."""
        return self._n_ticks >= _MIN_TICKS

    def features(self) -> np.ndarray:
        """
        Return the current feature vector as a 1-D float64 numpy array.
        Call only when self.ready is True.
        """
        snap = self._snaps[-1]
        mid = float(snap.get("mid_price", 0.0))
        spread = float(snap.get("spread", 0.0))
        bid_p0 = float(snap.get("bid_p0", 0.0))
        ask_p0 = float(snap.get("ask_p0", 0.0))
        bid_q0 = float(snap.get("bid_q0", 0.0))
        ask_q0 = float(snap.get("ask_q0", 0.0))

        rv50 = np.sqrt(max(0.0, self._rv_sq_50.value))
        rv300 = np.sqrt(max(0.0, self._rv_sq_300.value))

        # HAR lag: rv_300 shifted 50 ticks back
        har_rv_50 = self._rv300_hist[-51] if len(self._rv300_hist) > 50 else rv300

        spread_ma_50 = self._spread_50.value / max(1, self._spread_50.count)
        spread_std_50 = self._spread_welford.std
        spread_zscore = (spread - spread_ma_50) / max(spread_std_50, 1e-9)
        relative_spread = spread / max(mid, 1e-9)

        spread_trend_10 = self._spread_10.value / max(1, self._spread_10.count)
        spread_trend_20 = self._spread_20.value / max(1, self._spread_20.count)
        spread_trend_50 = spread_ma_50

        microprice = (ask_p0 * bid_q0 + bid_p0 * ask_q0) / (bid_q0 + ask_q0 + 1e-9)
        price_pressure = microprice - mid

        queue_imbalance = (bid_q0 - ask_q0) / (bid_q0 + ask_q0 + 1e-9)

        # Depth imbalance per level
        depth_imbs = []
        for lvl in range(min(5, self.lob_levels)):
            bq = float(snap.get(f"bid_q{lvl}", 0.0))
            aq = float(snap.get(f"ask_q{lvl}", 0.0))
            depth_imbs.append((bq - aq) / (bq + aq + 1e-9))

        # Total depth
        total_bid = sum(float(snap.get(f"bid_q{i}", 0.0)) for i in range(self.lob_levels))
        total_ask = sum(float(snap.get(f"ask_q{i}", 0.0)) for i in range(self.lob_levels))

        near_bid = sum(float(snap.get(f"bid_q{i}", 0.0)) for i in range(2))
        far_bid = sum(float(snap.get(f"bid_q{i}", 0.0)) for i in range(3, 5))
        near_ask = sum(float(snap.get(f"ask_q{i}", 0.0)) for i in range(2))
        far_ask = sum(float(snap.get(f"ask_q{i}", 0.0)) for i in range(3, 5))
        depth_near_far_bid = near_bid / (far_bid + 1e-9)
        depth_near_far_ask = near_ask / (far_ask + 1e-9)

        bid_qs = [float(snap.get(f"bid_q{i}", 0.0)) for i in range(self.lob_levels)]
        bid_level_concentration = max(bid_qs) / (sum(bid_qs) + 1e-9)

        far_lvl = min(4, self.lob_levels - 1)
        bid_slope = (bid_q0 - float(snap.get(f"bid_q{far_lvl}", 0.0))) / (far_lvl + 1)
        ask_slope = (ask_q0 - float(snap.get(f"ask_q{far_lvl}", 0.0))) / (far_lvl + 1)

        bid_p1 = float(snap.get("bid_p1", bid_p0))
        ask_p1 = float(snap.get("ask_p1", ask_p0))
        bid_price_gap = bid_p0 - bid_p1
        ask_price_gap = ask_p1 - ask_p0

        ofi_primary = self._ofi_primary.value / max(1, self._ofi_primary.count)
        ofi_1s = self._ofi_1s.value / max(1, self._ofi_1s.count)
        ofi_3s = self._ofi_3s.value / max(1, self._ofi_3s.count)
        ofi_5s = self._ofi_5s.value / max(1, self._ofi_5s.count)

        imb_std_10 = self._imb_std_10.std
        imb_min_10 = min(self._imb_min_10) if self._imb_min_10 else 0.0
        imb_max_10 = max(self._imb_max_10) if self._imb_max_10 else 0.0
        imb_std_30 = self._imb_std_30.std
        imb_min_30 = min(self._imb_min_30) if self._imb_min_30 else 0.0
        imb_max_30 = max(self._imb_max_30) if self._imb_max_30 else 0.0

        ofi_std_20 = self._ofi_welford.std
        ofi_min_20 = min(self._ofi_min_20) if self._ofi_min_20 else 0.0
        ofi_max_20 = max(self._ofi_max_20) if self._ofi_max_20 else 0.0

        trade_intensity_50 = self._intensity_50.value
        log_ret = self._log_returns[-1] if self._log_returns else 0.0
        squared_return = log_ret ** 2

        rv50_hist = list(self._rv50_hist)
        rv_momentum_10 = rv50_hist[-1] - rv50_hist[-11] if len(rv50_hist) > 10 else 0.0
        rv_momentum_20 = rv50_hist[-1] - rv50_hist[-21] if len(rv50_hist) > 20 else 0.0
        vol_of_vol_50 = self._vov_welford.std

        rets = list(self._log_returns)
        lag_return_1 = rets[-2] if len(rets) > 1 else 0.0
        lag_return_5 = rets[-6] if len(rets) > 5 else 0.0
        lag_return_10 = rets[-11] if len(rets) > 10 else 0.0
        lag_return_20 = rets[-21] if len(rets) > 20 else 0.0

        return_mean_10 = np.mean(rets[-10:]) if len(rets) >= 10 else 0.0
        return_mean_30 = np.mean(rets[-30:]) if len(rets) >= 30 else 0.0
        return_mean_50 = np.mean(rets[-50:]) if len(rets) >= 50 else 0.0

        mids = list(self._mid_hist)
        pv5 = (mids[-1] / mids[-6] - 1) if len(mids) > 5 else 0.0
        pv10 = (mids[-1] / mids[-11] - 1) if len(mids) > 10 else 0.0
        pv20 = (mids[-1] / mids[-21] - 1) if len(mids) > 20 else 0.0

        return np.array([
            queue_imbalance,
            ofi_primary,
            ofi_1s,
            ofi_3s,
            ofi_5s,
            rv50,
            rv300,
            har_rv_50,
            spread,
            spread_ma_50,
            spread_zscore,
            relative_spread,
            spread_trend_10,
            spread_trend_20,
            spread_trend_50,
            microprice,
            price_pressure,
            total_bid,
            total_ask,
            depth_near_far_bid,
            depth_near_far_ask,
            bid_level_concentration,
            bid_slope,
            ask_slope,
            bid_price_gap,
            ask_price_gap,
            *depth_imbs,
            imb_std_10,
            imb_min_10,
            imb_max_10,
            imb_std_30,
            imb_min_30,
            imb_max_30,
            ofi_std_20,
            ofi_min_20,
            ofi_max_20,
            trade_intensity_50,
            squared_return,
            rv_momentum_10,
            rv_momentum_20,
            vol_of_vol_50,
            lag_return_1,
            lag_return_5,
            lag_return_10,
            lag_return_20,
            return_mean_10,
            return_mean_30,
            return_mean_50,
            pv5,
            pv10,
            pv20,
        ], dtype=np.float64)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_ofi_tick(
        self,
        bid_p0: float,
        ask_p0: float,
        bid_q0: float,
        ask_q0: float,
    ) -> float:
        if self._prev_bid_p0 is None:
            return 0.0
        dbid_p = bid_p0 - self._prev_bid_p0
        dask_p = ask_p0 - self._prev_ask_p0
        dbid_q = bid_q0 - self._prev_bid_q0
        dask_q = ask_q0 - self._prev_ask_q0

        if dbid_p > 0:
            bid_ofi = self._prev_bid_q0
        elif dbid_p == 0:
            bid_ofi = dbid_q
        else:
            bid_ofi = -self._prev_bid_q0

        if dask_p < 0:
            ask_ofi = self._prev_ask_q0
        elif dask_p == 0:
            ask_ofi = dask_q
        else:
            ask_ofi = self._prev_ask_q0

        return bid_ofi - ask_ofi
