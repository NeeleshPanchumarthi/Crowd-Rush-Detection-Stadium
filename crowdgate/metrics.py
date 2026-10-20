"""Step 4: per-zone metrics in floor units (people/m^2, m/s, people/min)."""

from __future__ import annotations

from collections import deque

import numpy as np

from .geometry import line_normal, segment_crossing


def zone_density(count: float, zone_area_m2: float) -> float:
    """People per m^2 = count in zone / zone area.

    Sparse mode passes the number of tracked foot points inside the zone; dense mode passes the
    density map summed over the zone mask.
    """
    return float(count) / max(zone_area_m2, 1e-6)


def forward_speed(flow_m: np.ndarray, exit_dir: np.ndarray) -> float:
    """Mean velocity component towards the gate (m/s). flow_m is (N, 2) velocities in m/s."""
    if len(flow_m) == 0:
        return float("nan")
    return float(np.mean(flow_m @ exit_dir))


class MotionVariance:
    """Variance of zone flow vectors pooled over a short window (sway and turbulence).

    Pooling over time means both spatial spread (people moving differently) and temporal
    change of the mean vector (the whole block swaying back and forth) count.
    """

    def __init__(self, window_s: float = 2.0):
        self.window_s = window_s
        self.buf: deque = deque()  # (t, n, sum_vec, sum_sq)

    def update(self, t: float, flow_m: np.ndarray) -> float:
        if len(flow_m):
            self.buf.append((t, len(flow_m), flow_m.sum(0), float((flow_m**2).sum())))
        while self.buf and self.buf[0][0] < t - self.window_s:
            self.buf.popleft()
        return self.value()

    def value(self) -> float:
        n = sum(b[1] for b in self.buf)
        if n < 2:
            return 0.0
        s = np.sum([b[2] for b in self.buf], axis=0)
        ss = sum(b[3] for b in self.buf)
        mean = s / n
        return float(ss / n - mean @ mean)  # trace of the covariance, (m/s)^2


def motion_variance(flow_m: np.ndarray) -> float:
    """Single-frame version: trace of the covariance of the velocity vectors."""
    if len(flow_m) < 2:
        return 0.0
    return float(np.trace(np.cov(flow_m.T)))


class LineCounter:
    """Inflow at the approach line and outflow at the gate line.

    Sparse mode: tracked foot positions crossing the line in the exit direction.
    Dense mode: flux estimate = density near the line x normal speed x line length x dt.
    Events are kept for `window_s` so we can report rates per minute.
    """

    def __init__(self, a, b, exit_dir, window_s: float = 30.0):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.n = line_normal(self.a, self.b, np.asarray(exit_dir, dtype=np.float64))
        self.length = float(np.linalg.norm(self.b - self.a))
        self.window_s = window_s
        self.events: deque = deque()  # (t, amount)
        self.total = 0.0
        self.t0 = None

    def add(self, t: float, amount: float):
        if amount:
            self.events.append((t, amount))
            self.total += amount

    def update_tracks(self, t: float, prev_pos: dict, curr_pos: dict):
        """Count forward crossings from track positions (dict id -> floor xy)."""
        for tid, p1 in curr_pos.items():
            p0 = prev_pos.get(tid)
            if p0 is None:
                continue
            s = segment_crossing(np.asarray(p0), np.asarray(p1), self.a, self.b)
            if s:
                step = np.asarray(p1) - np.asarray(p0)
                self.add(t, 1.0 if step @ self.n > 0 else -1.0)

    def update_flux(self, t: float, dt: float, density_near: float, normal_speed: float):
        """Dense-mode estimate of people crossing during dt."""
        if np.isfinite(density_near) and np.isfinite(normal_speed):
            self.add(t, density_near * normal_speed * self.length * dt)

    def rate_per_min(self, t: float) -> float:
        if self.t0 is None:
            self.t0 = t
        while self.events and self.events[0][0] < t - self.window_s:
            self.events.popleft()
        span = min(self.window_s, max(t - self.t0, 1.0))
        return 60.0 * sum(e[1] for e in self.events) / span


def line_crossings(inflow: LineCounter, outflow: LineCounter, t: float) -> dict:
    """Rates per minute and accumulation = inflow - outflow."""
    i, o = inflow.rate_per_min(t), outflow.rate_per_min(t)
    return {"inflow_ppm": i, "outflow_ppm": o, "accumulation_ppm": i - o}


class EMA:
    """Exponential moving average so counts and density do not flicker."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None

    def __call__(self, x: float) -> float:
        if x is None or not np.isfinite(x):
            return self.value if self.value is not None else float("nan")
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def smooth(value: float, ema: EMA) -> float:
    return ema(value)
