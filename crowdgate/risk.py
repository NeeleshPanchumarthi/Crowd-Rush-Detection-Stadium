"""Step 5: risk score and green/amber/red level with hold-time hysteresis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GREEN, AMBER, RED = "green", "amber", "red"
LEVELS = [GREEN, AMBER, RED]
RANK = {GREEN: 0, AMBER: 1, RED: 2}


def _clip01(x: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def risk_score(metrics: dict, gate_cfg) -> tuple[float, dict]:
    """w1*density + w2*density*(1 - speed) + w3*motion_variance + w4*accumulation.

    Every term is normalised 0-1 against this gate's own limits:
      density      / red density limit
      speed        / free walking speed (stalled crowd -> 1 - speed close to 1)
      variance     / motion_variance_max
      accumulation / accumulation_max_ppm (net build-up per minute)
    Returns (score, parts) so the dashboard can show which term drives the score.
    """
    lim = gate_cfg.limits
    w = gate_cfg["weights"]
    d = _clip01(metrics["density"] / lim["density"]["red"])
    speed = metrics.get("speed")
    s = _clip01(speed / gate_cfg["free_speed_mps"]) if speed is not None and np.isfinite(speed) else 1.0
    v = _clip01(metrics.get("motion_variance", 0.0) / lim["motion_variance_max"])
    a = _clip01(metrics.get("accumulation_ppm", 0.0) / lim["accumulation_max_ppm"])
    parts = {
        "density": w["density"] * d,
        "stall": w["stall"] * d * (1.0 - s),
        "variance": w["variance"] * v,
        "accumulation": w["accumulation"] * a,
    }
    total = sum(w.values())
    return float(sum(parts.values()) / total), parts


def candidate_level(score: float, density: float, gate_cfg) -> str:
    """Instantaneous level: score thresholds, with the gate's absolute density limits as a floor."""
    lim = gate_cfg.limits
    lvl = GREEN
    if score >= lim["score"]["amber"] or density >= lim["density"]["amber"]:
        lvl = AMBER
    if score >= lim["score"]["red"] or density >= lim["density"]["red"]:
        lvl = RED
    return lvl


@dataclass
class LevelState:
    level: str = GREEN
    pending: str | None = None
    since: float = 0.0  # time the pending level first appeared
    changed_at: float = 0.0
    history: list = field(default_factory=list)


def risk_level(score: float, density: float, t: float, state: LevelState, gate_cfg) -> tuple[str, bool]:
    """Green/amber/red that changes only when the new condition holds for T seconds.

    Escalation uses hold_seconds.up, de-escalation the (longer) hold_seconds.down, so one noisy
    second neither raises nor clears an alert. Returns (level, changed).
    """
    hold = gate_cfg["hold_seconds"]
    cand = candidate_level(score, density, gate_cfg)
    if cand == state.level:
        state.pending = None
        return state.level, False
    if cand != state.pending:
        state.pending, state.since = cand, t
    need = hold["up"] if RANK[cand] > RANK[state.level] else hold["down"]
    if t - state.since >= need:
        state.history.append((t, state.level, cand))
        state.level, state.pending, state.changed_at = cand, None, t
        return state.level, True
    return state.level, False
