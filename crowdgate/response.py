"""Step 6: alerts, redirection choice and the event log."""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .risk import AMBER, GREEN, RED


@dataclass
class Alert:
    wall_time: float
    video_time: float
    gate_id: str
    gate_name: str
    level: str
    previous: str
    message: str
    redirect_to: str | None
    density: float
    speed: float
    score: float
    accumulation_ppm: float
    outflow_ppm: float
    latency_ms: float | None = None


def utilisation(state: dict, capacity_ppm: float) -> float:
    """Outflow / capacity; unknown outflow counts as idle."""
    out = state.get("outflow_ppm") if state else None
    if out is None or out != out:
        return 0.0
    return max(0.0, out) / max(capacity_ppm, 1e-6)


def choose_redirect(gate_cfg, all_states: dict, all_cfgs: dict) -> str | None:
    """Neighbour gate with the lowest utilisation that is not itself red."""
    best, best_u = None, None
    for nb in gate_cfg.get("neighbours", []):
        st = all_states.get(nb, {})
        if st.get("level") == RED:
            continue
        cap = all_cfgs[nb]["capacity_ppm"] if nb in all_cfgs else 1.0
        u = utilisation(st, cap)
        if best_u is None or u < best_u:
            best, best_u = nb, u
    return best


def make_alert(gate_cfg, level: str, previous: str, metrics: dict, video_time: float,
               all_states: dict | None = None, all_cfgs: dict | None = None) -> Alert:
    """Build the dispatch message for a level change."""
    name = gate_cfg.name
    redirect = None
    if level == RED:
        redirect = choose_redirect(gate_cfg, all_states or {}, all_cfgs or {})
        nb_name = all_cfgs[redirect].name if redirect and all_cfgs and redirect in all_cfgs else redirect
        msg = f"Send security to {name}."
        msg += f" Redirect crowd to {nb_name}." if redirect else " No neighbour gate free for redirection."
    elif level == AMBER:
        msg = f"{name} is crowding: stewards on standby, watch the approach."
        if previous == RED:
            msg = f"{name} easing (red -> amber): keep security in place."
    else:
        msg = f"{name} back to normal."
    return Alert(
        wall_time=time.time(), video_time=video_time, gate_id=gate_cfg.gate_id, gate_name=name,
        level=level, previous=previous, message=msg, redirect_to=redirect,
        density=round(metrics.get("density", 0.0), 3), speed=round(metrics.get("speed", float("nan")), 3),
        score=round(metrics.get("score", 0.0), 3), accumulation_ppm=round(metrics.get("accumulation_ppm", 0.0), 2),
        outflow_ppm=round(metrics.get("outflow_ppm", 0.0), 2),
    )


class EventLog:
    """Every level change goes to SQLite (for queries) and CSV (for opening in Excel)."""

    FIELDS = list(Alert.__dataclass_fields__.keys())

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.db_path = self.folder / "events.sqlite"
        self.csv_path = self.folder / "events.csv"
        self._lock = threading.Lock()
        with sqlite3.connect(self.db_path) as db:
            cols = ", ".join(f"{f} {'TEXT' if f in ('gate_id', 'gate_name', 'level', 'previous', 'message', 'redirect_to') else 'REAL'}"
                             for f in self.FIELDS)
            db.execute(f"CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log_event(self, alert: Alert):
        row = asdict(alert)
        with self._lock:
            with sqlite3.connect(self.db_path) as db:
                db.execute(f"INSERT INTO events ({', '.join(self.FIELDS)}) VALUES ({', '.join('?' * len(self.FIELDS))})",
                           [row[f] for f in self.FIELDS])
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([row[f] for f in self.FIELDS])

    def recent(self, n: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def clear(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as db:
                db.execute("DELETE FROM events")
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)


def log_event(log: EventLog, alert: Alert):
    log.log_event(alert)


def alert_to_json(alert: Alert) -> str:
    return json.dumps(asdict(alert))


__all__ = ["Alert", "EventLog", "make_alert", "log_event", "choose_redirect", "GREEN", "AMBER", "RED"]
