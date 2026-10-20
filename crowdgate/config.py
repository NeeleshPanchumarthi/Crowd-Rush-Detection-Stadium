"""Step 1: per-gate configuration.

Each gate has one YAML (or JSON) file. Everything that differs between gates lives here:
geometry (homography, zone, counting lines), limits used to normalise the risk score,
and the perception settings. See configs/gates/*.yaml for examples.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Defaults merged under every gate file, so a gate file only needs what is specific to it.
DEFAULTS: dict[str, Any] = {
    "name": None,
    "source": None,
    "source_fps": None,  # frame rate for image-folder sources (e.g. Mall frames)
    "width_m": 3.0,
    "capacity_ppm": 80.0,  # people per minute the gate is designed to pass
    "free_speed_mps": 1.3,  # normal walking speed towards the gate
    "exit_dir": [0.0, -1.0],  # unit vector (floor metres) pointing through the gate
    "roi": None,  # [x0, y0, x1, y1] crop in pixels; None = bounding box of the zone
    "homography": None,  # 3x3, image pixels -> floor metres
    "zone_m": None,  # polygon in floor metres
    "lines": {"approach": None, "gate": None},  # each [[x, y], [x, y]] in metres
    "neighbours": [],
    "limits": {
        "density": {"amber": 2.0, "red": 4.0},  # people per m^2
        "score": {"amber": 0.40, "red": 0.65},
        "motion_variance_max": 0.20,  # (m/s)^2 that maps to 1.0
        "accumulation_max_ppm": 60.0,  # net build-up (in - out) per minute that maps to 1.0
    },
    "weights": {"density": 0.40, "stall": 0.30, "variance": 0.15, "accumulation": 0.15},
    "hold_seconds": {"up": 3.0, "down": 6.0},
    "perception": {
        "switch_density": 2.0,  # people/m^2 where we move from YOLO to DM-Count
        "hysteresis": 0.5,  # +/- band around switch_density
        "min_mode_dwell_s": 10.0,  # stay dense at least this long before going back to sparse
        "first_count_every_s": 1.0,
        "first_count_scale": 0.5,  # resolution of the cheap first count
        "density_every_s": 1.0,
        "density_scale": 1.0,
        "dm_weights": "model_sh_B.pth",  # models/dmcount/*.pth; SHB by default (see README for Mall result)
        "yolo_conf": 0.25,
        "yolo_imgsz": 640,
        "flow_scale": 0.5,
        "flow_grid_px": 6,  # sample flow every N pixels of the downscaled frame
        "person_height_m": 1.7,
        "smoothing_alpha": 0.3,
        "variance_window_s": 2.0,
        "crossing_window_s": 30.0,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class GateConfig:
    gate_id: str
    raw: dict
    path: Path | None = None
    H: np.ndarray = field(default=None)  # image px -> floor m
    H_inv: np.ndarray = field(default=None)  # floor m -> image px

    # Convenience accessors ---------------------------------------------------------------
    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def name(self) -> str:
        return self.raw.get("name") or f"Gate {self.gate_id}"

    @property
    def perception(self) -> dict:
        return self.raw["perception"]

    @property
    def limits(self) -> dict:
        return self.raw["limits"]

    @property
    def zone_m(self) -> np.ndarray:
        return np.asarray(self.raw["zone_m"], dtype=np.float64)

    @property
    def exit_dir(self) -> np.ndarray:
        d = np.asarray(self.raw["exit_dir"], dtype=np.float64)
        return d / (np.linalg.norm(d) + 1e-9)

    def line(self, name: str) -> np.ndarray | None:
        ln = self.raw["lines"].get(name)
        return None if ln is None else np.asarray(ln, dtype=np.float64)

    def resolve(self, rel: str | None) -> str | None:
        """Resolve a path in the config relative to the project root."""
        if rel is None:
            return None
        if isinstance(rel, int) or str(rel).isdigit():
            return rel  # webcam index
        p = Path(rel)
        return str(p if p.is_absolute() else ROOT / p)


def load_gate_config(path: str | Path) -> GateConfig:
    """Read one gate's YAML/JSON file and fill in defaults.

    Required keys: gate_id, homography (3x3, image px -> floor m), zone_m (polygon in metres).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    raw = _deep_merge(DEFAULTS, data)
    if raw.get("homography") is None and raw.get("calibration_points"):
        # Ground points picked with the calibration tool: compute H on load.
        from .calibration import calibrate_homography

        cp = raw["calibration_points"]
        H, _ = calibrate_homography(cp["image"], cp["floor"])
        raw["homography"] = H.tolist()
    for key in ("gate_id", "homography", "zone_m"):
        if raw.get(key) is None:
            raise ValueError(f"{path}: missing required key '{key}' (or calibration_points)")
    H = np.asarray(raw["homography"], dtype=np.float64).reshape(3, 3)
    cfg = GateConfig(gate_id=str(raw["gate_id"]), raw=raw, path=path, H=H, H_inv=np.linalg.inv(H))
    return cfg


def load_all_gates(folder: str | Path) -> dict[str, GateConfig]:
    folder = Path(folder)
    gates = {}
    for p in sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")) + list(folder.glob("*.json"))):
        cfg = load_gate_config(p)
        gates[cfg.gate_id] = cfg
    return gates


def save_homography(path: str | Path, H: np.ndarray, img_pts=None, floor_pts=None) -> None:
    """Write a new homography (and the points it came from) into a gate file (calibration tool)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["homography"] = [[float(v) for v in row] for row in np.asarray(H)]
    if img_pts is not None and floor_pts is not None:
        data["calibration_points"] = {
            "image": [[float(a), float(b)] for a, b in img_pts],
            "floor": [[float(a), float(b)] for a, b in floor_pts],
        }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
