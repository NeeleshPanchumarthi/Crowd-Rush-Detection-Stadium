"""Dense mode: DM-Count density maps (VGG-19 back end, ShanghaiTech B weights).

The network definition is copied from github.com/cvlab-stonybrook/DM-Count (models.py) so the
released weights load directly. Inputs use ImageNet normalisation as in the repo's datasets/crowd.py.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ROOT

DMCOUNT_WEIGHTS = ROOT / "models" / "dmcount" / "model_sh_B.pth"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CFG_E = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]


class DMCount(nn.Module):
    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for v in _CFG_E:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.reg_layer(x)
        return self.density_layer(x)


_model = None
_model_lock = threading.Lock()


_weights = DMCOUNT_WEIGHTS


def use_weights(name_or_path: str | Path | None):
    """Pick the DM-Count weights file (e.g. 'model_qnrf.pth' in models/dmcount/). One set per process."""
    global _weights, _model
    if not name_or_path:
        return
    p = Path(name_or_path)
    p = p if p.is_absolute() else ROOT / "models" / "dmcount" / p.name
    with _model_lock:
        if p != _weights:
            _weights, _model = p, None


def get_model() -> DMCount:
    """Load DM-Count once per process and share it between gates."""
    global _model
    with _model_lock:
        if _model is None:
            m = DMCount()
            m.load_state_dict(torch.load(str(_weights), map_location="cpu"))
            m.eval()
            _model = m
    return _model


def estimate_density(frame_bgr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Run DM-Count and return a density map resized to the input frame size.

    The map is rescaled so its sum is preserved: sum(map) is the people count.
    """
    h, w = frame_bgr.shape[:2]
    img = frame_bgr if scale == 1.0 else cv2.resize(frame_bgr, (max(16, int(w * scale)), max(16, int(h * scale))),
                                                    interpolation=cv2.INTER_AREA)
    x = (img[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
    with torch.inference_mode():
        d = get_model()(x)[0, 0].numpy()
    total = float(d.sum())
    up = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    s = float(up.sum())
    if s > 1e-9:
        up *= total / s
    return up


def first_count(frame_bgr: np.ndarray, zone_mask: np.ndarray | None = None, scale: float = 0.5) -> tuple[float, np.ndarray]:
    """Cheap, low-resolution DM-Count pass. Returns (count inside zone, density map)."""
    d = estimate_density(frame_bgr, scale=scale)
    count = float((d * zone_mask).sum()) if zone_mask is not None else float(d.sum())
    return count, d


class DensityWorker:
    """Runs DM-Count off the frame loop so slow CPU inference does not stall video.

    Live mode: submit() drops a frame in, a background thread processes the newest one,
    latest() returns the most recent result. Offline mode (sync=True): submit() runs inline,
    which makes headless evaluation runs repeatable.
    """

    def __init__(self, sync: bool = False):
        self.sync = sync
        self._pending = None
        self._result = None  # (t_submitted, density_map, info)
        self._cv = threading.Condition()
        self._busy = False
        self._stop = False
        if not sync:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy or self._pending is not None

    def submit(self, t: float, frame: np.ndarray, scale: float, info=None):
        if self.sync:
            self._result = (t, estimate_density(frame, scale), info)
            return
        with self._cv:
            self._pending = (t, frame.copy(), scale, info)
            self._cv.notify()

    def latest(self):
        return self._result

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                t, frame, scale, info = self._pending
                self._pending = None
                self._busy = True
            try:
                self._result = (t, estimate_density(frame, scale), info)
            finally:
                self._busy = False

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
