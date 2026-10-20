"""Step 1: video input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Stream:
    cap: cv2.VideoCapture
    fps: float
    width: int
    height: int
    n_frames: int  # -1 for live sources
    source: str | int
    is_folder: bool = False
    _files: list | None = None
    _i: int = 0

    def read(self):
        """Return (ok, frame, frame_index)."""
        if self.is_folder:
            if self._i >= len(self._files):
                return False, None, self._i
            frame = cv2.imread(str(self._files[self._i]))
            self._i += 1
            return frame is not None, frame, self._i - 1
        ok, frame = self.cap.read()
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return ok, frame, idx

    def rewind(self):
        if self.is_folder:
            self._i = 0
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap is not None:
            self.cap.release()


def open_stream(source, folder_fps: float = 2.0) -> Stream:
    """Open a video file, an image folder (e.g. Mall frames) or a webcam index with OpenCV."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, str) and Path(source).is_dir():
        files = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise IOError(f"No images in {source}")
        h, w = cv2.imread(str(files[0])).shape[:2]
        return Stream(None, folder_fps, w, h, len(files), source, True, files)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1e-3 or fps > 240:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not isinstance(source, int) else -1
    return Stream(cap, fps, w, h, n, source)


def crop_to_zone(frame: np.ndarray, roi) -> np.ndarray:
    """Crop a frame to the gate's region [x0, y0, x1, y1] so models only see what matters."""
    x0, y0, x1, y1 = (int(v) for v in roi)
    return frame[y0:y1, x0:x1]
