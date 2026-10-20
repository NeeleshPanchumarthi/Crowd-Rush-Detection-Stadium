"""Always-on motion: Farnebäck dense optical flow (OpenCV) on a downscaled grey frame."""

from __future__ import annotations

import cv2
import numpy as np


def to_small_grey(frame_bgr: np.ndarray, scale: float) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return g


def compute_flow(prev_grey: np.ndarray, curr_grey: np.ndarray) -> np.ndarray:
    """Dense flow (H, W, 2) in pixels of the downscaled frame per frame step."""
    return cv2.calcOpticalFlowFarneback(prev_grey, curr_grey, None,
                                        pyr_scale=0.5, levels=3, winsize=13, iterations=3,
                                        poly_n=5, poly_sigma=1.1, flags=0)
