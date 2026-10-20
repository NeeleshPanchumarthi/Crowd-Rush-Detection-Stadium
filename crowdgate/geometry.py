"""Step 4 (geometry part): pixels <-> floor metres and small polygon/line helpers."""

from __future__ import annotations

import cv2
import numpy as np


def to_floor(points_px, H: np.ndarray) -> np.ndarray:
    """Map Nx2 image points (pixels) to floor coordinates (metres) with cv2.perspectiveTransform."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_image(points_m, H_inv: np.ndarray) -> np.ndarray:
    """Map Nx2 floor points (metres) back to image pixels."""
    return to_floor(points_m, H_inv)


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of a polygon given as Nx2 array."""
    x, y = np.asarray(poly, dtype=np.float64).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Boolean mask: which points lie inside the polygon."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in points])


def segment_crossing(p0: np.ndarray, p1: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    """Did the step p0 -> p1 cross segment a-b?

    Returns +1 if it crossed towards the left-hand normal of a->b's... in practice the caller
    compares the step direction with the exit direction, so we return +1/-1 for the side change
    (from negative to positive side of the line) and 0 for no crossing.
    """
    def side(p):
        return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

    s0, s1 = side(p0), side(p1)
    if s0 == 0 or s1 == 0 or s0 == s1:
        return 0
    # Check the crossing point lies within the segment (not on its infinite extension).
    d = p1 - p0
    e = b - a
    denom = d[0] * e[1] - d[1] * e[0]
    if abs(denom) < 1e-12:
        return 0
    t = ((a[0] - p0[0]) * e[1] - (a[1] - p0[1]) * e[0]) / denom
    u = ((a[0] - p0[0]) * d[1] - (a[1] - p0[1]) * d[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return int(s1)
    return 0


def line_normal(a: np.ndarray, b: np.ndarray, towards: np.ndarray) -> np.ndarray:
    """Unit normal of segment a-b, oriented to agree with the 'towards' direction."""
    e = b - a
    n = np.array([-e[1], e[0]], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-9
    if np.dot(n, towards) < 0:
        n = -n
    return n


def strip_polygon(a: np.ndarray, b: np.ndarray, half_width_m: float) -> np.ndarray:
    """Rectangle of +/- half_width around a line segment (for flow-based crossing estimates)."""
    n = line_normal(a, b, np.array([0.0, 1.0]))
    return np.array([a + n * half_width_m, b + n * half_width_m, b - n * half_width_m, a - n * half_width_m])


def downstream_strip(a: np.ndarray, b: np.ndarray, n: np.ndarray, depth_m: float) -> np.ndarray:
    """Rectangle of `depth_m` on the side of line a-b that the unit normal n points to."""
    a, b, n = (np.asarray(v, dtype=np.float64) for v in (a, b, n))
    return np.array([a, b, b + n * depth_m, a + n * depth_m])


def local_scale_px_per_m(pt_m: np.ndarray, H_inv: np.ndarray) -> float:
    """Pixels per metre on the floor at a floor point (horizontal direction)."""
    p = to_image(np.array([pt_m + [-0.5, 0.0], pt_m + [0.5, 0.0]]), H_inv)
    return float(np.linalg.norm(p[1] - p[0]))


def homography_jacobian(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    """2x2 Jacobian of the image->floor map at each pixel (N,2) -> (N,2,2), metres per pixel."""
    px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    x, y = px[:, 0], px[:, 1]
    h = H
    u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    J = np.empty((len(px), 2, 2))
    J[:, 0, 0] = (h[0, 0] * w - u * h[2, 0]) / w**2
    J[:, 0, 1] = (h[0, 1] * w - u * h[2, 1]) / w**2
    J[:, 1, 0] = (h[1, 0] * w - v * h[2, 0]) / w**2
    J[:, 1, 1] = (h[1, 1] * w - v * h[2, 1]) / w**2
    return J
