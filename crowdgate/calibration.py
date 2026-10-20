"""Step 2: calibration, done once per camera.

calibrate_homography() is the maths; click_calibration() is the small OpenCV click tool around it.
Run it with:  python scripts/calibrate.py configs/gates/gate_B.yaml
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import to_floor, to_image


def calibrate_homography(img_pts, floor_pts) -> tuple[np.ndarray, float]:
    """Compute H (image px -> floor m) from 4+ matched ground points.

    Returns (H, rms_error_m): the reprojection error of the clicked points in metres,
    which is a quick check that the clicks and the typed metre positions agree.
    """
    img_pts = np.asarray(img_pts, dtype=np.float64)
    floor_pts = np.asarray(floor_pts, dtype=np.float64)
    if len(img_pts) < 4 or len(img_pts) != len(floor_pts):
        raise ValueError("Need at least 4 matching image/floor points")
    method = 0 if len(img_pts) == 4 else cv2.RANSAC
    H, _ = cv2.findHomography(img_pts, floor_pts, method)
    if H is None:
        raise ValueError("findHomography failed: are 3 of the points collinear?")
    err = np.linalg.norm(to_floor(img_pts, H) - floor_pts, axis=1)
    return H, float(np.sqrt(np.mean(err**2)))


def zone_mask_from_floor(zone_m, H_inv: np.ndarray, shape) -> np.ndarray:
    """Convert a polygon drawn in floor metres into a uint8 pixel mask (1 inside) of image `shape`."""
    h, w = shape[:2]
    poly_px = to_image(np.asarray(zone_m, dtype=np.float64), H_inv)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly_px).astype(np.int32)], 1)
    return mask


def foot_offset_px(points_px, H: np.ndarray, H_inv: np.ndarray, lift_m: float) -> np.ndarray:
    """For pixels that sit `lift_m` above the floor on an upright person, estimate where the feet are.

    Uses the local floor scale (pixels per metre) at each point; two fixed-point iterations are enough.
    Returns the estimated foot pixel for every input pixel.
    """
    p = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    foot = p.copy()
    for _ in range(2):
        fl = to_floor(foot, H)
        a = to_image(fl - [0.5, 0.0], H_inv)
        b = to_image(fl + [0.5, 0.0], H_inv)
        scale = np.linalg.norm(b - a, axis=1)  # px per metre
        foot = p + np.stack([np.zeros(len(p)), lift_m * scale], 1)
    return foot


def lifted_zone_mask(zone_m, H: np.ndarray, H_inv: np.ndarray, shape, roi, lift_m: float) -> np.ndarray:
    """Pixel mask (cropped to roi) of where a point `lift_m` above the floor appears for people
    standing inside the zone.

    The floor polygon alone is wrong for counting: DM-Count puts the mass on heads, which appear
    above (in the image) the floor spot the person stands on, so people at the far edge of the zone
    would be missed and people behind it counted. lift_m ~ 1.6 m gives the head mask.
    """
    floor_mask = zone_mask_from_floor(zone_m, H_inv, shape)
    x0, y0, x1, y1 = roi
    ys, xs = np.mgrid[y0:y1, x0:x1]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    foot = np.round(foot_offset_px(pts, H, H_inv, lift_m)).astype(int)
    h, w = shape[:2]
    ok = (foot[:, 0] >= 0) & (foot[:, 0] < w) & (foot[:, 1] >= 0) & (foot[:, 1] < h)
    out = np.zeros(len(pts), dtype=np.uint8)
    out[ok] = floor_mask[foot[ok, 1], foot[ok, 0]]
    return out.reshape(y1 - y0, x1 - x0)


def click_calibration(frame: np.ndarray, window: str = "calibrate") -> list[tuple[float, float]]:
    """Let the user click ground points on a frame. Left click adds, right click undoes, Enter finishes."""
    pts: list[tuple[float, float]] = []
    vis = frame.copy()

    def redraw():
        nonlocal vis
        vis = frame.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, "L-click: add   R-click: undo   Enter: done   Esc: cancel", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
        redraw()

    redraw()
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            break
        if key == 27:
            pts.clear()
            break
    cv2.destroyWindow(window)
    return pts
