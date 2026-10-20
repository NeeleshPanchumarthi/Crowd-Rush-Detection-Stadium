"""Click tool: calibrate one camera's homography and save it into its gate config.

  python scripts/calibrate.py configs/gates/gate_A.yaml [--frame 0]

Click 4+ ground points on the frame, then type each point's floor position in metres (x y).
The result (H plus the points) is written back into the gate file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crowdgate.calibration import calibrate_homography, click_calibration  # noqa: E402
from crowdgate.config import load_gate_config, save_homography  # noqa: E402
from crowdgate.video import open_stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    cfg = load_gate_config(args.config)
    s = open_stream(cfg.resolve(cfg["source"]))
    for _ in range(args.frame + 1):
        ok, frame, _ = s.read()
    img_pts = click_calibration(frame)
    if len(img_pts) < 4:
        print("Need at least 4 points; nothing saved.")
        return
    floor = []
    for i, (x, y) in enumerate(img_pts, 1):
        a = input(f"Point {i} at pixel ({x:.0f}, {y:.0f}): floor x y in metres > ").split()
        floor.append((float(a[0]), float(a[1])))
    H, rms = calibrate_homography(img_pts, floor)
    print(f"RMS reprojection error: {rms:.3f} m")
    save_homography(args.config, H, img_pts, floor)
    print("Saved to", args.config)


if __name__ == "__main__":
    main()
