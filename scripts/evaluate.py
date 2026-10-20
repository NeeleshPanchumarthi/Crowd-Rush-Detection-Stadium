"""Step 7 checks: counting MAE on Mall frames (DM-Count SHB vs YOLO11n), plus speed and alert latency.

Mall has head labels for every frame (data/mall_dataset/mall_gt.mat). We compare whole-frame counts, so
no homography is needed. Usage:  python scripts/evaluate.py [--step 20]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crowdgate.perception.density import estimate_density  # noqa: E402
from crowdgate.perception.detector import PersonTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=20, help="evaluate every Nth Mall frame")
    args = ap.parse_args()
    gt = sio.loadmat(ROOT / "data" / "mall_dataset" / "mall_gt.mat")["count"].ravel()
    det = PersonTracker(conf=0.25, imgsz=640)
    rows = []
    t_dm, t_yolo = [], []
    for i in range(0, len(gt), args.step):
        img = cv2.imread(str(ROOT / "data" / "mall_dataset" / "frames" / f"seq_{i + 1:06d}.jpg"))
        t = time.perf_counter()
        d = estimate_density(img, 1.0)
        t_dm.append(time.perf_counter() - t)
        t = time.perf_counter()
        n_y = len(det.detect_heads(img).boxes)
        t_yolo.append(time.perf_counter() - t)
        rows.append((gt[i], d.sum(), n_y))
    r = np.array(rows)
    g = r[:, 0]
    out = ["Mall counting check (%d frames, every %d-th, whole frame)" % (len(r), args.step),
           f"  GT mean count {g.mean():.1f} (range {g.min():.0f}-{g.max():.0f})"]
    for name, col in (("DM-Count SHB", 1), ("YOLO11n COCO person", 2)):
        e = r[:, col] - g
        out.append(f"  {name:22s} MAE {np.abs(e).mean():5.2f}  RMSE {np.sqrt((e ** 2).mean()):5.2f}  bias {e.mean():+5.2f}")
    out.append(f"  speed: DM-Count {np.median(t_dm) * 1000:.0f} ms/frame (640x480), YOLO11n {np.median(t_yolo) * 1000:.0f} ms/frame")
    print("\n".join(out))
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "mall_counting_eval.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
