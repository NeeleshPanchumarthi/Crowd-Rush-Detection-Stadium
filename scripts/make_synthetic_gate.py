"""Make a synthetic exit-gate clip where a crowd builds up, with exact ground truth.

Why: no public dataset labels "dangerous compression at a gate", so Part 1 needs a scripted scenario
to show the alert going green -> amber -> red. This script:
  1. cuts real pedestrians out of OpenCV's sample clip (data/vtest.avi) using YOLO boxes and
     background subtraction, so the models see real people, not drawn blobs;
  2. simulates walkers on the floor plane (metres) heading for a gate whose usable opening shrinks
     over time (normal -> partly closed -> nearly blocked), with density-dependent walking speed
     (Weidmann) and a minimum spacing between people;
  3. renders them back into the camera view with the gate homography (configs/gates/gate_A.yaml)
     and writes the video plus a per-frame ground-truth CSV (count, density, speed in the zone).

Usage:  python scripts/make_synthetic_gate.py [--seconds 150] [--seed 1]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crowdgate.config import load_gate_config  # noqa: E402
from crowdgate.geometry import local_scale_px_per_m, points_in_polygon, polygon_area, to_image  # noqa: E402

SRC = ROOT / "data" / "vtest.avi"
OUT_DIR = ROOT / "data" / "synthetic"
FPS = 10.0
DT = 1.0 / FPS
MIN_GAP = 0.45  # metres between people (allows ~5 people/m^2 when packed)
GATE_X0, GATE_X1_FULL = 2.5, 5.5  # gate opening in metres (on the y = 0 line)
ZONE_X = (1.0, 7.0)
ZONE_Y = (0.0, 6.0)


def phase(t: float):
    """(arrival rate people/s, usable gate opening m) for each phase of the scenario."""
    if t < 40:
        return 0.7, 3.0  # normal egress
    if t < 85:
        return 2.6, 0.8  # big wave arriving, gate partly closed
    return 3.5, 0.2  # gate almost blocked


# --------------------------------------------------------------------------------------------
def median_background(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 5 == 0:
            frames.append(f)
        i += 1
    return np.median(np.stack(frames), 0).astype(np.uint8)


def extract_sprites(path: Path, bg: np.ndarray, max_sprites: int = 160) -> list[np.ndarray]:
    """RGBA cut-outs of single, unoccluded people from the sample clip."""
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
    cap = cv2.VideoCapture(str(path))
    sprites, i = [], 0
    while len(sprites) < max_sprites:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if i % 6:
            continue
        r = model.predict(f, classes=[0], conf=0.6, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        diff = (np.abs(f.astype(np.int16) - bg.astype(np.int16)).max(2) > 34).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        for j, b in enumerate(boxes):
            x0, y0, x1, y1 = b.astype(int)
            h, w = y1 - y0, x1 - x0
            if h < 55 or w < 14 or h / max(w, 1) < 1.6:
                continue
            # skip people overlapping someone else
            others = np.delete(boxes, j, 0)
            if len(others) and np.any((others[:, 0] < x1) & (others[:, 2] > x0) & (others[:, 1] < y1) & (others[:, 3] > y0)):
                continue
            x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
            x1, y1 = min(f.shape[1], x1 + 2), min(f.shape[0], y1 + 2)
            m = diff[y0:y1, x0:x1]
            if m.mean() < 0.25:
                continue
            # keep the biggest blob only (drops shadows / neighbours' limbs)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            if n <= 1:
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = cv2.erode((lab == k).astype(np.uint8), np.ones((2, 2), np.uint8))
            alpha = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)
            rgba = np.dstack([f[y0:y1, x0:x1], (alpha * 255).astype(np.uint8)])
            # crop to the mask's vertical extent so the bottom of the sprite is the feet
            rows = np.where(m.any(1))[0]
            rgba = rgba[rows.min():rows.max() + 1]
            sprites.append(rgba)
            sprites.append(rgba[:, ::-1].copy())
    return sprites


# --------------------------------------------------------------------------------------------
class Crowd:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.pos = np.zeros((0, 2))
        self.v0 = np.zeros(0)
        self.sprite = np.zeros(0, dtype=int)
        self.hscale = np.zeros(0)
        self.vel = np.zeros((0, 2))
        self.tokens = 0.0
        self.passed = 0

    def spawn(self, rate: float, n_sprites: int):
        k = self.rng.poisson(rate * DT)
        for _ in range(k):
            p = np.array([self.rng.uniform(1.5, 6.5), self.rng.uniform(9.3, 10.2)])
            if len(self.pos) and np.min(np.linalg.norm(self.pos - p, axis=1)) < MIN_GAP * 1.2:
                continue
            self.pos = np.vstack([self.pos, p])
            self.v0 = np.append(self.v0, self.rng.normal(1.3, 0.15))
            self.sprite = np.append(self.sprite, self.rng.integers(n_sprites))
            self.hscale = np.append(self.hscale, self.rng.uniform(0.93, 1.07))
            self.vel = np.vstack([self.vel, [0.0, -1.0]])

    def local_density(self) -> np.ndarray:
        if len(self.pos) == 0:
            return np.zeros(0)
        d = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=2)
        return (d < 1.0).sum(1) / np.pi  # people per m^2 within 1 m (includes self)

    def step(self, opening: float):
        if len(self.pos) == 0:
            return
        gx0, gx1 = GATE_X0, GATE_X0 + opening
        rho = self.local_density()
        # Weidmann fundamental diagram: walking speed drops as density rises.
        v_des = self.v0 * np.clip(1 - np.exp(-1.913 * (1 / np.maximum(rho, 0.3) - 1 / 5.4)), 0.0, 1.0)
        target = np.stack([np.clip(self.pos[:, 0], gx0 + 0.2, max(gx0 + 0.2, gx1 - 0.2)), np.zeros(len(self.pos)) - 1.0], 1)
        through = self.pos[:, 1] < 0.3
        direc = target - self.pos
        direc[through] = [0.0, -1.0]
        direc /= np.linalg.norm(direc, axis=1, keepdims=True) + 1e-9
        self.vel = 0.6 * self.vel + 0.4 * direc * v_des[:, None]
        new = self.pos + self.vel * DT

        # Gate service: crossing y = 0 needs a token (rate = 1 person per metre of opening per second)
        self.tokens = min(self.tokens + 1.0 * opening * DT, 1.5)
        order = np.argsort(self.pos[:, 1])
        for i in order:
            if self.pos[i, 1] >= 0 and new[i, 1] < 0:
                inside = gx0 <= new[i, 0] <= gx1
                if inside and self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.passed += 1
                else:
                    new[i, 1] = 0.02
        # Side railings inside the zone
        in_zone = (new[:, 1] >= 0) & (new[:, 1] <= ZONE_Y[1])
        new[in_zone, 0] = np.clip(new[in_zone, 0], ZONE_X[0] + 0.2, ZONE_X[1] - 0.2)
        # Minimum spacing (position-based projection, a few passes)
        for _ in range(4):
            d = new[:, None] - new[None]
            dist = np.linalg.norm(d, axis=2) + np.eye(len(new)) * 9
            close = dist < MIN_GAP
            if not close.any():
                break
            push = ((MIN_GAP - dist) * close / 2)[..., None] * d / dist[..., None]
            new += push.sum(1)
            new[in_zone, 0] = np.clip(new[in_zone, 0], ZONE_X[0] + 0.2, ZONE_X[1] - 0.2)
            # people still before the gate cannot be pushed through the wall
            blocked = (self.pos[:, 1] >= 0) & (new[:, 1] < 0.02) & ~((new[:, 0] >= gx0) & (new[:, 0] <= gx1) & (self.pos[:, 1] < 0.1))
            new[blocked, 1] = 0.02
        self.vel = (new - self.pos) / DT
        self.pos = new
        keep = self.pos[:, 1] > -2.5
        self.pos, self.v0, self.sprite, self.hscale, self.vel = (
            self.pos[keep], self.v0[keep], self.sprite[keep], self.hscale[keep], self.vel[keep])


# --------------------------------------------------------------------------------------------
def paste(canvas: np.ndarray, rgba: np.ndarray, foot_xy, alpha_mul: float = 1.0):
    h, w = rgba.shape[:2]
    x0 = int(round(foot_xy[0] - w / 2))
    y0 = int(round(foot_xy[1] - h))
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    s = rgba[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = (s[..., 3:4].astype(np.float32) / 255.0) * alpha_mul
    region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = (region * (1 - a) + s[..., :3].astype(np.float32) * a).astype(np.uint8)


def draw_gate(img: np.ndarray, H_inv: np.ndarray, opening: float):
    """Railings along the zone sides, walls either side of the gate and the closing gate panel."""
    def seg(p, q, col, th, height=1.0):
        a, b = to_image(np.array([p, q], dtype=float), H_inv)
        sa = local_scale_px_per_m(np.array(p, float), H_inv)
        sb = local_scale_px_per_m(np.array(q, float), H_inv)
        top_a, top_b = a - [0, height * sa], b - [0, height * sb]
        cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), col, th, cv2.LINE_AA)
        cv2.line(img, tuple(np.int32(top_a)), tuple(np.int32(top_b)), col, th, cv2.LINE_AA)
        for f in np.linspace(0, 1, max(2, int(np.linalg.norm(np.subtract(q, p)) / 1.0) + 1)):
            pb = a + (b - a) * f
            pt = top_a + (top_b - top_a) * f
            cv2.line(img, tuple(np.int32(pb)), tuple(np.int32(pt)), col, max(1, th - 1), cv2.LINE_AA)

    grey = (150, 150, 150)
    seg((ZONE_X[0], 0), (ZONE_X[0], ZONE_Y[1]), grey, 2)
    seg((ZONE_X[1], 0), (ZONE_X[1], ZONE_Y[1]), grey, 2)
    seg((ZONE_X[0], 0), (GATE_X0, 0), (90, 90, 160), 3, 2.0)
    seg((GATE_X1_FULL, 0), (ZONE_X[1], 0), (90, 90, 160), 3, 2.0)
    if opening < GATE_X1_FULL - GATE_X0 - 1e-6:
        seg((GATE_X0 + opening, 0), (GATE_X1_FULL, 0), (40, 60, 200), 3, 1.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=str(ROOT / "configs" / "gates" / "gate_A.yaml"))
    ap.add_argument("--out", default=str(OUT_DIR / "gate_A_buildup.mp4"))
    args = ap.parse_args()

    cfg = load_gate_config(args.config)
    H_inv = cfg.H_inv
    zone = cfg.zone_m
    area = polygon_area(zone)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building background and person cut-outs from", SRC)
    bg = median_background(SRC)
    sprites = extract_sprites(SRC, bg)
    print(f"  {len(sprites)} sprites")
    rng = np.random.default_rng(args.seed)
    crowd = Crowd(rng)

    h, w = bg.shape[:2]
    out_path = Path(args.out)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    gt_path = out_path.with_suffix(".gt.csv")
    gt_f = open(gt_path, "w", newline="")
    gt = csv.writer(gt_f)
    gt.writerow(["frame", "t", "arrival_rate", "opening_m", "count_zone", "density", "mean_forward_speed", "passed_total"])
    cache: dict = {}
    n_frames = int(args.seconds * FPS)
    for fi in range(n_frames):
        t = fi * DT
        rate, opening = phase(t)
        crowd.spawn(rate, len(sprites))
        crowd.step(opening)

        img = bg.copy()
        draw_gate(img, H_inv, opening)
        rho = crowd.local_density()
        # Sway: lateral oscillation that grows with local density (force waves in a packed crowd)
        amp = 0.12 * np.clip((rho - 2.5) / 2.5, 0, 1)
        sway_x = amp * np.sin(2 * np.pi * 0.45 * t + 0.9 * crowd.pos[:, 1]) if len(rho) else np.zeros(0)
        render_pos = crowd.pos + np.stack([sway_x, np.zeros_like(sway_x)], 1) if len(rho) else crowd.pos
        order = np.argsort(render_pos[:, 1])  # far (small y) first
        feet = to_image(render_pos, H_inv) if len(render_pos) else np.zeros((0, 2))
        for i in order:
            s = local_scale_px_per_m(render_pos[i], H_inv)
            hp = int(round(1.7 * crowd.hscale[i] * s))
            key = (int(crowd.sprite[i]), hp)
            if key not in cache:
                sp = sprites[key[0]]
                cache[key] = cv2.resize(sp, (max(4, int(sp.shape[1] * hp / sp.shape[0])), hp), interpolation=cv2.INTER_AREA)
            fade = float(np.clip((render_pos[i, 1] + 2.5) / 1.0, 0, 1))
            paste(img, cache[key], feet[i], fade)
        noise = rng.normal(0, 2.5, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        vw.write(img)

        inz = points_in_polygon(crowd.pos, zone) if len(crowd.pos) else np.zeros(0, bool)
        n = int(inz.sum())
        spd = float(np.mean(crowd.vel[inz] @ cfg.exit_dir)) if n else float("nan")
        gt.writerow([fi, round(t, 2), rate, opening, n, round(n / area, 3), round(spd, 3), crowd.passed])
        if fi % 100 == 0:
            print(f"  t={t:5.1f}s  in zone={n:3d}  density={n / area:4.2f}/m2  speed={spd:5.2f}  passed={crowd.passed}")
    vw.release()
    gt_f.close()
    print("Wrote", out_path, "and", gt_path)


if __name__ == "__main__":
    main()
