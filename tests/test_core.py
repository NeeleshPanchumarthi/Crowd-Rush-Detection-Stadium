"""Fast unit tests (no models): python -m pytest tests  (or python tests/test_core.py)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crowdgate.calibration import calibrate_homography  # noqa: E402
from crowdgate.config import load_gate_config  # noqa: E402
from crowdgate.geometry import polygon_area, segment_crossing, to_floor, to_image  # noqa: E402
from crowdgate.metrics import EMA, LineCounter, MotionVariance, forward_speed  # noqa: E402
from crowdgate.perception.mode import DENSE, SPARSE, select_mode  # noqa: E402
from crowdgate.risk import LevelState, risk_level, risk_score  # noqa: E402

CFG = load_gate_config(ROOT / "configs" / "gates" / "gate_A.yaml")


def test_homography_roundtrip():
    img = [[100, 100], [500, 100], [600, 400], [50, 400]]
    floor = [[0, 0], [4, 0], [4, 5], [0, 5]]
    H, rms = calibrate_homography(img, floor)
    assert rms < 1e-6
    assert np.allclose(to_floor(img, H), floor, atol=1e-6)
    assert np.allclose(to_image(to_floor(img, H), np.linalg.inv(H)), img, atol=1e-4)


def test_area_and_crossing():
    assert polygon_area(np.array([[0, 0], [4, 0], [4, 3], [0, 3]])) == 12
    a, b = np.array([0.0, 0.0]), np.array([4.0, 0.0])
    assert segment_crossing(np.array([2.0, 1.0]), np.array([2.0, -1.0]), a, b) != 0
    assert segment_crossing(np.array([6.0, 1.0]), np.array([6.0, -1.0]), a, b) == 0  # beyond the segment


def test_line_counter_direction():
    lc = LineCounter([0, 0], [4, 0], exit_dir=[0, -1])
    lc.update_tracks(1.0, {1: np.array([2.0, 1.0])}, {1: np.array([2.0, -1.0])})  # forward
    lc.update_tracks(2.0, {2: np.array([1.0, -1.0])}, {2: np.array([1.0, 1.0])})  # backwards
    assert lc.total == 0


def test_forward_speed_and_variance():
    v = np.array([[0.0, -1.0]] * 5)
    assert abs(forward_speed(v, np.array([0.0, -1.0])) - 1.0) < 1e-9
    mv = MotionVariance(2.0)
    assert mv.update(0.0, v) == 0.0
    assert mv.update(0.5, -v) > 0.5  # whole block reversing = sway


def test_mode_hysteresis():
    m = SPARSE
    for d in (1.0, 2.2, 2.4):
        m = select_mode(10, d, m, 2.0, 0.5)
    assert m == SPARSE  # inside the band
    m = select_mode(10, 2.6, m, 2.0, 0.5)
    assert m == DENSE
    m = select_mode(10, 1.8, m, 2.0, 0.5)
    assert m == DENSE  # still inside the band
    m = select_mode(3, 1.0, m, 2.0, 0.5, time_in_mode=30, min_dwell_s=10)
    assert m == SPARSE


def test_risk_hold_time():
    lim = CFG.limits["density"]["red"]
    st = LevelState()
    # a one-second spike must not raise the level
    assert risk_level(0.9, lim + 1, 0.0, st, CFG)[0] == "green"
    assert risk_level(0.9, lim + 1, 1.0, st, CFG)[0] == "green"
    assert risk_level(0.0, 0.1, 1.5, st, CFG)[0] == "green"
    # held for the up-time it escalates, then needs the longer down-time to clear
    for t in np.arange(10, 14, 0.5):
        lvl, _ = risk_level(0.9, lim + 1, t, st, CFG)
    assert lvl == "red"
    assert risk_level(0.0, 0.1, 14.0, st, CFG)[0] == "red"
    for t in np.arange(14.5, 22, 0.5):
        lvl, _ = risk_level(0.0, 0.1, t, st, CFG)
    assert lvl == "green"


def test_stalled_dense_scores_higher_than_moving_dense():
    base = {"density": 3.0, "motion_variance": 0.05, "accumulation_ppm": 0.0}
    moving, _ = risk_score({**base, "speed": 1.3}, CFG)
    stalled, _ = risk_score({**base, "speed": 0.0}, CFG)
    assert stalled > moving + 0.15


def test_ema():
    e = EMA(0.5)
    assert e(2.0) == 2.0
    assert e(4.0) == 3.0
    assert e(float("nan")) == 3.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
