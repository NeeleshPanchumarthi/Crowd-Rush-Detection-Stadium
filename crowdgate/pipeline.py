"""Glue: one GatePipeline per gate turns frames into metrics, a risk level and alerts.

Frame loop (per processed frame):
  crop to gate ROI -> optical flow (always) -> YOLO+ByteTrack (sparse mode) or DM-Count (dense mode,
  background thread) -> floor metrics -> smoothed -> risk score -> level with hold time -> alert + log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import foot_offset_px, lifted_zone_mask, zone_mask_from_floor
from .config import GateConfig
from .geometry import (downstream_strip, homography_jacobian, local_scale_px_per_m, points_in_polygon,
                       polygon_area, to_floor, to_image)
from .metrics import EMA, LineCounter, MotionVariance, forward_speed, line_crossings, zone_density
from .perception.density import DensityWorker, use_weights
from .perception.detector import Detections, PersonTracker
from .perception.flow import compute_flow, to_small_grey
from .perception.mode import DENSE, SPARSE, select_mode
from .response import EventLog, make_alert
from .risk import GREEN, LevelState, risk_level, risk_score
from .video import crop_to_zone

LEVEL_BGR = {"green": (60, 180, 60), "amber": (0, 165, 255), "red": (40, 40, 220)}


@dataclass
class GateState:
    gate_id: str
    name: str
    t: float
    frame_idx: int
    level: str
    mode: str
    count: float
    density: float
    speed: float
    motion_variance: float
    inflow_ppm: float
    outflow_ppm: float
    accumulation_ppm: float
    score: float
    parts: dict
    proc_ms: float
    latency_ms: float
    density_age_s: float | None
    alert: object | None = None
    frame: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self, with_frame: bool = False) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("frame", "alert")}
        d["alert"] = self.alert.message if self.alert else None
        if with_frame:
            d["frame"] = self.frame
        return d


class GatePipeline:
    def __init__(self, cfg: GateConfig, frame_shape, sync_density: bool = False,
                 event_log: EventLog | None = None, shared_states: dict | None = None,
                 all_cfgs: dict | None = None, tracker: PersonTracker | None = None):
        self.cfg = cfg
        p = cfg.perception
        self.p = p
        self.H, self.H_inv = cfg.H, cfg.H_inv
        self.frame_h, self.frame_w = frame_shape[:2]
        self.event_log = event_log
        self.shared_states = shared_states if shared_states is not None else {}
        self.all_cfgs = all_cfgs or {cfg.gate_id: cfg}

        # Zone geometry ------------------------------------------------------------------------
        self.zone_m = cfg.zone_m
        self.zone_area = polygon_area(self.zone_m)
        self.zone_px = to_image(self.zone_m, self.H_inv)
        self.zone_mask_full = zone_mask_from_floor(self.zone_m, self.H_inv, frame_shape)

        # Counting lines. For the dense-mode flux estimate each line gets a strip on its downstream
        # side: people just past the gate line are the ones who actually got out, whereas the queue
        # pressing on the line from behind barely moves.
        win = p["crossing_window_s"]
        self.lines = {}
        self.strip_mask, self.strip_area = {}, {}
        for name in ("approach", "gate"):
            ln = cfg.line(name)
            if ln is None:
                continue
            self.lines[name] = LineCounter(ln[0], ln[1], cfg.exit_dir, win)
            self.lines[name].strip_poly = downstream_strip(ln[0], ln[1], self.lines[name].n, 1.5)

        self.roi = self._auto_roi() if cfg.get("roi") is None else [int(v) for v in cfg["roi"]]
        x0, y0, x1, y1 = self.roi
        self.offset = (x0, y0)
        # Density maps put mass on heads, so count with the zone lifted to head height.
        self.head_lift_m = 0.9 * p["person_height_m"]
        self.zone_mask = lifted_zone_mask(self.zone_m, self.H, self.H_inv, frame_shape, self.roi,
                                          self.head_lift_m).astype(np.float32)
        for name, ln in self.lines.items():
            self.strip_mask[name] = lifted_zone_mask(ln.strip_poly, self.H, self.H_inv, frame_shape, self.roi,
                                                     self.head_lift_m).astype(np.float32)
            self.strip_area[name] = polygon_area(ln.strip_poly)

        # Flow sampling grid with per-sample Jacobian (metres per pixel at the estimated foot) ---
        self._build_flow_grid()

        # Models -----------------------------------------------------------------------------------
        self.tracker = tracker or PersonTracker(conf=p["yolo_conf"], imgsz=p["yolo_imgsz"])
        use_weights(p.get("dm_weights"))
        self.dworker = DensityWorker(sync=sync_density)

        # State --------------------------------------------------------------------------------
        self.mode = SPARSE
        self.mode_since = 0.0
        self.prev_grey = None
        self.prev_t = None
        self.prev_track_pos: dict = {}
        self.last_submit_t = -1e9
        self.last_density_t = None  # video time of the density map in use
        self.density_map = None  # crop-sized
        self.density_count = 0.0
        self.first_count_density = 0.0
        self.ema_density = EMA(p["smoothing_alpha"])
        self.ema_speed = EMA(p["smoothing_alpha"])
        self.ema_count = EMA(p["smoothing_alpha"])
        self.mvar = MotionVariance(p["variance_window_s"])
        self.level_state = LevelState()
        self.dets = Detections.empty()
        self.last_state: GateState | None = None

    # ------------------------------------------------------------------------------------------
    def _auto_roi(self):
        """Bounding box of the zone in the image, extended upwards by one person height so heads
        of people standing at the far edge are inside the crop."""
        ph = self.cfg.perception["person_height_m"]
        polys = [self.zone_m] + [ln.strip_poly for ln in self.lines.values()]
        corners = np.concatenate(polys)
        pts = [to_image(corners, self.H_inv)]
        for c in corners:
            s = local_scale_px_per_m(c, self.H_inv)
            foot = to_image(c[None], self.H_inv)[0]
            pts.append((foot - [0, ph * s])[None])
        allp = np.concatenate(pts)
        m = 20
        x0 = int(max(0, allp[:, 0].min() - m)); x1 = int(min(self.frame_w, allp[:, 0].max() + m))
        y0 = int(max(0, allp[:, 1].min() - m)); y1 = int(min(self.frame_h, allp[:, 1].max() + m))
        return [x0, y0, x1, y1]

    def _build_flow_grid(self):
        s = self.p["flow_scale"]
        g = self.p["flow_grid_px"]
        ph = self.p["person_height_m"]
        x0, y0, x1, y1 = self.roi
        ws, hs = int((x1 - x0) * s), int((y1 - y0) * s)
        gy, gx = np.mgrid[g // 2:hs:g, g // 2:ws:g]
        small = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
        full = small / s + [x0, y0]
        # Estimate where the floor is under each sampled body pixel: half a person height lower.
        foot = foot_offset_px(full, self.H, self.H_inv, 0.5 * ph)
        foot_m = to_floor(foot, self.H)
        scale = (foot[:, 1] - full[:, 1]) / (0.5 * ph)  # px per metre at the foot
        in_zone = points_in_polygon(foot_m, self.zone_m)
        inside = in_zone.copy()
        for ln in self.lines.values():
            inside |= points_in_polygon(foot_m, ln.strip_poly)
        self.grid_in_zone = in_zone[inside]
        self.grid_small = small[inside].astype(np.int32)  # (N, 2) in downscaled crop
        self.grid_crop = (small[inside] / s).astype(np.int32)  # in crop pixels
        self.grid_crop[:, 0] = np.clip(self.grid_crop[:, 0], 0, x1 - x0 - 1)
        self.grid_crop[:, 1] = np.clip(self.grid_crop[:, 1], 0, y1 - y0 - 1)
        self.grid_foot_m = foot_m[inside]
        self.grid_J = homography_jacobian(self.H, foot[inside])  # (N, 2, 2)
        sc = scale[inside]
        self.grid_person_area_px = 0.4 * (ph * sc) ** 2  # rough image area of one person
        self.grid_in_strip = {name: points_in_polygon(self.grid_foot_m, ln.strip_poly)
                              for name, ln in self.lines.items()}

    # ------------------------------------------------------------------------------------------
    def _presence(self, crop_shape) -> np.ndarray:
        """Which flow samples sit on a person (we want crowd motion, not the static floor)."""
        n = len(self.grid_crop)
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self.mode == DENSE and self.density_map is not None:
            blur = cv2.blur(self.density_map, (15, 15))
            local = blur[self.grid_crop[:, 1], self.grid_crop[:, 0]] * self.grid_person_area_px
            return local > 0.3
        if len(self.dets.boxes) == 0:
            return np.zeros(n, dtype=bool)
        pts = self.grid_crop + self.offset
        b = self.dets.boxes
        inside = ((pts[:, None, 0] >= b[None, :, 0]) & (pts[:, None, 0] <= b[None, :, 2]) &
                  (pts[:, None, 1] >= b[None, :, 1]) & (pts[:, None, 1] <= b[None, :, 3]))
        return inside.any(1)

    def _per_person(self, flow_m: np.ndarray, pts_full: np.ndarray) -> np.ndarray:
        """Sparse mode: one velocity per detected person (median of the flow samples on them),
        so background pixels inside a box do not inflate the motion variance."""
        b = self.dets.boxes
        inside = ((pts_full[:, None, 0] >= b[None, :, 0]) & (pts_full[:, None, 0] <= b[None, :, 2]) &
                  (pts_full[:, None, 1] >= b[None, :, 1]) & (pts_full[:, None, 1] <= b[None, :, 3]))
        owner = np.where(inside.any(1), inside.argmax(1), -1)
        out = [np.median(flow_m[owner == k], axis=0) for k in range(len(b)) if (owner == k).sum() >= 2]
        return np.array(out) if out else np.zeros((0, 2))

    def _handle_density_result(self):
        res = self.dworker.latest()
        if res is None or res[0] == self.last_density_t:
            return
        t_sub, dmap, kind = res
        self.last_density_t = t_sub
        count = float((dmap * self.zone_mask).sum())
        dens = zone_density(count, self.zone_area)
        if kind == "first":
            self.first_count_density = dens
        else:
            self.density_map = dmap
            self.density_count = count
        if kind == "first" and self.density_map is None:
            self.density_map = dmap  # something to show on the heat-map before dense mode

    def process(self, frame: np.ndarray, t: float, frame_idx: int = 0, capture_wall: float | None = None) -> GateState:
        t_start = time.perf_counter()
        capture_wall = capture_wall or time.time()
        p = self.p
        crop = crop_to_zone(frame, self.roi)

        # Optical flow (always on) ----------------------------------------------------------
        grey = to_small_grey(crop, p["flow_scale"])
        dt = (t - self.prev_t) if self.prev_t is not None else None
        flow_m = np.zeros((0, 2))
        flow = None
        if self.prev_grey is not None and dt and dt > 0:
            flow = compute_flow(self.prev_grey, grey)

        # Density worker: first count (sparse) or full density map (dense) ------------------
        self._handle_density_result()
        every = p["density_every_s"] if self.mode == DENSE else p["first_count_every_s"]
        if not self.dworker.busy and t - self.last_submit_t >= every:
            kind, scale = ("dense", p["density_scale"]) if self.mode == DENSE else ("first", p["first_count_scale"])
            self.dworker.submit(t, crop, scale, kind)
            self.last_submit_t = t
            self._handle_density_result()  # sync mode: result is already there

        # Detection + tracking (sparse mode) ------------------------------------------------
        curr_track_pos = {}
        if self.mode == SPARSE:
            self.dets = self.tracker.track(crop, offset=self.offset)
            feet_m = to_floor(self.dets.feet, self.H)
            in_zone = points_in_polygon(feet_m, self.zone_m)
            raw_count = float(in_zone.sum())
            curr_track_pos = {int(i): f for i, f in zip(self.dets.ids, feet_m) if i >= 0}
        else:
            self.dets = Detections.empty()
            raw_count = self.density_count

        # Flow -> floor velocities for samples that sit on people ----------------------------
        if flow is not None:
            pres_all = self._presence(crop.shape)
            pres = pres_all & self.grid_in_zone
            if pres.any():
                g = self.grid_small[pres]
                fpx = flow[g[:, 1], g[:, 0]] / p["flow_scale"]  # full-res pixels per step
                flow_m = np.einsum("nij,nj->ni", self.grid_J[pres], fpx) / dt
                if self.mode == SPARSE:
                    flow_m = self._per_person(flow_m, self.grid_crop[pres] + self.offset)
        else:
            pres_all = np.zeros(len(self.grid_small), dtype=bool)

        # Metrics -----------------------------------------------------------------------------
        if self.mode == SPARSE:
            dens_now = zone_density(raw_count, self.zone_area)
            density = self.ema_density(dens_now)
        else:
            # Dense mode: only feed the EMA when a new density map has arrived.
            if self.last_density_t is not None and getattr(self, "_fed_t", None) != self.last_density_t:
                self._fed_t = self.last_density_t
                self.ema_density(zone_density(self.density_count, self.zone_area))
            density = self.ema_density.value if self.ema_density.value is not None else 0.0
        count = density * self.zone_area
        speed_raw = forward_speed(flow_m, self.cfg.exit_dir) if len(flow_m) >= 1 else float("nan")
        if len(flow_m) < 1 and density < 0.05:
            speed_raw = float("nan")  # empty zone: nothing to measure
        speed = self.ema_speed(speed_raw)
        mvar = self.mvar.update(t, flow_m)

        # Line crossings ------------------------------------------------------------------------
        if self.mode == SPARSE:
            for ln in self.lines.values():
                ln.update_tracks(t, self.prev_track_pos, curr_track_pos)
        elif dt and self.density_map is not None and flow is not None:
            for name, ln in self.lines.items():
                d_near = float((self.density_map * self.strip_mask[name]).sum()) / self.strip_area[name]
                sel = pres_all & self.grid_in_strip[name]
                if sel.sum() >= 3:
                    g = self.grid_small[sel]
                    fpx = flow[g[:, 1], g[:, 0]] / p["flow_scale"]
                    v = np.einsum("nij,nj->ni", self.grid_J[sel], fpx) / dt
                    vn = float(np.mean(v @ ln.n))
                else:
                    vn = float(np.mean(flow_m @ ln.n)) if len(flow_m) else 0.0
                ln.update_flux(t, dt, d_near, max(vn, 0.0))
        if "approach" in self.lines and "gate" in self.lines:
            rates = line_crossings(self.lines["approach"], self.lines["gate"], t)
        else:
            rates = {"inflow_ppm": float("nan"), "outflow_ppm": float("nan"), "accumulation_ppm": 0.0}

        # Mode switch ------------------------------------------------------------------------------
        switch_d = max(density, self.first_count_density) if self.mode == SPARSE else density
        det_count = raw_count if self.mode == SPARSE else None
        new_mode = select_mode(count, switch_d, self.mode, p["switch_density"], p["hysteresis"],
                               first_count=self.first_count_density * self.zone_area, detector_count=det_count,
                               time_in_mode=t - self.mode_since, min_dwell_s=p["min_mode_dwell_s"])
        if new_mode != self.mode:
            if new_mode == SPARSE:
                self.tracker.reset()
                self.prev_track_pos = {}
            else:
                self.last_submit_t = -1e9  # get a full density map as soon as possible
                # Start the dense EMA from the current estimate so the level does not jump.
            self.mode = new_mode
            self.mode_since = t

        # Risk -----------------------------------------------------------------------------------
        metrics = {"density": density, "speed": speed, "motion_variance": mvar, **rates}
        score, parts = risk_score(metrics, self.cfg)
        metrics["score"] = score
        prev_level = self.level_state.level
        level, changed = risk_level(score, density, t, self.level_state, self.cfg)
        alert = None
        if changed:
            alert = make_alert(self.cfg, level, prev_level, metrics, t, self.shared_states, self.all_cfgs)
            alert.latency_ms = round((time.time() - capture_wall) * 1000, 1)
            if self.event_log is not None:
                self.event_log.log_event(alert)

        proc_ms = (time.perf_counter() - t_start) * 1000
        state = GateState(
            gate_id=self.cfg.gate_id, name=self.cfg.name, t=t, frame_idx=frame_idx, level=level, mode=self.mode,
            count=count, density=density, speed=speed, motion_variance=mvar,
            inflow_ppm=rates["inflow_ppm"], outflow_ppm=rates["outflow_ppm"],
            accumulation_ppm=rates["accumulation_ppm"], score=score, parts=parts, proc_ms=proc_ms,
            latency_ms=(time.time() - capture_wall) * 1000,
            density_age_s=(t - self.last_density_t) if self.last_density_t is not None else None,
            alert=alert,
        )
        self.shared_states[self.cfg.gate_id] = state.as_dict()
        self.prev_grey, self.prev_t = grey, t
        if self.mode == SPARSE:
            self.prev_track_pos = curr_track_pos
        self.last_state = state
        return state

    # ------------------------------------------------------------------------------------------
    def draw(self, frame: np.ndarray, state: GateState, heatmap: bool = True) -> np.ndarray:
        """Annotated frame: heat-map inside the zone, zone and lines, boxes, status bar."""
        vis = frame.copy()
        x0, y0, x1, y1 = self.roi
        if heatmap and self.density_map is not None:
            d = cv2.GaussianBlur(self.density_map, (0, 0), 3)
            norm = np.clip(d / (np.percentile(d, 99.5) + 1e-9), 0, 1)
            hm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            a = (0.55 * norm * (self.zone_mask > 0))[..., None]
            region = vis[y0:y1, x0:x1].astype(np.float32)
            vis[y0:y1, x0:x1] = (region * (1 - a) + hm.astype(np.float32) * a).astype(np.uint8)
        col = LEVEL_BGR[state.level]
        cv2.polylines(vis, [np.round(self.zone_px).astype(np.int32)], True, col, 2)
        for name, ln in self.lines.items():
            pa, pb = to_image(np.array([ln.a, ln.b]), self.H_inv)
            c = (255, 255, 0) if name == "approach" else (255, 0, 255)
            cv2.line(vis, tuple(np.int32(pa)), tuple(np.int32(pb)), c, 2)
        for b, i in zip(self.dets.boxes.astype(int), self.dets.ids):
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 1)
            if i >= 0:
                cv2.putText(vis, str(i), (b[0], b[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), col, -1)
        spd = "-" if not np.isfinite(state.speed) else f"{state.speed:.2f}"
        txt = (f"{self.cfg.name}  {state.level.upper()}  {state.mode}  n={state.count:.0f}  "
               f"{state.density:.2f}/m2  v={spd}m/s  score={state.score:.2f}")
        cv2.putText(vis, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def close(self):
        self.dworker.stop()
