"""Runs one gate's video through a GatePipeline on a background thread for the dashboard.

Playback follows the wall clock (times `speed`): if processing is slower than real time, frames are
skipped instead of the video falling behind, so the alert latency stays meaningful.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2

from .config import GateConfig
from .pipeline import GatePipeline
from .response import EventLog
from .video import open_stream


class GateRunner:
    def __init__(self, cfg: GateConfig, all_cfgs: dict, shared_states: dict, event_log: EventLog,
                 speed: float = 1.0, loop: bool = False, history_len: int = 600):
        self.cfg, self.speed, self.loop = cfg, speed, loop
        self.stream = open_stream(cfg.resolve(cfg["source"]), folder_fps=cfg.get("source_fps") or 2.0)
        self.pipe = GatePipeline(cfg, (self.stream.height, self.stream.width), sync_density=False,
                                 event_log=event_log, shared_states=shared_states, all_cfgs=all_cfgs)
        self.history: deque = deque(maxlen=history_len)  # dicts of metrics
        self.frame = None  # latest annotated BGR frame (with heat-map)
        self.frame_plain = None  # same without the heat-map
        self.state = None
        self.fps = 0.0
        self.finished = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        s, pipe = self.stream, self.pipe
        wall0 = time.perf_counter()
        last_wall = wall0
        while not self._stop.is_set():
            target_t = (time.perf_counter() - wall0) * self.speed
            ok, frame, idx = s.read()
            if not ok:
                if self.loop:
                    s.rewind()
                    wall0 = time.perf_counter()
                    continue
                self.finished = True
                return
            t = idx / s.fps
            if t < target_t - 1.0 / s.fps:
                continue  # behind the wall clock: skip this frame
            if t > target_t:
                time.sleep(min(0.5, (t - target_t) / self.speed))
            now = time.time()
            st = pipe.process(frame, t, idx, capture_wall=now)
            self.frame_plain = pipe.draw(frame, st, heatmap=False)
            self.frame = pipe.draw(frame, st, heatmap=True)
            self.state = st
            d = st.as_dict()
            self.history.append(d)
            tw = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(tw - last_wall, 1e-3)
            last_wall = tw

    def stop(self):
        self._stop.set()
        self.pipe.close()
        self.stream.release()
