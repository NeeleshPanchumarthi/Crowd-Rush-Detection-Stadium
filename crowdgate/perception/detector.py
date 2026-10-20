"""Sparse mode: YOLO11n (COCO person class) + ByteTrack, both from Ultralytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ROOT

YOLO_WEIGHTS = ROOT / "models" / "yolo11n.pt"


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) x0, y0, x1, y1 in full-frame pixels
    conf: np.ndarray  # (N,)
    ids: np.ndarray  # (N,) track id, -1 if not yet confirmed

    @property
    def feet(self) -> np.ndarray:
        """Bottom-centre of each box: the point that touches the floor."""
        if len(self.boxes) == 0:
            return np.zeros((0, 2))
        return np.stack([(self.boxes[:, 0] + self.boxes[:, 2]) / 2, self.boxes[:, 3]], axis=1)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))


class PersonTracker:
    """One instance per gate: ByteTrack keeps its state inside the Ultralytics predictor."""

    def __init__(self, weights=YOLO_WEIGHTS, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def detect_heads(self, frame: np.ndarray, offset=(0, 0)) -> Detections:
        """Detection only (no IDs). Pretrained COCO 'person' boxes; the head model is a Part 2 add-on."""
        r = self.model.predict(frame, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return self._to_dets(r, offset, with_ids=False)

    def track(self, frame: np.ndarray, offset=(0, 0)) -> Detections:
        """Detection + ByteTrack IDs across frames (persist=True keeps the tracker state)."""
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                             conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return self._to_dets(r, offset, with_ids=True)

    def reset(self):
        """Drop tracker state (used when we come back from dense mode)."""
        pred = getattr(self.model, "predictor", None)
        if pred is not None and getattr(pred, "trackers", None):
            for t in pred.trackers:
                t.reset()

    @staticmethod
    def _to_dets(r, offset, with_ids: bool) -> Detections:
        if r.boxes is None or len(r.boxes) == 0:
            return Detections.empty()
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        boxes[:, [0, 2]] += offset[0]
        boxes[:, [1, 3]] += offset[1]
        conf = r.boxes.conf.cpu().numpy()
        if with_ids and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids = -np.ones(len(boxes), dtype=int)
        return Detections(boxes, conf, ids)
