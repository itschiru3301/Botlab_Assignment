"""Detector wrapper with adaptive SAHI support."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import TINY_AREA_PX


class AdaptiveSAHIController:
    """Decide when tiled inference is worth its latency.

    Direct YOLO is fast and good for close subjects. SAHI improves tiny-person
    recall by running overlapping crops, but costs several extra forward passes.
    This controller watches recent detections and enables SAHI only for dense or
    high-altitude scenes, with hysteresis to avoid flickering.
    """

    def __init__(self, window: int = 10, high_threshold: float = 0.6, low_threshold: float = 0.3):
        self.history: deque[tuple[int, float]] = deque(maxlen=window)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.enabled = False

    def update(self, detections: np.ndarray) -> tuple[bool, float]:
        """Update complexity estimate from direct detections."""

        if len(detections) == 0:
            tiny_fraction = 0.0
        else:
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            tiny_fraction = float((areas < TINY_AREA_PX).mean())

        self.history.append((len(detections), tiny_fraction))
        mean_detections = float(np.mean([item[0] for item in self.history]))
        mean_tiny = float(np.mean([item[1] for item in self.history]))
        density_term = min(mean_detections / 40.0, 1.0)
        score = 0.5 * density_term + 0.5 * mean_tiny

        if score > self.high_threshold:
            self.enabled = True
        elif score < self.low_threshold:
            self.enabled = False
        return self.enabled, score


class PersonDetector:
    """YOLOv8 person detector optimized for lightweight drone inference."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "0",
        imgsz: int = 960,
        conf: float = 0.25,
        iou: float = 0.45,
        use_sahi: bool = False,
        slice_size: int = 640,
        overlap: float = 0.20,
    ):
        self.weights = str(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.overlap = overlap
        self.model = YOLO(self.weights)
        self.person_class_ids = [0]
        self._sahi_model = None

        if self.use_sahi:
            self._setup_sahi()

    def _setup_sahi(self) -> None:
        """Create a SAHI wrapper around the same YOLO model."""

        from sahi import AutoDetectionModel

        self._sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.weights,
            confidence_threshold=self.conf,
            device=f"cuda:{self.device}" if str(self.device).isdigit() else self.device,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return detections as [x1, y1, x2, y2, confidence]."""

        if self.use_sahi:
            return self._detect_sahi(frame)
        return self._detect_direct(frame)

    def _detect_direct(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.person_class_ids,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()[:, None]
        return np.hstack([xyxy, conf]).astype(np.float32)

    def _detect_sahi(self, frame: np.ndarray) -> np.ndarray:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            self._sahi_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=self.iou,
            verbose=0,
        )
        detections = []
        for pred in result.object_prediction_list:
            if int(pred.category.id) != 0:
                continue
            bbox = pred.bbox
            detections.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, pred.score.value])
        return np.asarray(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)
