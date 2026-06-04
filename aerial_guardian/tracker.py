"""CA-KF ByteTrack tracker with camera motion compensation."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import (
    DENSE_SCENE_TRACKS,
    HIGH_CONF_CROWDED,
    HIGH_CONF_STANDARD,
    IOU_MATCH_THRESHOLD,
    LOW_CONF_CROWDED,
    LOW_CONF_STANDARD,
)


class CAKalmanTrack:
    """Constant-acceleration Kalman filter for one drone-video track.

    The state is [cx, cy, w, h, vx, vy, vw, vh, ax, ay]. Camera ego-motion adds
    acceleration to apparent person movement; this model predicts that smoother
    motion before IoU matching and therefore reduces avoidable ID switches.
    """

    _next_id = 1

    def __init__(self, bbox: np.ndarray):
        x1, y1, x2, y2 = map(float, bbox)
        width, height = x2 - x1, y2 - y1
        self.x = np.array([x1 + width / 2, y1 + height / 2, width, height, 0, 0, 0, 0, 0, 0], dtype=float)
        self.p = np.eye(10) * 20.0
        self.p[4:, 4:] *= 50.0
        self.f = np.eye(10)
        self.f[0, 4] = self.f[1, 5] = self.f[2, 6] = self.f[3, 7] = 1.0
        self.f[0, 8] = self.f[1, 9] = 0.5
        self.f[4, 8] = self.f[5, 9] = 1.0
        self.h = np.zeros((4, 10))
        self.h[:4, :4] = np.eye(4)
        self.q = np.diag([0.2, 0.2, 0.1, 0.1, 1.0, 1.0, 0.4, 0.4, 3.0, 3.0])
        self.id = CAKalmanTrack._next_id
        CAKalmanTrack._next_id += 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.history: list[tuple[float, float]] = []

    @staticmethod
    def bbox_to_observation(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        width, height = x2 - x1, y2 - y1
        return np.array([x1 + width / 2, y1 + height / 2, width, height], dtype=float)

    @staticmethod
    def state_to_bbox(x: np.ndarray) -> np.ndarray:
        cx, cy, width, height = x[:4]
        width, height = max(width, 1.0), max(height, 1.0)
        return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])

    def predict(self) -> np.ndarray:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        return self.get_state()

    def update(self, bbox: np.ndarray) -> None:
        z = self.bbox_to_observation(bbox)
        area = max(z[2] * z[3], 1.0)
        r = np.eye(4) * max(1.0, 2000.0 / area)
        residual = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ residual
        self.p = (np.eye(10) - k @ self.h) @ self.p
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history.append((float(self.x[0]), float(self.x[1])))
        self.history = self.history[-60:]

    def warp_state(self, affine: np.ndarray) -> None:
        bbox = self.get_state()
        points = np.array([[bbox[0], bbox[1], 1.0], [bbox[2], bbox[3], 1.0]])
        warped = points @ affine.T
        self.x[:4] = self.bbox_to_observation(np.array([warped[0, 0], warped[0, 1], warped[1, 0], warped[1, 1]]))

    def get_state(self) -> np.ndarray:
        return self.state_to_bbox(self.x)


class SparseLKCameraMotion:
    """Sparse Lucas-Kanade CMC with RANSAC outlier rejection.

    Background keypoints share the camera transform; moving people do not.
    RANSAC rejects those inconsistent person correspondences and keeps the global
    affine motion used to warp predicted tracks before association.
    """

    def __init__(self, downscale: float = 0.5):
        self.prev_gray: np.ndarray | None = None
        self.downscale = downscale
        self.feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=20, blockSize=3)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def compute(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale != 1.0:
            gray = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale)

        identity = np.eye(2, 3, dtype=float)
        if self.prev_gray is None:
            self.prev_gray = gray
            return identity

        prev_points = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
        if prev_points is None or len(prev_points) < 4:
            self.prev_gray = gray
            return identity

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_points, None, **self.lk_params)
        good = status.reshape(-1) == 1
        if good.sum() < 4:
            self.prev_gray = gray
            return identity

        affine, _ = cv2.estimateAffinePartial2D(
            prev_points[good].reshape(-1, 2),
            curr_points[good].reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        self.prev_gray = gray
        if affine is None:
            return identity
        if self.downscale != 1.0:
            affine[0, 2] /= self.downscale
            affine[1, 2] /= self.downscale
        return affine


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for xyxy boxes."""

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


class AerialByteTracker:
    """ByteTrack-style two-stage association with CA-KF and CMC."""

    def __init__(self, max_age: int = 30, min_hits: int = 2, use_cmc: bool = True):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[CAKalmanTrack] = []
        self.frame_count = 0
        self.cmc = SparseLKCameraMotion() if use_cmc else None
        CAKalmanTrack._next_id = 1

    def _thresholds(self) -> tuple[float, float]:
        if len(self.tracks) > DENSE_SCENE_TRACKS:
            return HIGH_CONF_CROWDED, LOW_CONF_CROWDED
        return HIGH_CONF_STANDARD, LOW_CONF_STANDARD

    def _associate(self, tracks: list[CAKalmanTrack], detections: np.ndarray):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = 1.0 - iou_matrix(np.array([track.get_state() for track in tracks]), detections)
        rows, cols = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))
        for row, col in zip(rows, cols):
            if cost[row, col] <= 1.0 - IOU_MATCH_THRESHOLD:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_dets.discard(col)
        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    def update(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Update tracks with detections and return [x1,y1,x2,y2,id]."""

        self.frame_count += 1
        affine = self.cmc.compute(frame) if self.cmc is not None else np.eye(2, 3, dtype=float)
        for track in self.tracks:
            track.warp_state(affine)
            track.predict()

        high_threshold, low_threshold = self._thresholds()
        high = detections[detections[:, 4] >= high_threshold, :4] if len(detections) else np.empty((0, 4))
        low = detections[
            (detections[:, 4] >= low_threshold) & (detections[:, 4] < high_threshold), :4
        ] if len(detections) else np.empty((0, 4))

        matches, unmatched_tracks, unmatched_high = self._associate(self.tracks, high)
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high[det_idx])

        remaining_tracks = [self.tracks[idx] for idx in unmatched_tracks]
        low_matches, _, _ = self._associate(remaining_tracks, low)
        for local_track_idx, det_idx in low_matches:
            self.tracks[unmatched_tracks[local_track_idx]].update(low[det_idx])

        for det_idx in unmatched_high:
            self.tracks.append(CAKalmanTrack(high[det_idx]))

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

        outputs = []
        for track in self.tracks:
            if track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                outputs.append([*track.get_state(), track.id])
        return np.array(outputs, dtype=np.float32) if outputs else np.empty((0, 5), dtype=np.float32)

    def trajectories(self) -> dict[int, list[tuple[float, float]]]:
        """Return active trajectory histories."""

        return {track.id: track.history for track in self.tracks}
