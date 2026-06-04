"""End-to-end Aerial Guardian processing pipeline."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .detector import AdaptiveSAHIController, PersonDetector
from .tracker import AerialByteTracker
from .visualize import draw_tracks


class AerialGuardianPipeline:
    """Detector + adaptive SAHI + CA-KF ByteTrack + visualization."""

    def __init__(
        self,
        weights: str | Path,
        output_root: str | Path,
        device: str = "0",
        imgsz: int = 960,
        conf: float = 0.25,
        use_adaptive_sahi: bool = True,
        save_video: bool = True,
    ):
        self.output_root = Path(output_root)
        self.video_dir = self.output_root / "videos"
        self.track_dir = self.output_root / "tracks"
        self.metric_dir = self.output_root / "metrics"
        for directory in (self.video_dir, self.track_dir, self.metric_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.direct_detector = PersonDetector(weights, device=device, imgsz=imgsz, conf=conf, use_sahi=False)
        self.sahi_detector = PersonDetector(weights, device=device, imgsz=imgsz, conf=conf, use_sahi=True)
        self.sahi_controller = AdaptiveSAHIController()
        self.use_adaptive_sahi = use_adaptive_sahi
        self.save_video = save_video

    def process_sequence(self, sequence_dir: str | Path) -> dict:
        """Process one sequence and save video + MOT-format track file."""

        sequence_dir = Path(sequence_dir)
        frame_paths = sorted(sequence_dir.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {sequence_dir}")

        tracker = AerialByteTracker(max_age=30, min_hits=2, use_cmc=True)
        tails: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=30))
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]

        writer = None
        video_path = self.video_dir / f"{sequence_dir.name}.mp4"
        if self.save_video:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

        track_lines: list[str] = []
        timings = []
        sahi_frames = 0

        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=sequence_dir.name), start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            start = time.perf_counter()
            quick_detections = self.direct_detector.detect(frame)
            use_sahi, complexity = self.sahi_controller.update(quick_detections)
            if self.use_adaptive_sahi and use_sahi:
                detections = self.sahi_detector.detect(frame)
                sahi_frames += 1
            else:
                detections = quick_detections

            tracks = tracker.update(frame, detections)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            fps = 1.0 / max(elapsed, 1e-9)

            for x1, y1, x2, y2, track_id_float in tracks:
                track_id = int(track_id_float)
                tails[track_id].append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                track_lines.append(
                    f"{frame_idx},{track_id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n"
                )

            if writer is not None:
                writer.write(draw_tracks(frame, tracks, tails, fps, self.use_adaptive_sahi and use_sahi, complexity))

        if writer is not None:
            writer.release()

        track_path = self.track_dir / f"{sequence_dir.name}.txt"
        track_path.write_text("".join(track_lines), encoding="utf-8")

        avg_time = float(np.mean(timings)) if timings else 0.0
        stats = {
            "sequence": sequence_dir.name,
            "frames": len(frame_paths),
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "avg_ms": avg_time * 1000.0,
            "sahi_frames": sahi_frames,
            "video": str(video_path) if self.save_video else "",
            "tracks": str(track_path),
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stats

    def process_dataset(self, dataset_root: str | Path, sequences: list[str] | None = None) -> pd.DataFrame:
        """Process selected sequences, or every sequence if none are specified."""

        dataset_root = Path(dataset_root)
        if sequences is None:
            sequence_dirs = sorted((dataset_root / "sequences").iterdir())
        else:
            sequence_dirs = [dataset_root / "sequences" / name for name in sequences]

        rows = [self.process_sequence(sequence_dir) for sequence_dir in sequence_dirs]
        summary = pd.DataFrame(rows)
        summary.to_csv(self.metric_dir / "pipeline_fps.csv", index=False)
        return summary
