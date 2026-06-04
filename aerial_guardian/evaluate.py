"""MOT metrics for Aerial Guardian outputs."""

from __future__ import annotations

from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from .config import PERSON_CATEGORIES
from .dataset import read_annotation_file

if not hasattr(np, "asfarray"):
    np.asfarray = lambda values: np.asarray(values, dtype=float)  # motmetrics<1.5 compatibility with NumPy 2.x


def load_ground_truth(path: Path) -> dict[int, list[tuple[int, list[float]]]]:
    """Load person-only ground truth as frame -> [(id, xyxy)]."""

    frames: dict[int, list[tuple[int, list[float]]]] = {}
    for box in read_annotation_file(path):
        if box.category not in PERSON_CATEGORIES or box.score == 0:
            continue
        frames.setdefault(box.frame_id, []).append((box.track_id, [box.x, box.y, box.x + box.w, box.y + box.h]))
    return frames


def load_predictions(path: Path) -> dict[int, list[tuple[int, list[float]]]]:
    """Load MOT-format predictions as frame -> [(id, xyxy)]."""

    frames: dict[int, list[tuple[int, list[float]]]] = {}
    if not path.exists():
        return frames
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id, track_id = int(parts[0]), int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            frames.setdefault(frame_id, []).append((track_id, [x, y, x + w, y + h]))
    return frames


def evaluate_sequence(gt: dict, pred: dict) -> mm.MOTAccumulator:
    """Build a motmetrics accumulator for one sequence."""

    acc = mm.MOTAccumulator(auto_id=True)
    for frame_id in sorted(set(gt) | set(pred)):
        gt_items = gt.get(frame_id, [])
        pred_items = pred.get(frame_id, [])
        gt_ids = [item[0] for item in gt_items]
        pred_ids = [item[0] for item in pred_items]
        gt_boxes = np.array([item[1] for item in gt_items]) if gt_items else np.empty((0, 4))
        pred_boxes = np.array([item[1] for item in pred_items]) if pred_items else np.empty((0, 4))
        distances = (
            mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            if len(gt_boxes) and len(pred_boxes)
            else np.empty((len(gt_boxes), len(pred_boxes)))
        )
        acc.update(gt_ids, pred_ids, distances)
    return acc


def evaluate_outputs(dataset_root: Path, tracks_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Evaluate all track files that have matching ground-truth annotations."""

    accumulators = []
    names = []
    for gt_path in sorted((dataset_root / "annotations").glob("*.txt")):
        pred_path = tracks_dir / gt_path.name
        if not pred_path.exists():
            continue
        accumulators.append(evaluate_sequence(load_ground_truth(gt_path), load_predictions(pred_path)))
        names.append(gt_path.stem)

    if not accumulators:
        summary = pd.DataFrame()
    else:
        metrics = mm.metrics.create()
        summary = metrics.compute_many(
            accumulators,
            names=names,
            metrics=["mota", "idf1", "num_switches", "precision", "recall"],
            generate_overall=True,
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv)
    return summary
