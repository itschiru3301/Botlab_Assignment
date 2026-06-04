"""VisDrone MOT parsing and conversion utilities."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from .config import PERSON_CATEGORIES


@dataclass(frozen=True)
class VisDroneBox:
    """One VisDrone annotation row converted to a convenient object."""

    frame_id: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float
    category: int
    truncation: int = 0
    occlusion: int = 0


def read_annotation_file(path: Path) -> list[VisDroneBox]:
    """Read one VisDrone MOT annotation file.

    VisDrone stores MOT boxes as frame,id,x,y,w,h,score,class,truncation,occlusion.
    For this challenge we keep only person-like classes later, but parse all rows.
    """

    boxes: list[VisDroneBox] = []
    with path.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 8:
                continue
            boxes.append(
                VisDroneBox(
                    frame_id=int(row[0]),
                    track_id=int(row[1]),
                    x=float(row[2]),
                    y=float(row[3]),
                    w=float(row[4]),
                    h=float(row[5]),
                    score=float(row[6]),
                    category=int(row[7]),
                    truncation=int(row[8]) if len(row) > 8 else 0,
                    occlusion=int(row[9]) if len(row) > 9 else 0,
                )
            )
    return boxes


def convert_mot_to_yolo(
    dataset_root: Path,
    output_root: Path,
    val_sequences: int = 2,
    overwrite: bool = False,
) -> Path:
    """Convert available VisDrone MOT-val frames into a person-only YOLO dataset.

    The assignment permits using Task-4 MOT validation. Since no DET-train split is
    present in this workspace, we build a deterministic train/val split by sequence.
    Both VisDrone person labels, pedestrian and people, are mapped to class 0.
    """

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)

    yaml_path = output_root / "visdrone_person.yaml"
    if yaml_path.exists() and not overwrite:
        return yaml_path

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    sequence_dirs = sorted((dataset_root / "sequences").iterdir())
    val_set = {seq.name for seq in sequence_dirs[-val_sequences:]}

    for seq_dir in tqdm(sequence_dirs, desc="Converting VisDrone MOT to YOLO"):
        split = "val" if seq_dir.name in val_set else "train"
        rows_by_frame: dict[int, list[VisDroneBox]] = {}
        for box in read_annotation_file(dataset_root / "annotations" / f"{seq_dir.name}.txt"):
            if box.category in PERSON_CATEGORIES and box.score != 0:
                rows_by_frame.setdefault(box.frame_id, []).append(box)

        for image_path in sorted(seq_dir.glob("*.jpg")):
            frame_id = int(image_path.stem)
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            out_name = f"{seq_dir.name}_{image_path.name}"
            shutil.copy2(image_path, output_root / "images" / split / out_name)

            label_lines = []
            for box in rows_by_frame.get(frame_id, []):
                cx = (box.x + box.w / 2.0) / width
                cy = (box.y + box.h / 2.0) / height
                label_lines.append(
                    f"0 {cx:.6f} {cy:.6f} {box.w / width:.6f} {box.h / height:.6f}"
                )
            label_path = output_root / "labels" / split / out_name.replace(".jpg", ".txt")
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names: ['person']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def list_sequences(dataset_root: Path) -> list[str]:
    """Return sorted sequence names in the VisDrone MOT dataset."""

    return sorted(p.name for p in (Path(dataset_root) / "sequences").iterdir() if p.is_dir())
