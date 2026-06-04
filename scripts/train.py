#!/usr/bin/env python3
"""Fine-tune YOLOv8s for VisDrone person detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aerial_guardian.dataset import convert_mot_to_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Aerial Guardian detector on VisDrone MOT persons.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/b23_chiranjeevi/Botlab/VisDrone2019-MOT-val"))
    parser.add_argument("--yolo-data-root", type=Path, default=Path("data/visdrone_person_yolo"))
    parser.add_argument("--base-model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="aerial_guardian_yolov8s")
    parser.add_argument("--overwrite-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    yaml_path = convert_mot_to_yolo(args.dataset_root, args.yolo_data_root, overwrite=args.overwrite_data)
    model = YOLO(args.base_model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=min(3, args.epochs),
        mosaic=0.5,
        copy_paste=0.1,
        degrees=5.0,
        fliplr=0.5,
        scale=0.5,
        patience=max(args.epochs, 10),
        project=args.project,
        name=args.name,
        exist_ok=True,
        save_period=max(args.epochs, 1),
    )
    candidates = [
        Path(args.project) / args.name / "weights" / "best.pt",
        Path("runs/detect") / args.project / args.name / "weights" / "best.pt",
    ]
    best = next((path for path in candidates if path.exists()), candidates[0])
    if best.exists():
        size_mb = best.stat().st_size / (1024 * 1024)
        print(f"Best model: {best} ({size_mb:.1f} MB)")
        assert size_mb < 300, "Model exceeds 300 MB challenge limit"
        export_dir = Path("weights")
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / "aerial_guardian_best.pt"
        export_path.write_bytes(best.read_bytes())
        print(f"Submission checkpoint copied to: {export_path}")


if __name__ == "__main__":
    main()
