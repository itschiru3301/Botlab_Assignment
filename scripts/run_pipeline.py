#!/usr/bin/env python3
"""Run Aerial Guardian on VisDrone sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aerial_guardian.pipeline import AerialGuardianPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector + tracker pipeline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/b23_chiranjeevi/Botlab/VisDrone2019-MOT-val"))
    parser.add_argument("--weights", type=Path, default=Path("weights/aerial_guardian_best.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--no-adaptive-sahi", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights if args.weights.exists() else Path("yolov8s.pt")
    pipeline = AerialGuardianPipeline(
        weights=weights,
        output_root=args.output_root,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        use_adaptive_sahi=not args.no_adaptive_sahi,
        save_video=not args.no_video,
    )
    summary = pipeline.process_dataset(args.dataset_root, args.sequences)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
