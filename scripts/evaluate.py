#!/usr/bin/env python3
"""Evaluate MOT outputs against VisDrone ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aerial_guardian.evaluate import evaluate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Aerial Guardian MOT tracks.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/b23_chiranjeevi/Botlab/VisDrone2019-MOT-val"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("outputs/tracks"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/mot_metrics.csv"))
    return parser.parse_args()


def main() -> None:
    summary = evaluate_outputs(parse_args().dataset_root, parse_args().tracks_dir, parse_args().output_csv)
    print(summary.to_string())


if __name__ == "__main__":
    main()
