# Aerial Guardian

Lightweight drone-based person detection and multi-object tracking for the VisDrone2019 Task-4 MOT validation set.

This repo is a code-first submission: no notebooks are required. It includes a detector training script, an end-to-end tracking pipeline, output video generation with ID labels and trajectory tails, FPS reporting, and MOT evaluation.

## Quick Start

The machine used for this run already has a conda environment named `torch_env`.

```bash
cd /data/b23_chiranjeevi/aerial-guardian-submit
PYTHONNOUSERSITE=1 /data/b23_chiranjeevi/miniconda3/envs/torch_env/bin/python -m pip install -r requirements.txt
```

Train on GPU 0:

```bash
PYTHONNOUSERSITE=1 /data/b23_chiranjeevi/miniconda3/envs/torch_env/bin/python scripts/train.py \
  --dataset-root /data/b23_chiranjeevi/Botlab/VisDrone2019-MOT-val \
  --device 0 \
  --epochs 3 \
  --imgsz 960 \
  --batch 8
```

Run the tracker and save video/tracks:

```bash
PYTHONNOUSERSITE=1 /data/b23_chiranjeevi/miniconda3/envs/torch_env/bin/python scripts/run_pipeline.py \
  --dataset-root /data/b23_chiranjeevi/Botlab/VisDrone2019-MOT-val \
  --weights weights/aerial_guardian_best.pt \
  --device 0 \
  --sequences uav0000086_00000_v
```

Evaluate MOT metrics:

```bash
PYTHONNOUSERSITE=1 /data/b23_chiranjeevi/miniconda3/envs/torch_env/bin/python scripts/evaluate.py
```

## Dataset

Expected layout:

```text
/data/b23_chiranjeevi/Botlab/VisDrone2019-MOT-val/
  annotations/<sequence>.txt
  sequences/<sequence>/*.jpg
```

The target classes are persons: VisDrone categories `1` pedestrian and `2` people. Both are mapped to YOLO class `0`.

## Architecture

### Detector

Base model: `YOLOv8s`.

Why YOLOv8s:

- Small enough for drone/edge deployment, around 22 MB for weights.
- Better tiny-object speed/accuracy trade-off than larger YOLO variants.
- Anchor-free head avoids hand-tuned anchor assumptions for aerial person scale.
- C2f blocks and SPPF keep useful multi-scale features without a heavy transformer backbone.

Drone-specific adaptation added here:

- Person-only fine-tuning on VisDrone MOT frames converted to YOLO format.
- High-resolution training/inference (`imgsz=960` by default; 1280 can be used if latency budget allows).
- Reduced mosaic (`0.5`) because VisDrone scenes are already dense.
- Copy-paste augmentation (`0.1`) to simulate partial occlusions.
- Adaptive SAHI at inference: tiled detection is activated only when recent frames contain many detections or many tiny detections.

The model remains far below the challenge limit of 300 MB.

### Tracking

The tracker is a ByteTrack-style two-stage association system:

1. Match high-confidence detections to existing tracks using IoU.
2. Match remaining tracks to low-confidence detections to recover partially occluded persons.

Changes for drone footage:

- Constant-acceleration Kalman filter (CA-KF), with state `[cx, cy, w, h, vx, vy, vw, vh, ax, ay]`.
- Sparse Lucas-Kanade camera motion compensation (CMC).
- RANSAC affine estimation rejects moving-person keypoints and keeps background motion.
- Adaptive association thresholds in dense scenes.

This avoids the extra model size and latency of DeepSORT/ReID networks, which is important for edge deployment.

## Small Object Detection Strategy

Small aerial persons often occupy less than `32x32` pixels. The code handles this by:

- Training at higher image size so tiny boxes occupy more feature cells.
- Using person-only labels to remove irrelevant class competition.
- Activating SAHI only for hard frames. SAHI slices the frame into overlapping crops, detects at crop scale, then merges duplicate boxes. This improves tiny-person recall but costs extra forward passes, so the controller uses it selectively.

## ID Switch Mitigation

Drone ego-motion can shift every object between frames, causing IoU matching to fail even when the person did not move much. The pipeline estimates global camera motion between consecutive frames using sparse optical flow. RANSAC keeps the dominant background transform and rejects inconsistent moving-object tracks. Kalman predictions are warped by this transform before association.

Occlusion is handled by ByteTrack's second association stage: low-confidence boxes are not discarded immediately; they can recover existing tracks.

## Edge Hardware Adaptation

Recommended deployment path:

```python
from ultralytics import YOLO
YOLO("weights/aerial_guardian_best.pt").export(
    format="engine",
    imgsz=960,
    half=True,
    device=0,
)
```

Use FP16 TensorRT for Jetson by default. INT8 can be used with a representative calibration set, but tiny persons are sensitive to quantization noise. For multi-stream drone feeds, batch detector inference on GPU and run one CA-KF tracker per stream on CPU.

## Hardware Used For Test

The available test GPU is:

```text
NVIDIA RTX A6000, 48 GB VRAM, CUDA 12.4 driver stack
```

## Actual Run Results

Training was run on GPU 0 for 3 lightweight epochs at `imgsz=960`, `batch=8`.

Detector validation after the short fine-tune:

| Precision | Recall | mAP50 | mAP50-95 | Model Size |
|---:|---:|---:|---:|---:|
| 0.548 | 0.373 | 0.385 | 0.152 | 22.5 MB |

Pipeline run on `uav0000086_00000_v`:

| Frames | Avg FPS | Avg Latency | Adaptive SAHI Frames |
|---:|---:|---:|---:|
| 464 | 14.43 | 69.29 ms | 351 |

MOT evaluation on the generated track file:

| MOTA | IDF1 | ID Switches | Precision | Recall |
|---:|---:|---:|---:|---:|
| 82.55% | 69.57% | 217 | 93.67% | 89.59% |

Generated deliverables:

```text
weights/aerial_guardian_best.pt
outputs/videos/uav0000086_00000_v.mp4
outputs/tracks/uav0000086_00000_v.txt
outputs/metrics/pipeline_fps.csv
outputs/metrics/mot_metrics.csv
```

FPS is written to:

```text
outputs/metrics/pipeline_fps.csv
```

MOT metrics are written to:

```text
outputs/metrics/mot_metrics.csv
```

Example output video path:

```text
outputs/videos/uav0000086_00000_v.mp4
```

## Repository Structure

```text
aerial_guardian/
  dataset.py      # VisDrone parsing and YOLO conversion
  detector.py     # YOLO + adaptive SAHI controller
  tracker.py      # CA-KF ByteTrack + sparse LK CMC
  pipeline.py     # End-to-end processing and FPS logging
  evaluate.py     # MOT metrics
  visualize.py    # Boxes, IDs, trajectory tails
scripts/
  train.py
  run_pipeline.py
  evaluate.py
requirements.txt
```

## Notes

This project intentionally uses open-source components where sensible, but the drone-specific additions are the fine-tuning recipe, adaptive SAHI controller, CA-KF tracker, CMC warp, adaptive thresholds, and trajectory visualization.
