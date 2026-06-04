# Aerial Guardian

Lightweight person detection and multi-object tracking for drone video, built around the VisDrone2019 Task 4 MOT validation set.

The pipeline combines a small YOLO detector, selective tiled inference for tiny people, and a motion-only tracker adapted for moving drone cameras. The goal is not just high accuracy, but a practical speed/precision balance that can be moved toward edge hardware such as NVIDIA Jetson.

## Highlights

- Person-only YOLOv8s fine-tuning on VisDrone MOT frames converted to YOLO format
- Adaptive SAHI: tiled inference is enabled only for dense or tiny-object-heavy frames
- ByteTrack-style two-stage association for occlusion recovery
- Constant-acceleration Kalman filtering for drone motion dynamics
- Sparse Lucas-Kanade camera motion compensation with RANSAC
- Output videos with bounding boxes, track IDs, and short trajectory tails
- FPS logging and MOT metrics with `motmetrics`

## Results

Measured on sequence `uav0000086_00000_v` from VisDrone2019-MOT-val.

Hardware used: NVIDIA RTX A6000, 48 GB VRAM.

### Detector

The detector was fine-tuned for 3 lightweight epochs at `imgsz=960`, `batch=8`.

| Model | Precision | Recall | mAP50 | mAP50-95 | Size |
|---|---:|---:|---:|---:|---:|
| YOLOv8s person fine-tune | 0.548 | 0.373 | 0.385 | 0.152 | 22.5 MB |

### End-To-End Pipeline

| Frames | Avg FPS | Avg Latency | Adaptive SAHI Frames |
|---:|---:|---:|---:|
| 464 | 14.43 | 69.29 ms | 351 |

### MOT Metrics

| MOTA | IDF1 | ID Switches | Precision | Recall |
|---:|---:|---:|---:|---:|
| 82.55% | 69.57% | 217 | 93.67% | 89.59% |

Generated artifacts from this run:

```text
weights/aerial_guardian_best.pt
outputs/videos/uav0000086_00000_v.mp4
outputs/tracks/uav0000086_00000_v.txt
outputs/metrics/pipeline_fps.csv
outputs/metrics/mot_metrics.csv
```

## Demo Output

The generated video is saved at:

```text
outputs/videos/uav0000086_00000_v.mp4
```

It contains bounding boxes, unique track IDs, and fading trajectory tails for active tracks.

## Installation

Create or activate a Python environment with CUDA-enabled PyTorch, then install the project dependencies.

```bash
git clone <repo-url>
cd aerial-guardian-submit
python -m pip install -r requirements.txt
```

On shared machines, `PYTHONNOUSERSITE=1 python ...` can be useful if user-site packages shadow the active environment.

## Dataset

Download the VisDrone2019 Task 4 MOT validation set from the official VisDrone resources:

- VisDrone GitHub: https://github.com/VisDrone/VisDrone-Dataset
- Official dataset downloads are linked from the VisDrone repository.

Expected directory layout:

```text
VisDrone2019-MOT-val/
  annotations/
    <sequence>.txt
  sequences/
    <sequence>/
      0000001.jpg
      ...
```

The target labels are person-like VisDrone categories:

- `1`: pedestrian
- `2`: people

Both are mapped to a single YOLO class: `person`.

## Usage

Set the dataset path once:

```bash
export VISDRONE_ROOT=/path/to/VisDrone2019-MOT-val
```

### Train

The training script converts VisDrone MOT annotations into a person-only YOLO dataset, then fine-tunes YOLOv8s.

```bash
python scripts/train.py \
  --dataset-root "$VISDRONE_ROOT" \
  --device 0 \
  --epochs 3 \
  --imgsz 960 \
  --batch 8
```

The trained checkpoint is copied to:

```text
weights/aerial_guardian_best.pt
```

### Run Tracking

Run on one sequence:

```bash
python scripts/run_pipeline.py \
  --dataset-root "$VISDRONE_ROOT" \
  --weights weights/aerial_guardian_best.pt \
  --device 0 \
  --sequences uav0000086_00000_v
```

Run on every sequence:

```bash
python scripts/run_pipeline.py \
  --dataset-root "$VISDRONE_ROOT" \
  --weights weights/aerial_guardian_best.pt \
  --device 0
```

Outputs are written to:

```text
outputs/videos/
outputs/tracks/
outputs/metrics/pipeline_fps.csv
```

### Evaluate

```bash
python scripts/evaluate.py \
  --dataset-root "$VISDRONE_ROOT" \
  --tracks-dir outputs/tracks \
  --output-csv outputs/metrics/mot_metrics.csv
```

## Architecture

### Detector

Base detector: YOLOv8s.

YOLOv8s was selected because it is small enough for drone/edge deployment while still providing stronger tiny-object recall than the nano class of models. It uses an anchor-free detection head, C2f blocks, and SPPF context aggregation, all of which are useful without adding transformer-scale latency.

Drone-specific adaptation:

- Person-only fine-tuning on VisDrone frames
- High-resolution training and inference at `imgsz=960`
- Reduced mosaic augmentation because VisDrone is already visually dense
- Copy-paste augmentation to simulate partial person occlusion
- Adaptive SAHI for hard frames where tiny-person recall matters more than raw FPS

### Adaptive SAHI

SAHI improves small-object recall by slicing a full-resolution frame into overlapping tiles, running detection on each tile, and merging duplicate boxes. The downside is latency: every tile is another detector pass.

This project uses a simple controller over the recent detection stream:

```text
complexity = 0.5 * normalized_detection_density
           + 0.5 * tiny_detection_fraction
```

SAHI is enabled when the score is high, disabled when it is low, and held steady in a hysteresis band. This preserves speed on easy close-range frames and spends compute on high-altitude or crowded frames.

### Tracker

The tracker is ByteTrack-style, but adapted for drone motion.

ByteTrack keeps low-confidence detections for a second matching stage. That matters for aerial footage because people are often tiny, blurred, or partially occluded.

Drone-specific tracking additions:

- Constant-acceleration Kalman filter with state `[cx, cy, w, h, vx, vy, vw, vh, ax, ay]`
- Sparse Lucas-Kanade optical flow for camera motion compensation
- RANSAC affine estimation to reject moving-person keypoints and keep background motion
- Adaptive confidence thresholds for dense scenes

This keeps the tracker lightweight and avoids a ReID network, which would add model size and latency and may not transfer well from ground-level pedestrian data to aerial views.

## Edge Deployment

The trained model is far below the 300 MB constraint. The included checkpoint is about 22.5 MB.

Recommended Jetson path:

```python
from ultralytics import YOLO

YOLO("weights/aerial_guardian_best.pt").export(
    format="engine",
    imgsz=960,
    half=True,
    device=0,
)
```

Deployment notes:

- Use FP16 TensorRT as the default Jetson target.
- Use INT8 only with a representative VisDrone calibration set; tiny objects are sensitive to quantization error.
- For multiple drone feeds, batch detector inference on GPU and keep one CPU tracker per stream.
- Disable adaptive SAHI or raise its threshold when latency is the hard constraint.

## Repository Structure

```text
aerial_guardian/
  config.py       # Shared constants
  dataset.py      # VisDrone parsing and YOLO conversion
  detector.py     # YOLO wrapper and adaptive SAHI controller
  tracker.py      # CA-KF ByteTrack and camera motion compensation
  pipeline.py     # End-to-end processing and FPS logging
  evaluate.py     # MOT metrics
  visualize.py    # Video overlays
scripts/
  train.py
  run_pipeline.py
  evaluate.py
configs/
  default.yaml
outputs/
  metrics/
  tracks/
  videos/
weights/
  aerial_guardian_best.pt
```

## Design Trade-Offs

- YOLOv8s instead of a larger detector: lower latency and smaller deployment footprint.
- Motion-only tracking instead of DeepSORT: avoids a ReID network and keeps the tracker CPU-friendly.
- Adaptive SAHI instead of always-on SAHI: recovers tiny people when needed without paying the tiled-inference cost on every frame.
- Short fine-tune instead of a large training run: demonstrates dataset adaptation while respecting the lightweight-compute requirement.

## Limitations

- The included benchmark is reported on one validation sequence. Running all sequences will give a more representative average.
- The short 3-epoch fine-tune is intentionally lightweight; longer training can improve detector recall.
- Adaptive SAHI improves recall but reduces FPS when enabled for many frames.
