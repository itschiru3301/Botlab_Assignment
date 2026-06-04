"""Shared constants for the Aerial Guardian pipeline."""

from pathlib import Path

PERSON_CATEGORIES = {1, 2}  # VisDrone: pedestrian, people
TINY_AREA_PX = 32 * 32
DEFAULT_DATASET_ROOT = Path("/data/b23_chiranjeevi/Botlab/VisDrone2019-MOT-val")
DEFAULT_OUTPUT_ROOT = Path("outputs")

HIGH_CONF_STANDARD = 0.50
LOW_CONF_STANDARD = 0.15
HIGH_CONF_CROWDED = 0.40
LOW_CONF_CROWDED = 0.10
DENSE_SCENE_TRACKS = 20
IOU_MATCH_THRESHOLD = 0.30
