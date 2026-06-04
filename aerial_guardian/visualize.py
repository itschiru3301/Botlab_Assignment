"""Drawing helpers for tracked drone video."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Generate a stable BGR color from a track ID."""

    hue = (track_id * 37) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    tails: dict[int, deque[tuple[float, float]]],
    fps: float,
    sahi_enabled: bool,
    complexity: float,
) -> np.ndarray:
    """Draw boxes, IDs, short trajectory tails, and a compact status overlay."""

    output = frame.copy()
    for x1, y1, x2, y2, track_id_float in tracks:
        track_id = int(track_id_float)
        color = color_for_id(track_id)
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            output,
            f"ID {track_id}",
            (int(x1), max(18, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        points = list(tails.get(track_id, []))
        for idx in range(1, len(points)):
            alpha = idx / max(len(points) - 1, 1)
            faded = tuple(int(channel * (0.15 + 0.85 * alpha)) for channel in color)
            cv2.line(output, tuple(map(int, points[idx - 1])), tuple(map(int, points[idx])), faded, 2)

    overlay = output.copy()
    cv2.rectangle(overlay, (8, 8), (390, 82), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.55, output, 0.45, 0)
    cv2.putText(output, f"FPS {fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)
    cv2.putText(
        output,
        f"tracks {len(tracks)}  SAHI {sahi_enabled}  complexity {complexity:.2f}",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 255, 80),
        1,
        cv2.LINE_AA,
    )
    return output
