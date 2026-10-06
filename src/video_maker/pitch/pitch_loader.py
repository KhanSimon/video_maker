from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def load_pitch_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected pitch points shape (M,3), got {points.shape}")
    return points


def infer_pitch_segments(points: np.ndarray, max_segment_length: float = 8.0) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for idx in range(len(points) - 1):
        distance = float(np.linalg.norm(points[idx + 1] - points[idx]))
        if 1e-6 < distance <= max_segment_length:
            segments.append((idx, idx + 1))
    return segments


def parse_segments(raw: Iterable[Iterable[int]] | None, points: np.ndarray, max_segment_length: float) -> list[tuple[int, int]]:
    if raw:
        segments = [(int(a), int(b)) for a, b in raw]
        max_index = points.shape[0] - 1
        for a, b in segments:
            if not (0 <= a <= max_index and 0 <= b <= max_index):
                raise ValueError(f"Pitch segment {(a, b)} is outside 0..{max_index}")
        return segments
    return infer_pitch_segments(points, max_segment_length=max_segment_length)
