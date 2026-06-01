from __future__ import annotations

import cv2
import numpy as np


def alpha_between(frame_idx: int, start_frame: int, end_frame: int, invert: bool = False) -> float:
    if end_frame <= start_frame:
        return 0.0 if invert else 1.0
    alpha = (frame_idx - start_frame) / max(1, end_frame - start_frame - 1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 1.0 - alpha if invert else alpha


def crossfade_bgr(a: np.ndarray, b: np.ndarray, alpha_b: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0 - float(alpha_b), b, float(alpha_b), 0.0)
