from __future__ import annotations

import numpy as np


def camera_center_world(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera center in world coordinates for X_world=(X_cam-t)@R."""
    return -t @ r


def camera_forward_world(r: np.ndarray) -> np.ndarray:
    direction = np.array([0.0, 0.0, 1.0]) @ r
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
