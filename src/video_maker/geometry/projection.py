from __future__ import annotations

import numpy as np


def project_camera_points(x_cam: np.ndarray, k_matrix: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-space points with radial distortion k1/k2."""
    z = x_cam[..., 2]
    valid = np.isfinite(x_cam).all(axis=-1) & (z > 1e-6)
    safe_z = np.where(valid, z, 1.0)
    x = x_cam[..., 0] / safe_z
    y = x_cam[..., 1] / safe_z
    k1 = distortion[0] if distortion.size > 0 else 0.0
    k2 = distortion[1] if distortion.size > 1 else 0.0
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    u = k_matrix[0, 0] * x * factor + k_matrix[0, 2]
    v = k_matrix[1, 1] * y * factor + k_matrix[1, 2]
    points = np.stack([u, v], axis=-1)
    valid &= np.isfinite(points).all(axis=-1)
    return points, valid


def project_world_points(
    points_world: np.ndarray,
    k_matrix: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_cam = points_world @ r.T + t
    return project_camera_points(x_cam, k_matrix, distortion)
