from __future__ import annotations

import numpy as np


def reconstruct_x_cam(xrel_sam: np.ndarray, root_cam: np.ndarray) -> np.ndarray:
    """Reconstruct camera-space joints as X_rel_sam + root_cam."""
    if xrel_sam.ndim != 4 or xrel_sam.shape[2:] != (25, 3):
        raise ValueError(f"Expected xrel_sam shape (N,T,25,3), got {xrel_sam.shape}")
    if root_cam.shape != xrel_sam.shape[:2] + (3,):
        raise ValueError(f"Expected root_cam shape {xrel_sam.shape[:2] + (3,)}, got {root_cam.shape}")
    return xrel_sam + root_cam[:, :, None, :]


def camera_to_world_row_vector(x_cam: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert camera coordinates to world coordinates with X_world=(X_cam-t)@R."""
    if r.shape[:1] != t.shape[:1] or r.shape[1:] != (3, 3) or t.shape[1:] != (3,):
        raise ValueError(f"Invalid camera shapes R={r.shape}, t={t.shape}")
    centered = x_cam - t[None, :, None, :]
    return np.einsum("ntjc,tcd->ntjd", centered, r)


def world_to_camera_row_vector(x_world: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert world coordinates to camera coordinates, inverse of camera_to_world_row_vector."""
    return np.einsum("...d,dc->...c", x_world, r.T) + t
