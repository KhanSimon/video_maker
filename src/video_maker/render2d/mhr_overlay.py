from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pose_video_demo_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pose_video_demo_cache")

import pyvista as pv
from vtkmodules.vtkCommonMath import vtkMatrix4x4

from video_maker.io.mhr import MHRData
from video_maker.render3d.mhr_mesh import add_mhr_players


def camera_projection_matrix(k_matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """OpenCV intrinsics to OpenGL projection (view axes X, -Y, -Z)."""
    near, far = 0.01, 10000.0
    return np.array([
        [2 * k_matrix[0, 0] / width, -2 * k_matrix[0, 1] / width, 1 - 2 * k_matrix[0, 2] / width, 0],
        [0, 2 * k_matrix[1, 1] / height, 2 * k_matrix[1, 2] / height - 1, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ], dtype=np.float64)


class MHROverlay:
    """Render a depth-tested transparent surface in the calibrated video camera."""

    def __init__(self, data: MHRData, colors: dict, options: dict[str, Any]) -> None:
        self.data = data
        self.colors = colors
        self.options = options
        self.plotter: pv.Plotter | None = None
        self._size: tuple[int, int] | None = None
        self._map_key: tuple | None = None
        self._maps: tuple[np.ndarray, np.ndarray] | None = None

    def draw(self, frame: np.ndarray, frame_idx: int, k_matrix: np.ndarray, distortion: np.ndarray) -> int:
        height, width = frame.shape[:2]
        if self.plotter is None or self._size != (width, height):
            self.close()
            self.plotter = pv.Plotter(off_screen=True, window_size=(width, height))
            self.plotter.set_background("black")
            self._size = (width, height)
        plotter = self.plotter
        plotter.clear()
        count = add_mhr_players(
            plotter, self.data.camera_vertices(frame_idx), self.data.faces,
            self.data.sequence.teams, self.colors, bool(self.options.get("smooth_shading", True)),
        )
        if not count:
            return 0
        plotter.camera_position = [(0, 0, 0), (0, 0, 1), (0, -1, 0)]
        plotter.camera.clipping_range = (0.01, 10000.0)
        projection = camera_projection_matrix(k_matrix, width, height)
        matrix = vtkMatrix4x4()
        matrix.DeepCopy(projection.ravel())
        plotter.camera.SetExplicitProjectionTransformMatrix(matrix)
        plotter.camera.SetUseExplicitProjectionTransformMatrix(True)
        rgba = plotter.screenshot(return_img=True, transparent_background=True)
        # OpenGL uses a pinhole camera; warp its color and alpha together to match
        # the same radial k1/k2 distortion used by the pitch and joint overlays.
        if np.any(distortion):
            key = (width, height, k_matrix.tobytes(), distortion.tobytes())
            if key != self._map_key:
                yy, xx = np.indices((height, width), dtype=np.float32)
                pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
                coefficients = np.zeros(5, dtype=np.float64)
                coefficients[:2] = distortion[:2]
                undistorted = cv2.undistortPoints(pixels, k_matrix, coefficients, P=k_matrix).reshape(height, width, 2)
                self._maps = (undistorted[..., 0].copy(), undistorted[..., 1].copy())
                self._map_key = key
            rgba = cv2.remap(rgba, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        frame[:] = np.rint(rgba[..., 2::-1] * alpha + frame * (1 - alpha)).astype(np.uint8)
        return count if np.any(alpha) else 0

    def close(self) -> None:
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None
        self._maps = None
        self._map_key = None
