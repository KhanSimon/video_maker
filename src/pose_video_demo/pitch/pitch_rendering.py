from __future__ import annotations

import cv2
import numpy as np

from pose_video_demo.geometry.projection import project_world_points


def draw_projected_pitch_lines(
    frame: np.ndarray,
    points_world: np.ndarray,
    segments: list[tuple[int, int]],
    k_matrix: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    distortion: np.ndarray,
    color_bgr: tuple[int, int, int],
    line_width: int,
) -> int:
    points_uv, valid = project_world_points(points_world, k_matrix, r, t, distortion)
    count = 0
    h, w = frame.shape[:2]
    for a, b in segments:
        if not (valid[a] and valid[b]):
            continue
        pa = tuple(np.round(points_uv[a]).astype(int))
        pb = tuple(np.round(points_uv[b]).astype(int))
        if _near_image(pa, w, h) or _near_image(pb, w, h):
            cv2.line(frame, pa, pb, color_bgr, line_width, cv2.LINE_AA)
            count += 1
    return count


def draw_projected_floor(
    frame: np.ndarray,
    points_world: np.ndarray,
    k_matrix: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    distortion: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    mins = points_world.min(axis=0)
    maxs = points_world.max(axis=0)
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
        ],
        dtype=np.float32,
    )
    uv, valid = project_world_points(corners, k_matrix, r, t, distortion)
    if valid.sum() < 3:
        return
    polygon = np.round(uv[valid]).astype(np.int32)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, polygon, color_bgr)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0, frame)


def _near_image(point: tuple[int, int], width: int, height: int, margin: int = 160) -> bool:
    x, y = point
    return -margin <= x < width + margin and -margin <= y < height + margin
