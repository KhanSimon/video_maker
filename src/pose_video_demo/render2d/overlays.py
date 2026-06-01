from __future__ import annotations

import cv2
import numpy as np

from pose_video_demo.geometry.projection import project_camera_points
from pose_video_demo.skeleton.drawing import draw_projected_skeleton
from pose_video_demo.utils.colors import as_bgr


def draw_boxes(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    teams: dict[int, str],
    colors: dict[str, tuple[int, int, int]],
    line_width: int,
    alpha: float = 1.0,
) -> None:
    target = frame if alpha >= 0.999 else frame.copy()
    for person_idx, box in enumerate(boxes_xyxy):
        if not np.isfinite(box).all() or (box[2] <= box[0]) or (box[3] <= box[1]):
            continue
        color = as_bgr(colors.get(teams.get(person_idx, "fallback"), colors["fallback"]))
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(target, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
    if alpha < 0.999:
        cv2.addWeighted(target, float(alpha), frame, 1.0 - float(alpha), 0.0, frame)


def draw_stickmen_projected(
    frame: np.ndarray,
    x_cam_frame: np.ndarray,
    k_matrix: np.ndarray,
    distortion: np.ndarray,
    teams: dict[int, str],
    colors: dict[str, tuple[int, int, int]],
    line_width: int,
    joint_diameter: int,
) -> int:
    projected, valid = project_camera_points(x_cam_frame, k_matrix, distortion)
    rendered = 0
    for person_idx in range(x_cam_frame.shape[0]):
        person_valid = valid[person_idx]
        if person_valid.sum() < 2:
            continue
        color = as_bgr(colors.get(teams.get(person_idx, "fallback"), colors["fallback"]))
        draw_projected_skeleton(
            frame,
            projected[person_idx],
            person_valid,
            color,
            line_width=line_width,
            joint_diameter=joint_diameter,
        )
        rendered += 1
    return rendered
