from __future__ import annotations

import cv2
import numpy as np

from video_maker.skeleton.joints import BONES_25


def draw_projected_skeleton(
    frame: np.ndarray,
    joints_uv: np.ndarray,
    valid: np.ndarray,
    color_bgr: tuple[int, int, int],
    line_width: int = 2,
    joint_diameter: int = 4,
    bones: list[tuple[int, int]] | None = None,
) -> None:
    h, w = frame.shape[:2]
    bones = bones or BONES_25
    for a, b in bones:
        if a >= joints_uv.shape[0] or b >= joints_uv.shape[0] or not (valid[a] and valid[b]):
            continue
        pa = tuple(np.round(joints_uv[a]).astype(int))
        pb = tuple(np.round(joints_uv[b]).astype(int))
        if _near_image(pa, w, h) or _near_image(pb, w, h):
            cv2.line(frame, pa, pb, color_bgr, line_width, cv2.LINE_AA)
    radius = max(1, int(round(joint_diameter / 2)))
    for point, ok in zip(joints_uv, valid, strict=False):
        if not ok:
            continue
        p = tuple(np.round(point).astype(int))
        if _near_image(p, w, h):
            cv2.circle(frame, p, radius, color_bgr, -1, cv2.LINE_AA)


def _near_image(point: tuple[int, int], width: int, height: int, margin: int = 80) -> bool:
    x, y = point
    return -margin <= x < width + margin and -margin <= y < height + margin
