from __future__ import annotations

import numpy as np
import pyvista as pv

from pose_video_demo.skeleton.joints import BONES_25


def make_floor(points_world: np.ndarray) -> pv.PolyData:
    mins = points_world.min(axis=0)
    maxs = points_world.max(axis=0)
    z = float(mins[2])
    corners = np.array(
        [[mins[0], mins[1], z], [maxs[0], mins[1], z], [maxs[0], maxs[1], z], [mins[0], maxs[1], z]],
        dtype=np.float32,
    )
    return pv.PolyData(corners, faces=np.array([4, 0, 1, 2, 3]))


def make_pitch_lines(points_world: np.ndarray, segments: list[tuple[int, int]]) -> pv.PolyData:
    cells: list[int] = []
    for a, b in segments:
        cells.extend([2, int(a), int(b)])
    mesh = pv.PolyData(points_world)
    mesh.lines = np.asarray(cells, dtype=np.int64)
    return mesh


def make_pitch_boundary_lines(points_world: np.ndarray, z_offset_m: float = 0.02) -> pv.PolyData:
    mins = points_world.min(axis=0)
    maxs = points_world.max(axis=0)
    z = float(mins[2]) + float(z_offset_m)
    corners = np.array(
        [[mins[0], mins[1], z], [maxs[0], mins[1], z], [maxs[0], maxs[1], z], [mins[0], maxs[1], z]],
        dtype=np.float32,
    )
    mesh = pv.PolyData(corners)
    mesh.lines = np.asarray([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0], dtype=np.int64)
    return mesh


def make_skeleton_lines(x_world_frame: np.ndarray, person_indices: list[int]) -> pv.PolyData | None:
    points: list[np.ndarray] = []
    cells: list[int] = []
    for person_idx in person_indices:
        joints = x_world_frame[person_idx]
        for a, b in BONES_25:
            if a >= joints.shape[0] or b >= joints.shape[0]:
                continue
            if not (np.isfinite(joints[a]).all() and np.isfinite(joints[b]).all()):
                continue
            base = len(points)
            points.extend([joints[a], joints[b]])
            cells.extend([2, base, base + 1])
    if not points:
        return None
    mesh = pv.PolyData(np.asarray(points, dtype=np.float32))
    mesh.lines = np.asarray(cells, dtype=np.int64)
    return mesh


def make_joint_points(x_world_frame: np.ndarray, person_indices: list[int]) -> pv.PolyData | None:
    points: list[np.ndarray] = []
    for person_idx in person_indices:
        joints = x_world_frame[person_idx]
        valid = np.isfinite(joints).all(axis=-1)
        points.extend(joints[valid])
    if not points:
        return None
    return pv.PolyData(np.asarray(points, dtype=np.float32))
