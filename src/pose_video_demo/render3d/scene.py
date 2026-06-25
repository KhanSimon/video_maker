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


def make_pitch_lines(points_world: np.ndarray, segments: list[tuple[int, int]], z_offset_m: float = 0.02) -> pv.PolyData:
    points = points_world.copy()
    points[:, 2] += float(z_offset_m)
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = np.asarray(_polyline_cells(segments), dtype=np.int64)
    return mesh


def _polyline_cells(segments: list[tuple[int, int]]) -> list[int]:
    cells: list[int] = []
    current: list[int] = []
    for a, b in segments:
        a = int(a)
        b = int(b)
        if current and current[-1] == a:
            current.append(b)
            continue
        if current:
            cells.extend([len(current), *current])
        current = [a, b]
    if current:
        cells.extend([len(current), *current])
    return cells


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
