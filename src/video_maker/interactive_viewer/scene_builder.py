from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from video_maker.pitch.pitch_loader import infer_pitch_segments, load_pitch_points
from video_maker.utils.colors import color_table


@dataclass(frozen=True)
class BuiltScene:
    floor: o3d.geometry.TriangleMesh
    pitch_lines: o3d.geometry.LineSet
    axes: o3d.geometry.TriangleMesh
    pitch_center: np.ndarray
    pitch_extent: float
    bounds_min: np.ndarray
    bounds_max: np.ndarray


def build_scene(
    pitch_points_path,
    floor_color: tuple[float, float, float] | None = None,
    pitch_line_color: tuple[float, float, float] | None = None,
    max_segment_length: float = 8.0,
    pitch_line_z_offset: float = 0.05,
) -> BuiltScene:
    colors = color_table()
    points = load_pitch_points(pitch_points_path).astype(np.float64)
    segments = infer_pitch_segments(points, max_segment_length=max_segment_length)
    floor = build_floor(points, floor_color or _rgb01(colors["floor_green"]))
    line_points = points.copy()
    line_points[:, 2] += float(pitch_line_z_offset)
    pitch_lines = build_pitch_lines(line_points, segments, pitch_line_color or _rgb01(colors["pitch_lines"]))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0.0, 0.0, 0.0])
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    return BuiltScene(
        floor=floor,
        pitch_lines=pitch_lines,
        axes=axes,
        pitch_center=center,
        pitch_extent=extent,
        bounds_min=points.min(axis=0),
        bounds_max=points.max(axis=0),
    )


def build_floor(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    z = float(mins[2] - 0.01)
    vertices = np.array(
        [
            [mins[0], mins[1], z],
            [maxs[0], mins[1], z],
            [maxs[0], maxs[1], z],
            [mins[0], maxs[1], z],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def build_pitch_lines(
    points: np.ndarray,
    segments: list[tuple[int, int]],
    color: tuple[float, float, float],
) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(np.asarray(segments, dtype=np.int32)),
    )
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(segments), 1)))
    return line_set


def _rgb01(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(float(v) / 255.0 for v in rgb)
