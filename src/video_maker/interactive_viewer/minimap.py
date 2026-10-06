from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d

from video_maker.interactive_viewer.scene_builder import BuiltScene
from video_maker.interactive_viewer.skeleton_updater import SkeletonLayer


@dataclass(frozen=True)
class MinimapLayout:
    panel_left: float
    panel_top: float
    panel_right: float
    panel_bottom: float
    field_left: float
    field_top: float
    field_right: float
    field_bottom: float


@dataclass
class MinimapOverlay:
    source: SkeletonLayer
    pitch_points_xy: np.ndarray
    pitch_segments: np.ndarray
    pitch_bounds: np.ndarray
    panel: o3d.geometry.TriangleMesh
    pitch_lines: o3d.geometry.TriangleMesh
    markers: o3d.geometry.TriangleMesh
    width_px: int
    height_px: int
    margin_px: int
    padding_px: int
    marker_radius_px: float
    marker_circle_segments: int
    pitch_line_width_px: float
    depth_m: float
    panel_color: np.ndarray
    pitch_line_color: np.ndarray
    position_joint_index: int
    clip_positions: bool
    last_marker_count: int = 0
    last_frame_idx: int | None = None
    last_source_identity: int | None = None
    last_camera_signature: np.ndarray | None = None

    @property
    def geometries(self) -> list[o3d.geometry.Geometry]:
        return [self.panel, self.pitch_lines, self.markers]

    def update(
        self,
        frame_idx: int,
        camera_parameters: o3d.camera.PinholeCameraParameters,
    ) -> bool:
        camera_signature = _camera_signature(camera_parameters)
        if (
            self.last_frame_idx == frame_idx
            and self.last_source_identity == id(self.source)
            and self.last_camera_signature is not None
            and np.array_equal(self.last_camera_signature, camera_signature)
        ):
            return False
        intrinsic = camera_parameters.intrinsic
        layout = self._layout(int(intrinsic.width), int(intrinsic.height))
        panel_pixels = np.array(
            [
                [layout.panel_left, layout.panel_top],
                [layout.panel_right, layout.panel_top],
                [layout.panel_right, layout.panel_bottom],
                [layout.panel_left, layout.panel_bottom],
            ],
            dtype=np.float64,
        )
        self.panel.vertices = o3d.utility.Vector3dVector(
            _pixels_to_world(panel_pixels, self.depth_m, camera_parameters)
        )
        self.panel.compute_vertex_normals()

        pitch_pixels = self._pitch_to_pixels(self.pitch_points_xy, layout)
        self._update_pitch_lines(pitch_pixels, camera_parameters)

        marker_pixels: list[np.ndarray] = []
        marker_colors: list[np.ndarray] = []
        for player_idx in range(self.source.num_players):
            joints = self.source.visible_joints(player_idx, frame_idx)
            if joints is None:
                continue
            marker_pixels.append(
                self._pitch_to_pixels(joints[self.position_joint_index, :2][None, :], layout)[0]
            )
            marker_colors.append(self.source.player_colors[player_idx])
        self._update_markers(marker_pixels, marker_colors, camera_parameters, layout)
        self.last_marker_count = len(marker_pixels)
        self.last_frame_idx = int(frame_idx)
        self.last_source_identity = id(self.source)
        self.last_camera_signature = camera_signature
        return True

    def _update_pitch_lines(
        self,
        pitch_pixels: np.ndarray,
        camera_parameters: o3d.camera.PinholeCameraParameters,
    ) -> None:
        starts = pitch_pixels[self.pitch_segments[:, 0]]
        ends = pitch_pixels[self.pitch_segments[:, 1]]
        directions = ends - starts
        lengths = np.linalg.norm(directions, axis=1)
        valid = lengths > 1e-9
        starts = starts[valid]
        ends = ends[valid]
        directions = directions[valid]
        lengths = lengths[valid]
        perpendicular = np.column_stack([-directions[:, 1], directions[:, 0]])
        perpendicular *= (self.pitch_line_width_px * 0.5 / lengths)[:, None]
        pixel_vertices = np.stack(
            [
                starts + perpendicular,
                starts - perpendicular,
                ends - perpendicular,
                ends + perpendicular,
            ],
            axis=1,
        ).reshape(-1, 2)
        segment_count = len(starts)
        base = np.arange(segment_count, dtype=np.int32) * 4
        triangles = np.column_stack(
            [
                np.concatenate([base, base]),
                np.concatenate([base + 2, base + 3]),
                np.concatenate([base + 1, base + 2]),
            ]
        )
        vertices = _pixels_to_world(pixel_vertices, self.depth_m * 0.98, camera_parameters)
        self.pitch_lines.vertices = o3d.utility.Vector3dVector(vertices)
        self.pitch_lines.triangles = o3d.utility.Vector3iVector(triangles)
        self.pitch_lines.vertex_colors = o3d.utility.Vector3dVector(
            np.tile(self.pitch_line_color, (len(vertices), 1))
        )
        self.pitch_lines.compute_vertex_normals()

    def _layout(self, viewport_width: int, viewport_height: int) -> MinimapLayout:
        available_width = max(80, viewport_width - 2 * self.margin_px)
        available_height = max(60, viewport_height - 2 * self.margin_px)
        panel_width = float(min(self.width_px, available_width))
        panel_height = float(min(self.height_px, available_height))
        panel_right = float(viewport_width - self.margin_px)
        panel_left = panel_right - panel_width
        panel_top = float(self.margin_px)
        panel_bottom = panel_top + panel_height

        inner_width = max(1.0, panel_width - 2.0 * self.padding_px)
        inner_height = max(1.0, panel_height - 2.0 * self.padding_px)
        extent = self.pitch_bounds[1] - self.pitch_bounds[0]
        scale = min(inner_width / extent[0], inner_height / extent[1])
        field_width = extent[0] * scale
        field_height = extent[1] * scale
        field_left = panel_left + (panel_width - field_width) / 2.0
        field_top = panel_top + (panel_height - field_height) / 2.0
        return MinimapLayout(
            panel_left=panel_left,
            panel_top=panel_top,
            panel_right=panel_right,
            panel_bottom=panel_bottom,
            field_left=field_left,
            field_top=field_top,
            field_right=field_left + field_width,
            field_bottom=field_top + field_height,
        )

    def _pitch_to_pixels(self, xy: np.ndarray, layout: MinimapLayout) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        minimum, maximum = self.pitch_bounds
        normalized = (xy - minimum) / (maximum - minimum)
        if self.clip_positions:
            normalized = np.clip(normalized, 0.0, 1.0)
        u = layout.field_left + normalized[:, 0] * (layout.field_right - layout.field_left)
        v = layout.field_bottom - normalized[:, 1] * (layout.field_bottom - layout.field_top)
        return np.column_stack([u, v])

    def _update_markers(
        self,
        marker_pixels: list[np.ndarray],
        marker_colors: list[np.ndarray],
        camera_parameters: o3d.camera.PinholeCameraParameters,
        layout: MinimapLayout,
    ) -> None:
        if not marker_pixels:
            hidden_pixel = np.array([[layout.panel_left, layout.panel_top]] * 3, dtype=np.float64)
            vertices = _pixels_to_world(hidden_pixel, self.depth_m * 0.96, camera_parameters)
            triangles = np.array([[0, 2, 1]], dtype=np.int32)
            colors = np.tile(self.panel_color, (3, 1))
        else:
            pixel_vertices: list[np.ndarray] = []
            triangle_parts: list[np.ndarray] = []
            color_parts: list[np.ndarray] = []
            angles = np.linspace(0.0, 2.0 * np.pi, self.marker_circle_segments, endpoint=False)
            unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
            for marker_idx, (center, color) in enumerate(zip(marker_pixels, marker_colors)):
                vertex_start = marker_idx * (self.marker_circle_segments + 1)
                circle = center + self.marker_radius_px * unit_circle
                pixel_vertices.append(np.vstack([center, circle]))
                triangle_parts.append(
                    np.array(
                        [
                            [
                                vertex_start,
                                vertex_start + 1 + (segment_idx + 1) % self.marker_circle_segments,
                                vertex_start + 1 + segment_idx,
                            ]
                            for segment_idx in range(self.marker_circle_segments)
                        ],
                        dtype=np.int32,
                    )
                )
                color_parts.append(np.tile(color, (self.marker_circle_segments + 1, 1)))
            pixel_array = np.concatenate(pixel_vertices, axis=0)
            vertices = _pixels_to_world(pixel_array, self.depth_m * 0.96, camera_parameters)
            triangles = np.concatenate(triangle_parts, axis=0)
            colors = np.concatenate(color_parts, axis=0)

        self.markers.vertices = o3d.utility.Vector3dVector(vertices)
        self.markers.triangles = o3d.utility.Vector3iVector(triangles)
        self.markers.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.markers.compute_vertex_normals()


def build_minimap_overlay(
    source: SkeletonLayer,
    scene: BuiltScene,
    config: dict[str, Any],
) -> MinimapOverlay:
    pitch_points = np.asarray(scene.pitch_lines.points, dtype=np.float64)
    pitch_segments = np.asarray(scene.pitch_lines.lines, dtype=np.int32)
    if pitch_points.ndim != 2 or pitch_points.shape[1] != 3 or not len(pitch_points):
        raise ValueError("Cannot build minimap without pitch points")
    if pitch_segments.ndim != 2 or pitch_segments.shape[1] != 2 or not len(pitch_segments):
        raise ValueError("Cannot build minimap without pitch segments")
    bounds = np.stack([pitch_points[:, :2].min(axis=0), pitch_points[:, :2].max(axis=0)])
    if np.any(bounds[1] - bounds[0] <= 1e-9):
        raise ValueError(f"Invalid minimap pitch bounds: {bounds.tolist()}")

    width_px = _positive_int(config.get("width_px", 320), "width_px")
    height_px = _positive_int(config.get("height_px", 215), "height_px")
    margin_px = _non_negative_int(config.get("margin_px", 24), "margin_px")
    padding_px = _non_negative_int(config.get("padding_px", 12), "padding_px")
    marker_radius_px = _positive_float(config.get("marker_radius_px", 5.5), "marker_radius_px")
    pitch_line_width_px = _positive_float(
        config.get("pitch_line_width_px", 1.25),
        "pitch_line_width_px",
    )
    marker_circle_segments = max(
        6,
        _positive_int(config.get("marker_circle_segments", 12), "marker_circle_segments"),
    )
    depth_m = _positive_float(config.get("depth_m", 2.0), "depth_m")
    position_joint_index = int(config.get("position_joint_index", 8))
    if not 0 <= position_joint_index < 25:
        raise ValueError("minimap.position_joint_index must be inside 0..24")
    panel_color = _rgb01(config.get("panel_color", [38, 92, 58]), "panel_color")
    pitch_line_color = _rgb01(config.get("pitch_line_color", [245, 248, 246]), "pitch_line_color")

    panel = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.zeros((4, 3), dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(
            np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
        ),
    )
    panel.paint_uniform_color(panel_color)
    pitch_lines = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.zeros((4, 3), dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(
            np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
        ),
    )
    markers = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.zeros((3, 3), dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(np.array([[0, 2, 1]], dtype=np.int32)),
    )
    return MinimapOverlay(
        source=source,
        pitch_points_xy=pitch_points[:, :2],
        pitch_segments=pitch_segments,
        pitch_bounds=bounds,
        panel=panel,
        pitch_lines=pitch_lines,
        markers=markers,
        width_px=width_px,
        height_px=height_px,
        margin_px=margin_px,
        padding_px=padding_px,
        marker_radius_px=marker_radius_px,
        marker_circle_segments=marker_circle_segments,
        pitch_line_width_px=pitch_line_width_px,
        depth_m=depth_m,
        panel_color=panel_color,
        pitch_line_color=pitch_line_color,
        position_joint_index=position_joint_index,
        clip_positions=bool(config.get("clip_positions", True)),
    )


def _pixels_to_world(
    pixels: np.ndarray,
    depth: float,
    camera_parameters: o3d.camera.PinholeCameraParameters,
) -> np.ndarray:
    intrinsic = np.asarray(camera_parameters.intrinsic.intrinsic_matrix, dtype=np.float64)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid minimap camera focal lengths fx={fx}, fy={fy}")
    pixels = np.asarray(pixels, dtype=np.float64)
    camera_points = np.column_stack(
        [
            (pixels[:, 0] - cx) * depth / fx,
            (pixels[:, 1] - cy) * depth / fy,
            np.full(len(pixels), depth, dtype=np.float64),
        ]
    )
    extrinsic = np.asarray(camera_parameters.extrinsic, dtype=np.float64)
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    return (camera_points - translation) @ rotation


def _camera_signature(
    camera_parameters: o3d.camera.PinholeCameraParameters,
) -> np.ndarray:
    intrinsic = camera_parameters.intrinsic
    return np.concatenate(
        [
            np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64).reshape(-1),
            np.asarray(camera_parameters.extrinsic, dtype=np.float64).reshape(-1),
            np.array([intrinsic.width, intrinsic.height], dtype=np.float64),
        ]
    )


def _rgb01(raw: Any, name: str) -> np.ndarray:
    color = np.asarray(raw, dtype=np.float64)
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError(f"minimap.{name} must be an RGB triple")
    if float(color.max()) > 1.0:
        color = color / 255.0
    if np.any((color < 0.0) | (color > 1.0)):
        raise ValueError(f"minimap.{name} values must be inside 0..1 or 0..255")
    return color


def _positive_int(raw: Any, name: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError(f"minimap.{name} must be > 0")
    return value


def _non_negative_int(raw: Any, name: str) -> int:
    value = int(raw)
    if value < 0:
        raise ValueError(f"minimap.{name} must be >= 0")
    return value


def _positive_float(raw: Any, name: str) -> float:
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"minimap.{name} must be > 0")
    return value
