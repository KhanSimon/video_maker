from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d

from video_maker.interactive_viewer.skeleton_updater import SkeletonLayer
from video_maker.skeleton.joints import BONES_25


_BONES = np.asarray(BONES_25, dtype=np.int32)
_HIDDEN_POINT = np.array([0.0, 0.0, -1_000_000.0], dtype=np.float64)


@dataclass
class TemporalContextLayer:
    source: SkeletonLayer
    player_indices: tuple[int, ...]
    endpoint_offsets: frozenset[int]
    sampled_offsets: tuple[int, ...]
    linesets: list[o3d.geometry.LineSet]
    pointclouds: list[o3d.geometry.PointCloud]
    background_color: np.ndarray
    intermediate_pastel_coefficient: float
    endpoint_opacity: float
    endpoint_color: np.ndarray | None = None
    last_visible_offsets: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def geometries(self) -> list[o3d.geometry.Geometry]:
        return [*self.linesets, *self.pointclouds]

    def update(self, current_frame: int) -> None:
        for geometry_idx, player_idx in enumerate(self.player_indices):
            points_parts: list[np.ndarray] = []
            lines_parts: list[np.ndarray] = []
            line_color_parts: list[np.ndarray] = []
            point_color_parts: list[np.ndarray] = []
            visible_offsets: list[int] = []
            player_color = self._player_color(player_idx)

            for offset in self.sampled_offsets:
                joints = self.source.visible_joints(player_idx, current_frame + offset)
                if joints is None:
                    continue
                color = self._offset_color(offset, player_color)
                point_start = len(points_parts) * 25
                points_parts.append(joints)
                lines_parts.append(_BONES + point_start)
                line_color_parts.append(np.tile(color, (len(_BONES), 1)))
                point_color_parts.append(np.tile(color, (25, 1)))
                visible_offsets.append(offset)

            self._set_geometry(
                geometry_idx,
                points_parts,
                lines_parts,
                line_color_parts,
                point_color_parts,
            )
            self.last_visible_offsets[player_idx] = tuple(visible_offsets)

    def _player_color(self, player_idx: int) -> np.ndarray:
        colors = np.asarray(self.source.linesets[player_idx].colors, dtype=np.float64)
        if colors.shape[0]:
            return colors[0]
        return np.array([0.7, 0.7, 0.7], dtype=np.float64)

    def _offset_color(self, offset: int, player_color: np.ndarray) -> np.ndarray:
        intermediate_color = self._intermediate_color(player_color)
        if offset in self.endpoint_offsets:
            endpoint_color = self.endpoint_color if self.endpoint_color is not None else player_color
            return (
                intermediate_color * (1.0 - self.endpoint_opacity)
                + endpoint_color * self.endpoint_opacity
            )
        return intermediate_color

    def _intermediate_color(self, player_color: np.ndarray) -> np.ndarray:
        coefficient = self.intermediate_pastel_coefficient
        return player_color * (1.0 - coefficient) + np.ones(3, dtype=np.float64) * coefficient

    def _set_geometry(
        self,
        geometry_idx: int,
        points_parts: list[np.ndarray],
        lines_parts: list[np.ndarray],
        line_color_parts: list[np.ndarray],
        point_color_parts: list[np.ndarray],
    ) -> None:
        if points_parts:
            points = np.concatenate(points_parts, axis=0)
            lines = np.concatenate(lines_parts, axis=0)
            line_colors = np.concatenate(line_color_parts, axis=0)
            point_colors = np.concatenate(point_color_parts, axis=0)
        else:
            points = np.tile(_HIDDEN_POINT, (25, 1))
            lines = _BONES.copy()
            line_colors = np.tile(self.background_color, (len(_BONES), 1))
            point_colors = np.tile(self.background_color, (25, 1))

        line_set = self.linesets[geometry_idx]
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        point_cloud = self.pointclouds[geometry_idx]
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(point_colors)


def build_temporal_context_layer(
    source: SkeletonLayer,
    config: dict[str, Any],
    background_color: list[float] | tuple[float, ...] | np.ndarray,
) -> TemporalContextLayer:
    player_indices = _unique_ints(config.get("player_indices", []), "player_indices")
    if not player_indices:
        raise ValueError("temporal_context.player_indices must contain at least one player")
    invalid_players = [idx for idx in player_indices if not 0 <= idx < source.num_players]
    if invalid_players:
        raise ValueError(
            f"temporal_context player indices {invalid_players} are outside 0..{source.num_players - 1}"
        )

    endpoint_offsets = tuple(
        offset
        for offset in _unique_ints(config.get("frame_offsets", []), "frame_offsets")
        if offset != 0
    )
    if not endpoint_offsets:
        raise ValueError("temporal_context.frame_offsets must contain at least one non-zero offset")
    stride = int(config.get("intermediate_stride", 1))
    if stride <= 0:
        raise ValueError("temporal_context.intermediate_stride must be >= 1")
    sampled_offsets = _expanded_offsets(
        endpoint_offsets,
        show_intermediate=bool(config.get("show_intermediate", True)),
        stride=stride,
    )

    background = _rgb01(background_color, "background_color")
    endpoint_raw = config.get("endpoint_color")
    endpoint_color = _rgb01(endpoint_raw, "endpoint_color") if endpoint_raw is not None else None
    pastel_coefficient = _opacity(
        config.get("intermediate_pastel_coefficient", 0.55),
        "intermediate_pastel_coefficient",
    )
    endpoint_opacity = _opacity(config.get("endpoint_opacity", 0.85), "endpoint_opacity")

    line_sets = [o3d.geometry.LineSet() for _ in player_indices]
    point_clouds = [o3d.geometry.PointCloud() for _ in player_indices]
    layer = TemporalContextLayer(
        source=source,
        player_indices=player_indices,
        endpoint_offsets=frozenset(endpoint_offsets),
        sampled_offsets=sampled_offsets,
        linesets=line_sets,
        pointclouds=point_clouds,
        background_color=background,
        intermediate_pastel_coefficient=pastel_coefficient,
        endpoint_opacity=endpoint_opacity,
        endpoint_color=endpoint_color,
    )
    layer.update(0)
    return layer


def _expanded_offsets(
    endpoint_offsets: tuple[int, ...],
    show_intermediate: bool,
    stride: int,
) -> tuple[int, ...]:
    if not show_intermediate:
        return tuple(sorted(endpoint_offsets))
    offsets: set[int] = set(endpoint_offsets)
    for endpoint in endpoint_offsets:
        direction = 1 if endpoint > 0 else -1
        offsets.update(range(direction * stride, endpoint + direction, direction * stride))
        offsets.add(endpoint)
    return tuple(sorted(offsets))


def _unique_ints(raw: Any, name: str) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"temporal_context.{name} must be a list")
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"temporal_context.{name} must contain integers, got {value!r}")
        integer = int(value)
        if integer not in values:
            values.append(integer)
    return tuple(values)


def _rgb01(raw: Any, name: str) -> np.ndarray:
    color = np.asarray(raw, dtype=np.float64)
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError(f"temporal_context.{name} must be an RGB triple")
    if float(color.max()) > 1.0:
        color = color / 255.0
    if np.any((color < 0.0) | (color > 1.0)):
        raise ValueError(f"temporal_context.{name} values must be inside 0..1 or 0..255")
    return color


def _opacity(raw: Any, name: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"temporal_context.{name} must be inside 0..1")
    return value
