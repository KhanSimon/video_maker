from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from video_maker.io.loaders import load_team_assignments
from video_maker.skeleton.joints import BONES_25
from video_maker.utils.colors import color_table


_HIDDEN_POINT = np.array([0.0, 0.0, -1_000_000.0], dtype=np.float64)


@dataclass
class SkeletonLayer:
    name: str
    linesets: list[o3d.geometry.LineSet]
    pointclouds: list[o3d.geometry.PointCloud]
    data: np.ndarray
    valid_mask: np.ndarray
    frame_numbers: np.ndarray
    player_colors: np.ndarray
    player_display_mask: np.ndarray
    z_offset: float = 0.0

    @property
    def num_frames(self) -> int:
        return int(self.data.shape[1])

    @property
    def num_players(self) -> int:
        return int(self.data.shape[0])

    def update(self, frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, self.num_frames - 1))
        for person_idx, line_set in enumerate(self.linesets):
            joints = self.display_joints(person_idx, frame_idx)
            line_set.points = o3d.utility.Vector3dVector(joints)
            self.pointclouds[person_idx].points = o3d.utility.Vector3dVector(joints)

    @property
    def geometries(self) -> list[o3d.geometry.Geometry]:
        return [*self.linesets, *self.pointclouds]

    def visible_joints(self, person_idx: int, frame_idx: int) -> np.ndarray | None:
        if not 0 <= person_idx < self.num_players or not 0 <= frame_idx < self.num_frames:
            return None
        if not self.player_display_mask[person_idx]:
            return None
        if not self.valid_mask[person_idx, frame_idx]:
            return None
        joints = self.data[person_idx, frame_idx].astype(np.float64, copy=True)
        if not np.isfinite(joints).all():
            return None
        joints[:, 2] += self.z_offset
        return joints

    def display_joints(self, person_idx: int, frame_idx: int) -> np.ndarray:
        joints = self.visible_joints(person_idx, frame_idx)
        return joints if joints is not None else np.tile(_HIDDEN_POINT, (25, 1))

    def source_frame_number(self, frame_idx: int) -> int:
        frame_idx = int(np.clip(frame_idx, 0, self.num_frames - 1))
        return int(self.frame_numbers[frame_idx])

    def set_player_filter(self, player_indices: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
        if player_indices is None:
            self.player_display_mask[:] = True
            return tuple(range(self.num_players))
        if not isinstance(player_indices, (list, tuple)):
            raise ValueError("interactive_viewer.player_indices must be a list of integers")
        if not player_indices:
            self.player_display_mask[:] = True
            return tuple(range(self.num_players))
        selected: list[int] = []
        for value in player_indices:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"interactive_viewer.player_indices must contain integers, got {value!r}"
                )
            player_idx = int(value)
            if not 0 <= player_idx < self.num_players:
                raise ValueError(
                    f"interactive_viewer player index {player_idx} is outside 0..{self.num_players - 1}"
                )
            if player_idx not in selected:
                selected.append(player_idx)
        self.player_display_mask[:] = False
        self.player_display_mask[selected] = True
        return tuple(selected)


def load_x_world(path, data_key: str | None = None) -> np.ndarray:
    path = str(path)
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            selected_key = _select_npz_key(
                archive,
                data_key,
                candidates=("joints_pred_world_m", "x_world", "arr_0", "data"),
                kind="world-pose",
            )
            data = np.asarray(archive[selected_key], dtype=np.float32)
    else:
        if data_key is not None:
            raise ValueError(f"data_key={data_key!r} is only supported for .npz pose sources")
        data = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if data.ndim != 4 or data.shape[2:] != (25, 3):
        raise ValueError(f"Expected x_world shape (N,T,25,3), got {data.shape}")
    if data.shape[0] > data.shape[1]:
        raise ValueError(
            f"Expected x_world convention (N,T,25,3), got likely (T,N,25,3): {data.shape}"
        )
    return data


def load_pose_metadata(
    path,
    data: np.ndarray,
    valid_mask_key: str | None = None,
    frame_numbers_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(data).all(axis=(2, 3))
    frame_numbers = np.arange(data.shape[1], dtype=np.int64)
    path = str(path)
    if not path.lower().endswith(".npz"):
        return finite_mask, frame_numbers

    with np.load(path, allow_pickle=False) as archive:
        mask_key = _optional_npz_key(archive, valid_mask_key, "valid_mask")
        if mask_key is not None:
            valid_mask = np.asarray(archive[mask_key], dtype=bool)
            if valid_mask.shape == (data.shape[1], data.shape[0]):
                valid_mask = valid_mask.T
            if valid_mask.shape != data.shape[:2]:
                raise ValueError(
                    f"Expected {mask_key} shape {data.shape[:2]} or {data.shape[:2][::-1]}, "
                    f"got {valid_mask.shape}"
                )
            finite_mask &= valid_mask

        numbers_key = _optional_npz_key(archive, frame_numbers_key, "frame_numbers")
        if numbers_key is not None:
            frame_numbers = np.asarray(archive[numbers_key], dtype=np.int64)
            if frame_numbers.shape != (data.shape[1],):
                raise ValueError(
                    f"Expected {numbers_key} shape ({data.shape[1]},), got {frame_numbers.shape}"
                )
    return finite_mask, frame_numbers


def build_skeleton_layer(
    x_world_path,
    teams_path=None,
    colors_config: dict | None = None,
    name: str = "poses",
    uniform_color: tuple[int, int, int] | None = None,
    z_offset: float = 0.0,
    data_key: str | None = None,
    valid_mask_key: str | None = None,
    frame_numbers_key: str | None = None,
) -> SkeletonLayer:
    data = load_x_world(x_world_path, data_key=data_key)
    valid_mask, frame_numbers = load_pose_metadata(
        x_world_path,
        data,
        valid_mask_key=valid_mask_key,
        frame_numbers_key=frame_numbers_key,
    )
    teams = load_team_assignments(teams_path) if teams_path else {}
    colors = color_table(colors_config)
    initial_joints = [
        _initial_joints(data, valid_mask, person_idx, z_offset)
        for person_idx in range(data.shape[0])
    ]
    team_colors = [
        _team_color(person_idx, teams, colors)
        for person_idx in range(data.shape[0])
    ]
    render_colors = [uniform_color or color for color in team_colors]
    line_sets = [
        build_player_lineset(joints, render_colors[person_idx])
        for person_idx, joints in enumerate(initial_joints)
    ]
    point_clouds = [
        build_player_pointcloud(joints, render_colors[person_idx])
        for person_idx, joints in enumerate(initial_joints)
    ]
    return SkeletonLayer(
        name=name,
        linesets=line_sets,
        pointclouds=point_clouds,
        data=data,
        valid_mask=valid_mask,
        frame_numbers=frame_numbers,
        player_colors=np.asarray(team_colors, dtype=np.float64) / 255.0,
        player_display_mask=np.ones(data.shape[0], dtype=bool),
        z_offset=z_offset,
    )


def build_player_lineset(joints: np.ndarray, color_rgb: tuple[int, int, int]) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(joints.astype(np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(BONES_25, dtype=np.int32)),
    )
    color = np.asarray(color_rgb, dtype=np.float64) / 255.0
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(BONES_25), 1)))
    return line_set


def build_player_pointcloud(
    joints: np.ndarray,
    color_rgb: tuple[int, int, int],
) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(joints.astype(np.float64))
    )
    color = np.asarray(color_rgb, dtype=np.float64) / 255.0
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(joints), 1)))
    return point_cloud


def _team_color(person_idx: int, teams: dict[int, str], colors: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    return colors.get(teams.get(person_idx, "fallback"), colors["fallback"])


def _initial_joints(
    data: np.ndarray,
    valid_mask: np.ndarray,
    person_idx: int,
    z_offset: float,
) -> np.ndarray:
    if not valid_mask[person_idx, 0]:
        return np.tile(_HIDDEN_POINT, (25, 1))
    joints = data[person_idx, 0].astype(np.float64, copy=True)
    joints[:, 2] += float(z_offset)
    return joints


def _select_npz_key(
    archive: np.lib.npyio.NpzFile,
    requested: str | None,
    candidates: tuple[str, ...],
    kind: str,
) -> str:
    if requested is not None:
        if requested not in archive.files:
            raise KeyError(f"NPZ is missing requested {kind} key {requested!r}; keys={archive.files}")
        return requested
    for candidate in candidates:
        if candidate in archive.files:
            return candidate
    raise KeyError(f"Could not select a {kind} array from NPZ; keys={archive.files}")


def _optional_npz_key(
    archive: np.lib.npyio.NpzFile,
    requested: str | None,
    default: str,
) -> str | None:
    if requested is not None:
        if requested not in archive.files:
            raise KeyError(f"NPZ is missing requested key {requested!r}; keys={archive.files}")
        return requested
    return default if default in archive.files else None
