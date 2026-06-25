from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from pose_video_demo.io.loaders import load_team_assignments
from pose_video_demo.skeleton.joints import BONES_25
from pose_video_demo.utils.colors import color_table


@dataclass
class SkeletonLayer:
    name: str
    linesets: list[o3d.geometry.LineSet]
    data: np.ndarray
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
            joints = self.data[person_idx, frame_idx].astype(np.float64, copy=True)
            if not np.isfinite(joints).all():
                joints[~np.isfinite(joints)] = 0.0
            joints[:, 2] += self.z_offset
            line_set.points = o3d.utility.Vector3dVector(joints)


def load_x_world(path) -> np.ndarray:
    data = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if data.ndim != 4 or data.shape[2:] != (25, 3):
        raise ValueError(f"Expected x_world shape (N,T,25,3), got {data.shape}")
    if data.shape[0] > data.shape[1]:
        raise ValueError(
            f"Expected x_world convention (N,T,25,3), got likely (T,N,25,3): {data.shape}"
        )
    return data


def build_skeleton_layer(
    x_world_path,
    teams_path=None,
    colors_config: dict | None = None,
    name: str = "poses",
    uniform_color: tuple[int, int, int] | None = None,
    z_offset: float = 0.0,
) -> SkeletonLayer:
    data = load_x_world(x_world_path)
    teams = load_team_assignments(teams_path) if teams_path else {}
    colors = color_table(colors_config)
    line_sets = [
        build_player_lineset(data[person_idx, 0], uniform_color or _team_color(person_idx, teams, colors))
        for person_idx in range(data.shape[0])
    ]
    return SkeletonLayer(name=name, linesets=line_sets, data=data, z_offset=z_offset)


def build_player_lineset(joints: np.ndarray, color_rgb: tuple[int, int, int]) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(joints.astype(np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(BONES_25, dtype=np.int32)),
    )
    color = np.asarray(color_rgb, dtype=np.float64) / 255.0
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(BONES_25), 1)))
    return line_set


def _team_color(person_idx: int, teams: dict[int, str], colors: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    return colors.get(teams.get(person_idx, "fallback"), colors["fallback"])
