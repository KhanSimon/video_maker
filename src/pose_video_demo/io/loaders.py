from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pose_video_demo.config import resolve_config_paths
from pose_video_demo.geometry.transforms import camera_to_world_row_vector, reconstruct_x_cam
from pose_video_demo.io.video_io import VideoInfo, read_video_info
from pose_video_demo.utils.colors import team_key


@dataclass(frozen=True)
class CameraData:
    k_matrix: np.ndarray
    r: np.ndarray
    t: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True)
class PoseData:
    xrel_sam: np.ndarray
    root_cam: np.ndarray
    x_cam: np.ndarray
    x_world: np.ndarray


@dataclass(frozen=True)
class SequenceData:
    name: str
    video_path: Path
    video_info: VideoInfo
    cameras: CameraData
    poses: PoseData
    boxes: np.ndarray | None
    teams: dict[int, str]


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 1:
            return data[data.files[0]]
        for key in ("arr_0", "root_cam", "root_cam_pred", "data"):
            if key in data.files:
                return data[key]
        raise ValueError(f"Could not choose array inside {path}; keys={data.files}")
    return np.load(path, allow_pickle=False)


def load_cameras(path: Path) -> CameraData:
    data = np.load(path)
    for key in ("K", "R", "t"):
        if key not in data:
            raise KeyError(f"Camera file {path} is missing key {key}")
    distortion = data["k"] if "k" in data else np.zeros((data["K"].shape[0], 2), dtype=np.float64)
    if distortion.ndim == 1:
        distortion = np.tile(distortion[None, :], (data["K"].shape[0], 1))
    return CameraData(
        k_matrix=np.asarray(data["K"], dtype=np.float64),
        r=np.asarray(data["R"], dtype=np.float64),
        t=np.asarray(data["t"], dtype=np.float64),
        distortion=np.asarray(distortion[:, :2], dtype=np.float64),
    )


def load_boxes(path: Path | None, num_players: int, num_frames: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    boxes = np.asarray(_load_array(path), dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes shape (N,T,4) or (T,N,4), got {boxes.shape}")
    if boxes.shape[:2] == (num_players, num_frames):
        return boxes
    if boxes.shape[:2] == (num_frames, num_players):
        return np.transpose(boxes, (1, 0, 2))
    raise ValueError(
        f"Could not align boxes shape {boxes.shape} with N={num_players}, T={num_frames}"
    )


def load_team_assignments(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    teams: dict[int, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            person_idx = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid team line {line_number} in {path}: {raw}") from exc
        label = parts[1] if len(parts) > 1 else None
        teams[person_idx] = team_key(label)
    return teams


def load_sequence(cfg: dict[str, Any]) -> SequenceData:
    paths = resolve_config_paths(cfg)
    video_path = paths["video"]
    info = read_video_info(video_path)
    cameras = load_cameras(paths["cameras"])
    xrel_sam = np.asarray(_load_array(paths["xrel_sam"]), dtype=np.float32)
    root_cam = np.asarray(_load_array(paths["root_cam"]), dtype=np.float32)
    x_cam = reconstruct_x_cam(xrel_sam, root_cam)
    usable_t = min(x_cam.shape[1], cameras.k_matrix.shape[0], cameras.r.shape[0], cameras.t.shape[0])
    if usable_t != x_cam.shape[1]:
        xrel_sam = xrel_sam[:, :usable_t]
        root_cam = root_cam[:, :usable_t]
        x_cam = x_cam[:, :usable_t]
    x_world = camera_to_world_row_vector(x_cam, cameras.r[:usable_t], cameras.t[:usable_t])
    boxes = load_boxes(paths.get("boxes"), x_cam.shape[0], x_cam.shape[1])
    teams = load_team_assignments(paths.get("teams"))
    return SequenceData(
        name=str(cfg["sequence"]["name"]),
        video_path=video_path,
        video_info=info,
        cameras=CameraData(
            k_matrix=cameras.k_matrix[:usable_t],
            r=cameras.r[:usable_t],
            t=cameras.t[:usable_t],
            distortion=cameras.distortion[:usable_t],
        ),
        poses=PoseData(xrel_sam=xrel_sam, root_cam=root_cam, x_cam=x_cam, x_world=x_world),
        boxes=boxes[:, :usable_t] if boxes is not None else None,
        teams=teams,
    )


def describe_sequence(data: SequenceData) -> list[str]:
    lines = [
        f"sequence={data.name}",
        f"video={data.video_info.width}x{data.video_info.height} fps={data.video_info.fps:.3f} frames={data.video_info.frame_count}",
        f"K={data.cameras.k_matrix.shape} R={data.cameras.r.shape} t={data.cameras.t.shape} k={data.cameras.distortion.shape}",
        f"xrel_sam={data.poses.xrel_sam.shape} root_cam={data.poses.root_cam.shape}",
        f"x_cam={data.poses.x_cam.shape} x_world={data.poses.x_world.shape}",
        f"boxes={None if data.boxes is None else data.boxes.shape}",
        f"teams={len(data.teams)} assignments",
    ]
    if data.poses.x_cam.shape[2:] != (25, 3):
        raise ValueError(f"Expected reconstructed X_cam shape (N,T,25,3), got {data.poses.x_cam.shape}")
    if data.video_info.frame_count and data.video_info.frame_count != data.poses.x_cam.shape[1]:
        lines.append("warning=video frame count differs from pose/camera T; rendering uses the requested intersection")
    return lines
