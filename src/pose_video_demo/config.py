from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from pose_video_demo.utils.colors import DEFAULT_COLORS
from pose_video_demo.utils.paths import project_root, resolve_path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG: dict[str, Any] = {
    "sequence": {
        "name": "ARG_FRA_182345",
        "fps": None,
        "start_frame": 0,
        "num_frames": None,
    },
    "paths": {
        "video": "data/videos/ARG_FRA_182345.mov",
        "cameras": "data/cameras/ARG_FRA_182345.npz",
        "boxes": "data/boxes/ARG_FRA_182345.npy",
        "xrel_sam": "data/xrel_sam/ARG_FRA_182345.npy",
        "root_cam": "data/root_cam/ARG_FRA_182345.npy",
        "pitch_points": "data/pitch/pitch_points.txt",
        "teams": "data/teams/ARG_FRA_182345.txt",
    },
    "pitch": {
        "segments": None,
        "infer_segments": True,
        "max_segment_length": 8.0,
    },
    "render": {
        "width": None,
        "height": None,
        "output_fps": None,
        "stickman_line_width": 2,
        "joint_diameter": 4,
        "bbox_line_width": 2,
        "pitch_line_width": 2,
        "floor_alpha": 0.55,
        "colors": DEFAULT_COLORS,
        "debug_frames": [0],
        "debug_dir": "outputs/frames",
    },
    "broadcast": {
        "show_rgb": True,
        "show_pitch_lines": True,
        "show_boxes": True,
        "show_stickmen": True,
        "show_green_floor": False,
        "stickman_mode": "projected_3d",
        "start_frame": None,
        "end_frame": None,
        "output_path": "outputs/clips/broadcast.mp4",
    },
    "drone": {
        "width": 1920,
        "height": 1080,
        "start_frame": None,
        "end_frame": None,
        "output_path": "outputs/clips/drone.mp4",
        "camera_motion": {
            "type": "travel",
            "smoothstep": True,
            "look_at_distance_m": 100.0,
            "start": {
                "position": {
                    "mode": "spherical",
                    "origin": "players_center",
                    "radius_m": 50.0,
                    "azimuth_deg": 270.0,
                    "elevation_deg": 24.0,
                },
                "look_at": None,
                "zoom": {
                    "mode": "broadcast_camera",
                    "factor": 1.0,
                },
            },
            "end": {
                "position": {
                    "mode": "spherical",
                    "origin": "players_center",
                    "radius_m": 25.0,
                    "azimuth_deg": 220.0,
                    "elevation_deg": 22.0,
                },
                "look_at": None,
                "zoom": {
                    "mode": "broadcast_camera",
                    "factor": 1.0,
                },
            },
        },
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULT_CONFIG, loaded)
    cfg["_config_path"] = str(config_path)
    cfg["_base_dir"] = str(project_root())
    return cfg


def resolve_config_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    base = cfg.get("_base_dir") or project_root()
    return {key: resolve_path(value, base) for key, value in cfg["paths"].items() if value}


def frame_range(cfg: dict[str, Any], section: str, total_frames: int) -> tuple[int, int]:
    seq_start = int(cfg.get("sequence", {}).get("start_frame") or 0)
    seq_num = cfg.get("sequence", {}).get("num_frames")
    default_end = min(total_frames, seq_start + int(seq_num)) if seq_num else total_frames
    part = cfg.get(section, {})
    start = part.get("start_frame")
    end = part.get("end_frame")
    start_frame = seq_start if start is None else int(start)
    end_frame = default_end if end is None else int(end)
    if not 0 <= start_frame < total_frames:
        raise ValueError(f"Invalid start_frame={start_frame} for {total_frames} frames")
    if not start_frame < end_frame <= total_frames:
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}) for {total_frames} frames")
    return start_frame, end_frame
