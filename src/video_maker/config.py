from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from video_maker.utils.colors import DEFAULT_COLORS
from video_maker.utils.paths import project_root, resolve_path
from video_maker.timing import source_frame_range, timeline_frame_ranges


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG: dict[str, Any] = {
    "MHR": False,
    "mhr": {
        "lod": 1,
        "input_layout": "auto",
        "smooth_shading": True,
    },
    "sequence": {
        "name": "ARG_FRA_182345",
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
        "green_floor_start_frame": None,
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


def frame_range(
    cfg: dict[str, Any], section: str, total_frames: int, options: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Use the renderer's range, or infer it from its contiguous timeline clips."""
    part = cfg.get(section, {}) if options is None else options
    start = part.get("start_frame")
    end = part.get("end_frame")
    default_start, default_end = 0, total_frames
    if (start is None or end is None) and cfg.get("timeline"):
        ranges = timeline_frame_ranges(cfg["timeline"])
        selected = [bounds for item, bounds in zip(cfg["timeline"], ranges) if item.get("renderer", "broadcast") == section]
        if not selected or any(a[1] != b[0] for a, b in zip(selected, selected[1:])):
            raise ValueError(f"No single contiguous {section} range in the timeline; use render_full_demo.py or a separate video config.")
        default_start, default_end = selected[0][0], selected[-1][1]
    return source_frame_range(
        default_start if start is None else start,
        default_end if end is None else end,
        label=section, total_frames=total_frames,
    )
