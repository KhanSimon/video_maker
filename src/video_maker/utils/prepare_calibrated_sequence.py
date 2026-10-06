from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from video_maker.geometry.calibration import (
    camera_track_from_json,
    load_calibration_input_lines,
)
from video_maker.geometry.ground_translation import estimate_ground_ray_translation
from video_maker.geometry.transforms import (
    camera_to_world_row_vector,
    normalize_player_frame_array,
    normalize_pose_array,
    reconstruct_x_cam,
)
from video_maker.pitch.generator import (
    generate_calibrated_pitch_points,
    generate_standard_pitch_points,
)
from video_maker.pitch.pitch_loader import infer_pitch_segments
from video_maker.utils.paths import ensure_parent, project_root, resolve_path


def prepare_sequence(config_path: str | Path, overwrite: bool = False) -> dict[str, Path]:
    config_file = resolve_path(config_path, project_root())
    with config_file.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base = project_root()
    sequence = str(cfg.get("sequence", {}).get("name", "")).strip()
    if not sequence:
        raise ValueError("sequence.name is required")

    inputs = cfg.get("inputs", {})
    outputs = cfg.get("outputs", {})
    translation_cfg = cfg.get("translation", {})
    translation_method = str(translation_cfg.get("method", "input")).lower()
    xrel_path = _required_path(inputs, "xrel_sam", base)
    boxes_path = _required_path(inputs, "boxes", base)
    calibration_paths = _calibration_paths(inputs.get("calibration_json"), base)
    output_paths = {
        "cameras": _output_path(outputs, "cameras", f"data/cameras/{sequence}.npz", base),
        "pitch_points": _output_path(
            outputs, "pitch_points", f"data/pitch/{sequence}.txt", base
        ),
        "teams": _output_path(outputs, "teams", f"data/teams/{sequence}.txt", base),
        "x_world": _output_path(
            outputs, "x_world", f"data/x_world_estim/{sequence}.npy", base
        ),
    }
    if translation_method in {"ground_ray", "ray_ground", "geometric"}:
        output_paths["root_cam"] = _output_path(
            outputs,
            "root_cam",
            f"data/geometric_estim/{sequence}/root_cam.npy",
            base,
        )
        output_paths["diagnostics"] = _output_path(
            outputs,
            "diagnostics",
            f"data/geometric_estim/{sequence}/diagnostics.npz",
            base,
        )
    _check_outputs(output_paths, overwrite)

    pose_cfg = cfg.get("poses", {})
    raw_xrel = np.asarray(_load_array(xrel_path), dtype=np.float32)
    xrel_sam = normalize_pose_array(
        raw_xrel,
        input_layout=str(pose_cfg.get("input_layout", "auto")),
        axis_signs=tuple(pose_cfg.get("xrel_axis_signs", [1, 1, 1])),
    )
    num_players, num_frames = xrel_sam.shape[:2]
    boxes = normalize_player_frame_array(
        np.asarray(_load_array(boxes_path), dtype=np.float32),
        num_players=num_players,
        num_frames=num_frames,
        name="boxes",
    )

    camera_cfg = cfg.get("camera", {})
    camera_track = camera_track_from_json(
        calibration_paths,
        num_frames=num_frames,
        source_to_world_axis_signs=tuple(
            camera_cfg.get("source_to_world_axis_signs", [1, -1, -1])
        ),
    )
    diagnostics: dict[str, np.ndarray] | None = None
    if translation_method in {"input", "translations", "root_cam"}:
        translations_path = _required_path(inputs, "translations", base)
        root_cam = normalize_player_frame_array(
            np.asarray(_load_array(translations_path), dtype=np.float32),
            num_players=num_players,
            num_frames=num_frames,
            name="translations",
        )
        if root_cam.shape != (num_players, num_frames, 3):
            raise ValueError(f"Expected translations shape {(num_players, num_frames, 3)}, got {root_cam.shape}")
        x_cam = reconstruct_x_cam(xrel_sam, root_cam)
        x_world = camera_to_world_row_vector(
            x_cam,
            camera_track["R"],
            camera_track["t"],
        ).astype(np.float32, copy=False)
    elif translation_method in {"ground_ray", "ray_ground", "geometric"}:
        keypoints_2d_path = _required_path(inputs, "x_2d", base)
        keypoints_2d = normalize_player_frame_array(
            np.asarray(_load_array(keypoints_2d_path), dtype=np.float32),
            num_players=num_players,
            num_frames=num_frames,
            name="x_2d",
        )
        smoothing_cfg = translation_cfg.get("smoothing", {})
        result = estimate_ground_ray_translation(
            xrel_sam,
            keypoints_2d,
            camera_track["K"],
            camera_track["R"],
            camera_track["t"],
            distortion=camera_track["k"],
            ground_z_m=float(translation_cfg.get("ground_z_m", 0.0)),
            contact_selection=str(translation_cfg.get("contact_selection", "world_z")),
            relative_vertical_axis=translation_cfg.get("relative_vertical_axis", "y"),
            relative_ground_direction=str(
                translation_cfg.get("relative_ground_direction", "max")
            ),
            candidate_joint_indices=translation_cfg.get("candidate_joint_indices"),
            smoothing_enabled=bool(smoothing_cfg.get("enabled", True)),
            smoothing_fps=float(
                smoothing_cfg.get("fps", cfg.get("sequence", {}).get("fps", 50.0))
            ),
            smoothing_median_window=int(smoothing_cfg.get("median_window", 7)),
            smoothing_cutoff_hz=float(smoothing_cfg.get("cutoff_hz", 3.0)),
            smoothing_max_speed_m_s=(
                float(smoothing_cfg["max_speed_m_s"])
                if smoothing_cfg.get("max_speed_m_s") is not None
                else None
            ),
            preserve_ground_contact=bool(
                smoothing_cfg.get("preserve_ground_contact", True)
            ),
        )
        root_cam = result.root_cam.astype(np.float32)
        x_world = result.x_world.astype(np.float32)
        diagnostics = {
            "contact_joint_indices": result.contact_joint_indices,
            "valid": result.valid,
            "ground_points_raw": result.ground_points_raw.astype(np.float32),
            "contact_points_smoothed": result.contact_points_smoothed.astype(np.float32),
            "root_world_raw": result.root_world_raw.astype(np.float32),
            "root_world_smoothed": result.root_world_smoothed.astype(np.float32),
        }
    else:
        raise ValueError(f"Unsupported translation.method={translation_method}")

    pitch_cfg = cfg.get("pitch", {})
    pitch_mode = str(pitch_cfg.get("mode", "standard")).lower()
    if pitch_mode in {"calibration_lines", "input_lines", "calibrated"}:
        pitch_points = generate_calibrated_pitch_points(
            load_calibration_input_lines(calibration_paths[0]),
            k_matrix=camera_track["K"][0],
            rotation_world_to_camera=camera_track["R"][0],
            camera_center_world=camera_track["camera_center"][0],
            ground_z_m=float(pitch_cfg.get("ground_z_m", 0.0)),
            floor_extent_m=_optional_extent(pitch_cfg.get("floor_extent_m")),
            sample_spacing_m=float(pitch_cfg.get("sample_spacing_m", 1.0)),
            max_segment_length_m=float(pitch_cfg.get("max_segment_length_m", 8.0)),
        )
    elif pitch_mode == "standard":
        pitch_points = generate_standard_pitch_points(
            length_m=float(pitch_cfg.get("length_m", 105.0)),
            width_m=float(pitch_cfg.get("width_m", 68.0)),
            penalty_area_depth_m=float(pitch_cfg.get("penalty_area_depth_m", 16.5)),
            penalty_area_width_m=float(pitch_cfg.get("penalty_area_width_m", 40.32)),
            goal_area_depth_m=float(pitch_cfg.get("goal_area_depth_m", 5.5)),
            goal_area_width_m=float(pitch_cfg.get("goal_area_width_m", 18.32)),
            center_circle_radius_m=float(pitch_cfg.get("center_circle_radius_m", 9.15)),
            penalty_spot_distance_m=float(pitch_cfg.get("penalty_spot_distance_m", 11.0)),
            corner_arc_radius_m=float(pitch_cfg.get("corner_arc_radius_m", 1.0)),
            sample_spacing_m=float(pitch_cfg.get("sample_spacing_m", 1.0)),
            max_segment_length_m=float(pitch_cfg.get("max_segment_length_m", 8.0)),
        )
    else:
        raise ValueError(f"Unsupported pitch.mode={pitch_mode}")
    teams_text = _fictitious_teams(num_players, cfg.get("teams", {}))

    for output in output_paths.values():
        ensure_parent(output)
    np.savez(output_paths["cameras"], **camera_track)
    np.savetxt(output_paths["pitch_points"], pitch_points, fmt="%.6f")
    output_paths["teams"].write_text(teams_text, encoding="utf-8")
    np.save(output_paths["x_world"], x_world)
    if "root_cam" in output_paths:
        np.save(output_paths["root_cam"], root_cam)
    if "diagnostics" in output_paths and diagnostics is not None:
        np.savez(output_paths["diagnostics"], **diagnostics)

    segments = infer_pitch_segments(
        pitch_points,
        max_segment_length=float(pitch_cfg.get("max_segment_length_m", 8.0)),
    )
    valid_tracks = np.isfinite(boxes).all(axis=-1).any(axis=1)
    valid_pose_ratio = float(np.isfinite(x_world).all(axis=-1).mean())
    print(
        f"{sequence}: N={num_players} T={num_frames} "
        f"active_box_tracks={int(valid_tracks.sum())}/{num_players}"
    )
    print(
        f"camera_keyframes={len(camera_track['keyframe_positions'])} "
        f"camera_center_first={camera_track['camera_center'][0].round(4).tolist()}"
    )
    print(
        f"pitch_mode={pitch_mode} pitch_points={len(pitch_points)} pitch_segments={len(segments)} "
        f"translation_method={translation_method} x_world={x_world.shape} finite={valid_pose_ratio:.3f}"
    )
    for name, path in output_paths.items():
        print(f"{name} -> {path}")
    return output_paths


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if len(archive.files) == 1:
            return archive[archive.files[0]]
        for key in (
            "arr_0",
            "xrel_sam",
            "x_2d",
            "keypoints_2d",
            "root_cam",
            "translations",
            "boxes",
            "data",
        ):
            if key in archive.files:
                return archive[key]
        raise ValueError(f"Could not choose an array inside {path}; keys={archive.files}")
    return np.load(path, allow_pickle=False)


def _required_path(values: dict[str, Any], key: str, base: Path) -> Path:
    value = values.get(key)
    if not value:
        raise ValueError(f"inputs.{key} is required")
    path = resolve_path(value, base)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _calibration_paths(value: Any, base: Path) -> list[Path]:
    raw_paths = value if isinstance(value, list) else [value]
    if not raw_paths or raw_paths == [None]:
        raise ValueError("inputs.calibration_json is required")
    paths = [resolve_path(path, base) for path in raw_paths]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing calibration JSON: {missing}")
    return paths


def _output_path(values: dict[str, Any], key: str, fallback: str, base: Path) -> Path:
    return resolve_path(values.get(key, fallback), base)


def _optional_extent(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    extent = tuple(float(item) for item in value)
    if len(extent) != 4:
        raise ValueError("pitch.floor_extent_m must contain [x_min, x_max, y_min, y_max]")
    return extent


def _check_outputs(outputs: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; use --overwrite:\n{joined}")


def _fictitious_teams(num_players: int, cfg: dict[str, Any]) -> str:
    strategy = str(cfg.get("strategy", "split")).lower()
    split_index = int(cfg.get("split_index", (num_players + 1) // 2))
    lines: list[str] = []
    for person_idx in range(num_players):
        if strategy == "alternating":
            label = "A" if person_idx % 2 == 0 else "B"
        elif strategy == "split":
            label = "A" if person_idx < split_index else "B"
        else:
            raise ValueError(f"Unsupported fictitious team strategy={strategy}")
        lines.append(f"{person_idx} {label} fictif_{person_idx:02d}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare cameras, pitch, teams and world poses from a calibration JSON."
    )
    parser.add_argument("--config", required=True, help="Preparation YAML file")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_sequence(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
