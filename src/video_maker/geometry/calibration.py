from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationKeyframe:
    k_matrix: np.ndarray
    rotation_source_to_camera: np.ndarray
    camera_center_source: np.ndarray
    distortion: np.ndarray
    frame_index: float | None = None


def load_calibration_keyframes(paths: Iterable[str | Path]) -> list[CalibrationKeyframe]:
    keyframes: list[CalibrationKeyframe] = []
    for path in paths:
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in _calibration_records(payload):
            keyframes.append(_parse_keyframe(record, source))
    if not keyframes:
        raise ValueError("No camera calibration was found in the supplied JSON files")
    return keyframes


def load_calibration_input_lines(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for record in _calibration_records(payload):
        calibration = record.get("calibration", record)
        if not isinstance(calibration, dict):
            continue
        raw_lines = calibration.get("input_lines")
        if not isinstance(raw_lines, dict):
            continue
        lines: dict[str, np.ndarray] = {}
        for name, raw_points in raw_lines.items():
            if not isinstance(raw_points, list) or len(raw_points) < 2:
                continue
            points = np.asarray(
                [[point["x"], point["y"]] for point in raw_points],
                dtype=np.float64,
            )
            if points.shape != (len(raw_points), 2) or not np.isfinite(points).all():
                raise ValueError(f"Invalid input line {name!r} in {source}")
            lines[str(name)] = points
        if lines:
            return lines
    raise ValueError(f"No usable calibration.input_lines found in {source}")


def build_camera_track(
    keyframes: list[CalibrationKeyframe],
    num_frames: int,
    source_to_world_axis_signs: Iterable[float] = (1.0, -1.0, -1.0),
) -> dict[str, np.ndarray]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not keyframes:
        raise ValueError("At least one calibration keyframe is required")

    axis_signs = np.asarray(tuple(source_to_world_axis_signs), dtype=np.float64)
    if axis_signs.shape != (3,) or not np.all(np.isin(axis_signs, (-1.0, 1.0))):
        raise ValueError("source_to_world_axis_signs must contain three values equal to -1 or 1")
    source_from_world = np.diag(axis_signs)
    positions = _resolve_keyframe_positions(keyframes, num_frames)

    k_matrices = np.empty((num_frames, 3, 3), dtype=np.float64)
    rotations = np.empty((num_frames, 3, 3), dtype=np.float64)
    translations = np.empty((num_frames, 3), dtype=np.float64)
    distortions = np.empty((num_frames, 5), dtype=np.float64)
    camera_centers = np.empty((num_frames, 3), dtype=np.float64)

    for frame_idx in range(num_frames):
        left, right, alpha = _surrounding_keyframes(positions, frame_idx)
        first = keyframes[left]
        second = keyframes[right]
        k_matrix = _lerp(first.k_matrix, second.k_matrix, alpha)
        rotation_source = _interpolate_rotation(
            first.rotation_source_to_camera,
            second.rotation_source_to_camera,
            alpha,
        )
        center_source = _lerp(first.camera_center_source, second.camera_center_source, alpha)
        distortion = _lerp(first.distortion, second.distortion, alpha)

        center_world = center_source * axis_signs
        rotation_world_to_camera = rotation_source @ source_from_world
        translation_world_to_camera = -rotation_world_to_camera @ center_world

        k_matrices[frame_idx] = k_matrix
        rotations[frame_idx] = rotation_world_to_camera
        translations[frame_idx] = translation_world_to_camera
        distortions[frame_idx] = distortion
        camera_centers[frame_idx] = center_world

    rt = np.concatenate([rotations, translations[:, :, None]], axis=2)
    return {
        "K": k_matrices,
        "R": rotations,
        "t": translations,
        "k": distortions,
        "Rt": rt,
        "camera_center": camera_centers,
        "source_to_world_axis_signs": axis_signs,
        "keyframe_positions": positions,
    }


def camera_track_from_json(
    paths: Iterable[str | Path],
    num_frames: int,
    source_to_world_axis_signs: Iterable[float] = (1.0, -1.0, -1.0),
) -> dict[str, np.ndarray]:
    return build_camera_track(
        load_calibration_keyframes(paths),
        num_frames=num_frames,
        source_to_world_axis_signs=source_to_world_axis_signs,
    )


def _calibration_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_calibration_records(item))
        return records
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object or list, got {type(payload).__name__}")
    for key in ("frames", "calibrations", "camera_keyframes"):
        if isinstance(payload.get(key), list):
            return _calibration_records(payload[key])
    return [payload]


def _parse_keyframe(record: dict[str, Any], source: Path) -> CalibrationKeyframe:
    calibration = record.get("calibration", record)
    if not isinstance(calibration, dict):
        raise ValueError(f"Invalid calibration object in {source}")
    camera_parameters = calibration.get("camera_parameters", calibration)
    if not isinstance(camera_parameters, dict):
        raise ValueError(f"Invalid camera_parameters object in {source}")
    params = camera_parameters.get("cam_params", camera_parameters)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid cam_params object in {source}")

    fx = _required_float(params, "x_focal_length", source)
    fy = _required_float(params, "y_focal_length", source)
    principal = np.asarray(params.get("principal_point"), dtype=np.float64)
    if principal.shape != (2,):
        raise ValueError(f"principal_point must be [cx, cy] in {source}")
    k_matrix = np.array(
        [[fx, 0.0, principal[0]], [0.0, fy, principal[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    rotation = np.asarray(params.get("rotation_matrix"), dtype=np.float64)
    center = np.asarray(params.get("position_meters"), dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3,3) in {source}")
    if center.shape != (3,):
        raise ValueError(f"position_meters must have shape (3,) in {source}")
    rotation = _nearest_rotation(rotation, source)

    radial = np.asarray(params.get("radial_distortion", []), dtype=np.float64).reshape(-1)
    tangential = np.asarray(params.get("tangential_distortion", []), dtype=np.float64).reshape(-1)
    distortion = np.zeros(5, dtype=np.float64)
    if radial.size:
        distortion[0] = radial[0]
    if radial.size > 1:
        distortion[1] = radial[1]
    if tangential.size:
        distortion[2] = tangential[0]
    if tangential.size > 1:
        distortion[3] = tangential[1]
    if radial.size > 2:
        distortion[4] = radial[2]

    frame_index = _find_frame_index(record, calibration, camera_parameters, params)
    return CalibrationKeyframe(
        k_matrix=k_matrix,
        rotation_source_to_camera=rotation,
        camera_center_source=center,
        distortion=distortion,
        frame_index=frame_index,
    )


def _required_float(params: dict[str, Any], key: str, source: Path) -> float:
    if key not in params:
        raise KeyError(f"Missing {key} in {source}")
    return float(params[key])


def _find_frame_index(*objects: dict[str, Any]) -> float | None:
    for obj in objects:
        for key in ("frame_index", "frame_idx", "frame", "index"):
            value = obj.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                return float(value)
    return None


def _nearest_rotation(rotation: np.ndarray, source: Path) -> np.ndarray:
    if not np.isfinite(rotation).all():
        raise ValueError(f"rotation_matrix contains non-finite values in {source}")
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    if np.max(np.abs(projected - rotation)) > 1e-3:
        raise ValueError(f"rotation_matrix is not a valid rotation in {source}")
    return projected


def _resolve_keyframe_positions(
    keyframes: list[CalibrationKeyframe], num_frames: int
) -> np.ndarray:
    explicit = [keyframe.frame_index for keyframe in keyframes]
    if len(keyframes) == 1:
        return np.array([0.0], dtype=np.float64)
    if all(value is None for value in explicit):
        return np.linspace(0.0, float(num_frames - 1), len(keyframes), dtype=np.float64)
    if any(value is None for value in explicit):
        raise ValueError("Either all camera keyframes must define frame_index, or none of them")
    positions = np.asarray(explicit, dtype=np.float64)
    order = np.argsort(positions)
    positions = positions[order]
    keyframes[:] = [keyframes[int(index)] for index in order]
    if np.any(np.diff(positions) <= 0.0):
        raise ValueError(f"Camera keyframe positions must be strictly increasing, got {positions}")
    if positions[0] < 0.0 or positions[-1] > num_frames - 1:
        raise ValueError(
            f"Camera keyframe positions must be inside 0..{num_frames - 1}, got {positions}"
        )
    return positions


def _surrounding_keyframes(positions: np.ndarray, frame_idx: int) -> tuple[int, int, float]:
    if len(positions) == 1 or frame_idx <= positions[0]:
        return 0, 0, 0.0
    if frame_idx >= positions[-1]:
        last = len(positions) - 1
        return last, last, 0.0
    right = int(np.searchsorted(positions, frame_idx, side="right"))
    left = right - 1
    alpha = (float(frame_idx) - positions[left]) / (positions[right] - positions[left])
    return left, right, float(alpha)


def _interpolate_rotation(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0 or np.array_equal(first, second):
        return first
    if alpha >= 1.0:
        return second
    relative = second @ first.T
    rotation_vector, _ = cv2.Rodrigues(relative)
    increment, _ = cv2.Rodrigues(rotation_vector * float(alpha))
    return increment @ first


def _lerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) * first + float(alpha) * second
