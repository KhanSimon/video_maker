from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from video_maker.geometry.transforms import (
    camera_to_world_row_vector,
    reconstruct_x_cam,
)


@dataclass(frozen=True)
class GroundTranslationResult:
    root_cam: np.ndarray
    x_cam: np.ndarray
    x_world: np.ndarray
    root_world_raw: np.ndarray
    root_world_smoothed: np.ndarray
    ground_points_raw: np.ndarray
    contact_points_smoothed: np.ndarray
    contact_joint_indices: np.ndarray
    valid: np.ndarray


def estimate_ground_ray_translation(
    xrel_cam: np.ndarray,
    keypoints_2d: np.ndarray,
    k_matrices: np.ndarray,
    rotations_world_to_camera: np.ndarray,
    translations_world_to_camera: np.ndarray,
    distortion: np.ndarray | None = None,
    ground_z_m: float = 0.0,
    contact_selection: str = "world_z",
    relative_vertical_axis: str | int = "y",
    relative_ground_direction: str = "max",
    candidate_joint_indices: list[int] | tuple[int, ...] | None = None,
    smoothing_enabled: bool = True,
    smoothing_fps: float = 50.0,
    smoothing_median_window: int = 7,
    smoothing_cutoff_hz: float = 3.0,
    smoothing_max_speed_m_s: float | None = None,
    preserve_ground_contact: bool = True,
) -> GroundTranslationResult:
    xrel_cam = np.asarray(xrel_cam, dtype=np.float64)
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float64)
    k_matrices = np.asarray(k_matrices, dtype=np.float64)
    rotations = np.asarray(rotations_world_to_camera, dtype=np.float64)
    translations = np.asarray(translations_world_to_camera, dtype=np.float64)
    _validate_inputs(xrel_cam, keypoints_2d, k_matrices, rotations, translations)
    num_players, num_frames, num_joints = xrel_cam.shape[:3]

    if distortion is None:
        distortion = np.zeros((num_frames, 5), dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64)
    if distortion.ndim == 1:
        distortion = np.tile(distortion[None, :], (num_frames, 1))
    if distortion.shape[0] != num_frames:
        raise ValueError(f"Expected distortion T={num_frames}, got {distortion.shape}")

    candidate_indices = _candidate_indices(candidate_joint_indices, num_joints)
    joint_valid = np.isfinite(xrel_cam).all(axis=-1) & np.isfinite(keypoints_2d).all(axis=-1)
    contact_indices, valid = _select_contact_joints(
        xrel_cam,
        rotations,
        joint_valid,
        candidate_indices,
        contact_selection=contact_selection,
        relative_vertical_axis=relative_vertical_axis,
        relative_ground_direction=relative_ground_direction,
    )
    selected_relative = _take_joints(xrel_cam, contact_indices, valid)
    selected_pixels = _take_joints(keypoints_2d, contact_indices, valid)

    ground_points, ray_valid = _intersect_pixels_with_ground(
        selected_pixels,
        valid,
        k_matrices,
        rotations,
        translations,
        distortion,
        ground_z_m=float(ground_z_m),
    )
    valid &= ray_valid
    selected_relative[~valid] = np.nan
    ground_points[~valid] = np.nan

    relative_world = np.einsum("ntjd,tdc->ntjc", xrel_cam, rotations)
    selected_relative_world = _take_joints(relative_world, contact_indices, valid)
    root_world_raw = ground_points - selected_relative_world
    root_world_raw[~valid] = np.nan

    if smoothing_enabled:
        root_world_smoothed = smooth_track_positions(
            root_world_raw,
            valid,
            fps=float(smoothing_fps),
            median_window=int(smoothing_median_window),
            cutoff_hz=float(smoothing_cutoff_hz),
            max_speed_m_s=smoothing_max_speed_m_s,
        )
    else:
        root_world_smoothed = root_world_raw.copy()
    if preserve_ground_contact:
        root_world_smoothed[..., 2] = root_world_raw[..., 2]
    root_world_smoothed[~valid] = np.nan

    root_cam = np.einsum("ntd,tcd->ntc", root_world_smoothed, rotations) + translations[None, :, :]
    root_cam[~valid] = np.nan
    x_cam = reconstruct_x_cam(xrel_cam, root_cam)
    x_world = camera_to_world_row_vector(x_cam, rotations, translations)
    contact_points_smoothed = _take_joints(x_world, contact_indices, valid)
    return GroundTranslationResult(
        root_cam=root_cam,
        x_cam=x_cam,
        x_world=x_world,
        root_world_raw=root_world_raw,
        root_world_smoothed=root_world_smoothed,
        ground_points_raw=ground_points,
        contact_points_smoothed=contact_points_smoothed,
        contact_joint_indices=contact_indices,
        valid=valid,
    )


def smooth_track_positions(
    positions: np.ndarray,
    valid: np.ndarray,
    fps: float,
    median_window: int,
    cutoff_hz: float,
    max_speed_m_s: float | None = None,
) -> np.ndarray:
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if median_window <= 0 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    if not 0.0 < cutoff_hz < fps / 2.0:
        raise ValueError(f"cutoff_hz must be inside (0, {fps / 2.0}), got {cutoff_hz}")
    if max_speed_m_s is not None and max_speed_m_s <= 0.0:
        raise ValueError(f"max_speed_m_s must be positive, got {max_speed_m_s}")
    output = np.full_like(positions, np.nan, dtype=np.float64)
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / fps)
    radius = median_window // 2
    for person_idx in range(positions.shape[0]):
        valid_indices = np.flatnonzero(valid[person_idx])
        if not len(valid_indices):
            continue
        split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
        for indices in np.split(valid_indices, split_points):
            values = positions[person_idx, indices]
            padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
            median = np.stack(
                [np.median(padded[index : index + median_window], axis=0) for index in range(len(values))]
            )
            forward = median.copy()
            backward = median.copy()
            for index in range(1, len(values)):
                forward[index] = alpha * median[index] + (1.0 - alpha) * forward[index - 1]
            for index in range(len(values) - 2, -1, -1):
                backward[index] = alpha * median[index] + (1.0 - alpha) * backward[index + 1]
            filtered = (forward + backward) / 2.0
            if max_speed_m_s is not None:
                filtered = _limit_bidirectional_speed(
                    filtered,
                    max_step=float(max_speed_m_s) / fps,
                )
            output[person_idx, indices] = filtered
    return output


def _limit_bidirectional_speed(values: np.ndarray, max_step: float) -> np.ndarray:
    if len(values) <= 1:
        return values
    forward = values.copy()
    backward = values.copy()
    for index in range(1, len(values)):
        delta = forward[index] - forward[index - 1]
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            forward[index] = forward[index - 1] + delta * (max_step / distance)
    for index in range(len(values) - 2, -1, -1):
        delta = backward[index] - backward[index + 1]
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            backward[index] = backward[index + 1] + delta * (max_step / distance)
    return (forward + backward) / 2.0


def _select_contact_joints(
    xrel_cam: np.ndarray,
    rotations: np.ndarray,
    joint_valid: np.ndarray,
    candidates: np.ndarray,
    contact_selection: str,
    relative_vertical_axis: str | int,
    relative_ground_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_valid = joint_valid[:, :, candidates]
    person_valid = candidate_valid.any(axis=-1)
    mode = str(contact_selection).lower()
    if mode in {"world_z", "world_height", "world"}:
        relative_world = np.einsum("ntjd,tdc->ntjc", xrel_cam, rotations)
        scores = relative_world[:, :, candidates, 2]
        local_indices = np.where(candidate_valid, scores, np.inf).argmin(axis=-1)
    elif mode in {"relative_axis", "relative", "axis"}:
        axis = _axis_index(relative_vertical_axis)
        scores = xrel_cam[:, :, candidates, axis]
        direction = str(relative_ground_direction).lower()
        if direction == "max":
            local_indices = np.where(candidate_valid, scores, -np.inf).argmax(axis=-1)
        elif direction == "min":
            local_indices = np.where(candidate_valid, scores, np.inf).argmin(axis=-1)
        else:
            raise ValueError("relative_ground_direction must be 'min' or 'max'")
    else:
        raise ValueError(f"Unsupported contact_selection={contact_selection}")
    contact_indices = candidates[local_indices].astype(np.int16)
    contact_indices[~person_valid] = -1
    return contact_indices, person_valid


def _intersect_pixels_with_ground(
    pixels: np.ndarray,
    valid: np.ndarray,
    k_matrices: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    distortion: np.ndarray,
    ground_z_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    ground = np.full(pixels.shape[:2] + (3,), np.nan, dtype=np.float64)
    ray_valid = np.zeros(pixels.shape[:2], dtype=bool)
    camera_centers = -np.einsum("tc,tcd->td", translations, rotations)
    for frame_idx in range(pixels.shape[1]):
        person_indices = np.flatnonzero(valid[:, frame_idx])
        if not len(person_indices):
            continue
        frame_pixels = pixels[person_indices, frame_idx]
        normalized = cv2.undistortPoints(
            frame_pixels.reshape(-1, 1, 2),
            k_matrices[frame_idx],
            distortion[frame_idx],
        ).reshape(-1, 2)
        directions_camera = np.column_stack([normalized, np.ones(len(normalized))])
        directions_world = directions_camera @ rotations[frame_idx]
        denominator = directions_world[:, 2]
        usable = np.abs(denominator) > 1e-9
        distance = np.full(len(person_indices), np.nan, dtype=np.float64)
        distance[usable] = (
            ground_z_m - camera_centers[frame_idx, 2]
        ) / denominator[usable]
        usable &= distance > 0.0
        points = camera_centers[frame_idx] + distance[:, None] * directions_world
        points[:, 2] = ground_z_m
        accepted = person_indices[usable]
        ground[accepted, frame_idx] = points[usable]
        ray_valid[accepted, frame_idx] = True
    return ground, ray_valid


def _take_joints(data: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe_indices = np.maximum(indices, 0)
    selected = np.take_along_axis(data, safe_indices[..., None, None], axis=2).squeeze(2).copy()
    selected[~valid] = np.nan
    return selected


def _candidate_indices(values: list[int] | tuple[int, ...] | None, num_joints: int) -> np.ndarray:
    candidates = np.arange(num_joints, dtype=np.int64) if values is None else np.asarray(values, dtype=np.int64)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("candidate_joint_indices must be a non-empty list")
    if np.any(candidates < 0) or np.any(candidates >= num_joints):
        raise ValueError(f"candidate_joint_indices must be inside 0..{num_joints - 1}")
    return np.unique(candidates)


def _axis_index(value: str | int) -> int:
    if isinstance(value, str):
        mapping = {"x": 0, "y": 1, "z": 2}
        if value.lower() not in mapping:
            raise ValueError(f"Unsupported relative_vertical_axis={value}")
        return mapping[value.lower()]
    index = int(value)
    if index not in (0, 1, 2):
        raise ValueError(f"relative_vertical_axis must be x/y/z or 0/1/2, got {value}")
    return index


def _validate_inputs(
    xrel_cam: np.ndarray,
    keypoints_2d: np.ndarray,
    k_matrices: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> None:
    if xrel_cam.ndim != 4 or xrel_cam.shape[2:] != (25, 3):
        raise ValueError(f"Expected xrel_cam shape (N,T,25,3), got {xrel_cam.shape}")
    if keypoints_2d.shape != xrel_cam.shape[:3] + (2,):
        raise ValueError(f"Expected keypoints_2d shape {xrel_cam.shape[:3] + (2,)}, got {keypoints_2d.shape}")
    num_frames = xrel_cam.shape[1]
    if k_matrices.shape != (num_frames, 3, 3):
        raise ValueError(f"Expected K shape {(num_frames, 3, 3)}, got {k_matrices.shape}")
    if rotations.shape != (num_frames, 3, 3) or translations.shape != (num_frames, 3):
        raise ValueError(f"Invalid camera shapes R={rotations.shape} t={translations.shape}")
