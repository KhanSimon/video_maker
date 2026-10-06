from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def generate_calibrated_pitch_points(
    image_lines: dict[str, np.ndarray],
    k_matrix: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    camera_center_world: np.ndarray,
    ground_z_m: float = 0.0,
    floor_extent_m: tuple[float, float, float, float] | None = None,
    sample_spacing_m: float = 1.0,
    max_segment_length_m: float = 8.0,
) -> np.ndarray:
    """Back-project annotated image polylines onto the world ground plane."""
    sample_spacing_m = _positive("sample_spacing_m", sample_spacing_m)
    k_matrix = np.asarray(k_matrix, dtype=np.float64)
    rotation = np.asarray(rotation_world_to_camera, dtype=np.float64)
    camera_center = np.asarray(camera_center_world, dtype=np.float64)
    if k_matrix.shape != (3, 3) or rotation.shape != (3, 3) or camera_center.shape != (3,):
        raise ValueError(
            f"Invalid camera shapes K={k_matrix.shape} R={rotation.shape} C={camera_center.shape}"
        )

    inverse_k = np.linalg.inv(k_matrix)
    polylines: list[np.ndarray] = []
    for name, image_points in image_lines.items():
        image_points = np.asarray(image_points, dtype=np.float64)
        if image_points.ndim != 2 or image_points.shape[1] != 2 or len(image_points) < 2:
            raise ValueError(f"Pitch image line {name!r} must have shape (M,2), M >= 2")
        homogeneous = np.column_stack([image_points, np.ones(len(image_points))])
        directions_camera = (inverse_k @ homogeneous.T).T
        directions_world = directions_camera @ rotation
        vertical = directions_world[:, 2]
        if np.any(np.abs(vertical) <= 1e-9):
            raise ValueError(f"Pitch image line {name!r} contains a ray parallel to the ground")
        distance = (float(ground_z_m) - camera_center[2]) / vertical
        if np.any(distance <= 0.0):
            raise ValueError(f"Pitch image line {name!r} projects behind the camera")
        points_world = camera_center + distance[:, None] * directions_world
        points_world[:, 2] = float(ground_z_m)
        polylines.append(_sample_world_polyline(points_world, sample_spacing_m))

    if not polylines:
        raise ValueError("No calibrated pitch line was generated")
    measured_points = np.concatenate(polylines, axis=0)
    if floor_extent_m is None:
        bounds = (
            float(measured_points[:, 0].min()),
            float(measured_points[:, 0].max()),
            float(measured_points[:, 1].min()),
            float(measured_points[:, 1].max()),
        )
    else:
        bounds = _validate_floor_extent(floor_extent_m)
        x_min, x_max, y_min, y_max = bounds
        extent_corners = np.array(
            [
                [x_min, y_min, ground_z_m],
                [x_max, y_min, ground_z_m],
                [x_max, y_max, ground_z_m],
                [x_min, y_max, ground_z_m],
            ],
            dtype=np.float64,
        )
        polylines.extend(corner[None, :] for corner in extent_corners)

    return _join_with_inference_breaks(
        polylines,
        bounds=bounds,
        max_segment_length_m=max_segment_length_m,
    ).astype(np.float32)


def generate_standard_pitch_points(
    length_m: float = 105.0,
    width_m: float = 68.0,
    penalty_area_depth_m: float = 16.5,
    penalty_area_width_m: float = 40.32,
    goal_area_depth_m: float = 5.5,
    goal_area_width_m: float = 18.32,
    center_circle_radius_m: float = 9.15,
    penalty_spot_distance_m: float = 11.0,
    corner_arc_radius_m: float = 1.0,
    sample_spacing_m: float = 1.0,
    max_segment_length_m: float = 8.0,
) -> np.ndarray:
    """Build sampled 3D polylines for a regulation football pitch."""
    length_m = _positive("length_m", length_m)
    width_m = _positive("width_m", width_m)
    sample_spacing_m = _positive("sample_spacing_m", sample_spacing_m)
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    if penalty_area_depth_m >= half_length or goal_area_depth_m >= half_length:
        raise ValueError("Area depths must be smaller than half the pitch length")
    if penalty_area_width_m >= width_m or goal_area_width_m >= width_m:
        raise ValueError("Area widths must be smaller than the pitch width")

    polylines: list[np.ndarray] = []
    polylines.append(
        _sample_polyline(
            [
                (-half_length, -half_width),
                (half_length, -half_width),
                (half_length, half_width),
                (-half_length, half_width),
                (-half_length, -half_width),
            ],
            sample_spacing_m,
        )
    )
    polylines.append(_sample_segment((0.0, -half_width), (0.0, half_width), sample_spacing_m))
    polylines.append(
        _sample_arc(
            (0.0, 0.0),
            center_circle_radius_m,
            0.0,
            2.0 * np.pi,
            sample_spacing_m,
        )
    )

    for side in (-1.0, 1.0):
        goal_x = side * half_length
        penalty_x = goal_x - side * penalty_area_depth_m
        goal_area_x = goal_x - side * goal_area_depth_m
        penalty_half_width = penalty_area_width_m / 2.0
        goal_area_half_width = goal_area_width_m / 2.0
        polylines.append(
            _sample_polyline(
                [
                    (goal_x, -penalty_half_width),
                    (penalty_x, -penalty_half_width),
                    (penalty_x, penalty_half_width),
                    (goal_x, penalty_half_width),
                ],
                sample_spacing_m,
            )
        )
        polylines.append(
            _sample_polyline(
                [
                    (goal_x, -goal_area_half_width),
                    (goal_area_x, -goal_area_half_width),
                    (goal_area_x, goal_area_half_width),
                    (goal_x, goal_area_half_width),
                ],
                sample_spacing_m,
            )
        )

        penalty_spot_x = goal_x - side * penalty_spot_distance_m
        intersection_dx = penalty_area_depth_m - penalty_spot_distance_m
        if 0.0 < intersection_dx < center_circle_radius_m:
            half_angle = float(np.arccos(intersection_dx / center_circle_radius_m))
            center_angle = 0.0 if side < 0.0 else np.pi
            polylines.append(
                _sample_arc(
                    (penalty_spot_x, 0.0),
                    center_circle_radius_m,
                    center_angle - half_angle,
                    center_angle + half_angle,
                    sample_spacing_m,
                )
            )

    if corner_arc_radius_m > 0.0:
        corner_specs = (
            ((-half_length, -half_width), 0.0, np.pi / 2.0),
            ((half_length, -half_width), np.pi / 2.0, np.pi),
            ((half_length, half_width), np.pi, 3.0 * np.pi / 2.0),
            ((-half_length, half_width), 3.0 * np.pi / 2.0, 2.0 * np.pi),
        )
        for center, start_angle, end_angle in corner_specs:
            polylines.append(
                _sample_arc(
                    center,
                    corner_arc_radius_m,
                    start_angle,
                    end_angle,
                    sample_spacing_m,
                )
            )

    return _join_with_inference_breaks(
        polylines,
        bounds=(-half_length, half_length, -half_width, half_width),
        max_segment_length_m=max_segment_length_m,
    ).astype(np.float32)


def _sample_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    spacing: float,
) -> np.ndarray:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    count = max(2, int(np.ceil(np.linalg.norm(end_array - start_array) / spacing)) + 1)
    xy = np.linspace(start_array, end_array, count)
    return np.column_stack([xy, np.zeros(count, dtype=np.float64)])


def _sample_polyline(points: Iterable[tuple[float, float]], spacing: float) -> np.ndarray:
    vertices = list(points)
    sampled = [
        _sample_segment(vertices[index], vertices[index + 1], spacing)
        for index in range(len(vertices) - 1)
    ]
    return np.concatenate([sampled[0], *(part[1:] for part in sampled[1:])], axis=0)


def _sample_world_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / spacing)) + 1)
        segment = np.linspace(start, end, count)
        sampled.append(segment if index == 0 else segment[1:])
    return np.concatenate(sampled, axis=0)


def _sample_arc(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    spacing: float,
) -> np.ndarray:
    arc_length = abs(end_angle - start_angle) * radius
    count = max(3, int(np.ceil(arc_length / spacing)) + 1)
    angles = np.linspace(start_angle, end_angle, count)
    center_array = np.asarray(center, dtype=np.float64)
    xy = center_array + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.column_stack([xy, np.zeros(count, dtype=np.float64)])


def _join_with_inference_breaks(
    polylines: list[np.ndarray],
    bounds: tuple[float, float, float, float],
    max_segment_length_m: float,
) -> np.ndarray:
    if not polylines:
        return np.empty((0, 3), dtype=np.float64)
    x_min, x_max, y_min, y_max = bounds
    ground_z = float(polylines[0][0, 2])
    corners = np.array(
        [
            [x_min, y_min, ground_z],
            [x_max, y_min, ground_z],
            [x_max, y_max, ground_z],
            [x_min, y_max, ground_z],
        ],
        dtype=np.float64,
    )
    output = [polylines[0]]
    previous_end = polylines[0][-1]
    for polyline in polylines[1:]:
        next_start = polyline[0]
        if np.linalg.norm(next_start - previous_end) <= max_segment_length_m:
            distances = np.minimum(
                np.linalg.norm(corners - previous_end, axis=1),
                np.linalg.norm(corners - next_start, axis=1),
            )
            separator = corners[int(np.argmax(distances))]
            if float(np.max(distances)) <= max_segment_length_m:
                raise ValueError("Could not insert a safe pitch polyline separator")
            output.append(separator[None, :])
        output.append(polyline)
        previous_end = polyline[-1]
    return np.concatenate(output, axis=0)


def _validate_floor_extent(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    extent = np.asarray(values, dtype=np.float64)
    if extent.shape != (4,) or not np.isfinite(extent).all():
        raise ValueError("floor_extent_m must be [x_min, x_max, y_min, y_max]")
    x_min, x_max, y_min, y_max = (float(value) for value in extent)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Invalid floor_extent_m={values}")
    return x_min, x_max, y_min, y_max


def _positive(name: str, value: float) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return result
