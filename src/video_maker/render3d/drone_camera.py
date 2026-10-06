from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from video_maker.geometry.camera import camera_center_world, camera_forward_world
from video_maker.timing import frame_number, source_frame_range


@dataclass(frozen=True)
class _TravelStop:
    frame: int
    last_hold_frame: int
    position: np.ndarray
    look_at: np.ndarray
    fov: float


def players_center(x_world_frame: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x_world_frame).all(axis=-1)
    if valid.sum() == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    points = x_world_frame[valid]
    center = np.nanmedian(points, axis=0).astype(np.float64)
    center[2] = float(np.nanmedian(points[:, 2]))
    return center


def spherical_to_world(
    radius_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    origin: np.ndarray,
) -> np.ndarray:
    """Convert spherical coordinates to world XYZ meters.

    Azimuth is measured in the XY plane from +X toward +Y. Elevation is measured
    above the XY plane toward +Z.
    """
    radius = float(radius_m)
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    xy_radius = radius * np.cos(elevation)
    return origin + np.array(
        [
            xy_radius * np.cos(azimuth),
            xy_radius * np.sin(azimuth),
            radius * np.sin(elevation),
        ],
        dtype=np.float64,
    )


def resolve_world_point(
    spec: dict[str, Any] | list[float] | tuple[float, float, float] | None,
    x_world_frame: np.ndarray,
    r_broadcast: np.ndarray,
    t_broadcast: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    """Resolve a YAML point spec to world XYZ coordinates."""
    if spec is None:
        return default.astype(np.float64)
    if isinstance(spec, (list, tuple)):
        return _xyz(spec)
    mode = str(spec.get("mode", spec.get("type", "cartesian"))).lower()
    if mode in {"cartesian", "xyz", "world"}:
        return _xyz(spec.get("xyz", spec.get("value", default)))
    if mode in {"spherical", "polar"}:
        origin = resolve_origin(spec.get("origin", "world_origin"), x_world_frame, r_broadcast, t_broadcast)
        offset = _xyz(spec.get("origin_offset", [0.0, 0.0, 0.0]))
        return spherical_to_world(
            radius_m=float(spec["radius_m"]),
            azimuth_deg=float(spec["azimuth_deg"]),
            elevation_deg=float(spec["elevation_deg"]),
            origin=origin + offset,
        )
    if mode in {"target", "anchor"}:
        origin = resolve_origin(spec.get("target", spec.get("origin", "players_center")), x_world_frame, r_broadcast, t_broadcast)
        return origin + _xyz(spec.get("offset", [0.0, 0.0, 0.0]))
    raise ValueError(f"Unsupported drone point mode: {mode}")


def resolve_origin(
    raw: str | list[float] | tuple[float, float, float],
    x_world_frame: np.ndarray,
    r_broadcast: np.ndarray,
    t_broadcast: np.ndarray,
) -> np.ndarray:
    if isinstance(raw, (list, tuple)):
        return _xyz(raw)
    name = str(raw).lower()
    if name in {"world_origin", "origin", "zero"}:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if name in {"field_center", "pitch_center"}:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if name in {"players_center", "players", "team_center"}:
        return players_center(x_world_frame)
    if name in {"broadcast_camera", "real_camera"}:
        return camera_center_world(r_broadcast, t_broadcast)
    raise ValueError(f"Unsupported drone point origin: {raw}")


def camera_pose_for_frame(
    motion: dict[str, Any],
    frame_idx: int,
    x_world_frame: np.ndarray,
    r_broadcast: np.ndarray,
    t_broadcast: np.ndarray,
    *,
    clip_start_frame: int,
    clip_end_frame: int,
    fov_default: float = 45.0,
    start_fov_default: float | None = None,
    end_fov_default: float | None = None,
    start_x_world_frame: np.ndarray | None = None,
    end_x_world_frame: np.ndarray | None = None,
    start_r_broadcast: np.ndarray | None = None,
    end_r_broadcast: np.ndarray | None = None,
    start_t_broadcast: np.ndarray | None = None,
    end_t_broadcast: np.ndarray | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    """Evaluate the camera at an absolute source frame, independent of output FPS."""
    source_frame_range(clip_start_frame, clip_end_frame, label="drone")
    frame_idx = frame_number(frame_idx, "camera frame_idx")
    if not clip_start_frame <= frame_idx < clip_end_frame:
        raise ValueError(f"Camera frame {frame_idx} is outside [{clip_start_frame}, {clip_end_frame}).")
    motion_type = str(motion.get("type", "travel")).lower()
    fov = float(motion.get("fov_deg", fov_default))
    if _has_explicit_travel(motion):
        return explicit_travel_pose(
            motion,
            frame_idx,
            fov,
            start_fov_default if start_fov_default is not None else fov_default,
            end_fov_default if end_fov_default is not None else fov_default,
            start_x_world_frame if start_x_world_frame is not None else x_world_frame,
            end_x_world_frame if end_x_world_frame is not None else x_world_frame,
            start_r_broadcast if start_r_broadcast is not None else r_broadcast,
            end_r_broadcast if end_r_broadcast is not None else r_broadcast,
            start_t_broadcast if start_t_broadcast is not None else t_broadcast,
            end_t_broadcast if end_t_broadcast is not None else t_broadcast,
            clip_start_frame,
            clip_end_frame,
        )
    if motion_type == "static_broadcast_like":
        return static_broadcast_like_pose(motion, x_world_frame, r_broadcast, t_broadcast, fov)
    alpha = (frame_idx - clip_start_frame) / max(1, clip_end_frame - clip_start_frame - 1)
    return legacy_orbit_pose(motion, alpha, x_world_frame, fov)


def explicit_travel_pose(
    motion: dict[str, Any],
    frame_idx: int,
    fov: float,
    start_fov_default: float,
    end_fov_default: float,
    start_x_world_frame: np.ndarray,
    end_x_world_frame: np.ndarray,
    start_r_broadcast: np.ndarray,
    end_r_broadcast: np.ndarray,
    start_t_broadcast: np.ndarray,
    end_t_broadcast: np.ndarray,
    clip_start_frame: int,
    clip_end_frame: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    start = motion.get("start", {})
    end = motion.get("end", start)
    midpoints = travel_midpoint_schedule(motion, clip_start_frame, clip_end_frame)
    start_target_default = players_center(start_x_world_frame) + np.array([0.0, 0.0, 1.0])
    start_position_default = start_target_default + np.array([-45.0, -35.0, 25.0])
    start_position = resolve_world_point(
        start.get("position"),
        start_x_world_frame,
        start_r_broadcast,
        start_t_broadcast,
        start_position_default,
    )
    end_position = resolve_world_point(
        end.get("position"),
        end_x_world_frame,
        end_r_broadcast,
        end_t_broadcast,
        start_position,
    )
    look_at_distance = float(motion.get("look_at_distance_m", 100.0))
    start_look_at_default = broadcast_orientation_target(start_position, start_r_broadcast, look_at_distance)
    end_look_at_default = broadcast_orientation_target(end_position, end_r_broadcast, look_at_distance)
    start_look_at = resolve_world_point(
        start.get("look_at"),
        start_x_world_frame,
        start_r_broadcast,
        start_t_broadcast,
        start_look_at_default,
    )
    end_look_at = resolve_world_point(
        end.get("look_at"),
        end_x_world_frame,
        end_r_broadcast,
        end_t_broadcast,
        end_look_at_default,
    )
    smooth = bool(motion.get("smoothstep", True))
    start_fov = resolve_zoom_fov(start.get("zoom"), start_fov_default)
    end_fov = resolve_zoom_fov(end.get("zoom"), end_fov_default)
    stops = [_TravelStop(clip_start_frame, clip_start_frame, start_position, start_look_at, start_fov)]
    for _, mid, frame, last_hold_frame in midpoints:
        mid_anchor = str(mid.get("anchor_frame", "end")).lower()
        if mid_anchor in {"start", "begin", "first"}:
            mid_x_world_frame = start_x_world_frame
            mid_r_broadcast = start_r_broadcast
            mid_t_broadcast = start_t_broadcast
        else:
            mid_x_world_frame = end_x_world_frame
            mid_r_broadcast = end_r_broadcast
            mid_t_broadcast = end_t_broadcast
        alpha = (frame - clip_start_frame) / max(1, clip_end_frame - clip_start_frame - 1)
        mid_position_default = _lerp(start_position, end_position, alpha)
        mid_look_at_default = _lerp(start_look_at, end_look_at, alpha)
        mid_fov_default = float((1.0 - alpha) * start_fov + alpha * end_fov)
        mid_position = resolve_world_point(
            mid.get("position"),
            mid_x_world_frame,
            mid_r_broadcast,
            mid_t_broadcast,
            mid_position_default,
        )
        mid_look_at = resolve_world_point(
            mid.get("look_at"),
            mid_x_world_frame,
            mid_r_broadcast,
            mid_t_broadcast,
            mid_look_at_default,
        )
        mid_fov = resolve_zoom_fov(mid.get("zoom"), mid_fov_default)
        stops.append(_TravelStop(frame, last_hold_frame, mid_position, mid_look_at, mid_fov))
    if clip_end_frame - clip_start_frame > 1:
        stops.append(_TravelStop(clip_end_frame - 1, clip_end_frame - 1, end_position, end_look_at, end_fov))
    position, target, fov_from_zoom = _interpolate_travel_stops(frame_idx, stops, smooth)
    return _as_tuple3(position), _as_tuple3(target), (0.0, 0.0, 1.0), fov_from_zoom if np.isfinite(fov_from_zoom) else fov


def travel_midpoint_schedule(
    motion: dict[str, Any], clip_start_frame: int, clip_end_frame: int,
) -> list[tuple[str, dict[str, Any], int, int]]:
    """Return (name, spec, arrival frame, last held frame), in numeric mid order.

    hold_frames includes the arrival frame. Zero adds no pause; the camera still
    passes through the specified pose on its arrival frame.
    """
    source_frame_range(clip_start_frame, clip_end_frame, label="drone")
    legacy = [(name, motion[name]) for name in ("mid", "middle", "waypoint") if motion.get(name)]
    numbered = []
    for name in motion:
        match = re.fullmatch(r"mid([0-9]+)", str(name))
        if match:
            number = int(match[1])
            if number < 1 or str(name) != f"mid{number}":
                raise ValueError(f"Invalid camera midpoint {name!r}; use mid1, mid2, mid3, ...")
            numbered.append((number, name, motion[name]))
    if len(legacy) > 1 or (legacy and numbered):
        raise ValueError("Use either one legacy mid/middle/waypoint or numbered mid1, mid2, ...; rename mid to mid1.")
    specs = [(name, spec) for _, name, spec in sorted(numbered)] if numbered else legacy
    schedule = []
    previous_depart = clip_start_frame
    previous_name = "start"
    for name, spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(f"camera_motion.{name} must be a mapping of camera settings")
        obsolete = {"arrive_at", "at", "progress", "depart_at", "leave_at", "hold_duration_s", "hold_duration_progress", "hold_progress"}
        if obsolete.intersection(spec):
            raise ValueError(f"camera_motion.{name}: replace {', '.join(sorted(obsolete.intersection(spec)))} with frame and hold_frames.")
        frame = frame_number(spec.get("frame"), f"camera_motion.{name}.frame")
        hold_frames = frame_number(spec.get("hold_frames", 0), f"camera_motion.{name}.hold_frames")
        last_hold_frame = frame + max(hold_frames - 1, 0)
        if frame <= previous_depart:
            raise ValueError(
                f"camera_motion.{name}.frame={frame} must be after "
                f"{previous_name}'s last held frame {previous_depart}; leave frames for camera travel."
            )
        if last_hold_frame >= clip_end_frame - 1:
            raise ValueError(
                f"camera_motion.{name}: arrival and hold must finish before the end camera at frame {clip_end_frame - 1}."
            )
        schedule.append((name, spec, frame, last_hold_frame))
        previous_depart = last_hold_frame
        previous_name = name
    return schedule


def _interpolate_travel_stops(
    frame_idx: int, stops: list[_TravelStop], smooth: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    previous = stops[0]
    for stop in stops[1:]:
        if frame_idx < stop.frame:
            alpha = (frame_idx - previous.last_hold_frame) / (stop.frame - previous.last_hold_frame)
            alpha = _smoothstep(alpha) if smooth else alpha
            return (
                _lerp(previous.position, stop.position, alpha),
                _lerp(previous.look_at, stop.look_at, alpha),
                float((1.0 - alpha) * previous.fov + alpha * stop.fov),
            )
        if frame_idx <= stop.last_hold_frame:
            return stop.position, stop.look_at, stop.fov
        previous = stop
    return previous.position, previous.look_at, previous.fov


def broadcast_orientation_target(position: np.ndarray, r_broadcast: np.ndarray, distance_m: float) -> np.ndarray:
    """Return a look-at target that preserves the broadcast optical direction."""
    forward = camera_forward_world(r_broadcast)
    norm = np.linalg.norm(forward)
    if norm <= 1e-9 or not np.isfinite(forward).all():
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward = forward / norm
    return position + forward * float(distance_m)


def broadcast_vertical_fov_deg(k_matrix: np.ndarray, image_height: int) -> float:
    """Convert camera intrinsics to PyVista's vertical view angle in degrees."""
    fy = float(k_matrix[1, 1])
    if fy <= 1e-9:
        return 45.0
    return float(np.rad2deg(2.0 * np.arctan(float(image_height) / (2.0 * fy))))


def resolve_zoom_fov(spec: Any, broadcast_fov_deg: float) -> float:
    """Resolve a zoom spec to a vertical FOV in degrees.

    Higher zoom factors narrow the FOV. Absolute FOV remains available through
    mode=fov/fov_deg for cases where direct view-angle control is easier.
    """
    if spec is None:
        return float(broadcast_fov_deg)
    if isinstance(spec, str):
        if spec.lower() in {"auto", "broadcast", "broadcast_camera", "camera"}:
            return float(broadcast_fov_deg)
        return float(spec)
    if isinstance(spec, (int, float)):
        factor = max(float(spec), 1e-6)
        return zoom_factor_to_fov(broadcast_fov_deg, factor)
    mode = str(spec.get("mode", "broadcast_camera")).lower()
    if mode in {"broadcast", "broadcast_camera", "camera", "auto"}:
        factor = float(spec.get("factor", spec.get("zoom_factor", 1.0)))
        return zoom_factor_to_fov(broadcast_fov_deg, factor)
    if mode in {"factor", "zoom_factor", "relative"}:
        return zoom_factor_to_fov(broadcast_fov_deg, float(spec["factor"]))
    if mode in {"fov", "fov_deg", "view_angle"}:
        return float(spec.get("fov_deg", spec.get("value")))
    raise ValueError(f"Unsupported drone zoom mode: {mode}")


def zoom_factor_to_fov(base_fov_deg: float, factor: float) -> float:
    factor = max(float(factor), 1e-6)
    base_half = np.deg2rad(float(base_fov_deg)) / 2.0
    return float(np.rad2deg(2.0 * np.arctan(np.tan(base_half) / factor)))


def static_broadcast_like_pose(
    motion: dict[str, Any],
    x_world_frame: np.ndarray,
    r_broadcast: np.ndarray,
    t_broadcast: np.ndarray,
    fov: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    target = players_center(x_world_frame)
    target[2] = float(motion.get("look_at_height", 1.0))
    position = camera_center_world(r_broadcast, t_broadcast)
    if not np.isfinite(position).all() or np.linalg.norm(position - target) < 1.0:
        position = target + np.array([-45.0, -35.0, 25.0])
    return _as_tuple3(position), _as_tuple3(target), (0.0, 0.0, 1.0), fov


def legacy_orbit_pose(
    motion: dict[str, Any],
    progress: float,
    x_world_frame: np.ndarray,
    fov: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    target = players_center(x_world_frame)
    target[2] = float(motion.get("look_at_height", 1.0))
    motion_type = str(motion.get("type", "spiral_descend")).lower()
    radius_start = float(motion.get("radius_start", 45.0))
    radius_end = float(motion.get("radius_end", radius_start))
    height_start = float(motion.get("height_start", 25.0))
    height_end = float(motion.get("height_end", height_start))
    if motion_type == "arc_shot":
        height_end = height_start
    angle_start = np.deg2rad(float(motion.get("angle_start_deg", 0.0)))
    angle_end = np.deg2rad(float(motion.get("angle_end_deg", 160.0)))
    angle = angle_start + (angle_end - angle_start) * progress
    radius = radius_start + (radius_end - radius_start) * progress
    height = height_start + (height_end - height_start) * progress
    position = target + np.array([radius * np.cos(angle), radius * np.sin(angle), height])
    return _as_tuple3(position), _as_tuple3(target), (0.0, 0.0, 1.0), fov


def _has_explicit_travel(motion: dict[str, Any]) -> bool:
    return (
        "start" in motion or "end" in motion
        or str(motion.get("type", "")).lower() in {"travel", "travelling", "linear"}
        or any(name in {"mid", "middle", "waypoint"} or re.fullmatch(r"mid[0-9]+", str(name)) for name in motion)
    )


def _xyz(raw: Any) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected XYZ with 3 values, got {raw}")
    return values


def _lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start * (1.0 - alpha) + end * alpha


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _as_tuple3(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])
