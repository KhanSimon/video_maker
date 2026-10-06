"""Discrete player representation changes at camera waypoint source frames."""
from __future__ import annotations

from typing import Any

from video_maker.io.mhr import mhr_enabled
from video_maker.render3d.drone_camera import travel_midpoint_schedule


def drone_mhr_schedule(
    motion: dict[str, Any], start: int, end: int, default: bool = False,
) -> list[tuple[int, bool]]:
    """A waypoint's MHR value applies from arrival until the next explicit value.

    Missing values inherit the previous state. Like the camera pose, end applies
    at end_frame - 1, and a one-frame clip uses only start.
    """
    midpoints = travel_midpoint_schedule(motion, start, end)
    for name in ("start", "end"):
        if not isinstance(motion.get(name, {}), dict):
            raise ValueError(f"camera_motion.{name} must be a mapping of camera settings")
    state = mhr_enabled(motion.get("start", {}), default, field="camera_motion.start.MHR")
    schedule = [(start, state)]
    for name, spec, frame, _ in midpoints:
        state = mhr_enabled(spec, state, field=f"camera_motion.{name}.MHR")
        schedule.append((frame, state))
    state = mhr_enabled(motion.get("end", {}), state, field="camera_motion.end.MHR")
    if end - start > 1:
        schedule.append((end - 1, state))
    return schedule


def mhr_at_frame(schedule: list[tuple[int, bool]], frame_idx: int) -> bool:
    for frame, enabled in reversed(schedule):
        if frame <= frame_idx:
            return enabled
    raise ValueError(f"Frame {frame_idx} precedes the player display schedule")
