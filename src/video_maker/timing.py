"""Video timing uses absolute, zero-based source frames and exclusive range ends."""
from __future__ import annotations

from numbers import Integral
from typing import Any


def frame_number(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer frame, got {value!r}")
    return int(value)


def source_frame_range(
    start: Any, end: Any, *, label: str, total_frames: int | None = None,
) -> tuple[int, int]:
    start = frame_number(start, f"{label}.start_frame")
    end = frame_number(end, f"{label}.end_frame")
    if start >= end or (total_frames is not None and end > total_frames):
        available = f" for {total_frames} source frames" if total_frames is not None else ""
        raise ValueError(f"Invalid {label} range [{start}, {end}){available}; end_frame is exclusive.")
    return start, end


def timeline_frame_ranges(timeline: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Validate concatenated ranges so no source frame is replayed or skipped."""
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("The full demo config needs a non-empty timeline.")
    ranges: list[tuple[int, int]] = []
    names: set[str] = set()
    for index, item in enumerate(timeline):
        name = str(item["name"])
        if name in names:
            raise ValueError(f"Duplicate timeline name {name!r} would overwrite and replay the same clip.")
        names.add(name)
        if "start" in item or "end" in item:
            raise ValueError(f"timeline.{name}: replace start/end with integer start_frame/end_frame.")
        if item.get("renderer", "broadcast") not in {"broadcast", "drone"}:
            raise ValueError(f"timeline.{name}.renderer must be broadcast or drone.")
        start, end = source_frame_range(item.get("start_frame"), item.get("end_frame"), label=f"timeline.{name}")
        if ranges and start != ranges[-1][1]:
            previous_end = ranges[-1][1]
            delta = start - previous_end
            problem = "overlap" if delta < 0 else "gap"
            effect = "replayed" if delta < 0 else "skipped"
            raise ValueError(
                f"Timeline {problem} between {timeline[index - 1]['name']!r} and {name!r}: "
                f"{abs(delta)} source frames would be {effect}. "
                f"Set {name!r}.start_frame to {previous_end}; end_frame is exclusive."
            )
        ranges.append((start, end))
    return ranges


def debug_frame_indices(cfg: dict[str, Any]) -> set[int]:
    return {
        frame_number(value, "render.debug_frames")
        for value in cfg.get("render", {}).get("debug_frames", [])
    }


def green_floor_start_frame(options: dict[str, Any], clip_start: int) -> int:
    if "green_floor_start_delay_s" in options:
        raise ValueError("Replace drone.green_floor_start_delay_s with green_floor_start_frame (source frame).")
    value = options.get("green_floor_start_frame")
    return clip_start if value is None else frame_number(value, "drone.green_floor_start_frame")


def validate_video_timing(cfg: dict[str, Any]) -> None:
    """Reject obsolete/ambiguous video timing without changing viewer configuration."""
    for key in ("fps", "start_frame", "num_frames"):
        if key in cfg.get("sequence", {}):
            replacement = "render.output_fps" if key == "fps" else "start_frame/end_frame in the timeline or renderer section"
            raise ValueError(f"Remove sequence.{key} from video configs; use {replacement}.")
    debug_frame_indices(cfg)
    green_floor_start_frame(cfg.get("drone", {}), 0)
    if cfg.get("timeline") is not None:
        timeline_frame_ranges(cfg["timeline"])
        for section in ("broadcast", "drone"):
            for key in ("start_frame", "end_frame"):
                if cfg.get(section, {}).get(key) is not None:
                    raise ValueError(f"Remove {section}.{key}: the timeline already defines the source frame ranges.")
