from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from video_maker.io.loaders import load_sequence
from video_maker.io.mhr import MHRData, renderer_mhr_enabled
from video_maker.io.video_io import make_writer, resize_if_needed, save_debug_png
from video_maker.render2d.broadcast_renderer import BroadcastRenderer
from video_maker.render3d.pyvista_renderer import PyVistaDroneRenderer
from video_maker.render3d.player_display import drone_mhr_schedule
from video_maker.timing import debug_frame_indices, source_frame_range, timeline_frame_ranges, validate_video_timing
from video_maker.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def assemble_clips(clips: list[Path], output_path: Path, fps: float, width: int, height: int) -> Path:
    writer = make_writer(output_path, fps=fps, width=width, height=height)
    try:
        for clip in clips:
            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open clip: {clip}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in tqdm(range(total), desc=f"assemble {clip.name}"):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(resize_if_needed(frame, width, height))
            cap.release()
    finally:
        writer.release()
    LOGGER.info("wrote %s", output_path)
    return output_path


def render_full_demo(cfg: dict[str, Any]) -> Path:
    validate_video_timing(cfg)
    timeline = cfg.get("timeline") or []
    ranges = timeline_frame_ranges(timeline)
    # Validate every camera schedule before generating even the first clip.
    needs_mhr = False
    for item, (start, end) in zip(timeline, ranges):
        section = item.get("renderer", "broadcast")
        enabled = renderer_mhr_enabled(cfg, section, item)
        if section == "drone":
            schedule = drone_mhr_schedule(cfg.get("drone", {}).get("camera_motion", {}), start, end, enabled)
            enabled = any(value for _, value in schedule)
        needs_mhr |= bool(item.get("show_stickmen", section == "drone")) and enabled
    data = load_sequence(cfg)
    for item, (start, end) in zip(timeline, ranges):
        total = data.poses.x_world.shape[1]
        if item.get("renderer") != "drone":
            total = min(total, data.video_info.frame_count)
        source_frame_range(start, end, label=f"timeline.{item['name']}", total_frames=total)
    fps = float(cfg["render"].get("output_fps") or data.video_info.fps)
    width = int(cfg["render"].get("width") or data.video_info.width)
    height = int(cfg["render"].get("height") or data.video_info.height)
    clips_dir = resolve_path("outputs/clips", cfg.get("_base_dir"))
    mhr_data = MHRData(cfg, data) if needs_mhr else None
    broadcast = BroadcastRenderer(cfg, data=data, mhr_data=mhr_data)
    clips: list[Path] = []
    for item, (start, end) in zip(timeline, ranges):
        name = str(item["name"])
        LOGGER.info("timeline %s: source frames [%d, %d), %d frames", name, start, end, end - start)
        if item.get("renderer") == "drone":
            drone = PyVistaDroneRenderer(cfg, data=data, mhr_data=mhr_data)
            clip = drone.render(
                {
                    **{key: item[key] for key in ("MHR", "show_stickmen", "show_pitch_lines", "show_green_floor") if key in item},
                    "start_frame": start,
                    "end_frame": end,
                    "output_path": str(clips_dir / f"clip_{name}.mp4"),
                }
            )
        else:
            options = _broadcast_options_for_timeline_item(item, start, end, clips_dir / f"clip_{name}.mp4")
            if item.get("transition"):
                clip = _render_transition_clip(broadcast, item, start, end, options, fps, width, height)
            else:
                clip = broadcast.render(options)
        clips.append(clip)
    final_path = resolve_path(cfg.get("output_path", "outputs/final/final_demo.mp4"), cfg.get("_base_dir"))
    return assemble_clips(clips, final_path, fps=fps, width=width, height=height)


def _broadcast_options_for_timeline_item(
    item: dict[str, Any],
    start: int,
    end: int,
    output_path: Path,
) -> dict[str, Any]:
    return {
        **({"MHR": item["MHR"]} if "MHR" in item else {}),
        "start_frame": start,
        "end_frame": end,
        "output_path": str(output_path),
        "show_rgb": bool(item.get("show_rgb", False)),
        "show_pitch_lines": bool(item.get("show_pitch_lines", False)),
        "show_boxes": bool(item.get("show_boxes", False)),
        "show_stickmen": bool(item.get("show_stickmen", False)),
        "show_green_floor": bool(item.get("show_green_floor", False)),
    }


def _render_transition_clip(
    renderer: BroadcastRenderer,
    item: dict[str, Any],
    start: int,
    end: int,
    options: dict[str, Any],
    fps: float,
    width: int,
    height: int,
) -> Path:
    output_path = Path(options["output_path"])
    writer = make_writer(output_path, fps=fps, width=width, height=height)
    cap = cv2.VideoCapture(str(renderer.data.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {renderer.data.video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    transition = item.get("transition", {})
    debug_indices = debug_frame_indices(renderer.cfg)
    debug_dir = resolve_path(renderer.cfg["render"].get("debug_dir", "outputs/frames"), renderer.cfg.get("_base_dir"))
    try:
        for frame_idx in tqdm(range(start, end), desc=f"transition {item['name']}"):
            alpha = (frame_idx - start) / max(1, end - start - 1)
            opts = dict(options)
            opts["show_rgb"] = bool(transition.get("rgb_to_white", True))
            opts["rgb_alpha"] = 1.0 - alpha if transition.get("rgb_to_white", False) else None
            opts["show_boxes"] = bool(transition.get("boxes_fade_out", False))
            opts["boxes_alpha"] = 1.0 - alpha if transition.get("boxes_fade_out", False) else 1.0
            opts["show_green_floor"] = bool(transition.get("green_floor_fade_in", False))
            opts["floor_alpha"] = alpha * float(renderer.cfg["render"].get("floor_alpha", 0.55))
            frame = renderer.render_frame(frame_idx, opts, cap=cap)
            frame = resize_if_needed(frame, width, height)
            writer.write(frame)
            if frame_idx in debug_indices:
                save_debug_png(debug_dir / f"{renderer.data.name}_broadcast_{frame_idx:04d}.png", frame)
    finally:
        writer.release()
        cap.release()
        renderer.close()
    LOGGER.info("wrote %s", output_path)
    return output_path
