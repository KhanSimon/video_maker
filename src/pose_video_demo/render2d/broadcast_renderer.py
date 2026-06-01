from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from pose_video_demo.config import frame_range, resolve_config_paths
from pose_video_demo.io.loaders import SequenceData, describe_sequence, load_sequence
from pose_video_demo.io.video_io import make_writer, resize_if_needed, save_debug_png
from pose_video_demo.pitch.pitch_loader import load_pitch_points, parse_segments
from pose_video_demo.pitch.pitch_rendering import draw_projected_floor, draw_projected_pitch_lines
from pose_video_demo.render2d.overlays import draw_boxes, draw_stickmen_projected
from pose_video_demo.utils.colors import as_bgr, color_table
from pose_video_demo.utils.paths import ensure_parent, resolve_path

LOGGER = logging.getLogger(__name__)


class BroadcastRenderer:
    def __init__(self, cfg: dict[str, Any], data: SequenceData | None = None) -> None:
        self.cfg = cfg
        self.data = data or load_sequence(cfg)
        self.colors = color_table(cfg.get("render", {}).get("colors"))
        paths = resolve_config_paths(cfg)
        self.pitch_points = load_pitch_points(paths["pitch_points"])
        pitch_cfg = cfg.get("pitch", {})
        self.pitch_segments = parse_segments(
            pitch_cfg.get("segments"),
            self.pitch_points,
            float(pitch_cfg.get("max_segment_length", 8.0)),
        )

    def log_sanity(self) -> None:
        for line in describe_sequence(self.data):
            LOGGER.info(line)
        LOGGER.info("pitch_points=%s pitch_segments=%d", self.pitch_points.shape, len(self.pitch_segments))

    def render(self, overrides: dict[str, Any] | None = None) -> Path:
        self.log_sanity()
        options = dict(self.cfg.get("broadcast", {}))
        if overrides:
            options.update(overrides)
        start, end = self._range_from_options(options)
        render_cfg = self.cfg["render"]
        output_path = resolve_path(options["output_path"], self.cfg.get("_base_dir"))
        width = int(render_cfg.get("width") or self.data.video_info.width)
        height = int(render_cfg.get("height") or self.data.video_info.height)
        fps = float(render_cfg.get("output_fps") or self.data.video_info.fps)
        writer = make_writer(output_path, fps=fps, width=width, height=height)
        cap = cv2.VideoCapture(str(self.data.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.data.video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        debug_indices = set(int(v) for v in render_cfg.get("debug_frames", []))
        debug_dir = resolve_path(render_cfg.get("debug_dir", "outputs/frames"), self.cfg.get("_base_dir"))
        visible_pitch_on_debug = False
        visible_players_on_debug = False
        try:
            for local_idx, frame_idx in enumerate(tqdm(range(start, end), desc="broadcast")):
                frame = self._base_frame(cap, frame_idx, options, width, height)
                pitch_count, player_count = self._draw_overlays(frame, frame_idx, options)
                visible_pitch_on_debug |= pitch_count > 0
                visible_players_on_debug |= player_count > 0
                writer.write(frame)
                if local_idx in debug_indices:
                    save_debug_png(debug_dir / f"{self.data.name}_broadcast_{local_idx:04d}.png", frame)
        finally:
            writer.release()
            cap.release()
        if options.get("show_pitch_lines") and not visible_pitch_on_debug:
            LOGGER.warning("No projected pitch line was visible in the rendered range.")
        if options.get("show_stickmen") and not visible_players_on_debug:
            LOGGER.warning("No projected player stickman was visible in the rendered range.")
        LOGGER.info("wrote %s", output_path)
        return output_path

    def render_frame(self, frame_idx: int, options: dict[str, Any], cap: cv2.VideoCapture | None = None) -> np.ndarray:
        render_cfg = self.cfg["render"]
        width = int(render_cfg.get("width") or self.data.video_info.width)
        height = int(render_cfg.get("height") or self.data.video_info.height)
        own_cap = cap is None
        cap = cap or cv2.VideoCapture(str(self.data.video_path))
        if own_cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        frame = self._base_frame(cap, frame_idx, options, width, height)
        self._draw_overlays(frame, frame_idx, options)
        if own_cap:
            cap.release()
        return frame

    def _range_from_options(self, options: dict[str, Any]) -> tuple[int, int]:
        total = min(self.data.poses.x_cam.shape[1], self.data.video_info.frame_count)
        if options.get("start_frame") is None and options.get("end_frame") is None:
            return frame_range(self.cfg, "broadcast", total)
        start = int(options.get("start_frame") or 0)
        end = int(options.get("end_frame") or total)
        if not 0 <= start < end <= total:
            raise ValueError(f"Invalid broadcast range [{start}, {end}) for total={total}")
        return start, end

    def _base_frame(
        self,
        cap: cv2.VideoCapture,
        frame_idx: int,
        options: dict[str, Any],
        width: int,
        height: int,
    ) -> np.ndarray:
        bg_rgb = self.colors["background_white"]
        frame = np.full((height, width, 3), as_bgr(bg_rgb), dtype=np.uint8)
        if options.get("show_rgb", True) or options.get("rgb_alpha") is not None:
            ok, rgb_frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read video frame {frame_idx}")
            rgb_frame = resize_if_needed(rgb_frame, width, height)
            alpha = options.get("rgb_alpha")
            if alpha is None:
                frame = rgb_frame if options.get("show_rgb", True) else frame
            else:
                cv2.addWeighted(rgb_frame, float(alpha), frame, 1.0 - float(alpha), 0.0, frame)
        elif cap.get(cv2.CAP_PROP_POS_FRAMES) <= frame_idx:
            cap.grab()
        return frame

    def _draw_overlays(self, frame: np.ndarray, frame_idx: int, options: dict[str, Any]) -> tuple[int, int]:
        cam = self.data.cameras
        render_cfg = self.cfg["render"]
        k_matrix = self._scaled_k(frame_idx, frame.shape[1], frame.shape[0])
        floor_alpha = float(options.get("floor_alpha", render_cfg.get("floor_alpha", 0.55)))
        pitch_count = 0
        player_count = 0
        if options.get("show_green_floor"):
            draw_projected_floor(
                frame,
                self.pitch_points,
                k_matrix,
                cam.r[frame_idx],
                cam.t[frame_idx],
                cam.distortion[frame_idx],
                as_bgr(self.colors["floor_green"]),
                floor_alpha,
            )
        if options.get("show_pitch_lines"):
            pitch_count = draw_projected_pitch_lines(
                frame,
                self.pitch_points,
                self.pitch_segments,
                k_matrix,
                cam.r[frame_idx],
                cam.t[frame_idx],
                cam.distortion[frame_idx],
                as_bgr(self.colors["pitch_lines"]),
                int(render_cfg.get("pitch_line_width", 2)),
            )
        if options.get("show_boxes") and self.data.boxes is not None:
            boxes = self._scaled_boxes(self.data.boxes[:, frame_idx], frame.shape[1], frame.shape[0])
            draw_boxes(
                frame,
                boxes,
                self.data.teams,
                self.colors,
                int(render_cfg.get("bbox_line_width", 2)),
                alpha=float(options.get("boxes_alpha", 1.0)),
            )
        if options.get("show_stickmen"):
            player_count = draw_stickmen_projected(
                frame,
                self.data.poses.x_cam[:, frame_idx],
                k_matrix,
                cam.distortion[frame_idx],
                self.data.teams,
                self.colors,
                int(render_cfg.get("stickman_line_width", 2)),
                int(render_cfg.get("joint_diameter", 4)),
            )
        return pitch_count, player_count

    def _scaled_k(self, frame_idx: int, width: int, height: int) -> np.ndarray:
        k_matrix = self.data.cameras.k_matrix[frame_idx].copy()
        sx = width / float(self.data.video_info.width)
        sy = height / float(self.data.video_info.height)
        k_matrix[0, 0] *= sx
        k_matrix[0, 2] *= sx
        k_matrix[1, 1] *= sy
        k_matrix[1, 2] *= sy
        return k_matrix

    def _scaled_boxes(self, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        scaled = boxes.copy()
        sx = width / float(self.data.video_info.width)
        sy = height / float(self.data.video_info.height)
        scaled[:, [0, 2]] *= sx
        scaled[:, [1, 3]] *= sy
        return scaled
