from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pose_video_demo_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pose_video_demo_cache")

import pyvista as pv
from tqdm import tqdm

from pose_video_demo.config import frame_range, resolve_config_paths
from pose_video_demo.io.loaders import SequenceData, describe_sequence, load_sequence
from pose_video_demo.io.video_io import make_writer, save_debug_png
from pose_video_demo.pitch.pitch_loader import load_pitch_points, parse_segments
from pose_video_demo.render3d.drone_camera import broadcast_vertical_fov_deg, camera_pose_for_frame
from pose_video_demo.render3d.scene import (
    make_floor,
    make_joint_points,
    make_pitch_lines,
    make_skeleton_lines,
)
from pose_video_demo.utils.colors import color_table
from pose_video_demo.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


class PyVistaDroneRenderer:
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

    def render(self, overrides: dict[str, Any] | None = None) -> Path:
        for line in describe_sequence(self.data):
            LOGGER.info(line)
        options = dict(self.cfg.get("drone", {}))
        if overrides:
            options.update(overrides)
        start, end = self._range_from_options(options)
        width = int(options.get("width") or self.cfg["render"].get("width") or 1920)
        height = int(options.get("height") or self.cfg["render"].get("height") or 1080)
        fps = float(self.cfg["render"].get("output_fps") or self.data.video_info.fps)
        output_path = resolve_path(options["output_path"], self.cfg.get("_base_dir"))
        writer = make_writer(output_path, fps=fps, width=width, height=height)
        clip_duration_s = (end - start) / fps
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.set_background("white")
        if hasattr(plotter, "disable_parallel_projection"):
            plotter.disable_parallel_projection()
        debug_indices = set(int(v) for v in self.cfg["render"].get("debug_frames", []))
        debug_dir = resolve_path(self.cfg["render"].get("debug_dir", "outputs/frames"), self.cfg.get("_base_dir"))
        green_floor_start_delay_s = max(0.0, float(options.get("green_floor_start_delay_s", 0.0)))
        green_floor_start_frame = int(round(green_floor_start_delay_s * fps))
        rendered_players = False
        try:
            floor = make_floor(self.pitch_points)
            pitch_lines = make_pitch_lines(self.pitch_points, self.pitch_segments)
            total = max(1, end - start - 1)
            start_fov = broadcast_vertical_fov_deg(self.data.cameras.k_matrix[start], self.data.video_info.height)
            end_fov = broadcast_vertical_fov_deg(self.data.cameras.k_matrix[end - 1], self.data.video_info.height)
            for local_idx, frame_idx in enumerate(tqdm(range(start, end), desc="drone")):
                plotter.clear()
                if options.get("show_green_floor", True) and local_idx >= green_floor_start_frame:
                    plotter.add_mesh(floor, color=np.array(self.colors["floor_green"]) / 255.0, ambient=1.0, diffuse=0.0)
                if options.get("show_pitch_lines", True):
                    plotter.add_mesh(
                        pitch_lines,
                        color=np.array(self.colors["pitch_lines"]) / 255.0,
                        line_width=int(self.cfg["render"].get("pitch_line_width", 2)),
                        render_lines_as_tubes=False,
                        ambient=1.0,
                        diffuse=0.0,
                    )
                if options.get("show_stickmen", True):
                    self._add_players(plotter, frame_idx)
                rendered_players |= np.isfinite(self.data.poses.x_world[:, frame_idx]).any()
                progress = local_idx / total
                current_fov = broadcast_vertical_fov_deg(
                    self.data.cameras.k_matrix[frame_idx],
                    self.data.video_info.height,
                )
                position, target, up, fov = camera_pose_for_frame(
                    options.get("camera_motion", {}),
                    progress,
                    self.data.poses.x_world[:, frame_idx],
                    self.data.cameras.r[frame_idx],
                    self.data.cameras.t[frame_idx],
                    fov_default=current_fov,
                    start_fov_default=start_fov,
                    end_fov_default=end_fov,
                    start_x_world_frame=self.data.poses.x_world[:, start],
                    end_x_world_frame=self.data.poses.x_world[:, end - 1],
                    start_r_broadcast=self.data.cameras.r[start],
                    end_r_broadcast=self.data.cameras.r[end - 1],
                    start_t_broadcast=self.data.cameras.t[start],
                    end_t_broadcast=self.data.cameras.t[end - 1],
                    clip_duration_s=clip_duration_s,
                )
                plotter.camera_position = [position, target, up]
                plotter.camera.view_angle = fov
                rgb = plotter.screenshot(return_img=True)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                if local_idx in debug_indices:
                    save_debug_png(debug_dir / f"{self.data.name}_drone_{local_idx:04d}.png", bgr)
        finally:
            writer.release()
            plotter.close()
        if not rendered_players:
            LOGGER.warning("No valid 3D player point was available in the drone range.")
        LOGGER.info("wrote %s", output_path)
        return output_path

    def _range_from_options(self, options: dict[str, Any]) -> tuple[int, int]:
        total = self.data.poses.x_world.shape[1]
        if options.get("start_frame") is None and options.get("end_frame") is None:
            return frame_range(self.cfg, "drone", total)
        start = int(options.get("start_frame") or 0)
        end = int(options.get("end_frame") or total)
        if not 0 <= start < end <= total:
            raise ValueError(f"Invalid drone range [{start}, {end}) for total={total}")
        return start, end

    def _add_players(self, plotter: pv.Plotter, frame_idx: int) -> None:
        teams = self.data.teams
        for team_key in ("team_a", "team_b", "fallback", "referee"):
            indices = [
                idx
                for idx in range(self.data.poses.x_world.shape[0])
                if teams.get(idx, "fallback") == team_key
            ]
            if not indices:
                continue
            color = np.array(self.colors[team_key]) / 255.0
            line_mesh = make_skeleton_lines(self.data.poses.x_world[:, frame_idx], indices)
            if line_mesh is not None:
                plotter.add_mesh(
                    line_mesh,
                    color=color,
                    line_width=int(self.cfg["render"].get("stickman_line_width", 2)),
                    render_lines_as_tubes=False,
                )
            point_mesh = make_joint_points(self.data.poses.x_world[:, frame_idx], indices)
            if point_mesh is not None:
                plotter.add_points(
                    point_mesh,
                    color=color,
                    point_size=float(self.cfg["render"].get("joint_diameter", 4)),
                    render_points_as_spheres=True,
                )
