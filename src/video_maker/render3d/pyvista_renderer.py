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

from video_maker.config import frame_range, resolve_config_paths
from video_maker.io.loaders import SequenceData, describe_sequence, load_sequence
from video_maker.io.mhr import MHRData, renderer_mhr_enabled
from video_maker.io.video_io import make_writer, save_debug_png
from video_maker.pitch.pitch_loader import load_pitch_points, parse_segments
from video_maker.render3d.drone_camera import broadcast_vertical_fov_deg, camera_pose_for_frame
from video_maker.render3d.mhr_mesh import add_mhr_players
from video_maker.render3d.player_display import drone_mhr_schedule, mhr_at_frame
from video_maker.render3d.scene import (
    make_floor,
    make_joint_points,
    make_pitch_lines,
    make_skeleton_lines,
)
from video_maker.utils.colors import color_table
from video_maker.utils.paths import resolve_path
from video_maker.timing import debug_frame_indices, green_floor_start_frame, validate_video_timing

LOGGER = logging.getLogger(__name__)


class PyVistaDroneRenderer:
    def __init__(
        self, cfg: dict[str, Any], data: SequenceData | None = None, mhr_data: MHRData | None = None,
    ) -> None:
        self.cfg = cfg
        self.data = data or load_sequence(cfg)
        self.mhr_data = mhr_data
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
        validate_video_timing(self.cfg)
        for line in describe_sequence(self.data):
            LOGGER.info(line)
        options = dict(self.cfg.get("drone", {}))
        if overrides:
            options.update(overrides)
        start, end = self._range_from_options(options)
        width = int(options.get("width") or self.cfg["render"].get("width") or 1920)
        height = int(options.get("height") or self.cfg["render"].get("height") or 1080)
        fps = float(self.cfg["render"].get("output_fps") or self.data.video_info.fps)
        display_schedule = drone_mhr_schedule(
            options.get("camera_motion", {}), start, end, renderer_mhr_enabled(self.cfg, "drone", options),
        )
        floor_start = green_floor_start_frame(options, start)
        debug_indices = debug_frame_indices(self.cfg)
        if options.get("show_stickmen", True) and any(enabled for _, enabled in display_schedule):
            self._ensure_mhr_data()
        output_path = resolve_path(options["output_path"], self.cfg.get("_base_dir"))
        writer = make_writer(output_path, fps=fps, width=width, height=height)
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.set_background("white")
        if hasattr(plotter, "disable_parallel_projection"):
            plotter.disable_parallel_projection()
        debug_dir = resolve_path(self.cfg["render"].get("debug_dir", "outputs/frames"), self.cfg.get("_base_dir"))
        rendered_players = False
        try:
            floor = make_floor(self.pitch_points)
            pitch_lines = make_pitch_lines(self.pitch_points, self.pitch_segments)
            start_fov = broadcast_vertical_fov_deg(self.data.cameras.k_matrix[start], self.data.video_info.height)
            end_fov = broadcast_vertical_fov_deg(self.data.cameras.k_matrix[end - 1], self.data.video_info.height)
            for frame_idx in tqdm(range(start, end), desc="drone"):
                plotter.clear()
                if options.get("show_green_floor", True) and frame_idx >= floor_start:
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
                    rendered_players |= self._add_players(plotter, frame_idx, mhr_at_frame(display_schedule, frame_idx)) > 0
                current_fov = broadcast_vertical_fov_deg(
                    self.data.cameras.k_matrix[frame_idx],
                    self.data.video_info.height,
                )
                position, target, up, fov = camera_pose_for_frame(
                    options.get("camera_motion", {}),
                    frame_idx,
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
                    clip_start_frame=start,
                    clip_end_frame=end,
                )
                plotter.camera_position = [position, target, up]
                plotter.camera.view_angle = fov
                rgb = plotter.screenshot(return_img=True)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                if frame_idx in debug_indices:
                    save_debug_png(debug_dir / f"{self.data.name}_drone_{frame_idx:04d}.png", bgr)
        finally:
            writer.release()
            plotter.close()
        if not rendered_players:
            LOGGER.warning("No valid 3D player geometry was displayed in the drone range.")
        LOGGER.info("wrote %s", output_path)
        return output_path

    def _range_from_options(self, options: dict[str, Any]) -> tuple[int, int]:
        total = self.data.poses.x_world.shape[1]
        return frame_range(self.cfg, "drone", total, options)

    def _ensure_mhr_data(self) -> MHRData:
        if self.mhr_data is None:
            self.mhr_data = MHRData(self.cfg, self.data)
        return self.mhr_data

    def _add_players(self, plotter: pv.Plotter, frame_idx: int, use_mhr: bool) -> int:
        if use_mhr:
            mhr_data = self._ensure_mhr_data()
            return add_mhr_players(
                plotter,
                mhr_data.world_vertices(frame_idx),
                mhr_data.faces,
                self.data.teams,
                self.colors,
                bool(self.cfg.get("mhr", {}).get("smooth_shading", True)),
            )
        teams = self.data.teams
        count = 0
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
                count += 1
        return count
