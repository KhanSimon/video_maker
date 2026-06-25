from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from pose_video_demo.interactive_viewer.controls import ViewerState, register_controls
from pose_video_demo.interactive_viewer.mouse_capture import MacMouseCapture
from pose_video_demo.interactive_viewer.scene_builder import build_scene
from pose_video_demo.interactive_viewer.skeleton_updater import SkeletonLayer, build_skeleton_layer
from pose_video_demo.utils.paths import resolve_path


class InteractivePoseViewer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.viewer_cfg = cfg.get("interactive_viewer", {})
        self.state = ViewerState(
            frame_idx=int(self.viewer_cfg.get("start_frame", 0)),
            play=bool(self.viewer_cfg.get("play", False)),
            temporal_speed=float(self.viewer_cfg.get("temporal_speed", 1.0)),
            fps=float(self.viewer_cfg.get("fps", cfg.get("sequence", {}).get("fps") or 50.0)),
            move_step_m=float(self.viewer_cfg.get("move_step_m", 2.0)),
        )
        self.last_tick = time.monotonic()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.scene = None
        self.layers: dict[str, SkeletonLayer] = {}
        self.active_layer_names: set[str] = set()
        self.display_modes: list[str] = []
        self.display_mode = str(self.viewer_cfg.get("source", "gt")).lower()
        self.window_name = "Pose Video Demo - Open3D Viewer"
        self.mouse_look_enabled = bool(self.viewer_cfg.get("mouse_look", True))
        self.mouse_sensitivity = float(self.viewer_cfg.get("mouse_sensitivity", 0.35))
        self.max_mouse_delta = float(self.viewer_cfg.get("max_mouse_delta", 80.0))
        self.invert_mouse_y = bool(self.viewer_cfg.get("invert_mouse_y", False))
        self.capture_cursor = bool(self.viewer_cfg.get("capture_cursor", True))
        self.last_mouse_xy: tuple[float, float] | None = None
        self.mouse_capture: MacMouseCapture | None = None

    def run(self) -> None:
        width = int(self.viewer_cfg.get("window_width", 1400))
        height = int(self.viewer_cfg.get("window_height", 900))
        left = int(self.viewer_cfg.get("window_left", 80))
        top = int(self.viewer_cfg.get("window_top", 80))
        self.vis.create_window(self.window_name, width=width, height=height, left=left, top=top)
        self.mouse_capture = MacMouseCapture(left, top, width, height)
        try:
            self._build_scene()
            register_controls(self)
            self.vis.register_animation_callback(self._on_animation)
            self.reset_camera()
            if self.mouse_look_enabled and self.capture_cursor:
                if self.mouse_capture.available:
                    self.mouse_capture.activate()
                    self.last_mouse_xy = self.mouse_capture.center_local
                    print("mouse capture active: cursor hidden and recentered")
                else:
                    print("mouse capture unavailable on this platform; using bounded mouse-look")
            self.vis.run()
        finally:
            if self.mouse_capture is not None:
                self.mouse_capture.deactivate()
            self.vis.destroy_window()

    def _build_scene(self) -> None:
        base = self.cfg.get("_base_dir")
        paths = self.cfg.get("paths", {})
        pitch_points = resolve_path(paths.get("pitch_points", "data/pitch/pitch_points.txt"), base)
        teams_path = resolve_path(paths.get("teams", "data/teams/ARG_FRA_182345.txt"), base)
        source = str(self.viewer_cfg.get("source", "gt")).lower()
        self.scene = build_scene(
            pitch_points,
            pitch_line_z_offset=float(self.viewer_cfg.get("pitch_line_z_offset", 0.05)),
        )
        self._load_pose_layers(source, teams_path)

        self.vis.add_geometry(self.scene.floor)
        self.vis.add_geometry(self.scene.pitch_lines)
        if bool(self.viewer_cfg.get("show_axes", True)):
            self.vis.add_geometry(self.scene.axes)
        render = self.vis.get_render_option()
        render.background_color = np.asarray(self.viewer_cfg.get("background_color", [0.96, 0.98, 1.0]), dtype=np.float64)
        render.line_width = float(self.viewer_cfg.get("line_width", 3.0))
        render.point_size = float(self.viewer_cfg.get("point_size", 4.0))
        self._apply_display_mode()
        self._print_scene_diagnostics()
        self._push_frame_update()

    def _on_animation(self, vis) -> bool:
        total_frames = self._total_frames()
        if total_frames <= 0:
            return False
        now = time.monotonic()
        elapsed = now - self.last_tick
        frames_to_advance = int(elapsed * self.state.fps * self.state.temporal_speed)
        if self.state.play and frames_to_advance > 0:
            self.state.next_frame(total_frames, frames_to_advance)
            self.last_tick = now
            self._push_frame_update()
        return False

    def _push_frame_update(self) -> None:
        if not self.layers:
            return
        for layer in self.layers.values():
            layer.update(self.state.frame_idx)
            for line_set in layer.linesets:
                if layer.name in self.active_layer_names:
                    self.vis.update_geometry(line_set)
        self.vis.update_renderer()
        self._print_status()

    def toggle_play(self):
        self.state.play = not self.state.play
        self.last_tick = time.monotonic()
        self._push_frame_update()
        return False

    def step_frame(self, step: int):
        total_frames = self._total_frames()
        if total_frames <= 0:
            return False
        self.state.play = False
        if step >= 0:
            self.state.next_frame(total_frames, step)
        else:
            self.state.previous_frame(total_frames, abs(step))
        self._push_frame_update()
        return False

    def cycle_display_mode(self):
        if not self.display_modes:
            return False
        idx = self.display_modes.index(self.display_mode) if self.display_mode in self.display_modes else 0
        self.display_mode = self.display_modes[(idx + 1) % len(self.display_modes)]
        self._apply_display_mode()
        self._push_frame_update()
        return False

    def change_speed(self, delta: float):
        if delta > 0:
            self.state.faster()
        else:
            self.state.slower()
        self._push_frame_update()
        return False

    def move_camera(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0):
        control = self.vis.get_view_control()
        control.camera_local_translate(float(forward), float(right), float(up))
        self.vis.update_renderer()
        self._print_status()
        return False

    def on_mouse_move(self, x: float, y: float):
        if not self.mouse_look_enabled:
            return False
        if self.capture_cursor and self.mouse_capture is not None and self.mouse_capture.enabled:
            center_x, center_y = self.mouse_capture.center_local
            dx = float(x) - center_x
            dy = float(y) - center_y
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return False
            dx = float(np.clip(dx, -self.max_mouse_delta, self.max_mouse_delta))
            dy = float(np.clip(dy, -self.max_mouse_delta, self.max_mouse_delta))
            self._rotate_from_mouse_delta(dx, dy)
            self.mouse_capture.warp_to_center()
            self.last_mouse_xy = (center_x, center_y)
            return False
        if self.last_mouse_xy is None:
            self.last_mouse_xy = (float(x), float(y))
            return False
        last_x, last_y = self.last_mouse_xy
        dx = float(x) - last_x
        dy = float(y) - last_y
        self.last_mouse_xy = (float(x), float(y))
        dx = float(np.clip(dx, -self.max_mouse_delta, self.max_mouse_delta))
        dy = float(np.clip(dy, -self.max_mouse_delta, self.max_mouse_delta))
        self._rotate_from_mouse_delta(dx, dy)
        return False

    def _rotate_from_mouse_delta(self, dx: float, dy: float) -> None:
        y_direction = -1.0 if self.invert_mouse_y else 1.0
        self.vis.get_view_control().camera_local_rotate(
            -dx * self.mouse_sensitivity,
            y_direction * dy * self.mouse_sensitivity,
        )
        self.vis.update_renderer()

    def reset_camera(self):
        if self.scene is None:
            return False
        self.vis.reset_view_point(True)
        control = self.vis.get_view_control()
        center = self.scene.pitch_center
        initial_front = np.asarray(self.viewer_cfg.get("initial_front", [0.05, -0.62, -0.78]), dtype=np.float64)
        initial_front = self._normalize(initial_front, fallback=np.array([0.05, -0.62, -0.78]))
        control.set_lookat(center.tolist())
        control.set_front(initial_front.tolist())
        control.set_up([0.0, 0.0, 1.0])
        control.set_zoom(float(self.viewer_cfg.get("camera_zoom", 0.32)))
        self.last_mouse_xy = None
        if self.mouse_capture is not None and self.mouse_capture.enabled:
            self.mouse_capture.warp_to_center()
            self.last_mouse_xy = self.mouse_capture.center_local
        self.vis.update_renderer()
        self._print_status()
        return False

    def close(self):
        if self.mouse_capture is not None:
            self.mouse_capture.deactivate()
        self.vis.close()
        return False

    def _load_pose_layers(self, source: str, teams_path: Path) -> None:
        base = self.cfg.get("_base_dir")
        colors_config = self.cfg.get("render", {}).get("colors")
        gt_path = resolve_path(self.viewer_cfg.get("x_world_gt", "data/x_world_gt/ARG_FRA_182345.npy"), base)
        estim_path = resolve_path(self.viewer_cfg.get("x_world_estim", "data/x_world_estim/ARG_FRA_182345.npy"), base)
        gt_color = tuple(int(v) for v in self.viewer_cfg.get("gt_color", [255, 0, 0]))
        gt_z_offset = float(self.viewer_cfg.get("gt_z_offset", 0.04))

        if source not in {"gt", "x_world_gt", "estim", "estimated", "x_world_estim", "both"}:
            custom_path = resolve_path(source, base)
            self.layers["custom"] = build_skeleton_layer(
                custom_path,
                teams_path=teams_path,
                colors_config=colors_config,
                name="custom",
            )
            self.display_modes = ["custom"]
            self.display_mode = "custom"
            return

        if gt_path.exists():
            self.layers["gt"] = build_skeleton_layer(
                gt_path,
                teams_path=teams_path,
                colors_config=colors_config,
                name="gt",
                uniform_color=gt_color,
                z_offset=gt_z_offset,
            )
        if estim_path.exists():
            self.layers["estim"] = build_skeleton_layer(
                estim_path,
                teams_path=teams_path,
                colors_config=colors_config,
                name="estim",
            )
        if not self.layers:
            raise FileNotFoundError(f"No x_world layer found. Tried gt={gt_path} estim={estim_path}")

        self.display_modes = [mode for mode in ("gt", "estim") if mode in self.layers]
        if {"gt", "estim"}.issubset(self.layers):
            self.display_modes.append("both")
        if source in {"x_world_gt"}:
            source = "gt"
        elif source in {"estimated", "x_world_estim"}:
            source = "estim"
        if source not in self.display_modes:
            source = self.display_modes[0]
        self.display_mode = source

    def _apply_display_mode(self) -> None:
        desired = self._visible_layer_names(self.display_mode)
        for name in sorted(self.active_layer_names - desired):
            for line_set in self.layers[name].linesets:
                self.vis.remove_geometry(line_set, reset_bounding_box=False)
        for name in sorted(desired - self.active_layer_names):
            for line_set in self.layers[name].linesets:
                self.vis.add_geometry(line_set, reset_bounding_box=False)
        self.active_layer_names = desired
        print(f"display_mode={self.display_mode} visible={sorted(self.active_layer_names)}")

    def _visible_layer_names(self, mode: str) -> set[str]:
        if mode == "both":
            return {name for name in ("gt", "estim") if name in self.layers}
        if mode in self.layers:
            return {mode}
        return set()

    def _total_frames(self) -> int:
        if not self.layers:
            return 0
        return min(layer.num_frames for layer in self.layers.values())

    def _print_scene_diagnostics(self) -> None:
        if self.scene is None or not self.layers:
            return
        data = np.concatenate([layer.data.reshape(-1, 3) for layer in self.layers.values()], axis=0)
        finite = data[np.isfinite(data).all(axis=-1)]
        if finite.size:
            pose_min = finite.min(axis=0)
            pose_max = finite.max(axis=0)
        else:
            pose_min = pose_max = np.array([np.nan, np.nan, np.nan])
        print(
            "Open3D scene loaded: "
            f"layers={list(self.layers)} frames={self._total_frames()} "
            f"pitch_min={self.scene.bounds_min.round(2).tolist()} pitch_max={self.scene.bounds_max.round(2).tolist()} "
            f"pose_min={pose_min.round(2).tolist()} pose_max={pose_max.round(2).tolist()}"
        )

    def _camera_xyz_fov(self) -> tuple[np.ndarray, float]:
        control = self.vis.get_view_control()
        params = control.convert_to_pinhole_camera_parameters()
        extrinsic = np.asarray(params.extrinsic, dtype=np.float64)
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        camera_position = -(rotation.T @ translation)
        return camera_position, float(control.get_field_of_view())

    @staticmethod
    def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9 or not np.isfinite(vector).all():
            return fallback.astype(np.float64)
        return vector.astype(np.float64) / norm

    def _print_status(self) -> None:
        total_frames = self._total_frames()
        if total_frames <= 0:
            return
        camera_position, fov = self._camera_xyz_fov()
        print(
            f"frame={self.state.frame_idx + 1}/{total_frames} "
            f"mode={self.display_mode} play={self.state.play} speed={self.state.temporal_speed:.2f}x "
            f"camera_xyz=({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}) "
            f"fov={fov:.2f}"
        )
