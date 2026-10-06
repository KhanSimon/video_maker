from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from video_maker.interactive_viewer.controls import ViewerState, register_controls
from video_maker.interactive_viewer.minimap import MinimapOverlay, build_minimap_overlay
from video_maker.interactive_viewer.mouse_capture import MacMouseCapture
from video_maker.interactive_viewer.scene_builder import build_scene
from video_maker.interactive_viewer.skeleton_updater import SkeletonLayer, build_skeleton_layer
from video_maker.interactive_viewer.temporal_context import (
    TemporalContextLayer,
    build_temporal_context_layer,
)
from video_maker.utils.paths import resolve_path


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
        self.temporal_layers: dict[str, TemporalContextLayer] = {}
        self.minimap: MinimapOverlay | None = None
        self.minimap_cfg: dict[str, Any] = {}
        self.active_layer_names: set[str] = set()
        self.display_modes: list[str] = []
        self.display_mode = str(self.viewer_cfg.get("source", "gt")).lower()
        self.window_name = "Pose Video Demo - Open3D Viewer"
        self.camera_mode = str(
            self.viewer_cfg.get("camera_mode", self.viewer_cfg.get("view_mode", "free"))
        ).lower()
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
        teams_value = paths.get("teams")
        teams_path = resolve_path(teams_value, base) if teams_value else None
        source = str(self.viewer_cfg.get("source", "gt")).lower()
        self.scene = build_scene(
            pitch_points,
            pitch_line_z_offset=float(self.viewer_cfg.get("pitch_line_z_offset", 0.05)),
        )
        self._load_pose_layers(source, teams_path)
        self._configure_player_filter()
        self._build_temporal_layers()

        self.vis.add_geometry(self.scene.floor)
        self.vis.add_geometry(self.scene.pitch_lines)
        if bool(self.viewer_cfg.get("show_axes", True)):
            self.vis.add_geometry(self.scene.axes)
        render = self.vis.get_render_option()
        render.background_color = np.asarray(self.viewer_cfg.get("background_color", [0.96, 0.98, 1.0]), dtype=np.float64)
        render.line_width = float(self.viewer_cfg.get("line_width", 3.0))
        render.point_size = float(self.viewer_cfg.get("point_size", 4.0))
        floor_color_mode = str(self.viewer_cfg.get("floor_color_mode", "fixed")).lower()
        if floor_color_mode not in {"fixed", "lit"}:
            raise ValueError(f"Unsupported floor_color_mode={floor_color_mode}. Use 'fixed' or 'lit'.")
        render.light_on = floor_color_mode == "lit"
        render.mesh_show_back_face = bool(self.viewer_cfg.get("mesh_show_back_face", True))
        self._apply_display_mode()
        self._build_minimap()
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
        if self._field_orbit_enabled():
            self._apply_field_orbit_constraints()
        self._update_minimap(update_renderer=True)
        return False

    def _push_frame_update(self) -> None:
        if not self.layers:
            return
        for layer in self.layers.values():
            layer.update(self.state.frame_idx)
            if layer.name not in self.active_layer_names:
                continue
            for geometry in layer.geometries:
                self.vis.update_geometry(geometry)
            temporal_layer = self.temporal_layers.get(layer.name)
            if temporal_layer is not None:
                temporal_layer.update(self.state.frame_idx)
                for geometry in temporal_layer.geometries:
                    self.vis.update_geometry(geometry)
        self._update_minimap(update_renderer=False)
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
        if self._field_orbit_enabled():
            self._move_field_orbit(float(forward), float(right), float(up))
            return False
        control = self.vis.get_view_control()
        control.camera_local_translate(float(forward), float(right), float(up))
        self._update_minimap(update_renderer=False)
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
        if self._field_orbit_enabled():
            self._apply_field_orbit_constraints()
        self._update_minimap(update_renderer=False)
        self.vis.update_renderer()

    def reset_camera(self):
        if self.scene is None:
            return False
        self.vis.reset_view_point(True)
        initial_camera = self.viewer_cfg.get("initial_camera")
        if initial_camera:
            self._apply_initial_camera(initial_camera)
            if self._field_orbit_enabled():
                self._apply_field_orbit_constraints()
            self.last_mouse_xy = None
            if self.mouse_capture is not None and self.mouse_capture.enabled:
                self.mouse_capture.warp_to_center()
                self.last_mouse_xy = self.mouse_capture.center_local
            self._update_minimap(update_renderer=False)
            self.vis.update_renderer()
            self._print_status()
            return False
        control = self.vis.get_view_control()
        center = self.scene.pitch_center
        initial_front = np.asarray(self.viewer_cfg.get("initial_front", [0.05, -0.62, -0.78]), dtype=np.float64)
        initial_front = self._normalize(initial_front, fallback=np.array([0.05, -0.62, -0.78]))
        control.set_lookat(center.tolist())
        control.set_front(initial_front.tolist())
        control.set_up([0.0, 0.0, 1.0])
        control.set_zoom(float(self.viewer_cfg.get("camera_zoom", 0.32)))
        if self._field_orbit_enabled():
            self._apply_field_orbit_constraints()
        self.last_mouse_xy = None
        if self.mouse_capture is not None and self.mouse_capture.enabled:
            self.mouse_capture.warp_to_center()
            self.last_mouse_xy = self.mouse_capture.center_local
        self._update_minimap(update_renderer=False)
        self.vis.update_renderer()
        self._print_status()
        return False

    def close(self):
        if self.mouse_capture is not None:
            self.mouse_capture.deactivate()
        self.vis.close()
        return False

    def _load_pose_layers(self, source: str, teams_path: Path | None) -> None:
        base = self.cfg.get("_base_dir")
        paths = self.cfg.get("paths", {})
        colors_config = self.cfg.get("render", {}).get("colors")
        gt_path = resolve_path(self.viewer_cfg.get("x_world_gt", "data/x_world_gt/ARG_FRA_182345.npy"), base)
        estim_path = resolve_path(self.viewer_cfg.get("x_world_estim", "data/x_world_estim/ARG_FRA_182345.npy"), base)
        gt_color = tuple(int(v) for v in self.viewer_cfg.get("gt_color", [255, 0, 0]))
        gt_z_offset = float(self.viewer_cfg.get("gt_z_offset", 0.04))

        if source in {"npz", "sequence_npz"}:
            npz_value = self.viewer_cfg.get("sequence_npz", paths.get("sequence_npz"))
            if not npz_value:
                raise ValueError("source=npz requires paths.sequence_npz or interactive_viewer.sequence_npz")
            npz_path = resolve_path(npz_value, base)
            self.layers["npz"] = build_skeleton_layer(
                npz_path,
                teams_path=teams_path,
                colors_config=colors_config,
                name="npz",
                data_key=self.viewer_cfg.get("npz_pose_key", "joints_pred_world_m"),
                valid_mask_key=self.viewer_cfg.get("npz_valid_mask_key", "valid_mask"),
                frame_numbers_key=self.viewer_cfg.get("npz_frame_numbers_key", "frame_numbers"),
            )
            self.display_modes = ["npz"]
            self.display_mode = "npz"
            return

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

    def _configure_player_filter(self) -> None:
        raw = self.viewer_cfg.get("player_indices", [])
        selected: tuple[int, ...] | None = None
        for layer in self.layers.values():
            layer_selected = layer.set_player_filter(raw)
            layer.update(self.state.frame_idx)
            if selected is None:
                selected = layer_selected
            elif layer_selected != selected:
                raise ValueError("Pose layers disagree on interactive_viewer.player_indices")
        if raw:
            print(f"player_filter={list(selected or ())}")
        else:
            print("player_filter=all")

    def _apply_display_mode(self) -> None:
        desired = self._visible_layer_names(self.display_mode)
        for name in sorted(self.active_layer_names - desired):
            for geometry in self._display_geometries(name):
                self.vis.remove_geometry(geometry, reset_bounding_box=False)
        for name in sorted(desired - self.active_layer_names):
            for geometry in self._display_geometries(name):
                self.vis.add_geometry(geometry, reset_bounding_box=False)
        self.active_layer_names = desired
        self._sync_minimap_source()
        print(f"display_mode={self.display_mode} visible={sorted(self.active_layer_names)}")

    def _display_geometries(self, layer_name: str) -> list[o3d.geometry.Geometry]:
        geometries = list(self.layers[layer_name].geometries)
        temporal_layer = self.temporal_layers.get(layer_name)
        if temporal_layer is not None:
            geometries.extend(temporal_layer.geometries)
        return geometries

    def _build_temporal_layers(self) -> None:
        temporal_cfg = self.viewer_cfg.get("temporal_context")
        if not temporal_cfg or not bool(temporal_cfg.get("enabled", True)):
            return
        background = self.viewer_cfg.get("background_color", [0.96, 0.98, 1.0])
        self.temporal_layers = {
            name: build_temporal_context_layer(layer, temporal_cfg, background)
            for name, layer in self.layers.items()
        }
        first = next(iter(self.temporal_layers.values()))
        print(
            "temporal_context="
            f"players={list(first.player_indices)} endpoints={sorted(first.endpoint_offsets)} "
            f"sampled_frames={len(first.sampled_offsets)}"
        )

    def _build_minimap(self) -> None:
        raw = self.viewer_cfg.get("minimap")
        if not raw or not bool(raw.get("enabled", True)):
            return
        if self.scene is None:
            raise RuntimeError("Cannot build minimap before the pitch scene")
        self.minimap_cfg = dict(raw)
        source_name = self._select_minimap_source_name()
        self.minimap = build_minimap_overlay(
            self.layers[source_name],
            self.scene,
            self.minimap_cfg,
        )
        camera_parameters = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        self.minimap.update(self.state.frame_idx, camera_parameters)
        for geometry in self.minimap.geometries:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
        print(
            f"minimap=enabled source={source_name} size="
            f"{self.minimap.width_px}x{self.minimap.height_px}"
        )

    def _select_minimap_source_name(self) -> str:
        requested = str(self.minimap_cfg.get("source_layer", "auto")).lower()
        if requested != "auto":
            if requested not in self.layers:
                raise ValueError(
                    f"minimap.source_layer={requested!r} is unavailable; layers={list(self.layers)}"
                )
            return requested
        if self.display_mode in self.layers:
            return self.display_mode
        for candidate in ("estim", "npz", "custom", "gt"):
            if candidate in self.layers:
                return candidate
        raise ValueError("No pose layer is available for the minimap")

    def _sync_minimap_source(self) -> None:
        if self.minimap is None:
            return
        requested = str(self.minimap_cfg.get("source_layer", "auto")).lower()
        if requested == "auto":
            self.minimap.source = self.layers[self._select_minimap_source_name()]

    def _update_minimap(self, update_renderer: bool) -> None:
        if self.minimap is None:
            return
        self._sync_minimap_source()
        camera_parameters = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        changed = self.minimap.update(self.state.frame_idx, camera_parameters)
        if not changed:
            return
        for geometry in self.minimap.geometries:
            self.vis.update_geometry(geometry)
        if update_renderer:
            self.vis.update_renderer()

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

    def _field_orbit_enabled(self) -> bool:
        return self.camera_mode in {"field_orbit", "terrain_orbit", "pitch_orbit", "orbit"}

    def _field_orbit_center(self) -> np.ndarray:
        if self.scene is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        raw = self.viewer_cfg.get("field_orbit_center")
        if raw is not None:
            center = np.asarray(raw, dtype=np.float64)
            if center.shape != (3,):
                raise ValueError("field_orbit_center must be an XYZ triple")
            return center
        center = self.scene.pitch_center.astype(np.float64).copy()
        center[2] = float(self.viewer_cfg.get("field_orbit_center_z", 1.0))
        return center

    def _field_orbit_limits(self) -> tuple[float, float, float]:
        min_elevation = np.deg2rad(float(self.viewer_cfg.get("field_orbit_min_elevation_deg", 5.0)))
        max_elevation = np.deg2rad(float(self.viewer_cfg.get("field_orbit_max_elevation_deg", 75.0)))
        if max_elevation < min_elevation:
            min_elevation, max_elevation = max_elevation, min_elevation
        min_distance = float(self.viewer_cfg.get("field_orbit_min_distance_m", 5.0))
        return min_elevation, max_elevation, max(min_distance, 1e-3)

    def _field_orbit_angles(self) -> tuple[float, float, float]:
        center = self._field_orbit_center()
        position, _ = self._camera_xyz_fov()
        offset = position - center
        if np.linalg.norm(offset) <= 1e-9 or not np.isfinite(offset).all():
            offset = np.array([0.0, -85.0, 25.0], dtype=np.float64)
        radius = float(np.linalg.norm(offset))
        yaw = float(np.arctan2(offset[1], offset[0]))
        horizontal = float(np.linalg.norm(offset[:2]))
        elevation = float(np.arctan2(offset[2], horizontal))
        min_elevation, max_elevation, min_distance = self._field_orbit_limits()
        return yaw, float(np.clip(elevation, min_elevation, max_elevation)), max(radius, min_distance)

    def _field_orbit_position(self, yaw: float, elevation: float, radius: float) -> np.ndarray:
        min_elevation, max_elevation, min_distance = self._field_orbit_limits()
        elevation = float(np.clip(elevation, min_elevation, max_elevation))
        radius = max(float(radius), min_distance)
        center = self._field_orbit_center()
        horizontal = radius * np.cos(elevation)
        return center + np.array(
            [
                horizontal * np.cos(yaw),
                horizontal * np.sin(yaw),
                radius * np.sin(elevation),
            ],
            dtype=np.float64,
        )

    def _set_field_orbit_camera(self, yaw: float, elevation: float, radius: float, print_status: bool = True) -> None:
        position = self._field_orbit_position(yaw, elevation, radius)
        center = self._field_orbit_center()
        control = self.vis.get_view_control()
        params = control.convert_to_pinhole_camera_parameters()
        params.extrinsic = self._look_at_extrinsic(position, center, np.array([0.0, 0.0, 1.0]))
        control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        self.vis.update_renderer()
        if print_status:
            self._print_status()

    def _apply_field_orbit_constraints(self) -> None:
        yaw, elevation, radius = self._field_orbit_angles()
        self._set_field_orbit_camera(yaw, elevation, radius, print_status=False)

    def _move_field_orbit(self, forward: float, right: float, up: float) -> None:
        yaw, elevation, radius = self._field_orbit_angles()
        step_scale = max(self.state.move_step_m, 1e-9)
        yaw += np.deg2rad(float(self.viewer_cfg.get("field_orbit_yaw_step_deg", 4.0))) * (right / step_scale)
        elevation += np.deg2rad(float(self.viewer_cfg.get("field_orbit_pitch_step_deg", 3.0))) * (forward / step_scale)
        radius -= float(self.viewer_cfg.get("field_orbit_zoom_step_m", 5.0)) * (up / step_scale)
        self._set_field_orbit_camera(yaw, elevation, radius)

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

    def _apply_initial_camera(self, initial_camera: dict[str, Any]) -> None:
        position = np.asarray(initial_camera["position"], dtype=np.float64)
        look_at = np.asarray(initial_camera["look_at"], dtype=np.float64)
        up = np.asarray(initial_camera.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
        control = self.vis.get_view_control()
        params = control.convert_to_pinhole_camera_parameters()
        params.extrinsic = self._look_at_extrinsic(position, look_at, up)
        ok = control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        if not ok:
            print("warning: Open3D refused initial_camera extrinsic, falling back to lookat/front")
            front = self._normalize(look_at - position, fallback=np.array([0.05, -0.62, -0.78]))
            control.set_lookat(look_at.tolist())
            control.set_front(front.tolist())
            control.set_up(up.tolist())

    def _look_at_extrinsic(self, position: np.ndarray, look_at: np.ndarray, up: np.ndarray) -> np.ndarray:
        if position.shape != (3,) or look_at.shape != (3,) or up.shape != (3,):
            raise ValueError("initial_camera position/look_at/up must be XYZ triples")
        front = self._normalize(look_at - position, fallback=np.array([1.0, 0.0, 0.0]))
        up = self._normalize(up, fallback=np.array([0.0, 0.0, 1.0]))
        right = np.cross(front, up)
        if np.linalg.norm(right) <= 1e-9:
            right = np.cross(front, np.array([0.0, 1.0, 0.0]))
        right = self._normalize(right, fallback=np.array([1.0, 0.0, 0.0]))
        down = self._normalize(np.cross(front, right), fallback=-up)
        rotation = np.vstack([right, down, front])
        translation = -rotation @ position
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        return extrinsic

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
        active_layer = next(
            (self.layers[name] for name in sorted(self.active_layer_names) if name in self.layers),
            next(iter(self.layers.values())),
        )
        source_frame = active_layer.source_frame_number(self.state.frame_idx)
        print(
            f"frame={self.state.frame_idx + 1}/{total_frames} "
            f"source_frame={source_frame} "
            f"mode={self.display_mode} play={self.state.play} speed={self.state.temporal_speed:.2f}x "
            f"camera_xyz=({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}) "
            f"fov={fov:.2f}"
        )
