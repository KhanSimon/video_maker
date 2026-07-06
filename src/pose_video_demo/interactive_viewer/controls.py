from __future__ import annotations

from dataclasses import dataclass


KEY_ESCAPE = 256
KEY_RIGHT = 262
KEY_LEFT = 263


@dataclass
class ViewerState:
    frame_idx: int = 0
    play: bool = False
    temporal_speed: float = 1.0
    fps: float = 50.0
    move_step_m: float = 2.0

    def next_frame(self, total_frames: int, step: int = 1) -> None:
        self.frame_idx = (self.frame_idx + step) % total_frames

    def previous_frame(self, total_frames: int, step: int = 1) -> None:
        self.frame_idx = (self.frame_idx - step) % total_frames

    def faster(self) -> None:
        self.temporal_speed = min(2.0, self.temporal_speed + 0.25)

    def slower(self) -> None:
        self.temporal_speed = max(0.25, self.temporal_speed - 0.25)


def register_controls(viewer) -> None:
    vis = viewer.vis
    if viewer.mouse_look_enabled:
        vis.register_mouse_move_callback(lambda vis, x, y: viewer.on_mouse_move(x, y))
    vis.register_key_callback(ord(" "), lambda vis: viewer.toggle_play())
    vis.register_key_callback(KEY_RIGHT, lambda vis: viewer.step_frame(1))
    vis.register_key_callback(KEY_LEFT, lambda vis: viewer.step_frame(-1))
    vis.register_key_callback(ord("L"), lambda vis: viewer.cycle_display_mode())
    vis.register_key_callback(ord("R"), lambda vis: viewer.reset_camera())
    vis.register_key_callback(KEY_ESCAPE, lambda vis: viewer.close())

    vis.register_key_callback(ord("W"), lambda vis: viewer.move_camera(forward=viewer.state.move_step_m))
    vis.register_key_callback(ord("S"), lambda vis: viewer.move_camera(forward=-viewer.state.move_step_m))
    vis.register_key_callback(ord("A"), lambda vis: viewer.move_camera(right=-viewer.state.move_step_m))
    vis.register_key_callback(ord("D"), lambda vis: viewer.move_camera(right=viewer.state.move_step_m))
    vis.register_key_callback(ord("Q"), lambda vis: viewer.move_camera(up=viewer.state.move_step_m))
    vis.register_key_callback(ord("E"), lambda vis: viewer.move_camera(up=-viewer.state.move_step_m))


    vis.register_key_callback(ord("+"), lambda vis: viewer.change_speed(0.25))
    vis.register_key_callback(ord("="), lambda vis: viewer.change_speed(0.25))
    vis.register_key_callback(ord(")"), lambda vis: viewer.change_speed(-0.25))
