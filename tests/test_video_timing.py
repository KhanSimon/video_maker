from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_maker.assembly.timeline import _render_transition_clip, render_full_demo
from video_maker.config import frame_range, load_config
from video_maker.io.video_io import VideoInfo
from video_maker.render3d.pyvista_renderer import PyVistaDroneRenderer
from video_maker.timing import validate_video_timing


class VideoTimingTests(unittest.TestCase):
    def test_standalone_and_timeline_ranges_have_one_source_of_truth(self):
        # Fixtures are independent of the user's editable demo duration.
        timeline = [{"name": "b", "start_frame": 0, "end_frame": 400},
                    {"name": "d", "renderer": "drone", "start_frame": 400, "end_frame": 850}]
        for cfg, section, expected in (
            ({"drone": {"start_frame": 650, "end_frame": 1240}}, "drone", (650, 1240)),
            ({"broadcast": {"start_frame": 0, "end_frame": 1240}}, "broadcast", (0, 1240)),
            ({"timeline": timeline}, "drone", (400, 850)),
            ({"timeline": timeline}, "broadcast", (0, 400)),
        ):
            validate_video_timing(cfg)
            self.assertEqual(frame_range(cfg, section, 1241), expected)
        self.assertEqual(frame_range({}, "drone", 10), (0, 10))
        self.assertEqual(frame_range({}, "drone", 10, {"start_frame": 7}), (7, 10))
        self.assertEqual(frame_range({}, "drone", 10, {"end_frame": 3}), (0, 3))

    def test_ranges_are_never_coerced_rounded_or_silently_clamped(self):
        for options in (
            {"end_frame": 0}, {"end_frame": 11}, {"start_frame": 10},
            {"start_frame": 0.0}, {"end_frame": "10"}, {"start_frame": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                frame_range({}, "drone", 10, options)

    def test_legacy_and_duplicate_temporal_settings_fail_explicitly(self):
        for cfg, field in (
            ({"sequence": {"fps": 50}}, "sequence.fps"),
            ({"sequence": {"start_frame": 5}}, "sequence.start_frame"),
            ({"sequence": {"num_frames": 10}}, "sequence.num_frames"),
            ({"drone": {"green_floor_start_delay_s": 0}}, "green_floor_start_frame"),
            ({"render": {"debug_frames": [0.5]}}, "debug_frames"),
            ({"timeline": [{"name": "rgb", "start": 0, "end": 1}]}, "start_frame/end_frame"),
            ({"timeline": [{"name": "rgb", "start_frame": 0, "end_frame": 10}],
              "broadcast": {"start_frame": 0}}, "timeline already defines"),
        ):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, field):
                validate_video_timing(cfg)

    def test_invalid_drone_schedule_or_source_end_fails_before_any_clip(self):
        base = load_config("configs/sequence_demo.yaml")
        data = SimpleNamespace(video_info=VideoInfo(640, 360, 50, 1241),
                               poses=SimpleNamespace(x_world=np.zeros((1, 1241, 25, 3))))
        for invalid_schedule in (True, False):
            cfg = deepcopy(base)
            if invalid_schedule:
                cfg["drone"]["camera_motion"]["mid1"]["hold_frames"] = 1000
            else:
                cfg["timeline"][-1]["end_frame"] = 2000
            with (
                patch("video_maker.assembly.timeline.load_sequence", return_value=data),
                patch("video_maker.assembly.timeline.BroadcastRenderer") as broadcast,
                patch("video_maker.assembly.timeline.make_writer") as writer,
                self.assertRaises(ValueError),
            ):
                render_full_demo(cfg)
            broadcast.assert_not_called()
            writer.assert_not_called()

    def test_drone_fps_changes_only_encoding_not_camera_players_floor_or_debug(self):
        results = []
        for fps in (25, 50, 60):
            renderer = PyVistaDroneRenderer.__new__(PyVistaDroneRenderer)
            renderer.cfg = {
                "render": {"output_fps": fps, "debug_frames": [0, 404]},
                "drone": {"start_frame": 400, "end_frame": 411, "output_path": "unused.mp4",
                          "show_pitch_lines": False, "green_floor_start_frame": 405,
                          "camera_motion": {
                              "start": {"position": [0, 0, 10], "look_at": [0, 5, 0]},
                              "mid1": {"frame": 404, "hold_frames": 3,
                                       "position": [5, 0, 10], "look_at": [0, 5, 0]},
                              "end": {"position": [10, 0, 10], "look_at": [0, 5, 0]},
                          }},
            }
            renderer.data = SimpleNamespace(
                name="synthetic", video_info=VideoInfo(8, 8, 50, 411),
                poses=SimpleNamespace(x_world=np.zeros((1, 411, 25, 3))),
                cameras=SimpleNamespace(k_matrix=np.tile(np.diag([10., 10., 1.]), (411, 1, 1)),
                                        r=np.tile(np.eye(3), (411, 1, 1)), t=np.zeros((411, 3))),
            )
            renderer.pitch_points = np.zeros((4, 3))
            renderer.pitch_segments = []
            renderer.colors = {"floor_green": [0, 255, 0]}
            renderer._add_players = Mock(return_value=1)
            plotter = Mock()
            poses, floor_frames = [], []

            def screenshot(**kwargs):
                poses.append(plotter.camera_position)
                return np.zeros((8, 8, 3), dtype=np.uint8)

            plotter.screenshot.side_effect = screenshot
            plotter.add_mesh.side_effect = lambda *a, **kw: floor_frames.append(400 + len(poses))
            with (
                patch("video_maker.render3d.pyvista_renderer.describe_sequence", return_value=[]),
                patch("video_maker.render3d.pyvista_renderer.pv.Plotter", return_value=plotter),
                patch("video_maker.render3d.pyvista_renderer.make_floor"),
                patch("video_maker.render3d.pyvista_renderer.make_pitch_lines"),
                patch("video_maker.render3d.pyvista_renderer.make_writer") as writer,
                patch("video_maker.render3d.pyvista_renderer.save_debug_png") as debug,
            ):
                renderer.render()
                self.assertEqual(writer.call_args.kwargs["fps"], fps)
                self.assertEqual(writer.return_value.write.call_count, 11)
                self.assertEqual([call.args[1] for call in renderer._add_players.call_args_list], list(range(400, 411)))
                self.assertEqual(floor_frames, list(range(405, 411)))
                self.assertEqual([call.args[0].name for call in debug.call_args_list], ["synthetic_drone_0404.png"])
                np.testing.assert_array_equal(np.array(poses)[4:7, 0], [[5, 0, 10]] * 3)
                results.append(poses)
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])

    def test_transition_fades_and_debug_follow_source_frames_at_any_fps(self):
        for fps in (25, 60):
            renderer = Mock()
            renderer.cfg = {"render": {"debug_frames": [0, 402]}}
            renderer.data = SimpleNamespace(name="synthetic", video_path=Path("unused.mov"))
            renderer.render_frame.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
            with (
                patch("video_maker.assembly.timeline.cv2.VideoCapture"),
                patch("video_maker.assembly.timeline.make_writer") as writer,
                patch("video_maker.assembly.timeline.save_debug_png") as debug,
            ):
                _render_transition_clip(renderer, {"name": "fade", "transition": {"rgb_to_white": True}},
                                        400, 403, {"output_path": "unused.mp4"}, fps, 8, 8)
                self.assertEqual([c.args[0] for c in renderer.render_frame.call_args_list], [400, 401, 402])
                self.assertEqual([c.args[1]["rgb_alpha"] for c in renderer.render_frame.call_args_list], [1, .5, 0])
                self.assertEqual(writer.return_value.write.call_count, 3)
                self.assertEqual(debug.call_args.args[0].name, "synthetic_broadcast_0402.png")


if __name__ == "__main__":
    unittest.main()
