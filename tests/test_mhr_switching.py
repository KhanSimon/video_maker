from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_maker.assembly.timeline import render_full_demo
from video_maker.io.mhr import renderer_mhr_enabled
from video_maker.io.video_io import VideoInfo
from video_maker.render2d.broadcast_renderer import BroadcastRenderer
from video_maker.render3d.player_display import drone_mhr_schedule, mhr_at_frame
from video_maker.render3d.pyvista_renderer import PyVistaDroneRenderer
from video_maker.utils.colors import color_table


class MHRStageTests(unittest.TestCase):
    def setUp(self):
        self.motion = {
            "start": {"MHR": False},
            "mid1": {"frame": 404, "hold_frames": 3, "MHR": True},
            "mid2": {"frame": 408, "MHR": False},
            "end": {"MHR": True},
        }

    def test_switches_at_arrival_including_hold_and_last_frame(self):
        schedule = drone_mhr_schedule(self.motion, 400, 411)
        self.assertEqual([mhr_at_frame(schedule, f) for f in range(400, 411)],
                         [False] * 4 + [True] * 4 + [False] * 2 + [True])

    def test_missing_values_inherit_previous_step_and_start_inherits_clip(self):
        motion = deepcopy(self.motion)
        motion["start"].pop("MHR")
        motion["mid1"]["MHR"] = False
        motion["mid2"].pop("MHR")
        motion["end"].pop("MHR")
        schedule = drone_mhr_schedule(motion, 400, 411, default=True)
        self.assertEqual([mhr_at_frame(schedule, f) for f in range(400, 411)], [True] * 4 + [False] * 7)

    def test_numeric_mid_order_and_single_frame_start(self):
        shuffled = {"mid10": self.motion["mid2"], "end": self.motion["end"],
                    "mid1": self.motion["mid1"], "start": self.motion["start"]}
        self.assertEqual(drone_mhr_schedule(shuffled, 400, 411), drone_mhr_schedule(self.motion, 400, 411))
        self.assertEqual(drone_mhr_schedule({"start": {"MHR": False}, "end": {"MHR": True}}, 400, 401), [(400, False)])

    def test_local_false_overrides_legacy_true(self):
        cfg = {"MHR": True, "drone": {"MHR": False}}
        self.assertFalse(renderer_mhr_enabled(cfg, "drone", {}))
        self.assertTrue(renderer_mhr_enabled(cfg, "drone", {"MHR": True}))
        self.assertFalse(renderer_mhr_enabled({"MHR": True}, "broadcast", {"MHR": False}))
        self.assertFalse(renderer_mhr_enabled({}, "drone", {}))

    def test_invalid_boolean_at_any_step_is_rejected_before_rendering(self):
        for name in self.motion:
            for value in (None, 0, 1, "true", "false"):
                motion = deepcopy(self.motion)
                motion[name]["MHR"] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name + ".MHR.*YAML boolean"):
                    drone_mhr_schedule(motion, 400, 411)


class MHRSwitchingRendererTests(unittest.TestCase):
    def test_broadcast_can_return_to_skeleton_and_reuse_mhr_data(self):
        renderer = BroadcastRenderer.__new__(BroadcastRenderer)
        renderer.cfg = {"render": {}, "MHR": False}
        renderer.data = SimpleNamespace(
            cameras=SimpleNamespace(distortion=np.zeros((10, 2))), boxes=None, teams={},
            poses=SimpleNamespace(x_cam=np.zeros((1, 10, 25, 3))),
        )
        renderer.colors = color_table(None)
        renderer.mhr_data = renderer.mhr_overlay = None
        renderer._scaled_k = Mock(return_value=np.eye(3))
        with (
            patch("video_maker.render2d.broadcast_renderer.MHRData") as assets,
            patch("video_maker.render2d.mhr_overlay.MHROverlay") as overlay,
            patch("video_maker.render2d.broadcast_renderer.draw_stickmen_projected", return_value=1) as skeleton,
        ):
            overlay.return_value.draw.return_value = 1
            for frame, enabled in ((0, False), (1, True), (2, False), (3, True)):
                renderer._draw_overlays(np.zeros((8, 8, 3), dtype=np.uint8), frame,
                                        {"show_stickmen": True, "MHR": enabled})
            assets.assert_called_once()
            overlay.assert_called_once()
            self.assertEqual([c.args[1] for c in overlay.return_value.draw.call_args_list], [1, 3])
            self.assertEqual(skeleton.call_count, 2)

    def test_drone_cached_mesh_does_not_prevent_return_to_skeleton(self):
        renderer = PyVistaDroneRenderer.__new__(PyVistaDroneRenderer)
        renderer.cfg = {"render": {}}
        renderer.data = SimpleNamespace(teams={0: "team_a"}, poses=SimpleNamespace(x_world=np.zeros((1, 4, 25, 3))))
        renderer.colors = color_table(None)
        renderer.mhr_data = Mock()
        plotter = Mock()
        with (
            patch("video_maker.render3d.pyvista_renderer.add_mhr_players", return_value=1) as meshes,
            patch("video_maker.render3d.pyvista_renderer.make_skeleton_lines") as skeleton,
            patch("video_maker.render3d.pyvista_renderer.make_joint_points", return_value=None),
        ):
            renderer._add_players(plotter, 0, True)
            renderer._add_players(plotter, 1, False)
            renderer._add_players(plotter, 2, True)
            self.assertEqual(meshes.call_count, 2)
            skeleton.assert_called_once()
            self.assertEqual([c.args[0] for c in renderer.mhr_data.world_vertices.call_args_list], [0, 2])

    def test_drone_render_passes_each_frame_and_its_stage_flag_once(self):
        renderer = PyVistaDroneRenderer.__new__(PyVistaDroneRenderer)
        renderer.cfg = {"render": {}, "drone": {"start_frame": 400, "end_frame": 411,
                        "show_pitch_lines": False, "show_green_floor": False, "output_path": "unused.mp4",
                        "camera_motion": {"start": {"MHR": False},
                                          "mid1": {"frame": 404, "hold_frames": 3, "MHR": True},
                                          "end": {"MHR": False}}}}
        renderer.data = SimpleNamespace(video_info=VideoInfo(8, 8, 50, 411),
                                         poses=SimpleNamespace(x_world=np.zeros((1, 411, 25, 3))),
                                         cameras=SimpleNamespace(k_matrix=np.tile(np.eye(3), (411, 1, 1)),
                                                                 r=np.tile(np.eye(3), (411, 1, 1)), t=np.zeros((411, 3))))
        renderer.mhr_data = Mock()
        renderer.pitch_points = np.zeros((4, 3))
        renderer.pitch_segments = []
        renderer._add_players = Mock(return_value=1)
        plotter = Mock()
        plotter.screenshot.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            patch("video_maker.render3d.pyvista_renderer.describe_sequence", return_value=[]),
            patch("video_maker.render3d.pyvista_renderer.pv.Plotter", return_value=plotter),
            patch("video_maker.render3d.pyvista_renderer.make_writer") as writer,
            patch("video_maker.render3d.pyvista_renderer.make_floor"),
            patch("video_maker.render3d.pyvista_renderer.make_pitch_lines"),
        ):
            renderer.render()
        self.assertEqual([c.args[1] for c in renderer._add_players.call_args_list], list(range(400, 411)))
        self.assertEqual([c.args[2] for c in renderer._add_players.call_args_list], [False] * 4 + [True] * 6 + [False])
        self.assertEqual(writer.return_value.write.call_count, 11)

    def test_timeline_propagates_local_flags_including_transition_and_shares_assets(self):
        cfg = {"render": {}, "timeline": [
            {"name": "skeleton", "start_frame": 0, "end_frame": 2, "MHR": False, "show_stickmen": True},
            {"name": "mesh", "start_frame": 2, "end_frame": 4, "MHR": True, "show_stickmen": True},
            {"name": "fade", "start_frame": 4, "end_frame": 6, "MHR": False,
             "show_stickmen": True, "transition": {"rgb_to_white": True}},
            {"name": "drone", "renderer": "drone", "start_frame": 6, "end_frame": 10,
             "MHR": True, "show_stickmen": True},
        ]}
        data = SimpleNamespace(video_info=VideoInfo(8, 8, 50, 10), poses=SimpleNamespace(x_world=np.zeros((1, 10, 25, 3))))
        with (
            patch("video_maker.assembly.timeline.load_sequence", return_value=data),
            patch("video_maker.assembly.timeline.MHRData") as assets,
            patch("video_maker.assembly.timeline.BroadcastRenderer") as broadcast,
            patch("video_maker.assembly.timeline.PyVistaDroneRenderer") as drone,
            patch("video_maker.assembly.timeline._render_transition_clip") as transition,
            patch("video_maker.assembly.timeline.assemble_clips") as assemble,
        ):
            render_full_demo(cfg)
        assets.assert_called_once()
        self.assertIs(broadcast.call_args.kwargs["mhr_data"], assets.return_value)
        self.assertIs(drone.call_args.kwargs["mhr_data"], assets.return_value)
        self.assertEqual([c.args[0]["MHR"] for c in broadcast.return_value.render.call_args_list], [False, True])
        self.assertFalse(transition.call_args.args[4]["MHR"])
        self.assertTrue(drone.return_value.render.call_args.args[0]["MHR"])
        self.assertEqual(len(assemble.call_args.args[0]), 4)

    def test_bad_mid_flag_or_missing_assets_fails_before_first_clip(self):
        cfg = {"render": {}, "timeline": [
            {"name": "rgb", "start_frame": 0, "end_frame": 4, "MHR": False},
            {"name": "drone", "renderer": "drone", "start_frame": 4, "end_frame": 10}],
            "drone": {"camera_motion": {"mid1": {"frame": 6, "MHR": "false"}}}}
        data = SimpleNamespace(video_info=VideoInfo(8, 8, 50, 10), poses=SimpleNamespace(x_world=np.zeros((1, 10, 25, 3))))
        with (
            patch("video_maker.assembly.timeline.load_sequence", return_value=data),
            patch("video_maker.assembly.timeline.MHRData", side_effect=FileNotFoundError("assets")) as assets,
            patch("video_maker.assembly.timeline.BroadcastRenderer") as broadcast,
        ):
            with self.assertRaisesRegex(ValueError, "mid1.MHR"):
                render_full_demo(cfg)
            assets.assert_not_called()
            cfg["drone"]["camera_motion"]["mid1"]["MHR"] = True
            with self.assertRaisesRegex(FileNotFoundError, "assets"):
                render_full_demo(cfg)
            broadcast.assert_not_called()

    def test_hidden_mhr_or_all_skeleton_timeline_does_not_load_assets(self):
        data = SimpleNamespace(video_info=VideoInfo(8, 8, 50, 10), poses=SimpleNamespace(x_world=np.zeros((1, 10, 25, 3))))
        for hidden in (False, True):
            cfg = {"render": {}, "MHR": True, "timeline": [{"name": "drone", "renderer": "drone",
                   "start_frame": 0, "end_frame": 10, "show_stickmen": not hidden}],
                   "drone": {"camera_motion": {"start": {"MHR": hidden}, "end": {"MHR": hidden}}}}
            with (
                patch("video_maker.assembly.timeline.load_sequence", return_value=data),
                patch("video_maker.assembly.timeline.MHRData", side_effect=AssertionError("Assets opened")),
                patch("video_maker.assembly.timeline.BroadcastRenderer"),
                patch("video_maker.assembly.timeline.PyVistaDroneRenderer"),
                patch("video_maker.assembly.timeline.assemble_clips"),
            ):
                render_full_demo(cfg)


if __name__ == "__main__":
    unittest.main()
