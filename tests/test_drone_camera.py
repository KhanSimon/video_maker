from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from video_maker.io.video_io import VideoInfo
from video_maker.render3d.drone_camera import camera_pose_for_frame, travel_midpoint_schedule


class DroneCameraTests(unittest.TestCase):
    def setUp(self):
        self.players_start = np.tile([0.0, 0.0, 1.0], (2, 25, 1))
        self.players_end = self.players_start + [20.0, 10.0, 0.0]
        self.motion = {
            "type": "travel", "smoothstep": True,
            "start": self.stop([0, 0, 10], [0, 5, 1], 60),
            "mid1": {**self.stop([10, 5, 8], [2, 10, 1], 50), "frame": 560, "hold_frames": 81},
            "mid2": {**self.stop([-5, 15, 4], [10, -5, 1], 35), "frame": 800, "hold_frames": 41},
            "mid3": {**self.stop([3, 20, 6], [12, 0, 2], 45), "frame": 1040, "hold_frames": 21},
            "end": self.stop([20, 30, 12], [15, 3, 1], 25),
        }

    @staticmethod
    def stop(position, look_at, fov):
        return {"position": position, "look_at": look_at, "zoom": {"mode": "fov", "fov_deg": fov}}

    def pose(self, frame, motion=None, start=400, end=1201):
        return camera_pose_for_frame(
            motion if motion is not None else self.motion,
            frame, self.players_start, np.eye(3), np.array([0.0, 0.0, -10.0]),
            start_x_world_frame=self.players_start, end_x_world_frame=self.players_end,
            end_r_broadcast=np.eye(3), end_t_broadcast=np.array([3.0, 1.0, -12.0]),
            start_fov_default=60, end_fov_default=25, clip_start_frame=start, clip_end_frame=end,
        )

    @staticmethod
    def values(pose):
        position, look_at, _, fov = pose
        return np.r_[position, look_at, fov]

    @staticmethod
    def expected(stop):
        return np.r_[stop["position"], stop["look_at"], stop["zoom"]["fov_deg"]]

    def test_reaches_all_positions_orientations_and_zooms_and_holds_them(self):
        for name, times in (
            ("start", [400]), ("mid1", range(560, 641)),
            ("mid2", range(800, 841)), ("mid3", range(1040, 1061)), ("end", [1200]),
        ):
            for frame in times:
                with self.subTest(name=name, frame=frame):
                    np.testing.assert_allclose(self.values(self.pose(frame)), self.expected(self.motion[name]))

    def test_interpolates_each_travel_leg_from_previous_departure(self):
        self.motion["smoothstep"] = False
        for frame, start, end in (
            (480, "start", "mid1"), (720, "mid1", "mid2"),
            (940, "mid2", "mid3"), (1130, "mid3", "end"),
        ):
            with self.subTest(frame=frame):
                expected = (self.expected(self.motion[start]) + self.expected(self.motion[end])) / 2
                np.testing.assert_allclose(self.values(self.pose(frame)), expected)

    def test_smoothstep_eases_each_leg_at_integer_frames(self):
        # Quarter of the 160-frame leg from the last hold at 640 to arrival at 800.
        expected = self.expected(self.motion["mid1"]) * 0.84375 + self.expected(self.motion["mid2"]) * 0.15625
        np.testing.assert_allclose(self.values(self.pose(680)), expected)
        for boundary in (560, 640, 800, 840, 1040, 1060):
            before = self.values(self.pose(boundary - 1))
            at = self.values(self.pose(boundary))
            after = self.values(self.pose(boundary + 1))
            self.assertLess(np.linalg.norm(before - at), 0.01)
            self.assertLess(np.linalg.norm(after - at), 0.01)

    def test_numbered_midpoints_use_numeric_order_independent_of_yaml_order(self):
        motion = {"mid10": self.motion["mid3"], "mid2": self.motion["mid2"], "mid1": self.motion["mid1"]}
        for name in ("start", "end", "smoothstep"):
            motion[name] = self.motion[name]
        for frame in range(400, 1201, 20):
            np.testing.assert_allclose(self.values(self.pose(frame, motion)), self.values(self.pose(frame)))

    def test_arrival_frame_is_required_and_hold_defaults_to_zero(self):
        self.motion["mid1"].pop("hold_frames")
        self.assertEqual(travel_midpoint_schedule(self.motion, 400, 1201)[0][2:], (560, 560))
        self.motion["mid1"].pop("frame")
        with self.assertRaisesRegex(ValueError, "mid1.frame"):
            self.pose(400)

    def test_legacy_mid_aliases_and_direct_travel_keep_their_behaviour(self):
        for alias in ("mid", "middle", "waypoint", "mid1"):
            motion = {name: deepcopy(self.motion[name]) for name in ("start", "end")}
            motion[alias] = deepcopy(self.motion["mid1"])
            for frame, expected in (
                (480, (self.expected(motion["start"]) + self.expected(motion[alias])) / 2),
                (600, self.expected(motion[alias])),
                (920, (self.expected(motion[alias]) + self.expected(motion["end"])) / 2),
            ):
                with self.subTest(alias=alias, frame=frame):
                    np.testing.assert_allclose(self.values(self.pose(frame, motion)), expected)
        direct = {name: self.motion[name] for name in ("start", "end")}
        expected = self.expected(direct["start"]) * 0.84375 + self.expected(direct["end"]) * 0.15625
        np.testing.assert_allclose(self.values(self.pose(600, direct)), expected)

    def test_hold_count_is_exact_and_independent_of_clip_length(self):
        for end in (1201, 1501):
            for count in (0, 1, 2, 25):
                self.motion["mid1"]["hold_frames"] = count
                poses = [self.values(self.pose(frame, end=end)) for frame in range(559, 590)]
                equal = [np.array_equal(value, self.expected(self.motion["mid1"])) for value in poses]
                self.assertEqual(sum(equal), max(count, 1))

    def test_single_frame_clip_uses_start_pose(self):
        direct = {name: self.motion[name] for name in ("start", "end")}
        np.testing.assert_allclose(self.values(self.pose(400, direct, end=401)), self.expected(direct["start"]))

    def test_each_midpoint_resolves_its_own_spherical_or_target_anchor(self):
        self.motion["mid1"].update(
            anchor_frame="start",
            position={"mode": "spherical", "origin": "players_center", "radius_m": 2, "azimuth_deg": 0, "elevation_deg": 0},
            look_at={"mode": "target", "target": "broadcast_camera", "offset": [1, 0, 0]},
        )
        self.motion["mid2"].update(
            anchor_frame="end",
            position={"mode": "target", "target": "broadcast_camera", "offset": [2, 3, 4]},
            look_at={"mode": "target", "target": "players_center", "offset": [0, 0, 2]},
        )
        position, target, _, _ = self.pose(560)
        np.testing.assert_allclose(position, [2, 0, 1])
        np.testing.assert_allclose(target, [1, 0, 10])
        position, target, _, _ = self.pose(800)
        np.testing.assert_allclose(position, [-1, 2, 16])
        np.testing.assert_allclose(target, [20, 10, 3])

    def test_invalid_timing_and_ambiguous_midpoints_are_rejected(self):
        for spec in (
            {"frame": 400}, {"frame": 1200}, {"frame": 560.0}, {"frame": True},
            {"hold_frames": -1}, {"hold_frames": 1.5}, {"hold_frames": 700},
            {"arrive_at": 0.2}, {"hold_duration_s": 1}, {"depart_at": 0.3},
        ):
            motion = deepcopy(self.motion)
            motion["mid1"].update(spec)
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "mid1"):
                self.pose(400, motion)
        for frame in (640, 500):
            motion = deepcopy(self.motion)
            motion["mid2"]["frame"] = frame
            with self.subTest(frame=frame), self.assertRaisesRegex(ValueError, "mid2.*after.*mid1"):
                self.pose(400, motion)
        with self.assertRaisesRegex(ValueError, "rename mid to mid1"):
            self.pose(400, {**self.motion, "mid": self.motion["mid1"]})
        for name, spec in (("mid0", {}), ("mid01", {}), ("mid1", [])):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                travel_midpoint_schedule({name: spec}, 400, 1201)

    def test_renderer_validates_pauses_before_creating_output_or_plotter(self):
        from video_maker.render3d.pyvista_renderer import PyVistaDroneRenderer

        self.motion["mid1"]["hold_frames"] = 300  # overlaps mid2 arrival at 800
        renderer = PyVistaDroneRenderer.__new__(PyVistaDroneRenderer)
        renderer.cfg = {"render": {"output_fps": 50}, "drone": {"start_frame": 400, "end_frame": 1201, "camera_motion": self.motion}}
        renderer.data = SimpleNamespace(
            video_info=VideoInfo(640, 360, 50, 1201), poses=SimpleNamespace(x_world=np.zeros((1, 1201, 25, 3))),
        )
        with (
            patch("video_maker.render3d.pyvista_renderer.describe_sequence", return_value=[]),
            patch("video_maker.render3d.pyvista_renderer.make_writer") as writer,
            patch("video_maker.render3d.pyvista_renderer.pv.Plotter") as plotter,
        ):
            with self.assertRaisesRegex(ValueError, "mid2.*after.*mid1"):
                renderer.render()
            writer.assert_not_called()
            plotter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
