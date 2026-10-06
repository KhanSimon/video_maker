from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from video_maker.assembly.timeline import render_full_demo, timeline_frame_ranges
from video_maker.config import load_config
from video_maker.io.video_io import VideoInfo


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / "configs/sequence_demo.yaml")

    def test_demo_uses_every_source_frame_once_in_order(self):
        ranges = timeline_frame_ranges(self.cfg["timeline"])
        frames = [frame for start, end in ranges for frame in range(start, end)]
        self.assertEqual(frames, list(range(ranges[0][0], ranges[-1][1])))
        self.assertEqual(ranges[-2][1] - 1, 399)
        self.assertEqual(ranges[-1][0], 400)

    def test_original_broadcast_drone_overlap_reports_25_replayed_frames(self):
        timeline = deepcopy(self.cfg["timeline"])
        timeline[-1]["start_frame"] = 375
        with self.assertRaisesRegex(ValueError, "overlap.*25 source frames would be replayed.*start_frame to 400"):
            timeline_frame_ranges(timeline)

    def test_gap_reports_skipped_frames(self):
        timeline = deepcopy(self.cfg["timeline"])
        timeline[-1]["start_frame"] = 405
        with self.assertRaisesRegex(ValueError, "gap.*5 source frames would be skipped"):
            timeline_frame_ranges(timeline)

    def test_first_clip_can_start_at_any_absolute_source_frame(self):
        timeline = [
            {"name": "a", "start_frame": 427, "end_frame": 432},
            {"name": "b", "start_frame": 432, "end_frame": 437},
        ]
        self.assertEqual(timeline_frame_ranges(timeline), [(427, 432), (432, 437)])

    def test_duplicate_clip_names_cannot_overwrite_earlier_frames(self):
        timeline = [
            {"name": "same", "start_frame": 0, "end_frame": 1},
            {"name": "same", "start_frame": 1, "end_frame": 2},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate timeline name"):
            timeline_frame_ranges(timeline)

    def test_invalid_or_noninteger_frame_ranges_are_rejected(self):
        for start, end in ((0, 0), (-1, 1), (1, 0), (0, 0.001), (np.nan, 1), (0, np.inf), (0.0, 5), (False, 1), (0, "5")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                timeline_frame_ranges([{"name": "bad", "start_frame": start, "end_frame": end}])
        with self.assertRaisesRegex(ValueError, "non-empty timeline"):
            timeline_frame_ranges([])

    def test_overlap_fails_before_renderers_or_output_writers_are_created(self):
        self.cfg["timeline"][-1]["start_frame"] = 375
        data = SimpleNamespace(video_info=VideoInfo(1920, 1080, 50, 1241))
        with (
            patch("video_maker.assembly.timeline.load_sequence", return_value=data),
            patch("video_maker.assembly.timeline.BroadcastRenderer") as broadcast,
            patch("video_maker.assembly.timeline.PyVistaDroneRenderer") as drone,
            patch("video_maker.assembly.timeline.make_writer") as writer,
        ):
            with self.assertRaisesRegex(ValueError, "25 source frames"):
                render_full_demo(self.cfg)
            broadcast.assert_not_called()
            drone.assert_not_called()
            writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
