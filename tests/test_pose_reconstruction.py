from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from video_maker.geometry.projection import project_camera_points
from video_maker.geometry.transforms import normalize_pose_inputs, reconstruct_x_cam
from video_maker.io.loaders import CameraData, load_sequence


class PoseReconstructionTests(unittest.TestCase):
    def setUp(self):
        # A source pose whose origin is near the feet, with a nonzero pelvis.
        self.pose = np.zeros((2, 3, 25, 3), dtype=np.float64)
        self.pose[:, :, 8] = (0.1, -0.9, 0.2)
        self.pose[:, :, 0] = (0.1, -1.6, 0.2)
        self.root = np.broadcast_to((1.0, 2.0, 50.0), (2, 3, 3)).copy()
        self.root[1, :, 0] = 3.0
        self.root[:, :, 2] += np.arange(3)

    def test_mixed_layouts_place_pelvis_at_root_without_changing_bones(self):
        original = self.pose.copy()
        for pose_t in (False, True):
            for root_t in (False, True):
                with self.subTest(pose_t=pose_t, root_t=root_t):
                    pose, root = normalize_pose_inputs(
                        self.pose.swapaxes(0, 1) if pose_t else self.pose,
                        self.root.swapaxes(0, 1) if root_t else self.root,
                        expected_frames=3,
                    )
                    actual = reconstruct_x_cam(pose, root, root_joint_index=8)
                    np.testing.assert_allclose(actual[:, :, 8], self.root)
                    np.testing.assert_allclose(np.diff(actual, axis=2), np.diff(self.pose, axis=2))
        np.testing.assert_array_equal(self.pose, original)

    def test_source_origin_does_not_change_reconstruction(self):
        expected = reconstruct_x_cam(self.pose, self.root, root_joint_index=8)
        centered = self.pose - self.pose[:, :, 8:9]
        shifted = self.pose + np.array((3.0, -2.0, 1.0))
        for pose in (centered, shifted):
            np.testing.assert_allclose(reconstruct_x_cam(pose, self.root, root_joint_index=8), expected)

    def test_translation_mode_preserves_source_origin(self):
        expected = self.pose + self.root[:, :, None, :]
        np.testing.assert_array_equal(reconstruct_x_cam(self.pose, self.root), expected)
        np.testing.assert_array_equal(reconstruct_x_cam(self.pose, self.root, root_joint_index=None), expected)

    def test_other_root_joint_and_invalid_indices(self):
        actual = reconstruct_x_cam(self.pose, self.root, root_joint_index=0)
        np.testing.assert_allclose(actual[:, :, 0], self.root)
        for index in (-1, 25, 8.5, "8"):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "root_joint_index"):
                reconstruct_x_cam(self.pose, self.root, root_joint_index=index)

    def test_missing_pose_stays_hidden(self):
        self.pose[0, 0] = np.nan
        actual = reconstruct_x_cam(self.pose, self.root, root_joint_index=8)
        self.assertTrue(np.isnan(actual[0, 0]).all())
        np.testing.assert_allclose(actual[1, :, 8], self.root[1])

    def test_sequence_loader_anchors_broadcast_projection_and_world_pose(self):
        cameras = CameraData(
            k_matrix=np.tile([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]], (3, 1, 1)),
            r=np.tile(np.eye(3), (3, 1, 1)),
            t=np.zeros((3, 3)),
            distortion=np.zeros((3, 2)),
        )
        cfg = {
            "sequence": {"name": "test"},
            "paths": {name: name for name in ("video", "cameras", "xrel_sam", "root_cam")},
            "_base_dir": str(Path.cwd()),
        }
        for poses_cfg in ({}, {"root_joint_index": None}):
            cfg["poses"] = poses_cfg
            with (
                self.subTest(poses_cfg=poses_cfg),
                patch("video_maker.io.loaders.read_video_info"),
                patch("video_maker.io.loaders.load_cameras", return_value=cameras),
                patch("video_maker.io.loaders._load_array", side_effect=[self.pose.swapaxes(0, 1), self.root]),
            ):
                data = load_sequence(cfg)
                expected_root = self.root if not poses_cfg else self.root + self.pose[:, :, 8]
                np.testing.assert_allclose(data.poses.x_world[:, :, 8], expected_root)
                projected, valid = project_camera_points(data.poses.x_cam[:, 0], cameras.k_matrix[0], cameras.distortion[0])
                expected_uv = expected_root[:, 0, :2] / expected_root[:, 0, 2:3] * 1000 + (960, 540)
                np.testing.assert_allclose(projected[:, 8], expected_uv)
                self.assertTrue(valid[:, 8].all())


if __name__ == "__main__":
    unittest.main()
