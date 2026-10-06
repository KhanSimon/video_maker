from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_maker.io.loaders import CameraData, PoseData, SequenceData
from video_maker.io.mhr import LOD_VERTEX_COUNTS, MHRData, mhr_enabled
from video_maker.io.video_io import VideoInfo
from video_maker.geometry.transforms import camera_to_world_row_vector, reconstruct_x_cam


class MHRDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.vertices = np.zeros((2, 3, 18439, 3), dtype=np.float32)
        self.vertices[:] = (0.1, -0.9, 0.2)
        self.vertices[:, :, 1] += (0, -0.6, 0)
        self.vertices[:, :, 2] += (0.2, 0, 0)
        self.pose = np.zeros((2, 3, 25, 3), dtype=np.float32)
        self.pose[:, :, 8] = self.vertices[:, :, 0]
        self.pose[:, :, 0] = self.vertices[:, :, 1]
        self.root = np.broadcast_to((1.0, 2.0, 50.0), (2, 3, 3)).astype(np.float32).copy()
        self.root[1, :, 0] = 3.0
        self.root[:, :, 2] += np.arange(3)
        r = np.tile([[0.0, 1, 0], [0, 0, -1], [-1, 0, 0]], (3, 1, 1))
        self.cameras = CameraData(
            k_matrix=np.tile([[800.0, 0, 470], [0, 900.0, 250], [0, 0, 1]], (3, 1, 1)),
            r=r, t=np.tile((1.0, 2.0, 3.0), (3, 1)), distortion=np.zeros((3, 2)),
        )
        self.cfg = {
            "MHR": True, "mhr": {"lod": 1}, "_base_dir": str(self.folder),
            "paths": {"vertices": "vertices.npy", "mhr_assets": "assets"},
        }
        (self.folder / "assets").mkdir()
        np.save(self.folder / "assets/faces_lod1.npy", [[0, 1, 2]])
        self.save_vertices()

    def save_vertices(self, layout="nt"):
        np.save(self.folder / "vertices.npy", self.vertices if layout == "nt" else self.vertices.swapaxes(0, 1))

    def sequence(self):
        options = self.cfg.get("poses", {})
        pose = self.pose * np.asarray(options.get("xrel_axis_signs", [1, 1, 1]), dtype=np.float32)
        x_cam = reconstruct_x_cam(pose, self.root, options.get("root_joint_index", 8))
        return SequenceData(
            name="test", video_path=self.folder / "video.mov", video_info=VideoInfo(960, 540, 50, 3),
            cameras=self.cameras,
            poses=PoseData(pose, self.root, x_cam, camera_to_world_row_vector(x_cam, self.cameras.r, self.cameras.t)),
            boxes=None, teams={0: "team_a", 1: "team_b"},
        )

    def test_memory_mapped_layouts_follow_joint_camera_and_world_positions(self):
        for layout in ("nt", "tn"):
            self.save_vertices(layout)
            for pose_options in ({}, {"root_joint_index": None}, {"xrel_axis_signs": [1, -1, -1]}):
                with self.subTest(layout=layout, poses=pose_options):
                    self.cfg["poses"] = pose_options
                    sequence = self.sequence()
                    data = MHRData(self.cfg, sequence)
                    self.assertIsInstance(data.vertices, np.memmap)
                    self.assertFalse(data.vertices.flags.writeable)
                    for t in range(3):
                        np.testing.assert_allclose(data.camera_vertices(t)[:, 0], sequence.poses.x_cam[:, t, 8])
                        np.testing.assert_allclose(data.world_vertices(t)[:, 1], sequence.poses.x_world[:, t, 0])
                    original = np.load(self.folder / "vertices.npy")
                    np.testing.assert_array_equal(original, self.vertices if layout == "nt" else self.vertices.swapaxes(0, 1))

    def test_missing_vertices_and_missing_roots_hide_players(self):
        self.vertices[0, 0] = np.nan
        self.vertices[0, 1] = 0
        self.root[1, 2] = np.nan
        self.save_vertices()
        data = MHRData(self.cfg, self.sequence())
        for t, player in ((0, 0), (1, 0), (2, 1)):
            self.assertTrue(np.isnan(data.world_vertices(t)[player]).all())
        self.assertTrue(np.isfinite(data.world_vertices(0)[1]).all())

    def test_official_barycentric_mapping_is_applied_before_placement(self):
        count = LOD_VERTEX_COUNTS[6]
        np.save(self.folder / "assets/faces_lod6.npy", [[0, 1, 2]])
        np.savez(
            self.folder / "assets/lod1_to_lod6_mapping.npz",
            triangle_ids=np.zeros(count, dtype=np.int64),
            baryc_coords=np.tile([0.25, 0.5, 0.25], (count, 1)),
        )
        self.cfg["mhr"]["lod"] = 6
        data = MHRData(self.cfg, self.sequence())
        actual = data.camera_vertices(1)
        self.assertEqual(actual.shape, (2, count, 3))
        expected = self.root[:, 1] + (0.05, -0.3, 0)
        np.testing.assert_allclose(actual[:, 0], expected, atol=1e-6)

    def test_invalid_assets_and_inputs_fail_with_actionable_errors(self):
        for value in (-1, 7, 1.5, True):
            self.cfg["mhr"]["lod"] = value
            with self.subTest(lod=value), self.assertRaisesRegex(ValueError, "mhr.lod"):
                MHRData(self.cfg, self.sequence())
        self.cfg["mhr"]["lod"] = 1
        np.save(self.folder / "assets/faces_lod1.npy", [[0, 1, 18439]])
        with self.assertRaisesRegex(ValueError, "triangles"):
            MHRData(self.cfg, self.sequence())
        (self.folder / "assets/faces_lod1.npy").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "prepare_mhr_assets.py"):
            MHRData(self.cfg, self.sequence())
        (self.folder / "vertices.npy").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "paths.vertices"):
            MHRData(self.cfg, self.sequence())

    def test_vertex_layout_and_frame_coverage_are_validated(self):
        self.vertices = self.vertices[:, :2]
        self.save_vertices()
        with self.assertRaisesRegex(ValueError, "align MHR vertices"):
            MHRData(self.cfg, self.sequence())


class MHRRenderingTests(unittest.TestCase):
    def test_switch_is_a_boolean_and_defaults_to_existing_rendering(self):
        self.assertFalse(mhr_enabled({}))
        self.assertFalse(mhr_enabled({"MHR": False}))
        self.assertTrue(mhr_enabled({"MHR": True}))
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "YAML boolean"):
                mhr_enabled({"MHR": value})

    def test_broadcast_intrinsics_match_existing_projection(self):
        from video_maker.geometry.projection import project_camera_points
        from video_maker.render2d.mhr_overlay import camera_projection_matrix

        points = np.array([[0.1, -0.3, 3], [-0.8, 0.6, 5], [1.4, 2, 8]])
        k = np.array([[800.0, 0, 470], [0, 900.0, 250], [0, 0, 1]])
        width, height = 960, 540
        expected, _ = project_camera_points(points, k, np.zeros(2))
        view = np.column_stack((points * [1, -1, -1], np.ones(len(points))))
        clip = view @ camera_projection_matrix(k, width, height).T
        ndc = clip[:, :2] / clip[:, 3:4]
        actual = np.column_stack(((ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2))
        np.testing.assert_allclose(actual, expected)

    def test_meshes_keep_team_colors_and_omit_missing_players(self):
        from video_maker.render3d.mhr_mesh import add_mhr_players
        from video_maker.utils.colors import color_table

        plotter = Mock()
        vertices = np.tile([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], (3, 1, 1))
        vertices[1] = np.nan
        colors = color_table(None)
        count = add_mhr_players(plotter, vertices, np.array([[0, 1, 2]]), {0: "team_a", 1: "team_b"}, colors)
        self.assertEqual(count, 2)
        self.assertEqual(plotter.add_mesh.call_count, 2)
        self.assertEqual(plotter.add_points.call_count, 0)
        for call, team in zip(plotter.add_mesh.call_args_list, ("team_a", "fallback")):
            np.testing.assert_allclose(call.kwargs["color"], np.array(colors[team]) / 255)
            self.assertEqual(call.args[0].n_cells, 1)

    def test_disabled_renderers_never_open_mhr_files(self):
        from video_maker.render2d.broadcast_renderer import BroadcastRenderer
        from video_maker.render3d.pyvista_renderer import PyVistaDroneRenderer

        cfg = {"paths": {"pitch_points": "unused"}, "MHR": False}
        for module, renderer in (
            ("video_maker.render2d.broadcast_renderer", BroadcastRenderer),
            ("video_maker.render3d.pyvista_renderer", PyVistaDroneRenderer),
        ):
            with (
                self.subTest(renderer=renderer.__name__),
                patch(f"{module}.MHRData", side_effect=AssertionError("MHR files opened")),
                patch(f"{module}.load_pitch_points"),
                patch(f"{module}.parse_segments"),
            ):
                self.assertIsNone(renderer(cfg, data=Mock()).mhr_data)


if __name__ == "__main__":
    unittest.main()
