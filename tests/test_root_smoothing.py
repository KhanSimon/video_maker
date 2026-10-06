from pathlib import Path
import tempfile
import unittest

import numpy as np

from video_maker.geometry.root_smoothing import smooth_root_camera
from video_maker.utils.smooth_root_cam import smooth_root_file


class RootSmoothingTests(unittest.TestCase):
    def setUp(self):
        self.frames = 81
        self.r = np.tile(np.eye(3), (self.frames, 1, 1))
        self.t = np.zeros((self.frames, 3))
        f = np.arange(self.frames, dtype=float)
        self.world = np.stack([1 + .1 * f, 3 - .02 * f + .001 * f * f, 1 + .0001 * f * f], axis=-1)[None]

    def test_preserves_quadratic_motion_and_endpoints_without_delay(self):
        for layout in ("nt", "tn"):
            raw = self.world if layout == "nt" else self.world.swapaxes(0, 1)
            out, report = smooth_root_camera(raw, self.r, self.t)
            np.testing.assert_allclose(out, raw, atol=1e-12)
            self.assertEqual(report["layout"], layout)

    def test_reduces_noise_with_a_symmetric_centered_response(self):
        rng = np.random.default_rng(4)
        noisy = self.world + rng.normal(0, .15, self.world.shape)
        out, report = smooth_root_camera(noisy, self.r, self.t)
        self.assertLess(np.mean((out - self.world) ** 2), np.mean((noisy - self.world) ** 2) * .3)
        self.assertLess(report["second_difference_rms_after_m_per_frame2"], report["second_difference_rms_before_m_per_frame2"] * .2)
        impulse = np.ones_like(self.world)
        impulse[0, 40, 0] += 1
        response, _ = smooth_root_camera(impulse, self.r, self.t)
        np.testing.assert_allclose(response[0, 33:48, 0], response[0, 33:48, 0][::-1])
        self.assertEqual(np.argmax(response[0, :, 0]), 40)

    def test_compensates_for_moving_broadcast_camera(self):
        f = np.arange(self.frames)
        angle = .4 * np.sin(f * .8)
        r = self.r.copy()
        r[:, 0, 0] = r[:, 1, 1] = np.cos(angle)
        r[:, 0, 1], r[:, 1, 0] = -np.sin(angle), np.sin(angle)
        t = np.stack([np.sin(f), np.cos(f), 30 + np.sin(2 * f)], axis=-1)
        roots = np.einsum("ntd,tcd->ntc", self.world, r) + t
        out, _ = smooth_root_camera(roots, r, t)
        np.testing.assert_allclose(out, roots, atol=1e-12)

    def test_preserves_missing_samples_short_runs_and_player_isolation(self):
        raw = np.repeat(self.world, 2, axis=0).astype(np.float32)
        raw[0, :20] = np.nan
        raw[0, 20:23] = [4, 5, 6]  # short run stays bit-identical
        raw[0, 23:40] = [np.nan, 17, np.inf]
        raw[0, 40:43] = 0
        raw[0, 43:] = [100, 200, 300]
        original = raw.copy()
        out, _ = smooth_root_camera(raw, self.r, self.t)
        invalid = ~np.isfinite(raw).all(-1) | np.all(raw == 0, axis=-1)
        np.testing.assert_array_equal(out[invalid], raw[invalid])
        np.testing.assert_array_equal(out[0, 20:23], raw[0, 20:23])
        np.testing.assert_array_equal(out[0, 43:], raw[0, 43:])
        np.testing.assert_allclose(out[1], raw[1], atol=1e-6)
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(out.dtype, raw.dtype)
        self.assertEqual(out.shape, raw.shape)

    def test_all_missing_track_and_single_frame_are_unchanged(self):
        for raw, r, t in (
            (np.full_like(self.world, np.nan), self.r, self.t),
            (self.world[:, :1], self.r[:1], self.t[:1]),
        ):
            out, report = smooth_root_camera(raw, r, t, input_layout="nt")
            np.testing.assert_array_equal(out, raw)
            self.assertEqual(report["smoothed_samples"], 0)

    def test_invalid_parameters_and_camera_alignment_fail(self):
        for window in (0, 2, 4, 10, 15.0, True):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window_frames"):
                smooth_root_camera(self.world, self.r, self.t, window_frames=window)
        with self.assertRaisesRegex(ValueError, "match camera frames"):
            smooth_root_camera(self.world[:, :-1], self.r, self.t, input_layout="nt")
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            smooth_root_camera(self.world, self.r * 2, self.t)
        with self.assertRaisesRegex(ValueError, "infer root layout"):
            smooth_root_camera(np.ones((81, 81, 3)), self.r, self.t)

    def test_cli_file_workflow_is_dry_runnable_backed_up_and_not_cumulative(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, cameras, dest = base / "root.npy", base / "camera.npz", base / "smooth.npy"
            noisy = (self.world + np.random.default_rng(0).normal(0, .1, self.world.shape)).astype(np.float32)
            np.save(source, noisy)
            np.savez(cameras, R=self.r, t=self.t)
            original_bytes = source.read_bytes()
            smooth_root_file(source, cameras, dest, dry_run=True)
            self.assertFalse(dest.exists())
            self.assertEqual(sorted(p.name for p in base.iterdir()), ["camera.npz", "root.npy"])
            smooth_root_file(source, cameras, dest)
            self.assertEqual(source.read_bytes(), original_bytes)
            report = smooth_root_file(source, cameras, source)
            self.assertEqual(Path(report["input_backup_path"]).read_bytes(), original_bytes)
            np.testing.assert_array_equal(np.load(source), np.load(dest))
            once = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "already smoothed"):
                smooth_root_file(source, cameras, source)
            self.assertEqual(source.read_bytes(), once)
            # A newly supplied root file is processed even if an old report exists.
            np.save(source, noisy + 1)
            next_report = smooth_root_file(source, cameras, source)
            self.assertNotEqual(next_report["input_backup_path"], report["input_backup_path"])


if __name__ == "__main__":
    unittest.main()
