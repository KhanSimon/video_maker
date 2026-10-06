"""Offline root smoothing in world coordinates, with timing expressed in frames."""
from __future__ import annotations

from numbers import Integral

import numpy as np


def _quadratic_smooth(values: np.ndarray, window_frames: int) -> np.ndarray:
    """Savitzky-Golay quadratic fit; boundary values use the first/last window.

    The centered interior kernel is symmetric. Quadratic trajectories are
    preserved, including at the boundaries, without a causal filter's lag.
    """
    window = min(window_frames, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return values.copy()
    radius = window // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    design = np.polynomial.polynomial.polyvander(offsets, 2)
    inverse = np.linalg.pinv(design)
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    result = np.empty_like(values)
    result[radius:-radius] = np.einsum("tdw,w->td", windows, inverse[0])
    result[:radius] = design[:radius] @ (inverse @ values[:window])
    result[-radius:] = design[-radius:] @ (inverse @ values[-window:])
    return result


def smooth_root_camera(
    root_cam: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    window_frames: int = 15,
    input_layout: str = "auto",
) -> tuple[np.ndarray, dict]:
    """Smooth each contiguous valid player track, returning the original layout.

    Camera -> world -> filter -> camera keeps broadcast camera motion separate
    from player translation. NaNs, infinities and all-zero missing roots remain
    unchanged. No samples are inserted, removed or mixed between players/gaps.
    """
    if isinstance(window_frames, bool) or not isinstance(window_frames, Integral) or window_frames < 5 or window_frames % 2 == 0:
        raise ValueError("window_frames must be an odd integer of at least 5 frames")
    root = np.asarray(root_cam)
    if root.ndim != 3 or root.shape[-1] != 3 or not np.issubdtype(root.dtype, np.floating):
        raise ValueError("root_cam must be a floating-point (N,T,3) or (T,N,3) array")
    r = np.asarray(rotations, dtype=np.float64)
    t = np.asarray(translations, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or t.shape != (len(r), 3):
        raise ValueError("Cameras must contain R (T,3,3) and t (T,3)")
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("Camera R and t must be finite")
    if not np.allclose(r @ r.swapaxes(1, 2), np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("Camera R must contain proper orthonormal rotations")
    if input_layout == "auto":
        nt, tn = root.shape[1] == len(r), root.shape[0] == len(r)
        if nt == tn:
            raise ValueError("Cannot infer root layout from camera frame count; specify --input-layout nt or tn")
        input_layout = "nt" if nt else "tn"
    if input_layout not in {"nt", "tn"}:
        raise ValueError("input_layout must be auto, nt or tn")
    roots = root.swapaxes(0, 1) if input_layout == "tn" else root
    if roots.shape[1] != len(r):
        raise ValueError(f"Root frames ({roots.shape[1]}) must match camera frames ({len(r)}); no trimming is performed")
    valid = np.isfinite(roots).all(axis=-1) & np.any(roots != 0, axis=-1)
    # Substitute only for arithmetic; invalid source entries are copied unchanged below.
    safe_roots = np.where(valid[..., None], roots, 0).astype(np.float64)
    world = np.einsum("ntc,tcd->ntd", safe_roots - t[None], r)
    filtered = world.copy()
    processed = np.zeros_like(valid)
    for player in range(len(roots)):
        indices = np.flatnonzero(valid[player])
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(run) < 5:
                continue
            filtered[player, run] = _quadratic_smooth(world[player, run], window_frames)
            processed[player, run] = True
    output = roots.copy()
    camera = np.einsum("ntd,tcd->ntc", filtered, r) + t[None]
    output[processed] = camera[processed].astype(root.dtype)
    if not np.isfinite(output[valid]).all():
        raise ValueError("Smoothing produced non-finite coordinates; output was not saved")
    # Report the data that will actually be saved, including float32 rounding.
    saved_world = np.einsum("ntc,tcd->ntd", np.where(valid[..., None], output, 0) - t[None], r)
    correction = np.linalg.norm(saved_world[valid] - world[valid], axis=-1)
    triple = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    before = np.linalg.norm(np.diff(world, n=2, axis=1), axis=-1)[triple]
    after = np.linalg.norm(np.diff(saved_world, n=2, axis=1), axis=-1)[triple]
    report = {
        "method": "savitzky_golay", "polynomial_order": 2, "space": "world",
        "window_frames": int(window_frames), "layout": input_layout,
        "shape": list(root.shape), "dtype": str(root.dtype),
        "valid_samples": int(valid.sum()), "smoothed_samples": int(processed.sum()),
        "missing_samples_preserved": int((~valid).sum()),
        "correction_median_m": float(np.median(correction)) if correction.size else None,
        "correction_p95_m": float(np.percentile(correction, 95)) if correction.size else None,
        "correction_max_m": float(correction.max()) if correction.size else None,
        "second_difference_rms_before_m_per_frame2": float(np.sqrt(np.mean(before ** 2))) if before.size else None,
        "second_difference_rms_after_m_per_frame2": float(np.sqrt(np.mean(after ** 2))) if after.size else None,
    }
    return (output.swapaxes(0, 1) if input_layout == "tn" else output), report
