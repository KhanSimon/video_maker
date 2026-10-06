from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from video_maker.config import resolve_config_paths
from video_maker.io.loaders import SequenceData
from video_maker.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)
LOD_VERTEX_COUNTS = (73639, 18439, 10661, 4899, 2461, 971, 595)


def mhr_enabled(cfg: dict[str, Any], default: bool = False, *, field: str = "MHR") -> bool:
    enabled = cfg.get("MHR", default)
    if not isinstance(enabled, bool):
        raise ValueError(f"{field} must be a YAML boolean: true or false (without quotes)")
    return enabled


def renderer_mhr_enabled(cfg: dict[str, Any], section: str, options: dict[str, Any]) -> bool:
    """Local clip options override renderer defaults, then the legacy global switch."""
    default = mhr_enabled(cfg)
    default = mhr_enabled(cfg.get(section, {}), default, field=f"{section}.MHR")
    return mhr_enabled(options, default, field=f"{section} clip.MHR")


def _load_faces(path: Path, vertex_count: int) -> np.ndarray:
    faces = np.load(path, allow_pickle=False)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= vertex_count
    ):
        raise ValueError(f"Invalid MHR triangles in {path}: expected integer indices (F,3) in 0..{vertex_count - 1}")
    return faces.astype(np.int64, copy=False)


class MHRData:
    """Read exported SAM 3D Body LOD 1 vertices one frame at a time.

    Vertices and xrel_sam must share their source origin, camera axes and meters.
    The same root anchoring and camera calibration then place both in the world.
    No parametric model or shape/body-pose files are needed for playback.
    """

    def __init__(self, cfg: dict[str, Any], sequence: SequenceData) -> None:
        self.sequence = sequence
        options = cfg.get("mhr", {})
        self.lod = options.get("lod", 1)
        if type(self.lod) is not int or not 0 <= self.lod <= 6:
            raise ValueError("mhr.lod must be an integer from 0 to 6")
        paths = resolve_config_paths(cfg)
        vertices_path = paths.get("vertices") or resolve_path(
            f"data/vertices/{sequence.name}.npy", cfg.get("_base_dir")
        )
        if not vertices_path.is_file():
            raise FileNotFoundError(f"MHR: vertices missing at {vertices_path}; set paths.vertices or MHR: false")
        if vertices_path.suffix != ".npy":
            raise ValueError("paths.vertices must be an uncompressed .npy file for memory-mapped playback")
        vertices = np.load(vertices_path, mmap_mode="r", allow_pickle=False)
        if vertices.ndim != 4 or vertices.shape[2:] != (LOD_VERTEX_COUNTS[1], 3):
            raise ValueError(f"MHR: expected vertices (T,N,18439,3) or (N,T,18439,3), got {vertices.shape}")
        if not np.issubdtype(vertices.dtype, np.floating):
            raise ValueError("MHR vertices must contain floating-point coordinates in meters")
        num_players, num_frames = sequence.poses.x_world.shape[:2]
        layout = str(options.get("input_layout", "auto")).lower()
        if layout == "auto":
            nt = vertices.shape[0] == num_players and vertices.shape[1] >= num_frames
            tn = vertices.shape[1] == num_players and vertices.shape[0] >= num_frames
            if nt == tn:
                raise ValueError(
                    f"Cannot align MHR vertices {vertices.shape[:2]} with N={num_players}, T={num_frames}; "
                    "set mhr.input_layout to nt or tn"
                )
            layout = "nt" if nt else "tn"
        if layout not in {"nt", "tn"}:
            raise ValueError("mhr.input_layout must be auto, nt or tn")
        if layout == "tn":
            vertices = vertices.swapaxes(0, 1)
        if vertices.shape[0] != num_players or vertices.shape[1] < num_frames:
            raise ValueError(f"MHR vertices {vertices.shape[:2]} do not cover N={num_players}, T={num_frames}")
        self.vertices = vertices[:, :num_frames]
        pose_options = cfg.get("poses", {})
        self.axis_signs = np.asarray(pose_options.get("xrel_axis_signs", [1, 1, 1]), dtype=np.float32)
        if self.axis_signs.shape != (3,) or not np.isin(self.axis_signs, [-1, 1]).all():
            raise ValueError("poses.xrel_axis_signs must contain three values equal to -1 or 1")
        self.root_joint_index = pose_options.get("root_joint_index", 8)
        if self.root_joint_index is not None and (
            type(self.root_joint_index) is not int or not 0 <= self.root_joint_index < 25
        ):
            raise ValueError("poses.root_joint_index must be an integer in 0..24 or null")

        assets_dir = paths.get("mhr_assets") or resolve_path("data/mhr", cfg.get("_base_dir"))
        required = [assets_dir / "faces_lod1.npy", assets_dir / f"faces_lod{self.lod}.npy"]
        if self.lod != 1:
            required.append(assets_dir / f"lod1_to_lod{self.lod}_mapping.npz")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MHR topology/mapping missing: " + ", ".join(missing)
                + ". Run python scripts/prepare_mhr_assets.py (or set paths.mhr_assets)."
            )
        source_faces = _load_faces(required[0], LOD_VERTEX_COUNTS[1])
        self.faces = _load_faces(required[1], LOD_VERTEX_COUNTS[self.lod])
        self.mapping_indices: np.ndarray | None = None
        self.mapping_weights: np.ndarray | None = None
        if self.lod != 1:
            with np.load(required[2], allow_pickle=False) as mapping:
                triangle_ids = mapping["triangle_ids"]
                weights = mapping["baryc_coords"]
            target_count = LOD_VERTEX_COUNTS[self.lod]
            if (
                triangle_ids.shape != (target_count,)
                or not np.issubdtype(triangle_ids.dtype, np.integer)
                or triangle_ids.min() < 0
                or triangle_ids.max() >= len(source_faces)
                or weights.shape != (target_count, 3)
                or not np.isfinite(weights).all()
                or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
            ):
                raise ValueError(f"Invalid MHR barycentric mapping in {required[2]}")
            self.mapping_indices = source_faces[triangle_ids]
            self.mapping_weights = weights.astype(np.float32)
        if np.prod(self.axis_signs) < 0:
            self.faces = self.faces[:, ::-1].copy()
        LOGGER.info(
            "MHR vertices=%s (%s, memory mapped), LOD=%d, vertices/player=%d, triangles/player=%d",
            self.vertices.shape, vertices_path, self.lod, LOD_VERTEX_COUNTS[self.lod], len(self.faces),
        )

    def camera_vertices(self, frame_idx: int) -> np.ndarray:
        if not 0 <= frame_idx < self.vertices.shape[1]:
            raise IndexError(f"MHR frame {frame_idx} outside [0, {self.vertices.shape[1]})")
        source = self.vertices[:, frame_idx]
        valid = np.isfinite(source).all(axis=(1, 2))
        # Some exporters use all-zero arrays for missing players.
        valid &= np.any(source != 0, axis=(1, 2))
        if self.mapping_indices is None:
            vertices = np.array(source, dtype=np.float32, copy=True)
        else:
            vertices = np.einsum("nvkc,vk->nvc", source[:, self.mapping_indices], self.mapping_weights)
        vertices *= self.axis_signs
        poses = self.sequence.poses
        if self.root_joint_index is not None:
            vertices -= poses.xrel_sam[:, frame_idx, self.root_joint_index, None, :]
        vertices += poses.root_cam[:, frame_idx, None, :]
        vertices[~valid] = np.nan
        return vertices

    def world_vertices(self, frame_idx: int) -> np.ndarray:
        cameras = self.sequence.cameras
        return (self.camera_vertices(frame_idx) - cameras.t[frame_idx]) @ cameras.r[frame_idx]
