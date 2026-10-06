from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from video_maker.geometry.transforms import (
    camera_to_world_row_vector,
    normalize_pose_inputs,
    reconstruct_x_cam,
)
from video_maker.utils.paths import ensure_parent, project_root, resolve_path


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 1:
            return data[data.files[0]]
        for key in ("arr_0", "xrel_sam", "root_cam", "root_cam_pred", "data"):
            if key in data.files:
                return data[key]
        raise ValueError(f"Could not choose array inside {path}; keys={data.files}")
    return np.load(path, allow_pickle=False)


def sequence_names(xrel_dir: Path, requested: str | None) -> list[str]:
    if requested:
        return [requested]
    names = sorted(path.stem for path in xrel_dir.glob("*.npy"))
    names.extend(path.stem for path in xrel_dir.glob("*.npz"))
    if not names:
        raise FileNotFoundError(f"No .npy/.npz sequence found in {xrel_dir}")
    return sorted(set(names))


def find_existing_sequence_file(directory: Path, sequence: str) -> Path:
    for suffix in (".npy", ".npz"):
        path = directory / f"{sequence}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing file for sequence {sequence} in {directory}")


def build_x_world_for_sequence(
    sequence: str,
    xrel_dir: Path,
    root_dir: Path,
    cameras_dir: Path,
    output_dir: Path,
    dtype: np.dtype,
    layout: str,
    overwrite: bool,
    input_layout: str = "auto",
    xrel_axis_signs: tuple[float, float, float] = (1.0, 1.0, 1.0),
    root_joint_index: int | None = 8,
) -> Path:
    xrel_path = find_existing_sequence_file(xrel_dir, sequence)
    root_path = find_existing_sequence_file(root_dir, sequence)
    cameras_path = cameras_dir / f"{sequence}.npz"
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing camera file: {cameras_path}")

    output_path = output_dir / f"{sequence}.npy"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    xrel_sam = np.asarray(load_array(xrel_path), dtype=np.float32)
    root_cam = np.asarray(load_array(root_path), dtype=np.float32)
    cameras = np.load(cameras_path)
    r = np.asarray(cameras["R"], dtype=np.float64)
    t = np.asarray(cameras["t"], dtype=np.float64)

    xrel_sam, root_cam = normalize_pose_inputs(
        xrel_sam,
        root_cam,
        input_layout=input_layout,
        expected_frames=r.shape[0],
        xrel_axis_signs=xrel_axis_signs,
    )
    x_cam = reconstruct_x_cam(xrel_sam, root_cam, root_joint_index=root_joint_index)
    usable_t = min(x_cam.shape[1], r.shape[0], t.shape[0])
    if usable_t != x_cam.shape[1]:
        print(f"{sequence}: trimming T from {x_cam.shape[1]} to {usable_t}")
        x_cam = x_cam[:, :usable_t]
    x_world_ntj = camera_to_world_row_vector(x_cam, r[:usable_t], t[:usable_t]).astype(dtype, copy=False)
    if layout == "tnj3":
        x_world = np.transpose(x_world_ntj, (1, 0, 2, 3))
    elif layout == "ntj3":
        x_world = x_world_ntj
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    ensure_parent(output_path)
    np.save(output_path, x_world)
    print(f"{sequence}: x_cam={x_cam.shape} x_world_saved={x_world.shape} layout={layout} -> {output_path}")
    return output_path


def compare_to_gt(output_path: Path, gt_dir: Path) -> None:
    gt_path = gt_dir / output_path.name
    if not gt_path.exists():
        return
    pred = np.load(output_path, allow_pickle=False)
    gt = np.load(gt_path, allow_pickle=False)
    if pred.shape != gt.shape:
        print(f"{output_path.stem}: gt shape differs pred={pred.shape} gt={gt.shape}")
        return
    diff = np.abs(pred.astype(np.float64) - gt.astype(np.float64))
    print(
        f"{output_path.stem}: compare gt mean_abs={float(np.nanmean(diff)):.6f} "
        f"max_abs={float(np.nanmax(diff)):.6f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build estimated world-coordinate poses from xrel_sam/root_cam/cameras.")
    parser.add_argument("--sequence", default=None, help="Sequence name. If omitted, all sequences in xrel dir are processed.")
    parser.add_argument("--xrel-dir", default="data/xrel_sam")
    parser.add_argument("--root-dir", default="data/root_cam")
    parser.add_argument("--cameras-dir", default="data/cameras")
    parser.add_argument("--output-dir", default="data/x_world_estim")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--layout", choices=["ntj3", "tnj3"], default="ntj3", help="Saved layout. ntj3 is the project convention.")
    parser.add_argument(
        "--input-layout",
        choices=["auto", "nt", "tn"],
        default="auto",
        help="Layout of xrel/root inputs before conversion.",
    )
    parser.add_argument(
        "--root-joint-index",
        type=lambda value: None if value.lower() == "none" else int(value),
        default=8,
        help="Joint positioned at root_cam (default: 8, pelvis); 'none' for a source-origin translation.",
    )
    parser.add_argument(
        "--xrel-axis-signs",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Optional sign correction applied to relative camera-space poses.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compare-gt-dir", default="data/x_world_gt", help="Optional GT directory for a simple diff report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = project_root()
    xrel_dir = resolve_path(args.xrel_dir, base)
    root_dir = resolve_path(args.root_dir, base)
    cameras_dir = resolve_path(args.cameras_dir, base)
    output_dir = resolve_path(args.output_dir, base)
    gt_dir = resolve_path(args.compare_gt_dir, base) if args.compare_gt_dir else None
    dtype = np.dtype(args.dtype)

    for sequence in sequence_names(xrel_dir, args.sequence):
        output_path = build_x_world_for_sequence(
            sequence,
            xrel_dir=xrel_dir,
            root_dir=root_dir,
            cameras_dir=cameras_dir,
            output_dir=output_dir,
            dtype=dtype,
            layout=args.layout,
            overwrite=args.overwrite,
            input_layout=args.input_layout,
            xrel_axis_signs=tuple(args.xrel_axis_signs),
            root_joint_index=args.root_joint_index,
        )
        if gt_dir is not None:
            compare_to_gt(output_path, gt_dir)


if __name__ == "__main__":
    main()
