from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from video_maker.skeleton.joints import BONES_25
from video_maker.utils.paths import ensure_parent, project_root, resolve_path


def load_pose_array(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]])
        for key in ("arr_0", "x_world", "poses", "data"):
            if key in data.files:
                return np.asarray(data[key])
        raise ValueError(f"Could not choose an array inside {path}; keys={data.files}")
    return np.asarray(data)


def select_frame(points: np.ndarray, player_index: int, frame_index: int) -> tuple[np.ndarray, str]:
    if points.ndim == 4 and points.shape[-2:] == (25, 3):
        if not (0 <= player_index < points.shape[0]):
            raise IndexError(f"player_index={player_index} out of range for {points.shape[0]} players")
        if not (0 <= frame_index < points.shape[1]):
            raise IndexError(f"frame_index={frame_index} out of range for {points.shape[1]} frames")
        return np.asarray(points[player_index, frame_index], dtype=float), f"player={player_index} frame={frame_index}"

    if points.ndim == 3 and points.shape[-2:] == (25, 3):
        if not (0 <= frame_index < points.shape[0]):
            raise IndexError(f"frame_index={frame_index} out of range for {points.shape[0]} frames")
        return np.asarray(points[frame_index], dtype=float), f"frame={frame_index}"

    if points.ndim == 2 and points.shape == (25, 3):
        return np.asarray(points, dtype=float), "single_frame"

    raise ValueError(f"Unsupported pose shape {points.shape}; expected (N,T,25,3), (T,25,3), or (25,3)")


def set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_labeled_frame(
    pose_3d: np.ndarray,
    title: str,
    output_path: Path | None = None,
    show: bool = True,
    animate_rotation: bool = False,
    interval_ms: int = 40,
    rotation_frames: int = 120,
) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    coords = np.asarray(pose_3d, dtype=float)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    for a, b in BONES_25:
        if a >= coords.shape[0] or b >= coords.shape[0]:
            continue
        segment = coords[[a, b]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#4c4c4c", linewidth=2.0, alpha=0.85)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, coords.shape[0]))
    ax.scatter(xs, ys, zs, s=30, c=colors, depthshade=True)

    for idx, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, f" {idx}", fontsize=10, color="black")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=24, azim=-55)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(span * 0.18, np.array([0.4, 0.4, 0.4]))
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    set_axes_equal(ax)
    fig.tight_layout()

    if output_path is not None:
        ensure_parent(output_path)
        fig.savefig(output_path, dpi=180)

    if animate_rotation and show:
        def update(frame: int):
            ax.view_init(elev=24, azim=-55 + (360.0 * frame / max(rotation_frames, 1)))
            return (ax,)

        animation = FuncAnimation(fig, update, frames=rotation_frames, interval=interval_ms, blit=False)
        fig._pose_animation = animation  # keep a strong reference for the interactive window

    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a labeled 3D pose frame with joints numbered 0 to 24.")
    parser.add_argument("--input", default="data/x_world_estim/ARG_FRA_182345.npy", help="Path to the pose array (.npy or .npz).")
    parser.add_argument("--player", type=int, default=0, help="Player index when the input has shape (N,T,25,3).")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to display.")
    parser.add_argument("--output", default="outputs/frames/player_frame_labels.png", help="PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive Matplotlib window.")
    parser.add_argument("--animate-rotation", action="store_true", help="Rotate the 3D camera in the display window.")
    parser.add_argument("--rotation-frames", type=int, default=120, help="Number of frames for the rotation animation.")
    parser.add_argument("--interval-ms", type=int, default=40, help="Animation interval in milliseconds.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base = project_root()
    input_path = resolve_path(args.input, base)
    output_path = resolve_path(args.output, base) if args.output else None

    pose_data = load_pose_array(input_path)
    pose_3d, selection_label = select_frame(pose_data, player_index=args.player, frame_index=args.frame)

    title = f"Pose 3D labels - {input_path.stem} - {selection_label}"
    plot_labeled_frame(
        pose_3d,
        title=title,
        output_path=output_path,
        show=not args.no_show,
        animate_rotation=args.animate_rotation,
        interval_ms=args.interval_ms,
        rotation_frames=args.rotation_frames,
    )

    print(f"Saved labeled pose frame to {output_path}" if output_path else f"Rendered {selection_label}")


if __name__ == "__main__":
    main()


"""
python scripts/plot_player_frame.py --input data/x_world_estim/ARG_FRA_182345.npy --player 0 --frame 0 --output outputs/frames/player_frame_labels.png
"""