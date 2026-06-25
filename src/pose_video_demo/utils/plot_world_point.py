from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pose_video_demo.pitch.pitch_loader import infer_pitch_segments, load_pitch_points
from pose_video_demo.utils.paths import ensure_parent, project_root, resolve_path


# Option pratique quand tu veux lancer ce fichier/script sans arguments CLI.
# Convention spherique :
# - radius_m : distance en metres depuis origin
# - azimuth_deg : angle dans le plan XY, 0 deg = +X, 90 deg = +Y
# - elevation_deg : angle au-dessus du plan XY, 0 deg = sol, 90 deg = +Z
CARTESIAN_POINT: tuple[float, float, float] | None = None
SPHERICAL_POINT: dict[str, object] | None = None
CAMERAS_PATH: str | None = "data/cameras/ARG_FRA_182345.npz"
# Exemple :
# SPHERICAL_POINT = {
#     "radius_m": 30.0,
#     "azimuth_deg": 45.0,
#     "elevation_deg": 15.0,
#     "origin": (0.0, 0.0, 0.0),
# }


def spherical_to_world(
    radius_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Convert spherical coordinates to world XYZ meters.

    Azimuth is measured in the XY plane from +X toward +Y. Elevation is measured
    above the XY plane toward +Z.
    """
    radius = float(radius_m)
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    ox, oy, oz = origin
    xy_radius = radius * np.cos(elevation)
    x = ox + xy_radius * np.cos(azimuth)
    y = oy + xy_radius * np.sin(azimuth)
    z = oz + radius * np.sin(elevation)
    return float(x), float(y), float(z)


def spherical_dict_to_world(params: dict[str, object]) -> tuple[float, float, float]:
    return spherical_to_world(
        radius_m=float(params["radius_m"]),
        azimuth_deg=float(params["azimuth_deg"]),
        elevation_deg=float(params["elevation_deg"]),
        origin=tuple(params.get("origin", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
    )


def load_mean_camera_pose(cameras_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cameras and return mean world center plus mean optical direction."""
    data = np.load(cameras_path)
    r = np.asarray(data["R"], dtype=float)
    t = np.asarray(data["t"], dtype=float)
    centers = np.einsum("ti,tij->tj", -t, r)
    forwards = np.einsum("i,tij->tj", np.array([0.0, 0.0, 1.0]), r)
    mean_center = centers.mean(axis=0)
    mean_forward = forwards.mean(axis=0)
    norm = np.linalg.norm(mean_forward)
    if norm > 1e-9:
        mean_forward = mean_forward / norm
    return mean_center, mean_forward


def set_axes_equal(ax) -> None:
    """Keep x/y/z meters visually comparable in Matplotlib 3D."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_pitch_with_point(
    pitch_points_path: Path,
    point_xyz: tuple[float, float, float],
    cameras_path: Path | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    points = load_pitch_points(pitch_points_path)
    segments = infer_pitch_segments(points)
    point = np.asarray(point_xyz, dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for a, b in segments:
        line = points[[a, b]]
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="red", linewidth=1.0)

    ax.scatter([0.0], [0.0], [0.0], color="black", s=45, label="origine monde (0,0,0)")
    ax.scatter([point[0]], [point[1]], [point[2]], color="blue", s=80, label=f"point ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
    ax.text(point[0], point[1], point[2], f"  ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})", color="blue")

    extra_points = [point]
    if cameras_path is not None:
        camera_center, camera_forward = load_mean_camera_pose(cameras_path)
        extra_points.append(camera_center)
        ax.scatter(
            [camera_center[0]],
            [camera_center[1]],
            [camera_center[2]],
            color="purple",
            marker="^",
            s=100,
            label="camera moyenne",
        )
        ax.quiver(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            camera_forward[0],
            camera_forward[1],
            camera_forward[2],
            length=10.0,
            color="purple",
            linewidth=2.0,
        )
        ax.text(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            f"  camera ({camera_center[0]:.2f}, {camera_center[1]:.2f}, {camera_center[2]:.2f})",
            color="purple",
        )

    ax.set_xlabel("X monde (m)")
    ax.set_ylabel("Y monde (m)")
    ax.set_zlabel("Z monde (m)")
    ax.set_title("Terrain en coordonnees monde")
    ax.legend(loc="upper right")
    ax.view_init(elev=32, azim=-55)
    ax.set_box_aspect((1, 1, 0.25))

    extra_points_arr = np.asarray(extra_points)
    mins = np.minimum(points.min(axis=0), extra_points_arr.min(axis=0))
    maxs = np.maximum(points.max(axis=0), extra_points_arr.max(axis=0))
    pad = np.array([5.0, 5.0, 3.0])
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(min(mins[2] - pad[2], -1.0), max(maxs[2] + pad[2], 3.0))
    set_axes_equal(ax)
    fig.tight_layout()

    if output_path is not None:
        ensure_parent(output_path)
        fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_pitch_with_spherical_point(
    pitch_points_path: Path,
    radius_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cameras_path: Path | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> tuple[float, float, float]:
    point_xyz = spherical_to_world(radius_m, azimuth_deg, elevation_deg, origin)
    plot_pitch_with_point(
        pitch_points_path,
        point_xyz,
        cameras_path=cameras_path,
        output_path=output_path,
        show=show,
    )
    return point_xyz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Afficher le terrain et un point en coordonnees monde.")
    parser.add_argument("--x", type=float, default=None, help="Coordonnee X monde en metres.")
    parser.add_argument("--y", type=float, default=None, help="Coordonnee Y monde en metres.")
    parser.add_argument("--z", type=float, default=None, help="Coordonnee Z monde en metres.")
    parser.add_argument("--pitch-points", default="data/pitch/pitch_points.txt", help="Chemin vers pitch_points.txt.")
    parser.add_argument("--cameras", default=CAMERAS_PATH, help="Chemin vers le .npz cameras, ou vide pour masquer la camera.")
    parser.add_argument("--output", default=None, help="Chemin PNG optionnel pour sauvegarder la figure.")
    parser.add_argument("--no-show", action="store_true", help="Ne pas ouvrir la fenetre Matplotlib.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pitch_points_path = resolve_path(args.pitch_points)
    cameras_path = resolve_path(args.cameras) if args.cameras else None
    output_path = resolve_path(args.output) if args.output else None

    if args.x is not None and args.y is not None and args.z is not None:
        point_xyz = (args.x, args.y, args.z)
        plot_pitch_with_point(
            pitch_points_path,
            point_xyz,
            cameras_path=cameras_path,
            output_path=output_path,
            show=not args.no_show,
        )
        print(f"Point monde: {point_xyz}")
        return

    if SPHERICAL_POINT is not None:
        point_xyz = plot_pitch_with_spherical_point(
            pitch_points_path,
            radius_m=float(SPHERICAL_POINT["radius_m"]),
            azimuth_deg=float(SPHERICAL_POINT["azimuth_deg"]),
            elevation_deg=float(SPHERICAL_POINT["elevation_deg"]),
            origin=tuple(SPHERICAL_POINT.get("origin", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            cameras_path=cameras_path,
            output_path=output_path,
            show=not args.no_show,
        )
    else:
        point_xyz = plot_pitch_with_spherical_point(
            pitch_points_path,
            radius_m=30.0,
            azimuth_deg=45.0,
            elevation_deg=15.0,
            origin=(0.0, 0.0, 0.0),
            cameras_path=cameras_path,
            output_path=output_path,
            show=not args.no_show,
        )

    print(f"Point monde: {point_xyz}")


# -----------------------------------------------------------------------------
#
# Lancer :
#   uv run python scripts/plot_world_point.py --x 1 --y 2 --z 3
#
# Pour un point spherique, omettre `--x/--y/--z` et laisser
# `SPHERICAL_POINT` dans ce fichier, ou modifier la valeur par defaut dans `main()`.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
