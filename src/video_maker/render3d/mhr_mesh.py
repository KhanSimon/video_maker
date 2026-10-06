from __future__ import annotations

import numpy as np
import pyvista as pv


def add_mhr_players(
    plotter: pv.Plotter,
    vertices: np.ndarray,
    faces: np.ndarray,
    teams: dict[int, str],
    colors: dict,
    smooth_shading: bool = True,
) -> int:
    """Add one surface per team; omit players with missing/invalid geometry."""
    # Plotter.clear() also removes the lights between video frames.
    plotter.enable_lightkit()
    valid = np.isfinite(vertices).all(axis=(1, 2))
    count = 0
    for team in ("team_a", "team_b", "fallback", "referee"):
        indices = [idx for idx in np.flatnonzero(valid) if teams.get(idx, "fallback") == team]
        if not indices:
            continue
        points = vertices[indices].reshape(-1, 3)
        offsets = np.arange(len(indices), dtype=np.int64) * vertices.shape[1]
        triangles = (faces[None] + offsets[:, None, None]).reshape(-1, 3)
        cells = np.column_stack((np.full(len(triangles), 3, dtype=np.int64), triangles))
        mesh = pv.PolyData(points, faces=cells.ravel())
        plotter.add_mesh(
            mesh,
            color=np.asarray(colors[team]) / 255.0,
            smooth_shading=smooth_shading,
            show_edges=False,
            ambient=0.35,
            diffuse=0.65,
            specular=0.1,
            specular_power=20.0,
        )
        count += len(indices)
    return count
