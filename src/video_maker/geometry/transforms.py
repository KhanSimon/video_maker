from __future__ import annotations

import numpy as np


def normalize_pose_array(
    xrel_sam: np.ndarray,
    input_layout: str = "auto",
    expected_frames: int | None = None,
    axis_signs: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return a pose array in the internal (N,T,25,3) convention."""
    xrel_sam = np.asarray(xrel_sam)
    if xrel_sam.ndim != 4 or xrel_sam.shape[2:] != (25, 3):
        raise ValueError(f"Expected pose shape (N,T,25,3) or (T,N,25,3), got {xrel_sam.shape}")
    layout = _resolve_pose_layout(
        xrel_sam.shape[:2],
        input_layout=input_layout,
        expected_frames=expected_frames,
    )
    if layout == "tn":
        xrel_sam = np.transpose(xrel_sam, (1, 0, 2, 3))
    signs = np.asarray(axis_signs, dtype=xrel_sam.dtype)
    if signs.shape != (3,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("axis_signs must contain three values equal to -1 or 1")
    return xrel_sam if np.all(signs == 1.0) else xrel_sam * signs


def normalize_pose_inputs(
    xrel_sam: np.ndarray,
    root_cam: np.ndarray,
    input_layout: str = "auto",
    expected_frames: int | None = None,
    xrel_axis_signs: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return pose inputs in (N,T,...), accepting independently ordered inputs.

    input_layout describes xrel_sam; root_cam is aligned to its player/frame axes.
    """
    root_cam = np.asarray(root_cam)
    if root_cam.ndim != 3 or root_cam.shape[2:] != (3,):
        raise ValueError(f"Expected root_cam shape (N,T,3) or (T,N,3), got {root_cam.shape}")
    raw_xrel = np.asarray(xrel_sam)
    if raw_xrel.shape[:2] != root_cam.shape[:2] and raw_xrel.shape[:2] == root_cam.shape[1::-1]:
        root_cam = np.swapaxes(root_cam, 0, 1)
    if raw_xrel.shape[:2] != root_cam.shape[:2]:
        raise ValueError(
            f"xrel_sam and root_cam player/frame dimensions differ: "
            f"{raw_xrel.shape[:2]} vs {root_cam.shape[:2]}"
        )

    layout = _resolve_pose_layout(
        raw_xrel.shape[:2],
        input_layout=input_layout,
        expected_frames=expected_frames,
    )
    xrel_sam = normalize_pose_array(
        raw_xrel,
        input_layout=layout,
        expected_frames=expected_frames,
        axis_signs=xrel_axis_signs,
    )
    if layout == "tn":
        root_cam = np.transpose(root_cam, (1, 0, 2))
    return xrel_sam, root_cam


def normalize_player_frame_array(
    data: np.ndarray,
    num_players: int,
    num_frames: int,
    name: str = "array",
) -> np.ndarray:
    """Align an array beginning with (N,T) or (T,N) to (N,T)."""
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError(f"Expected {name} to have at least two dimensions, got {data.shape}")
    if data.shape[:2] == (num_players, num_frames):
        return data
    if data.shape[:2] == (num_frames, num_players):
        axes = (1, 0, *range(2, data.ndim))
        return np.transpose(data, axes)
    raise ValueError(
        f"Could not align {name} shape {data.shape} with N={num_players}, T={num_frames}"
    )


def _resolve_pose_layout(
    leading_shape: tuple[int, int],
    input_layout: str,
    expected_frames: int | None,
) -> str:
    aliases = {
        "auto": "auto",
        "nt": "nt",
        "ntj3": "nt",
        "players_first": "nt",
        "tn": "tn",
        "tnj3": "tn",
        "frames_first": "tn",
    }
    layout = aliases.get(str(input_layout).lower())
    if layout is None:
        raise ValueError(f"Unsupported input_layout={input_layout}")
    if layout != "auto":
        return layout

    first, second = leading_shape
    if expected_frames is not None:
        first_matches = first == expected_frames
        second_matches = second == expected_frames
        if first_matches != second_matches:
            return "tn" if first_matches else "nt"
    if first == second:
        raise ValueError(
            f"Cannot infer pose layout from square leading dimensions {leading_shape}; "
            "set input_layout to 'nt' or 'tn'"
        )
    return "nt" if first < second else "tn"


def reconstruct_x_cam(
    xrel_sam: np.ndarray,
    root_cam: np.ndarray,
    root_joint_index: int | None = None,
) -> np.ndarray:
    """Reconstruct camera-space joints using a translation or a joint position.

    With root_joint_index=None, root_cam translates the source pose origin.
    Otherwise, center the source pose on that joint before placing it at root_cam.
    BODY_25 uses joint 8 for the pelvis.
    """
    if xrel_sam.ndim != 4 or xrel_sam.shape[2:] != (25, 3):
        raise ValueError(f"Expected xrel_sam shape (N,T,25,3), got {xrel_sam.shape}")
    if root_cam.shape != xrel_sam.shape[:2] + (3,):
        raise ValueError(f"Expected root_cam shape {xrel_sam.shape[:2] + (3,)}, got {root_cam.shape}")
    if root_joint_index is not None:
        if not isinstance(root_joint_index, (int, np.integer)) or not 0 <= root_joint_index < xrel_sam.shape[2]:
            raise ValueError(f"root_joint_index must be an integer in 0..24 or None, got {root_joint_index}")
        xrel_sam = xrel_sam - xrel_sam[:, :, root_joint_index : root_joint_index + 1, :]
    return xrel_sam + root_cam[:, :, None, :]


def camera_to_world_row_vector(x_cam: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert camera coordinates to world coordinates with X_world=(X_cam-t)@R."""
    if r.shape[:1] != t.shape[:1] or r.shape[1:] != (3, 3) or t.shape[1:] != (3,):
        raise ValueError(f"Invalid camera shapes R={r.shape}, t={t.shape}")
    centered = x_cam - t[None, :, None, :]
    return np.einsum("ntjc,tcd->ntjd", centered, r)


def world_to_camera_row_vector(x_world: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert world coordinates to camera coordinates, inverse of camera_to_world_row_vector."""
    return np.einsum("...d,dc->...c", x_world, r.T) + t
