from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_COLORS: dict[str, list[int]] = {
    "team_a": [30, 120, 255],
    "team_b": [255, 140, 0],
    "fallback": [180, 180, 180],
    "referee": [230, 230, 230],
    "pitch_lines": [255, 0, 0],
    "floor_green": [190, 235, 190],
    "background_white": [255, 255, 255],
}


def as_rgb(value: Iterable[int] | np.ndarray) -> tuple[int, int, int]:
    vals = [int(v) for v in value]
    if len(vals) != 3:
        raise ValueError(f"Expected RGB color with 3 values, got {vals}")
    return tuple(max(0, min(255, v)) for v in vals)


def as_bgr(value: Iterable[int] | np.ndarray) -> tuple[int, int, int]:
    r, g, b = as_rgb(value)
    return b, g, r


def color_table(config_colors: dict | None = None) -> dict[str, tuple[int, int, int]]:
    colors = dict(DEFAULT_COLORS)
    if config_colors:
        colors.update(config_colors)
    return {key: as_rgb(value) for key, value in colors.items()}


def team_key(value: str | None) -> str:
    if value is None:
        return "fallback"
    cleaned = value.strip().lower()
    if cleaned in {"a", "blue", "bleu", "team_a", "france"}:
        return "team_a"
    if cleaned in {"b", "orange", "team_b", "argentina", "argentine"}:
        return "team_b"
    if cleaned in {"ref", "referee", "arbitre"}:
        return "referee"
    return "fallback"
