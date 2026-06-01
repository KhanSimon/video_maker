from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (Path(base_dir) if base_dir else project_root()) / value


def ensure_parent(path: str | Path) -> Path:
    value = Path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    return value
