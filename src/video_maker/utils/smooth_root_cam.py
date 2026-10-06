"""Explicit preprocessing command; video rendering never invokes this filter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from video_maker.geometry.root_smoothing import smooth_root_camera
from video_maker.utils.paths import resolve_path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def smooth_root_file(
    input_path: Path,
    cameras_path: Path,
    output_path: Path,
    *,
    window_frames: int = 15,
    input_layout: str = "auto",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    input_path, cameras_path, output_path = (p.resolve() for p in (input_path, cameras_path, output_path))
    if input_path.suffix != ".npy" or output_path.suffix != ".npy":
        raise ValueError("Input and output must be .npy files")
    same_file = input_path == output_path or (output_path.exists() and input_path.samefile(output_path))
    if not same_file and output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{output_path} exists; use --overwrite to replace it with a backup")
    input_hash = _sha256(input_path)
    report_path = output_path.with_suffix(".smoothing.json")
    if same_file and report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("output_sha256") == input_hash:
            raise ValueError(
                "This file is already smoothed. Use --input with the original backup "
                f"({previous.get('input_backup_path')}) to change the window without filtering twice."
            )
    raw = np.load(input_path, allow_pickle=False)
    with np.load(cameras_path, allow_pickle=False) as cameras:
        output, report = smooth_root_camera(raw, cameras["R"], cameras["t"],
                                            window_frames=window_frames, input_layout=input_layout)
    report.update(input_path=str(input_path), output_path=str(output_path), cameras_path=str(cameras_path),
                  input_sha256=input_hash, cameras_sha256=_sha256(cameras_path), dry_run=dry_run)
    if dry_run:
        return report
    # Complete computation/validation before any source or destination mutation.
    if _sha256(input_path) != input_hash:
        raise RuntimeError("Input changed during smoothing; nothing was saved. Run again with the new input.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output_path.exists():
        backup_dir = output_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = backup_dir / f"{output_path.stem}.{stamp}.npy"
        shutil.copy2(output_path, backup)
        report["replaced_file_backup_path"] = str(backup)
        report["replaced_file_sha256"] = _sha256(backup)
    report["input_backup_path"] = str(backup if same_file else input_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npy", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, output, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        report["output_sha256"] = _sha256(temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth root_cam separately from rendering, using a centered world-space filter.")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--input", help="Default: data/root_cam/<sequence>.npy")
    parser.add_argument("--cameras", help="Default: data/cameras/<sequence>.npz")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", help="Default: data/root_cam_smoothed/<sequence>.npy")
    destination.add_argument("--in-place", action="store_true", help="Replace the input after making a timestamped backup")
    parser.add_argument("--window-frames", type=int, default=15, help="Odd frame window, at least 5; larger values smooth more")
    parser.add_argument("--input-layout", choices=["auto", "nt", "tn"], default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing a separate output, with a backup")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics without writing any file")
    args = parser.parse_args()
    source = resolve_path(args.input or f"data/root_cam/{args.sequence}.npy")
    destination_path = source if args.in_place else resolve_path(args.output or f"data/root_cam_smoothed/{args.sequence}.npy")
    report = smooth_root_file(source, resolve_path(args.cameras or f"data/cameras/{args.sequence}.npz"), destination_path,
                              window_frames=args.window_frames, input_layout=args.input_layout,
                              overwrite=args.overwrite, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
