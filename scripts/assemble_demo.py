from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from video_maker.assembly.timeline import assemble_clips
from video_maker.config import load_config
from video_maker.io.loaders import load_sequence
from video_maker.utils.logging import setup_logging
from video_maker.utils.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/assemble_demo.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    data = load_sequence(cfg)
    fps = float(cfg["render"].get("output_fps") or data.video_info.fps)
    width = int(cfg["render"].get("width") or data.video_info.width)
    height = int(cfg["render"].get("height") or data.video_info.height)
    clips = [resolve_path(path, cfg.get("_base_dir")) for path in cfg.get("clips", [])]
    output_path = resolve_path(cfg.get("output_path", "outputs/final/final_demo.mp4"), cfg.get("_base_dir"))
    assemble_clips(clips, Path(output_path), fps=fps, width=width, height=height)


if __name__ == "__main__":
    main()
