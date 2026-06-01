from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pose_video_demo.assembly.timeline import render_full_demo
from pose_video_demo.config import load_config
from pose_video_demo.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sequence_demo.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    render_full_demo(cfg)


if __name__ == "__main__":
    main()
