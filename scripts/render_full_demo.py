from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from video_maker.assembly.timeline import render_full_demo
from video_maker.config import load_config
from video_maker.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sequence_demo.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    render_full_demo(cfg)


if __name__ == "__main__":
    main()
