from __future__ import annotations

import argparse

from video_maker.config import load_config
from video_maker.interactive_viewer.viewer import InteractivePoseViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Open3D interactive pose viewer.")
    parser.add_argument("--config", default="configs/interactive_viewer.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    InteractivePoseViewer(cfg).run()


if __name__ == "__main__":
    main()
