from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from video_maker.config import load_config
from video_maker.render3d.pyvista_renderer import PyVistaDroneRenderer
from video_maker.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/render_drone.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    PyVistaDroneRenderer(cfg).render()


if __name__ == "__main__":
    main()
