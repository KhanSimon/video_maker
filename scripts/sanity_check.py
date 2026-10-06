from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from video_maker.config import load_config
from video_maker.io.loaders import describe_sequence, load_sequence
from video_maker.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sequence_demo.yaml")
    args = parser.parse_args()
    setup_logging()
    data = load_sequence(load_config(args.config))
    for line in describe_sequence(data):
        print(line)


if __name__ == "__main__":
    main()
