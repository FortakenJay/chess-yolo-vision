"""CLI entrypoint for chess vision recorder."""

from __future__ import annotations

import argparse
import logging

from src.pipeline import ChessVisionPipeline
from src.settings import load_settings
from src.types import Orientation

_ORIENTATION_MENU = """\
Camera orientation:
  1 - Overhead, white at BOTTOM  (egocentric top-down, you sit behind white)
  2 - Overhead, white at TOP     (egocentric top-down, you sit behind black)
  3 - Side / tripod, white on LEFT
  4 - Side / tripod, white on RIGHT
"""

_ORIENTATION_MAP: dict[str, Orientation] = {
    "1": Orientation.OVERHEAD_WHITE_BOTTOM,
    "2": Orientation.OVERHEAD_WHITE_TOP,
    "3": Orientation.SIDE_WHITE_LEFT,
    "4": Orientation.SIDE_WHITE_RIGHT,
}


def _prompt_orientation() -> Orientation:
    print(_ORIENTATION_MENU)
    while True:
        choice = input("Enter 1-4: ").strip()
        if choice in _ORIENTATION_MAP:
            return _ORIENTATION_MAP[choice]
        print("Invalid choice, please enter 1, 2, 3, or 4.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chess Vision Recorder")
    parser.add_argument("--start", action="store_true", help="Start game immediately")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    settings = load_settings(args.config)
    orientation = _prompt_orientation()

    pipeline = ChessVisionPipeline(settings=settings, orientation=orientation)
    if args.start:
        pipeline.start_game()
    pipeline.run()


if __name__ == "__main__":
    main()

