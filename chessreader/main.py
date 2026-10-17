"""Chess reader entry point.

    .\\.venv\\Scripts\\python run.py play          # play a game (default command)
    .\\.venv\\Scripts\\python run.py play --start  # auto-calibrate and start the game
    .\\.venv\\Scripts\\python run.py replay sessions\\<folder> --pgn game.pgn
    .\\.venv\\Scripts\\python run.py check         # network + camera settings, no windows
    .\\.venv\\Scripts\\python run.py calibrate     # calibrate (empty board or start position), then view
    .\\.venv\\Scripts\\python run.py view          # live raw + top-down view
    .\\.venv\\Scripts\\python run.py record        # save the live stream to a video file
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, CalibrationError, load_reference, run_calibration
from .display import RED, draw_grid, draw_hud, draw_quad
from .settings import DEFAULT_CONFIG, Settings, load_settings, resolve_path
from .stream import CameraControl, CameraUnreachable, FrameGrabber, bypass_proxy, check_camera_network

logger = logging.getLogger("chessreader")

RAW_WINDOW = "Camera"
BOARD_WINDOW = "Board"


def setup_logging(settings: Settings) -> Path:
    log_dir = resolve_path(settings.logging.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(settings.logging.level.upper())
    console.setFormatter(fmt)
    file = logging.FileHandler(log_file, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers[:] = [console, file]
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def prepare_camera(settings: Settings) -> bool:
    """Check the network and push sensor settings. Returns False if the camera is unreachable."""
    cam = settings.camera
    bypass_proxy(cam.host)
    try:
        check_camera_network(cam.host, cam.http_port, cam.connect_timeout_s)
    except CameraUnreachable as exc:
        logger.error("%s", exc)
        return False
    if cam.apply_controls:
        try:
            CameraControl(cam).apply()
        except Exception as exc:  # camera settings are best effort; the stream may still work
            logger.warning("Couldn't apply camera settings: %s", exc)
    return True


def load_calibration(settings: Settings) -> Calibration | None:
    try:
        return Calibration.load(resolve_path(settings.calibration.dir))
    except FileNotFoundError:
        return None


def rotate_saved(cal: Calibration, settings: Settings) -> Calibration:
    """Turn the top-down view 90 deg clockwise and re-save it with its reference images."""
    directory = resolve_path(settings.calibration.dir)
    raw, _, _ = load_reference(directory)
    rotated = cal.rotated(1)
    rotated.save(directory, raw, settings.calibration.canny_low, settings.calibration.canny_high)
    return rotated


def save_snapshot(settings: Settings, raw: np.ndarray, warped: np.ndarray | None) -> None:
    directory = resolve_path(settings.view.captures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(str(directory / f"{stamp}_raw.png"), raw)
    if warped is not None:
        cv2.imwrite(str(directory / f"{stamp}_board.png"), warped)
    logger.info("Snapshot saved: %s", directory / f"{stamp}_*.png")


def cmd_check(settings: Settings) -> int:
    if not prepare_camera(settings):
        return 1
    try:
        status = CameraControl(settings.camera).status()
        logger.info("Camera status: %s", {k: status.get(k) for k in ("framesize", "quality", "vflip", "hmirror")})
    except Exception as exc:
        logger.warning("Couldn't read camera status: %s", exc)
    grabber = FrameGrabber.for_camera(settings.camera).start()
    try:
        first = grabber.wait_for_frame(timeout=10)
        if first is None:
            logger.error("Connected to the camera but got no frames from %s", settings.camera.stream_url)
            return 1
        h, w = first.image.shape[:2]
        t0, last = time.monotonic(), first.index
        while time.monotonic() - t0 < 3:
            frame = grabber.wait_for_frame(last, timeout=3)
            if frame is None:
                break
            last = frame.index
        logger.info("Stream OK: %dx%d, %.1f fps (%d frames in 3 s)", w, h, grabber.fps, last - first.index)
        return 0
    finally:
        grabber.stop()


def cmd_view(settings: Settings, force_calibrate: bool) -> int:
    if not prepare_camera(settings):
        logger.info("Will keep retrying in the background. Press Q in the window to quit.")
    grabber = FrameGrabber.for_camera(settings.camera).start()
    cal = None if force_calibrate else load_calibration(settings)
    try:
        if cal is None:
            cal = run_calibration(grabber, settings.calibration, settings.view.show_labels)
            if cal is None:
                logger.info("Calibration cancelled.")
                return 1
        else:
            logger.info("Loaded %s calibration from %s (created %s)", cal.method, settings.calibration.dir, cal.created_at)

        cv2.namedWindow(RAW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(BOARD_WINDOW, cv2.WINDOW_NORMAL)
        last_index = -1
        raw = warped = None
        while True:
            frame = grabber.wait_for_frame(last_index, timeout=0.5)
            if frame is not None:
                last_index = frame.index
                raw = frame.image
                h, w = raw.shape[:2]
                vis = raw.copy()
                draw_quad(vis, cal.outer_corners)
                status = "live" if grabber.connected else "RECONNECTING"
                draw_hud(vis, [f"{w}x{h}  {grabber.fps:.1f} fps  {status}",
                               "Q quit  C recalibrate  O rotate board  P snapshot"])
                cv2.imshow(RAW_WINDOW, vis)
                try:
                    warped = cal.warp(raw)
                    board = warped.copy()
                    draw_grid(board, settings.view.show_labels)
                except CalibrationError as exc:
                    warped = None
                    board = np.zeros((cal.warp_size, cal.warp_size, 3), np.uint8)
                    draw_hud(board, [str(exc)[:60], "Press C to recalibrate"], color=RED)
                cv2.imshow(BOARD_WINDOW, board)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(BOARD_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key == ord("c"):
                new = run_calibration(grabber, settings.calibration, settings.view.show_labels)
                cal = new or cal
            elif key == ord("o"):
                cal = rotate_saved(cal, settings)
                logger.info("Board view rotated 90 deg clockwise")
            elif key == ord("p") and raw is not None:
                save_snapshot(settings, raw, warped)
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


def cmd_record(settings: Settings) -> int:
    """Write the newest frame at a fixed rate so playback runs in real time."""
    if not prepare_camera(settings):
        return 1
    grabber = FrameGrabber.for_camera(settings.camera).start()
    first = grabber.wait_for_frame(timeout=10)
    if first is None:
        logger.error("No frames from the camera.")
        grabber.stop()
        return 1

    directory = resolve_path(settings.record.dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"game_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    h, w = first.image.shape[:2]
    fps = settings.record.fps
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    logger.info("Recording %dx%d at %.0f fps to %s. Press Q to stop.", w, h, fps, path)

    cv2.namedWindow(RAW_WINDOW, cv2.WINDOW_NORMAL)
    written, start = 0, time.monotonic()
    try:
        while True:
            frame = grabber.latest()
            if frame is not None and frame.image.shape[:2] == (h, w):
                writer.write(frame.image)
                written += 1
                vis = frame.image.copy()
                draw_hud(vis, [f"REC {written / fps:.0f}s  {grabber.fps:.1f} camera fps", "Q stop"], color=RED)
                cv2.imshow(RAW_WINDOW, vis)
            next_t = start + written / fps
            wait_ms = max(1, int((next_t - time.monotonic()) * 1000))
            if (cv2.waitKey(wait_ms) & 0xFF) == ord("q"):
                break
    finally:
        writer.release()
        grabber.stop()
        cv2.destroyAllWindows()
    logger.info("Saved %s (%d frames, %.0f s)", path, written, written / fps)
    return 0


def cmd_play(settings: Settings, auto_start: bool = False) -> int:
    from .play import run_play

    if not prepare_camera(settings):
        logger.info("Will keep retrying in the background.")
    grabber = FrameGrabber.for_camera(settings.camera).start()
    try:
        return run_play(settings, grabber, auto_start)
    finally:
        grabber.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chess board reader")
    parser.add_argument("command", nargs="?", default="play",
                        choices=["play", "replay", "check", "calibrate", "view", "record"])
    parser.add_argument("source", nargs="?", help="replay: a session folder or a video file")
    parser.add_argument("--pgn", help="replay: ground-truth PGN to score against")
    parser.add_argument("--llm", action="store_true", help="replay: allow tier 3 (costs API calls)")
    parser.add_argument("--start", action="store_true", help="play: calibrate and start the game automatically")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    log_file = setup_logging(settings)
    logger.info("Logging to %s", log_file)

    if args.command == "check":
        return cmd_check(settings)
    if args.command == "record":
        return cmd_record(settings)
    if args.command == "play":
        return cmd_play(settings, args.start)
    if args.command == "replay":
        if not args.source:
            parser.error("replay needs a session folder or video file")
        from .replay import run_replay

        summary = run_replay(settings, Path(args.source), Path(args.pgn) if args.pgn else None, args.llm)
        return 0 if summary["wrong_accepts"] == 0 else 2
    return cmd_view(settings, force_calibrate=args.command == "calibrate")


if __name__ == "__main__":
    sys.exit(main())
