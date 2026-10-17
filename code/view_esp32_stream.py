"""Quick viewer for ESP32-CAM MJPEG stream with FPS counter."""

from __future__ import annotations

import argparse
import collections
import time

import cv2


_FPS_WINDOW = 30  # number of recent frames used for the rolling FPS average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View ESP32 MJPEG stream")
    parser.add_argument("--url", required=True, help="Example: http://192.168.1.50:81/stream")
    return parser.parse_args()


def draw_fps(frame: cv2.typing.MatLike, fps: float) -> None:
    label = f"FPS: {fps:.1f}"
    x, y = 10, 30
    cv2.putText(frame, label, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y),         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open stream: {args.url}")

    timestamps: collections.deque[float] = collections.deque(maxlen=_FPS_WINDOW)

    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed; exiting.")
            break

        now = time.monotonic()
        timestamps.append(now)

        if len(timestamps) >= 2:
            fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        else:
            fps = 0.0

        draw_fps(frame, fps)
        cv2.imshow("ESP32 Stream", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

