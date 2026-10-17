"""Receive phone JPEG frames and expose them as an MJPEG stream."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import cv2
import numpy as np

LOGGER = logging.getLogger("phone_stream_bridge")
BOUNDARY = "frame"
BOUNDARY_BYTES = f"\r\n--{BOUNDARY}\r\n".encode("ascii")


def is_valid_jpeg(data: bytes) -> bool:
    return len(data) > 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


@dataclass
class FrameStore:
    """Thread-safe latest-frame store with ingest stats."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    condition: threading.Condition = field(init=False)
    latest_frame: bytes | None = None
    frame_id: int = 0
    frame_ts: float = 0.0
    sender_ip: str = ""
    ingest_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    dropped_frames: int = 0

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)

    def ingest(self, frame: bytes, sender_ip: str) -> None:
        now = time.monotonic()
        with self.condition:
            self.latest_frame = frame
            self.frame_id += 1
            self.frame_ts = now
            self.sender_ip = sender_ip
            self.ingest_timestamps.append(now)
            self.condition.notify_all()

    def mark_drop(self) -> None:
        with self.lock:
            self.dropped_frames += 1

    def ingest_fps(self, window_seconds: float = 5.0) -> float:
        cutoff = time.monotonic() - window_seconds
        with self.lock:
            while self.ingest_timestamps and self.ingest_timestamps[0] < cutoff:
                self.ingest_timestamps.popleft()
            count = len(self.ingest_timestamps)
        return count / window_seconds

    def health(self) -> dict[str, object]:
        now = time.monotonic()
        with self.lock:
            age_ms = int((now - self.frame_ts) * 1000) if self.frame_ts else None
            return {
                "frame_id": self.frame_id,
                "last_frame_age_ms": age_ms,
                "ingest_fps_5s": round(self.ingest_fps(), 2),
                "last_sender_ip": self.sender_ip or None,
                "dropped_frames": self.dropped_frames,
                "has_frame": self.latest_frame is not None,
            }


STORE = FrameStore()


def apply_requested_flip(frame: bytes, flip_h: bool, flip_v: bool) -> bytes | None:
    """Apply optional phone-side orientation controls before publishing."""

    if not flip_h and not flip_v:
        return frame

    decoded = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return None

    if flip_h and flip_v:
        flip_code = -1
    elif flip_h:
        flip_code = 1
    else:
        flip_code = 0

    flipped = cv2.flip(decoded, flip_code)
    ok, encoded = cv2.imencode(".jpg", flipped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return encoded.tobytes()


class BridgeHandler(BaseHTTPRequestHandler):
    """HTTP bridge endpoints: POST /frame, GET /stream, GET /health."""

    server_version = "PhoneMJPEGBridge/0.1"
    protocol_version = "HTTP/1.0"
    stream_clients: ClassVar[int] = 0
    stream_clients_lock: ClassVar[threading.Lock] = threading.Lock()

    def log_message(self, fmt: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Flip-H, X-Flip-V")

    def do_OPTIONS(self) -> None:  # noqa: N802
        if self.path not in {"/frame", "/health"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        LOGGER.info("do_POST path=%s from=%s len=%s", self.path, self.client_address[0], self.headers.get("Content-Length"))
        if self.path != "/frame":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return

        if content_length <= 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "Empty body")
            return

        frame = self.rfile.read(content_length)
        if not is_valid_jpeg(frame):
            STORE.mark_drop()
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JPEG markers")
            return

        flip_h = self.headers.get("X-Flip-H", "0") == "1"
        flip_v = self.headers.get("X-Flip-V", "0") == "1"
        transformed = apply_requested_flip(frame, flip_h=flip_h, flip_v=flip_v)
        if transformed is None:
            STORE.mark_drop()
            self.send_error(HTTPStatus.BAD_REQUEST, "JPEG transform failed")
            return

        STORE.ingest(transformed, sender_ip=self.client_address[0])

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_GET(self) -> None:  # noqa: N802
        LOGGER.info("do_GET path=%s from=%s", self.path, self.client_address[0])
        if self.path == "/health":
            self._send_health()
            return
        if self.path == "/stream":
            self._stream_mjpeg()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _send_health(self) -> None:
        payload = STORE.health()
        with self.stream_clients_lock:
            payload["stream_clients"] = self.stream_clients

        data = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream_mjpeg(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        with self.stream_clients_lock:
            BridgeHandler.stream_clients += 1

        last_sent_frame_id = -1
        try:
            while True:
                with STORE.condition:
                    STORE.condition.wait_for(lambda: STORE.frame_id != last_sent_frame_id, timeout=1.0)
                    frame = STORE.latest_frame
                    current_id = STORE.frame_id

                if frame is None or current_id == last_sent_frame_id:
                    continue

                header = (
                    BOUNDARY_BYTES
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_sent_frame_id = current_id
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        finally:
            with self.stream_clients_lock:
                BridgeHandler.stream_clients = max(0, BridgeHandler.stream_clients - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phone JPEG -> MJPEG bridge")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    server.daemon_threads = True
    LOGGER.info("Bridge listening on http://%s:%s", args.host, args.port)
    LOGGER.info("Endpoints: POST /frame, GET /stream, GET /health")

    def status_logger() -> None:
        while True:
            time.sleep(5.0)
            health = STORE.health()
            LOGGER.info(
                "ingest_fps=%.2f clients=%d last_sender=%s drops=%d age_ms=%s",
                health["ingest_fps_5s"],
                BridgeHandler.stream_clients,
                health["last_sender_ip"],
                health["dropped_frames"],
                health["last_frame_age_ms"],
            )

    logger_thread = threading.Thread(target=status_logger, daemon=True)
    logger_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down bridge.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
