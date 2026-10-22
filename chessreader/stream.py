"""ESP32 CameraWebServer intake.

CameraControl pushes sensor settings over /control and verifies them.
FrameGrabber reads the MJPEG stream nonstop in a background thread and keeps only
the newest frame, reconnecting whenever the stream drops.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import requests

from .settings import CameraSettings

logger = logging.getLogger(__name__)

# framesize_t from esp32-camera as bundled with arduino-esp32 3.3.x (sensor.h).
# Older esp32-camera releases lack 128X128 and 320X320, which shifts every index,
# so CameraControl.apply() checks the real frame size after setting it.
FRAMESIZES: dict[str, tuple[int, tuple[int, int]]] = {
    "96X96": (0, (96, 96)),
    "QQVGA": (1, (160, 120)),
    "128X128": (2, (128, 128)),
    "QCIF": (3, (176, 144)),
    "HQVGA": (4, (240, 176)),
    "240X240": (5, (240, 240)),
    "QVGA": (6, (320, 240)),
    "320X320": (7, (320, 320)),
    "CIF": (8, (400, 296)),
    "HVGA": (9, (480, 320)),
    "VGA": (10, (640, 480)),
    "SVGA": (11, (800, 600)),
    "XGA": (12, (1024, 768)),
    "HD": (13, (1280, 720)),
    "SXGA": (14, (1280, 1024)),
    "UXGA": (15, (1600, 1200)),
}


class CameraUnreachableError(RuntimeError):
    """The ESP32 can't be reached from this PC."""


def framesize_code(name: str) -> tuple[int, tuple[int, int]]:
    key = name.strip().upper()
    if key not in FRAMESIZES:
        raise ValueError(f"Unknown framesize '{name}'. Use one of: {', '.join(FRAMESIZES)}")
    return FRAMESIZES[key]


def bypass_proxy(host: str) -> None:
    """Keep camera traffic off any campus HTTP proxy (requests and FFmpeg both honor no_proxy)."""
    for var in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(var, "")
        hosts = [h for h in current.split(",") if h]
        if host not in hosts:
            os.environ[var] = ",".join([*hosts, host])


def check_camera_network(host: str, port: int, timeout: float) -> None:
    """Raise CameraUnreachableError with a hint about which network is wrong."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((host, port))
            local_ip = probe.getsockname()[0]
    except OSError as exc:
        raise CameraUnreachableError(f"No route to {host}: {exc}. Is Wi-Fi connected to ChessCam?") from exc

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        hint = "Is Wi-Fi connected to ChessCam?"
        if not local_ip.startswith(host.rsplit(".", 1)[0] + "."):
            hint = f"This PC's address on that route is {local_ip}, not on the camera network. {hint}"
        else:
            hint = f"PC is on the camera network ({local_ip}) but the ESP32 isn't answering. Check its serial log."
        raise CameraUnreachableError(f"Can't reach ESP32 at {host}:{port} ({exc}). {hint}") from exc


class CameraControl:
    """Settings and stills over the CameraWebServer HTTP API (port 80)."""

    def __init__(self, cam: CameraSettings) -> None:
        self.cam = cam
        self.session = requests.Session()
        self.session.trust_env = False  # never send camera requests through a proxy

    def status(self) -> dict:
        resp = self.session.get(f"{self.cam.base_url}/status", timeout=self.cam.connect_timeout_s)
        resp.raise_for_status()
        return resp.json()

    def set(self, var: str, val: int) -> None:
        params: dict[str, str | int] = {"var": var, "val": int(val)}
        resp = self.session.get(
            f"{self.cam.base_url}/control",
            params=params,
            timeout=self.cam.connect_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"/control {var}={val} failed with HTTP {resp.status_code}")
        logger.debug("camera %s=%s", var, val)

    def capture(self) -> np.ndarray:
        resp = self.session.get(f"{self.cam.base_url}/capture", timeout=self.cam.read_timeout_s)
        resp.raise_for_status()
        image = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("/capture returned an undecodable image")
        return image

    def apply(self) -> tuple[int, int]:
        """Push configured settings, then confirm the sensor really switched size. Returns (w, h)."""
        code, expected = framesize_code(self.cam.framesize)
        self.set("framesize", code)
        self.set("quality", self.cam.quality)
        self.set("vflip", int(self.cam.vflip))
        self.set("hmirror", int(self.cam.hmirror))
        for var, val in self.cam.extra_controls.items():
            self.set(var, val)

        size = (0, 0)
        for _ in range(5):
            time.sleep(0.3)  # the sensor drops a frame or two after a size change
            reported = self.status().get("framesize")
            h, w = self.capture().shape[:2]
            size = (w, h)
            if reported == code and size == expected:
                logger.info("Camera set to %s %dx%d, quality %d", self.cam.framesize.upper(), w, h, self.cam.quality)
                return size
        logger.warning(
            "Asked for %s (code %d, %dx%d) but camera reports framesize=%s and sends %dx%d. "
            "The firmware's esp32-camera version may number framesizes differently.",
            self.cam.framesize,
            code,
            *expected,
            reported,
            *size,
        )
        return size


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


class FrameGrabber:
    """Background reader that always holds the newest decoded frame."""

    def __init__(
        self,
        source: str,
        open_timeout_s: float = 5.0,
        read_timeout_s: float = 5.0,
        backoff_s: float = 1.0,
        backoff_max_s: float = 10.0,
        reachability: Callable[[], None] | None = None,
    ) -> None:
        self.source = source
        self.open_timeout_ms = int(open_timeout_s * 1000)
        self.read_timeout_ms = int(read_timeout_s * 1000)
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        self.reachability = reachability

        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._index = -1
        self._fps = 0.0
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_warning = ("", 0.0)

    @classmethod
    def for_camera(cls, cam: CameraSettings) -> FrameGrabber:
        return cls(
            cam.stream_url if cam.source == "stream" else cam.capture_url,
            open_timeout_s=cam.connect_timeout_s,
            read_timeout_s=cam.read_timeout_s,
            backoff_s=cam.reconnect_backoff_s,
            backoff_max_s=cam.reconnect_backoff_max_s,
            # probe port 80, not the single-viewer stream port
            reachability=lambda: check_camera_network(cam.host, cam.http_port, cam.connect_timeout_s),
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> FrameGrabber:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="FrameGrabber", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self.read_timeout_ms / 1000 + 2)
            self._thread = None

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    def wait_for_frame(self, after_index: int = -1, timeout: float = 5.0) -> Frame | None:
        """Block until a frame newer than after_index arrives (or timeout)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while (self._latest is None or self._latest.index <= after_index) and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._latest if self._latest and self._latest.index > after_index else None

    def _warn(self, message: str) -> None:
        # Log a repeated failure at most every 10 s so the console stays readable.
        last_msg, last_t = self._last_warning
        now = time.monotonic()
        if message != last_msg or now - last_t > 10:
            logger.warning(message)
            self._last_warning = (message, now)

    def _read_mjpeg(self) -> None:
        """Read an HTTP multipart MJPEG stream directly: no FFmpeg probing delay, no frame buffer."""
        session = requests.Session()
        session.trust_env = False
        timeout = (self.open_timeout_ms / 1000, self.read_timeout_ms / 1000)
        with session.get(self.source, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            logger.info("Stream connected: %s", self.source)
            self._connected = True
            buf = bytearray()
            last_t = None
            for chunk in resp.iter_content(chunk_size=16384):
                if self._stop.is_set():
                    return
                buf += chunk
                while True:
                    start = buf.find(b"\xff\xd8")
                    if start < 0:
                        buf.clear()
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        del buf[:start]
                        break
                    jpeg = bytes(buf[start : end + 2])
                    del buf[: end + 2]
                    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    now = time.monotonic()
                    if last_t is not None:
                        rate = 1.0 / max(now - last_t, 1e-3)
                        self._fps = rate if self._fps == 0 else 0.8 * self._fps + 0.2 * rate
                    last_t = now
                    self._publish(image)
        raise ConnectionError("stream ended")

    def _read_snapshots(self) -> None:
        """Poll a still-image URL (CameraWebServer /capture). Works even if a browser holds the stream."""
        session = requests.Session()
        session.trust_env = False
        timeout = (self.open_timeout_ms / 1000, self.read_timeout_ms / 1000)
        last_t = None
        while not self._stop.is_set():
            resp = session.get(self.source, timeout=timeout)
            resp.raise_for_status()
            image = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ConnectionError("undecodable snapshot")
            if not self._connected:
                logger.info("Polling snapshots: %s", self.source)
                self._connected = True
            now = time.monotonic()
            if last_t is not None:
                rate = 1.0 / max(now - last_t, 1e-3)
                self._fps = rate if self._fps == 0 else 0.8 * self._fps + 0.2 * rate
            last_t = now
            self._publish(image)

    def _publish(self, image: np.ndarray) -> None:
        with self._cond:
            self._index += 1
            self._latest = Frame(self._index, time.time(), image)
            self._cond.notify_all()

    def _open(self) -> cv2.VideoCapture:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.read_timeout_ms,
        ]
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._read_loop()
            except Exception:  # never let the reader thread die silently
                logger.exception("Frame reader crashed; restarting")
                self._connected = False
                self._stop.wait(self.backoff_s)

    def _read_loop(self) -> None:
        backoff = self.backoff_s
        while not self._stop.is_set():
            if self.reachability is not None:
                try:
                    self.reachability()
                except CameraUnreachableError as exc:
                    self._warn(str(exc))
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, self.backoff_max_s)
                    continue

            if self.source.startswith(("http://", "https://")):
                try:
                    if self.source.rstrip("/").endswith("/stream"):
                        self._read_mjpeg()
                    else:
                        self._read_snapshots()
                    backoff = self.backoff_s
                except Exception as exc:
                    self._warn(
                        f"Stream {self.source} failed ({type(exc).__name__}: {exc}). "
                        "Is a browser tab already viewing it? Reconnecting."
                    )
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, self.backoff_max_s)
                finally:
                    self._connected = False
                continue

            cap = self._open()
            if not cap.isOpened():
                cap.release()
                self._warn(f"Couldn't open stream {self.source}. Is a browser tab already viewing it?")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.backoff_max_s)
                continue

            logger.info("Stream connected: %s", self.source)
            self._connected = True
            backoff = self.backoff_s
            last_t = None
            try:
                while not self._stop.is_set():
                    ok, image = cap.read()
                    if not ok or image is None:
                        self._warn(f"Stream dropped: {self.source}. Reconnecting.")
                        break
                    now = time.monotonic()
                    if last_t is not None:
                        rate = 1.0 / max(now - last_t, 1e-3)  # the Windows clock can tick 0 between frames
                        self._fps = rate if self._fps == 0 else 0.9 * self._fps + 0.1 * rate
                    last_t = now
                    self._publish(image)
            finally:
                self._connected = False
                cap.release()
