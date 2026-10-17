"""MJPEG stream reader for phone bridge or ESP32 sources."""

from __future__ import annotations

import socket
import time
from urllib.parse import urlparse

import cv2
import numpy as np

from .errors import StreamConnectionError


class MJPEGStreamReader:
    """Read and validate frames from an MJPEG stream source."""

    def __init__(
        self,
        url: str,
        min_width: int,
        min_height: int,
        retries: int = 5,
        backoff_seconds: float = 0.5,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.url = self._validate_url(url)
        self.min_width = min_width
        self.min_height = min_height
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self._capture: cv2.VideoCapture | None = None

        self._check_reachable()
        self._open_capture()

    @staticmethod
    def _validate_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise StreamConnectionError(url, "invalid URL format")
        if not (parsed.path.endswith("/stream") or parsed.path.endswith("/video")):
            raise StreamConnectionError(url, "URL path must end with /stream or /video")
        return url

    def _check_reachable(self) -> None:
        parsed = urlparse(self.url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            raise StreamConnectionError(self.url, "missing hostname")
        try:
            with socket.create_connection((host, port), timeout=self.timeout_seconds):
                return
        except OSError as exc:
            raise StreamConnectionError(self.url, f"host unreachable: {exc}") from exc

    def _open_capture(self) -> None:
        self._capture = cv2.VideoCapture(self.url)
        if not self._capture.isOpened():
            raise StreamConnectionError(self.url, "unable to open cv2.VideoCapture")

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _validate_frame(self, frame: np.ndarray | None) -> np.ndarray:
        if frame is None:
            raise ValueError("frame is None")
        if frame.size == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be BGR with 3 channels")
        height, width = frame.shape[:2]
        if width < self.min_width or height < self.min_height:
            raise ValueError(f"frame too small: {width}x{height}")
        return frame

    def read_frame(self) -> np.ndarray:
        """Read one validated frame, retrying on connection loss."""
        if self._capture is None:
            self._open_capture()

        assert self._capture is not None
        success, frame = self._capture.read()
        if success:
            return self._validate_frame(frame)

        reason = "read failed"
        for attempt in range(1, self.retries + 1):
            time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
            self.close()
            try:
                self._check_reachable()
                self._open_capture()
                assert self._capture is not None
                success, frame = self._capture.read()
                if success:
                    return self._validate_frame(frame)
                reason = "read failed after reconnect"
            except StreamConnectionError as exc:
                reason = str(exc)

        raise StreamConnectionError(self.url, f"max retries exceeded: {reason}")

