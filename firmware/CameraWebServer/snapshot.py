"""Grab a still image from the ESP32 CameraWebServer and save it as a JPEG.

Usage:
    python snapshot.py 192.168.1.50            # saves snap_<timestamp>.jpg
    python snapshot.py 192.168.1.50 -n 5 -d 1  # 5 shots, 1 second apart
"""
import argparse
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def snap(ip: str, out_dir: Path) -> Path:
    with urllib.request.urlopen(f"http://{ip}/capture", timeout=10) as resp:
        data = resp.read()
    path = out_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    path.write_bytes(data)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ip", help="IP printed in the Serial Monitor, e.g. 192.168.1.50")
    p.add_argument("-n", "--count", type=int, default=1, help="number of snapshots")
    p.add_argument("-d", "--delay", type=float, default=0.5, help="seconds between shots")
    p.add_argument("-o", "--out", default="snapshots", help="output folder")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        print("saved", snap(args.ip, out_dir))
        if i < args.count - 1:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
