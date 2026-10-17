"""Generate the 4 ArUco markers used by the corner detector.

Produces:
  - aruco_markers/marker_0_TL.png  (place at TOP-LEFT corner of the board)
  - aruco_markers/marker_1_TR.png  (TOP-RIGHT)
  - aruco_markers/marker_2_BR.png  (BOTTOM-RIGHT)
  - aruco_markers/marker_3_BL.png  (BOTTOM-LEFT)
  - aruco_markers/markers_sheet.png  (all 4 markers on a single A4-ish sheet
                                       for easy printing)

Printing & placement:
  - Print at 100% scale on plain paper. The default 800 px markers print at
    ~6.8 cm if your printer renders at 300 dpi; that's a comfortable size.
  - Each marker MUST be oriented the same way the PNG is (do not rotate).
  - The marker's INNER corner (the one closest to the centre of the board)
    should sit exactly on the corner of the playing surface, with a small
    white border (>= 5 mm) around every marker.

Usage:
  python code/generate_aruco_markers.py
  python code/generate_aruco_markers.py --size 1000 --dictionary DICT_4X4_50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "aruco_markers"

CORNER_LABELS: dict[int, str] = {
    0: "TL",
    1: "TR",
    2: "BR",
    3: "BL",
}


def _generate_marker(dictionary, marker_id: int, size_px: int) -> np.ndarray:
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    border = max(8, size_px // 25)
    padded = cv2.copyMakeBorder(
        img, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )
    label = f"ID {marker_id} ({CORNER_LABELS[marker_id]})"
    cv2.putText(
        padded,
        label,
        (border, border + size_px + border - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, size_px / 1200.0),
        0,
        max(1, size_px // 400),
        cv2.LINE_AA,
    )
    return padded


def _save_sheet(markers: dict[int, np.ndarray], path: Path) -> None:
    # 2x2 layout: TL  TR
    #             BL  BR
    h, w = markers[0].shape[:2]
    sheet = np.full((h * 2 + 60, w * 2 + 60, 1), 255, dtype=np.uint8).squeeze()
    positions = {0: (20, 20), 1: (20, w + 40), 2: (h + 40, w + 40), 3: (h + 40, 20)}
    for marker_id, (y, x) in positions.items():
        sheet[y : y + h, x : x + w] = markers[marker_id]
    cv2.imwrite(str(path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chess-board ArUco markers")
    parser.add_argument("--size", type=int, default=800, help="Marker pixel size (default 800)")
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name (must match src/corners.py)",
    )
    args = parser.parse_args()

    dict_id = getattr(cv2.aruco, args.dictionary, None)
    if dict_id is None:
        raise SystemExit(f"Unknown ArUco dictionary: {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    markers: dict[int, np.ndarray] = {}
    for marker_id in (0, 1, 2, 3):
        img = _generate_marker(dictionary, marker_id, args.size)
        out_path = OUT_DIR / f"marker_{marker_id}_{CORNER_LABELS[marker_id]}.png"
        cv2.imwrite(str(out_path), img)
        markers[marker_id] = img
        print(f"Wrote {out_path}")

    sheet_path = OUT_DIR / "markers_sheet.png"
    _save_sheet(markers, sheet_path)
    print(f"Wrote {sheet_path}  (single sheet for printing)")
    print()
    print("Placement reminder:")
    print("  ID 0  -> TOP-LEFT       ID 1  -> TOP-RIGHT")
    print("  ID 3  -> BOTTOM-LEFT    ID 2  -> BOTTOM-RIGHT")
    print("Keep every marker in its printed orientation and align its inner")
    print("corner with the corner of the playing surface.")


if __name__ == "__main__":
    main()
