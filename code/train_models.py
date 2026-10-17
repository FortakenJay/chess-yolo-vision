"""Train the YOLO pieces detection model in 2 stages.

Stage 1 — Pre-train (broad generalization):
  Train on a diverse combined dataset (your Roboflow set + any added Universe
  datasets, all remapped to the same 12 class names). Produces
  ``models/pieces_pretrain.pt``.

Stage 2 — Fine-tune (specialize to your specific board):
  Continue training from ``models/pieces_pretrain.pt`` using only photos of
  YOUR physical board (labelled on Roboflow). Uses a low learning rate and
  freezes the backbone for the first few epochs so the model adapts without
  forgetting general features. Produces ``models/pieces.pt`` (the file
  ``main.py`` actually uses).

The board corner detector no longer uses a trained model — it uses ArUco
markers (see ``src/corners.py`` and ``code/generate_aruco_markers.py``).

Note: yolo26m at imgsz=800, batch=8 is tight on 8 GB VRAM. If training hits
CUDA out-of-memory, drop batch to 4 in the call sites below.

Usage:
  python code/train_models.py --stage pretrain
  python code/train_models.py --stage finetune
  python code/train_models.py --stage both
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"

PRETRAIN_WEIGHTS = MODELS_DIR / "pieces_pretrain.pt"
FINETUNE_WEIGHTS = MODELS_DIR / "pieces.pt"

# Datasets that count as "broad / pre-train data" (your Roboflow set + any
# Universe datasets you've merged into the same project)
PRETRAIN_DATASET_CANDIDATES = [
    "Chess_Pieces_Combined.yolo26",
    "Chess Pieces Combined.yolo26",
    "Chess_Pieces.yolo26",
    "Chess Pieces.yolo26",
    "Chess Pieces.yolov8",
    "Chess_Pieces.yolov8",
]

# Datasets that count as "your specific board" (fine-tune data)
FINETUNE_DATASET_CANDIDATES = [
    "My_Board.yolo26",
    "My Board.yolo26",
    "My_Board.yolov8",
    "MyBoard.yolo26",
    "MyBoard",
]


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _resolve_dataset_yaml(candidates: list[str]) -> Path:
    all_yamls = list(DATASETS_DIR.glob("**/data.yaml"))
    if not all_yamls:
        raise FileNotFoundError(f"No data.yaml files found under: {DATASETS_DIR}")

    normalized_candidates = [_norm(c) for c in candidates]
    for yaml_path in all_yamls:
        parent_norm = _norm(str(yaml_path.parent))
        if any(candidate in parent_norm for candidate in normalized_candidates):
            return yaml_path

    raise FileNotFoundError(
        f"Could not match any dataset yaml for candidates={candidates}. "
        f"Found: {[str(p.parent) for p in all_yamls]}"
    )


def _copy_best_weight(train_name: str, destination: Path) -> None:
    best = ROOT / "runs" / "train" / train_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best weight missing: {best}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)


def _prepare_ultralytics_yaml(source_yaml: Path) -> Path:
    """Create a training YAML with absolute paths and train-only fallback."""
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = source_yaml.parent

    train_rel = data.get("train", "train/images")
    val_rel = data.get("val", "valid/images")
    test_rel = data.get("test", "test/images")

    train_abs = (dataset_root / train_rel).resolve()
    val_abs = (dataset_root / val_rel).resolve()
    test_abs = (dataset_root / test_rel).resolve()

    same_dir_train = (dataset_root / "train" / "images").resolve()
    same_dir_val = (dataset_root / "valid" / "images").resolve()
    same_dir_test = (dataset_root / "test" / "images").resolve()

    if not train_abs.exists() and same_dir_train.exists():
        train_abs = same_dir_train
    if not val_abs.exists() and same_dir_val.exists():
        val_abs = same_dir_val
    if not test_abs.exists() and same_dir_test.exists():
        test_abs = same_dir_test

    if not train_abs.exists():
        raise FileNotFoundError(f"Train images path missing: {train_abs}")

    if not val_abs.exists():
        print(f"[warn] Missing val path for {source_yaml.name}: {val_abs}")
        print("[warn] Falling back to train/images for validation.")
        val_abs = train_abs

    if not test_abs.exists():
        test_abs = val_abs

    fixed = {
        "path": str(dataset_root),
        "train": str(train_abs),
        "val": str(val_abs),
        "test": str(test_abs),
        "nc": data["nc"],
        "names": data["names"],
    }
    fixed_yaml = source_yaml.with_name("data.autofix.yaml")
    fixed_yaml.write_text(yaml.safe_dump(fixed, sort_keys=False), encoding="utf-8")
    return fixed_yaml


def train_pretrain(device: int = 0) -> None:
    """Stage 1: broad pre-training on the combined dataset."""
    dataset_yaml = _prepare_ultralytics_yaml(_resolve_dataset_yaml(PRETRAIN_DATASET_CANDIDATES))
    print(f"[pretrain] dataset: {dataset_yaml}")

    model = YOLO("yolo26m.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=80,
        imgsz=800,
        batch=8,
        device=device,
        workers=0,
        cache=False,
        patience=25,
        lr0=0.001,
        project=str(ROOT / "runs" / "train"),
        name="pieces_pretrain",
        exist_ok=True,
    )
    _copy_best_weight("pieces_pretrain", PRETRAIN_WEIGHTS)
    print(f"[pretrain] saved -> {PRETRAIN_WEIGHTS}")


def train_finetune(device: int = 0) -> None:
    """Stage 2: fine-tune the pre-trained model on YOUR specific board."""
    dataset_yaml = _prepare_ultralytics_yaml(_resolve_dataset_yaml(FINETUNE_DATASET_CANDIDATES))
    print(f"[finetune] dataset: {dataset_yaml}")

    if PRETRAIN_WEIGHTS.exists():
        starting_weights = str(PRETRAIN_WEIGHTS)
        print(f"[finetune] starting from pretrain weights: {PRETRAIN_WEIGHTS}")
    elif FINETUNE_WEIGHTS.exists():
        starting_weights = str(FINETUNE_WEIGHTS)
        print(
            f"[finetune] pretrain weights missing, starting from existing {FINETUNE_WEIGHTS} instead"
        )
    else:
        starting_weights = "yolo26m.pt"
        print("[finetune] no pretrain weights found, starting from yolo26m.pt base")

    model = YOLO(starting_weights)
    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=800,
        batch=8,
        device=device,
        workers=0,
        cache=False,
        patience=15,
        # Low LR + freezing prevents the model from forgetting what it learned in
        # stage 1 while it adapts to your specific board.
        lr0=0.0001,
        freeze=10,
        project=str(ROOT / "runs" / "train"),
        name="pieces_finetune",
        exist_ok=True,
    )
    _copy_best_weight("pieces_finetune", FINETUNE_WEIGHTS)
    print(f"[finetune] saved -> {FINETUNE_WEIGHTS}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=("pretrain", "finetune", "both"),
        default="both",
        help="Which training stage to run",
    )
    parser.add_argument("--device", default=0, help="CUDA device index or 'cpu'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu to train on CPU.")
    if torch.cuda.is_available():
        print(f"CUDA available | GPU: {torch.cuda.get_device_name(0)}")
    device = args.device if args.device == "cpu" else int(args.device)

    if args.stage in ("pretrain", "both"):
        print("=" * 60)
        print("STAGE 1 / Pre-train on combined dataset")
        print("=" * 60)
        train_pretrain(device=device)

    if args.stage in ("finetune", "both"):
        print("=" * 60)
        print("STAGE 2 / Fine-tune on your specific board")
        print("=" * 60)
        train_finetune(device=device)

    print()
    print("Done.")
    if PRETRAIN_WEIGHTS.exists():
        print(f"- Pre-train weights: {PRETRAIN_WEIGHTS}")
    if FINETUNE_WEIGHTS.exists():
        print(f"- Final weights:     {FINETUNE_WEIGHTS}  (used by main.py)")
    print()
    print("Reminder: corner detection uses ArUco markers (no training required).")
    print("Run 'python code/generate_aruco_markers.py' if you haven't already.")


if __name__ == "__main__":
    main()
