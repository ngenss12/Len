"""
train_yolov12_cls.py
====================
Transfer learning klasifikasi kapal menggunakan YOLOv12 (n/s/m/l).

Alur:
  1. Konversi TIFF (VV+VH) ke PNG RGB dari final/train.csv, val.csv, test.csv
  2. Susun folder yolo_cls_data/<split>/<class_name>/
  3. Training yolo12{variant}-cls.yaml diinisialisasi dari bobot yolo12{variant}.pt
  4. Simpan summary.json di experiments/yolov12/<run_name>/

Penggunaan:
  python train_yolov12_cls.py --model yolov12n --epochs 50 --batch 64 --seed 42
  python train_yolov12_cls.py --model yolov12s --epochs 50 --batch 64 --seed 42
  python train_yolov12_cls.py --model yolov12m --epochs 50 --batch 32 --seed 42
  python train_yolov12_cls.py --model yolov12l --epochs 50 --batch 16 --seed 42
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff

CLASS_NAMES = {
    0: "bulk_carrier",
    1: "container_ship",
    2: "fishing",
    3: "tanker",
}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _tif_to_png_rgb(src: Path, dst: Path):
    """Konversi TIFF (2-channel VV/VH) ke PNG RGB 3-channel."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    img = tiff.imread(str(src)).astype(np.float32)
    if img.ndim == 2:
        c = _to_uint8(img)
        rgb = np.stack([c, c, c], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 2:
        vv = _to_uint8(img[..., 0])
        vh = _to_uint8(img[..., 1])
        avg = ((vv.astype(np.float32) + vh.astype(np.float32)) * 0.5).astype(np.uint8)
        rgb = np.stack([vv, vh, avg], axis=-1)
    elif img.ndim == 3 and img.shape[-1] >= 3:
        rgb = np.stack([_to_uint8(img[..., i]) for i in range(3)], axis=-1)
    else:
        c = _to_uint8(img[..., 0] if img.ndim == 3 else img)
        rgb = np.stack([c, c, c], axis=-1)
    cv2.imwrite(str(dst), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def prepare_yolo_cls_dataset(
    root: str = "yolo_cls_data",
    train_csv: str = "final/train.csv",
    val_csv: str = "final/val.csv",
    test_csv: str = "final/test.csv",
) -> Path:
    root = Path(root)
    for split, csv_path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            label = int(row["label"])
            class_name = CLASS_NAMES[label]
            src = Path(str(row["img_path"]))
            dst = root / split / class_name / f"{src.stem}.png"
            _tif_to_png_rgb(src, dst)
    print(f"Dataset siap di: {root}")
    return root


def resolve_variant(model_name: str) -> str:
    key = model_name.lower().replace(".pt", "").replace(".yaml", "").replace("yolov", "yolo")
    if not key.startswith("yolo12") or len(key) < 7:
        raise ValueError(f"Model harus salah satu dari: yolov12n, yolov12s, yolov12m, yolov12l. Diterima: {model_name}")
    variant = key[6]
    if variant not in ("n", "s", "m", "l", "x"):
        raise ValueError(f"Variant tidak valid: '{variant}'. Harus n/s/m/l.")
    return variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov12n",
                    help="Nama model: yolov12n | yolov12s | yolov12m | yolov12l")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="yolo_cls_data",
                    help="Direktori output dataset YOLO cls")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    seed_everything(args.seed)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics belum terinstall. Jalankan: pip install ultralytics") from e

    # Siapkan dataset
    data_root = prepare_yolo_cls_dataset(root=args.data_root)

    # Resolve variant dan bobot
    variant = resolve_variant(args.model)
    cls_yaml = f"yolo12{variant}-cls.yaml"
    det_weight = f"yolo12{variant}.pt"

    weight_path = Path(det_weight)
    if not weight_path.exists():
        print(f"Bobot '{det_weight}' tidak ditemukan, mencoba download otomatis...")
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(det_weight)
        except Exception as e:
            raise FileNotFoundError(
                f"Gagal mendownload '{det_weight}': {e}\n"
                "Download manual dari repositori YOLOv12 dan letakkan di direktori ini."
            ) from e

    print(f"\nModel   : {args.model} -> yaml={cls_yaml}, weight={det_weight}")
    print(f"Epochs  : {args.epochs}, Batch: {args.batch}, imgsz: {args.imgsz}, seed: {args.seed}")

    yolo_model = YOLO(cls_yaml).load(det_weight)

    project = Path("experiments") / "yolov12"
    project.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.model}_e{args.epochs}_b{args.batch}_s{args.seed}"

    train_res = yolo_model.train(
        data=str(data_root),
        task="classify",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        workers=args.workers,
        verbose=True,
    )

    val_res = yolo_model.val(
        data=str(data_root),
        task="classify",
        imgsz=args.imgsz,
        batch=args.batch,
        split="val",
    )

    test_res = yolo_model.val(
        data=str(data_root),
        task="classify",
        imgsz=args.imgsz,
        batch=args.batch,
        split="test",
    )

    out_dir = project / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "variant": variant,
        "weight": det_weight,
        "yaml": cls_yaml,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "val_top1": float(val_res.results_dict.get("metrics/accuracy_top1", 0)),
        "test_top1": float(test_res.results_dict.get("metrics/accuracy_top1", 0)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary disimpan: {out_dir / 'summary.json'}")
    print(f"Val  Top-1: {summary['val_top1']*100:.2f}%")
    print(f"Test Top-1: {summary['test_top1']*100:.2f}%")


if __name__ == "__main__":
    main()
