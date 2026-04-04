"""
compare_architectures.py
========================
Membandingkan ResNet50 (PyTorch) vs YOLOv12 (n/s/m/l) pada test set 4-kelas
(Bulk Carrier, Container Ship, Fishing, Tanker).

Langkah:
  1. Evaluasi ResNet50 (load checkpoint terbaik) pada test set
  2. Evaluasi setiap varian YOLOv12 yang sudah dilatih pada test set
  3. Tampilkan tabel perbandingan lengkap

Penggunaan:
  python compare_architectures.py
  python compare_architectures.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from dataload import FinalDataset
from model import ResNet50WithRT

# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FINAL_DIR = BASE_DIR / "final"
CLASS_NAMES = ["Bulk Carrier", "Container Ship", "Fishing", "Tanker"]
NUM_CLASSES = 4
OUTPUT_DIR = BASE_DIR / "comparison_results"

YOLO_CLASS_NAMES = {
    0: "bulk_carrier",
    1: "container_ship",
    2: "fishing",
    3: "tanker",
}
YOLO_VARIANTS = ["yolov12n", "yolov12s", "yolov12m", "yolov12l"]


# ─────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─────────────────────────────────────────────────────────────
# EVALUASI RESNET50
# ─────────────────────────────────────────────────────────────
def _find_best_resnet50_ckpt() -> Path | None:
    candidates = list((BASE_DIR / "experiments").glob("resnet50_*/model.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def evaluate_resnet50(device: torch.device, loader: DataLoader) -> dict:
    ckpt_path = _find_best_resnet50_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError(
            "Tidak ditemukan checkpoint ResNet50 di experiments/resnet50_*/model.pth. "
            "Jalankan 'python main.py' terlebih dahulu."
        )
    print(f"  Memuat ResNet50 dari: {ckpt_path}")
    model = ResNet50WithRT(rt_dim=8, num_classes=NUM_CLASSES, pretrained=False).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            rt = batch["rt"].to(device)
            lbl = batch["label"]
            out = model(img, rt)
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())

    return _compute_metrics(np.array(labels), np.array(preds))


# ─────────────────────────────────────────────────────────────
# EVALUASI YOLO
# ─────────────────────────────────────────────────────────────
def _find_yolo_ckpt(variant: str) -> Path | None:
    """Cari best.pt dari run terbaru untuk varian ini."""
    pattern = f"{variant}_e*_b*_s*"
    candidates = list((BASE_DIR / "experiments" / "yolov12").glob(f"{pattern}/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def evaluate_yolo_variant(variant: str, data_root: Path, imgsz: int = 64) -> dict | None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics belum terinstall. Jalankan: pip install ultralytics")

    ckpt = _find_yolo_ckpt(variant)
    if ckpt is None:
        print(f"  [{variant}] Checkpoint tidak ditemukan, skip.")
        return None

    print(f"  [{variant}] Memuat dari: {ckpt}")
    model = YOLO(str(ckpt))

    # Kumpulkan prediksi pada test set secara manual agar dapat metrik lengkap
    test_dir = data_root / "test"
    preds, labels = [], []
    for class_idx, class_name in YOLO_CLASS_NAMES.items():
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        img_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        for img_path in img_files:
            result = model.predict(str(img_path), imgsz=imgsz, verbose=False)
            pred_class = int(result[0].probs.top1)
            preds.append(pred_class)
            labels.append(class_idx)

    if not preds:
        print(f"  [{variant}] Tidak ada gambar test ditemukan.")
        return None

    return _compute_metrics(np.array(labels), np.array(preds))


# ─────────────────────────────────────────────────────────────
# METRIK
# ─────────────────────────────────────────────────────────────
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    label_ids = list(range(NUM_CLASSES))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "fscore_weighted": float(fw),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "fscore_per_class": f.tolist(),
        "support": s.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
    }


# ─────────────────────────────────────────────────────────────
# PRINT TABEL
# ─────────────────────────────────────────────────────────────
def print_comparison(results: dict[str, dict]):
    print(f"\n{'='*80}\n{'MODEL COMPARISON — TEST SET':^80}\n{'='*80}")
    header = f"{'Model':<14} {'Acc':>7} {'BalAcc':>8} {'F1-W':>7} {'Prec-W':>8} {'Rec-W':>7} {'Kappa':>7}"
    print(header)
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<14} "
              f"{r['accuracy']:>6.2f}% "
              f"{r['balanced_accuracy']:>7.2f}% "
              f"{r['fscore_weighted']*100:>6.2f}% "
              f"{r['precision_weighted']*100:>7.2f}% "
              f"{r['recall_weighted']*100:>6.2f}% "
              f"{r['kappa']:>7.4f}")

    print("\n" + "-" * 80)
    print("Per-Class F1 Score (%):")
    print(f"{'Model':<14} " + "  ".join(f"{c:<15}" for c in CLASS_NAMES))
    print("-" * 80)
    for name, r in results.items():
        per = "  ".join(f"{v*100:>6.2f}%{' '*8}" for v in r["fscore_per_class"])
        print(f"{name:<14} {per}")

    print("\n" + "-" * 80)
    best_name = max(results, key=lambda k: results[k]["fscore_weighted"])
    print(f"Model terbaik (F1 Weighted): {best_name.upper()}")
    r = results[best_name]
    print(f"  Acc={r['accuracy']:.2f}%  BalAcc={r['balanced_accuracy']:.2f}%  "
          f"F1={r['fscore_weighted']*100:.2f}%  Kappa={r['kappa']:.4f}")
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=64,
                        help="Ukuran gambar untuk inferensi YOLO")
    parser.add_argument("--yolo-data-root", default="yolo_cls_data",
                        help="Root folder dataset YOLO cls (hasil train_yolov12_cls.py)")
    args = parser.parse_args()

    seed_everything(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoader untuk ResNet50
    g = torch.Generator()
    g.manual_seed(args.seed)
    test_ds = FinalDataset(FINAL_DIR / "test.csv")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
    )
    print(f"Test set: {len(test_ds)} sampel\n")

    results = {}

    # ── ResNet50 ─────────────────────────────────────────────
    print("[ResNet50]")
    results["resnet50"] = evaluate_resnet50(device, test_loader)
    r = results["resnet50"]
    print(f"  Acc={r['accuracy']:.2f}%  F1={r['fscore_weighted']*100:.2f}%  Kappa={r['kappa']:.4f}")
    with open(OUTPUT_DIR / "resnet50_test_results.json", "w") as f:
        json.dump(results["resnet50"], f, indent=2)

    # ── YOLOv12 variants ─────────────────────────────────────
    yolo_data_root = BASE_DIR / args.yolo_data_root
    if not yolo_data_root.exists():
        print(f"\nPeringatan: folder '{yolo_data_root}' tidak ditemukan.")
        print("Jalankan 'python train_yolov12_cls.py --model yolov12n' terlebih dahulu,")
        print("atau jalankan 'python run_experiments.py --only-yolo'.")
    else:
        for variant in YOLO_VARIANTS:
            print(f"\n[{variant}]")
            res = evaluate_yolo_variant(variant, yolo_data_root, imgsz=args.imgsz)
            if res is not None:
                results[variant] = res
                r = results[variant]
                print(f"  Acc={r['accuracy']:.2f}%  F1={r['fscore_weighted']*100:.2f}%  Kappa={r['kappa']:.4f}")
                with open(OUTPUT_DIR / f"{variant}_test_results.json", "w") as f:
                    json.dump(res, f, indent=2)

    if len(results) < 2:
        print("\nHanya ResNet50 yang berhasil dievaluasi. Latih YOLO terlebih dahulu.")
        return

    # ── Simpan & tampilkan ────────────────────────────────────
    with open(OUTPUT_DIR / "comparison_summary.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "test_size": len(test_ds),
            "class_names": CLASS_NAMES,
            "results": results,
        }, f, indent=2)

    print_comparison(results)
    print(f"Hasil tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
