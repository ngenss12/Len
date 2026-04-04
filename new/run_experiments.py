"""
run_experiments.py
==================
Orchestrator untuk menjalankan semua eksperimen secara otomatis:

  1. Sweep batch size & epoch   (ResNet50 via main.py)
  2. Variasi backbone/metode    (ResNet50, VGG19, AlexNet, Baseline via compare_architectures.py)
  3. Variasi YOLOv12            (n, s, m, l via train_yolov12_cls.py)

Semua eksperimen menggunakan seed yang sama untuk konsistensi.

Penggunaan:
  python run_experiments.py                   # jalankan semua
  python run_experiments.py --skip-yolo       # skip eksperimen YOLO
  python run_experiments.py --only-sweep      # hanya batch/epoch sweep
  python run_experiments.py --seed 123        # ganti seed global
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str], env_extra: dict | None = None) -> dict:
    merged_env = os.environ.copy()
    if env_extra:
        merged_env.update({k: str(v) for k, v in env_extra.items()})
    cmd_str = " ".join(cmd)
    print(f"\n[RUN] {cmd_str}")
    proc = subprocess.run(cmd, env=merged_env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        print(f"  [STDERR] {stderr[-2000:]}")
    else:
        print(f"  [OK] returncode=0")
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": stdout[-6000:],
        "stderr": stderr[-2000:],
    }


# ─────────────────────────────────────────────────────────────
# 1. SWEEP BATCH SIZE & EPOCH (ResNet50)
# ─────────────────────────────────────────────────────────────
BATCH_EPOCH_GRID = [
    {"batch_size": 16, "epochs": 30},
    {"batch_size": 32, "epochs": 30},
    {"batch_size": 64, "epochs": 30},
    {"batch_size": 32, "epochs": 50},
    {"batch_size": 32, "epochs": 80},
]


def run_batch_epoch_sweep(seed: int) -> list[dict]:
    print("\n" + "=" * 60)
    print("EKSPERIMEN 1: Sweep Batch Size & Epoch (ResNet50)")
    print("=" * 60)
    results = []
    for cfg in BATCH_EPOCH_GRID:
        out = run_cmd(
            [sys.executable, "main.py"],
            env_extra={
                "SEED": seed,
                "BATCH_SIZE": cfg["batch_size"],
                "EPOCHS": cfg["epochs"],
                "LR": "0.0001",
                "EARLY_STOP_PATIENCE": "0",
            },
        )
        out["experiment"] = {"type": "batch_epoch_sweep", **cfg, "seed": seed}
        results.append(out)
    return results


# ─────────────────────────────────────────────────────────────
# 2. PERBANDINGAN ResNet50 vs YOLOv12
# ─────────────────────────────────────────────────────────────
def run_comparison(seed: int) -> dict:
    print("\n" + "=" * 60)
    print("EKSPERIMEN 2: Perbandingan ResNet50 vs YOLOv12 (n/s/m/l)")
    print("=" * 60)
    out = run_cmd([sys.executable, "compare_architectures.py", "--seed", str(seed)])
    return {"experiment": {"type": "resnet50_vs_yolo", "seed": seed}, **out}


# ─────────────────────────────────────────────────────────────
# 3. VARIASI YOLOV12 (n, s, m, l)
# ─────────────────────────────────────────────────────────────
YOLO_VARIANTS = [
    {"model": "yolov12n", "batch": 64, "epochs": 50},
    {"model": "yolov12s", "batch": 64, "epochs": 50},
    {"model": "yolov12m", "batch": 32, "epochs": 50},
    {"model": "yolov12l", "batch": 16, "epochs": 50},
]


def run_yolo_sweep(seed: int) -> list[dict]:
    print("\n" + "=" * 60)
    print("EKSPERIMEN 3: Variasi YOLOv12 (n / s / m / l)")
    print("=" * 60)
    results = []
    for cfg in YOLO_VARIANTS:
        out = run_cmd([
            sys.executable, "train_yolov12_cls.py",
            "--model", cfg["model"],
            "--epochs", str(cfg["epochs"]),
            "--batch", str(cfg["batch"]),
            "--seed", str(seed),
        ])
        out["experiment"] = {"type": "yolov12_sweep", **cfg, "seed": seed}
        results.append(out)
    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip eksperimen YOLOv12")
    parser.add_argument("--only-sweep", action="store_true",
                        help="Hanya jalankan batch/epoch sweep")
    parser.add_argument("--only-compare", action="store_true",
                        help="Hanya jalankan perbandingan ResNet50 vs YOLO")
    parser.add_argument("--only-yolo", action="store_true",
                        help="Hanya jalankan YOLO sweep (training)")
    args = parser.parse_args()

    seed = args.seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("experiments") / f"experiment_report_{timestamp}.json"
    Path("experiments").mkdir(exist_ok=True)

    print(f"\nSeed global : {seed}")
    print(f"Timestamp   : {timestamp}")
    print(f"Report      : {report_path}")

    report = {
        "seed": seed,
        "timestamp": timestamp,
        "python": sys.executable,
    }

    run_all = not (args.only_sweep or args.only_compare or args.only_yolo)

    if run_all or args.only_sweep:
        report["batch_epoch_sweep"] = run_batch_epoch_sweep(seed)

    if (run_all or args.only_yolo) and not args.skip_yolo:
        report["yolo_sweep"] = run_yolo_sweep(seed)

    if run_all or args.only_compare:
        report["comparison"] = run_comparison(seed)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Semua eksperimen selesai.")
    print(f"Report tersimpan: {report_path}")
    print(f"{'='*60}")

    # Ringkasan status
    def _check(results, key="returncode"):
        if isinstance(results, list):
            failed = [r for r in results if r.get(key, 1) != 0]
            return len(failed), len(results)
        elif isinstance(results, dict):
            ok = 1 if results.get(key, 1) == 0 else 0
            return (1 - ok), 1
        return 0, 0

    for section_key in ["batch_epoch_sweep", "yolo_sweep", "comparison"]:
        if section_key in report:
            failed, total = _check(report[section_key])
            status = "OK" if failed == 0 else f"{failed}/{total} GAGAL"
            print(f"  {section_key:<25}: {status}")

    # Export hasil ke Excel
    print("\nMengekspor hasil ke Excel...")
    run_cmd([sys.executable, "export_reports_to_excel.py"])


if __name__ == "__main__":
    main()
