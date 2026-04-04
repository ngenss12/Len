"""
export_reports_to_excel.py
==========================
Mengumpulkan semua hasil eksperimen dan mengekspor ke satu file Excel.

Sheet yang dihasilkan:
  1. Model_Comparison    — ResNet50 vs YOLOv12 (n/s/m/l), metrik keseluruhan
  2. PerClass_Metrics    — F1/Precision/Recall per kelas per model
  3. Confusion_Matrices  — Confusion matrix per model
  4. Batch_Epoch_Sweep   — Hasil variasi batch size & epoch (ResNet50)
  5. YOLO_Variants       — Summary training setiap varian YOLO

Penggunaan:
  python export_reports_to_excel.py
  python export_reports_to_excel.py --out hasil_eksperimen.xlsx
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = BASE_DIR / "comparison_results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

CLASS_NAMES = ["Bulk Carrier", "Container Ship", "Fishing", "Tanker"]
YOLO_VARIANTS = ["yolov12n", "yolov12s", "yolov12m", "yolov12l"]


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _style_header(ws, header_fill, header_font, thin_border):
    """Warnai baris pertama sebagai header."""
    from openpyxl.styles import Alignment
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _auto_col_width(ws):
    for col in ws.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)


def _write_df(ws, df: pd.DataFrame, header_fill, header_font, thin_border):
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import Alignment, PatternFill

    alt_fill = PatternFill(start_color="EEF2FF", end_color="EEF2FF", fill_type="solid")

    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=1):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if r_idx == 1:
                cell.fill = header_fill
                cell.font = header_font
            elif r_idx % 2 == 0:
                cell.fill = alt_fill

    _auto_col_width(ws)


# ─────────────────────────────────────────────────────────────
# SHEET 1 & 2: Model Comparison + Per-Class
# ─────────────────────────────────────────────────────────────
def build_comparison_dfs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Kembalikan (df_overall, df_perclass, df_cm)."""
    summary = _load_json(COMPARISON_DIR / "comparison_summary.json")
    results = {}

    if summary and "results" in summary:
        results = summary["results"]
    else:
        # Fallback: baca file per model
        for name in ["resnet50"] + YOLO_VARIANTS:
            p = COMPARISON_DIR / f"{name}_test_results.json"
            d = _load_json(p)
            if d:
                results[name] = d

    if not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Overall ──────────────────────────────────────────────
    rows_overall = []
    for model, r in results.items():
        rows_overall.append({
            "Model": model,
            "Accuracy (%)": round(r.get("accuracy", 0), 2),
            "Balanced Acc (%)": round(r.get("balanced_accuracy", 0), 2),
            "F1 Weighted (%)": round(r.get("fscore_weighted", 0) * 100, 2),
            "Precision Weighted (%)": round(r.get("precision_weighted", 0) * 100, 2),
            "Recall Weighted (%)": round(r.get("recall_weighted", 0) * 100, 2),
            "Kappa": round(r.get("kappa", 0), 4),
        })
    df_overall = pd.DataFrame(rows_overall)

    # ── Per-class F1 ─────────────────────────────────────────
    rows_pc = []
    for model, r in results.items():
        f = r.get("fscore_per_class", [0] * 4)
        p = r.get("precision_per_class", [0] * 4)
        rec = r.get("recall_per_class", [0] * 4)
        sup = r.get("support", [0] * 4)
        for i, cls in enumerate(CLASS_NAMES):
            rows_pc.append({
                "Model": model,
                "Class": cls,
                "F1 (%)": round(f[i] * 100 if i < len(f) else 0, 2),
                "Precision (%)": round(p[i] * 100 if i < len(p) else 0, 2),
                "Recall (%)": round(rec[i] * 100 if i < len(rec) else 0, 2),
                "Support": int(sup[i]) if i < len(sup) else 0,
            })
    df_perclass = pd.DataFrame(rows_pc)

    # ── Confusion Matrices ────────────────────────────────────
    cm_rows = []
    for model, r in results.items():
        cm = r.get("confusion_matrix", [])
        if not cm:
            continue
        for i, row_vals in enumerate(cm):
            for j, val in enumerate(row_vals):
                cm_rows.append({
                    "Model": model,
                    "Actual": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i),
                    "Predicted": CLASS_NAMES[j] if j < len(CLASS_NAMES) else str(j),
                    "Count": int(val),
                })
    df_cm = pd.DataFrame(cm_rows)

    return df_overall, df_perclass, df_cm


# ─────────────────────────────────────────────────────────────
# SHEET 4: Batch/Epoch Sweep
# ─────────────────────────────────────────────────────────────
def build_batch_epoch_df() -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(EXPERIMENTS_DIR.glob("resnet50_*/"), key=lambda p: p.stat().st_mtime):
        history_path = exp_dir / "history.json"
        results_path = exp_dir / "results.txt"
        h = _load_json(history_path)
        if h is None:
            continue

        # Ekstrak config dari nama folder
        # Format: resnet50_YYYYMMDD_HHMMSS_lr{lr}
        parts = exp_dir.name.split("_")
        lr = next((p.replace("lr", "") for p in parts if p.startswith("lr")), "?")

        val_f1 = h.get("val_f1", [])
        val_acc = h.get("val_acc", [])
        val_bal = h.get("val_balanced_acc", [])
        epoch_times = h.get("epoch_time_sec", [])

        rows.append({
            "Experiment": exp_dir.name,
            "LR": lr,
            "Epochs Run": len(val_f1),
            "Best Val F1 (%)": round(max(val_f1), 2) if val_f1 else None,
            "Best Val Acc (%)": round(max(val_acc), 2) if val_acc else None,
            "Best Balanced Acc (%)": round(max(val_bal), 2) if val_bal else None,
            "Avg Epoch Time (s)": round(float(np.mean(epoch_times)), 1) if epoch_times else None,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# SHEET 5: YOLO Variants Summary
# ─────────────────────────────────────────────────────────────
def build_yolo_summary_df() -> pd.DataFrame:
    rows = []
    yolo_dir = EXPERIMENTS_DIR / "yolov12"
    if not yolo_dir.exists():
        return pd.DataFrame()

    for run_dir in sorted(yolo_dir.glob("yolov12*_e*_b*_s*/"), key=lambda p: p.stat().st_mtime):
        summary = _load_json(run_dir / "summary.json")
        if summary is None:
            continue
        rows.append({
            "Run": run_dir.name,
            "Model": summary.get("model", ""),
            "Variant": summary.get("variant", ""),
            "Epochs": summary.get("epochs", ""),
            "Batch": summary.get("batch", ""),
            "imgsz": summary.get("imgsz", ""),
            "Seed": summary.get("seed", ""),
            "Val Top-1 (%)": round(summary.get("val_top1", 0) * 100, 2),
            "Test Top-1 (%)": round(summary.get("test_top1", 0) * 100, 2),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="", help="Path output file Excel")
    args = parser.parse_args()

    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Border, Side
    except ImportError:
        raise RuntimeError("openpyxl belum terinstall. Jalankan: pip install openpyxl")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else BASE_DIR / f"experiment_report_{timestamp}.xlsx"

    # Style
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    thin = Side(style="thin", color="CCCCCC")
    thin_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # hapus sheet kosong default

    # ── Sheet 1: Model Comparison ─────────────────────────────
    df_overall, df_perclass, df_cm = build_comparison_dfs()
    ws1 = wb.create_sheet("Model_Comparison")
    if not df_overall.empty:
        _write_df(ws1, df_overall, header_fill, header_font, thin_border)
        print(f"  Sheet 'Model_Comparison': {len(df_overall)} model")
    else:
        ws1.cell(1, 1, "Belum ada hasil perbandingan. Jalankan compare_architectures.py terlebih dahulu.")
        print("  Sheet 'Model_Comparison': kosong (belum ada data)")

    # ── Sheet 2: Per-Class Metrics ────────────────────────────
    ws2 = wb.create_sheet("PerClass_Metrics")
    if not df_perclass.empty:
        _write_df(ws2, df_perclass, header_fill, header_font, thin_border)
        print(f"  Sheet 'PerClass_Metrics': {len(df_perclass)} baris")
    else:
        ws2.cell(1, 1, "Belum ada data.")

    # ── Sheet 3: Confusion Matrices ───────────────────────────
    ws3 = wb.create_sheet("Confusion_Matrices")
    if not df_cm.empty:
        _write_df(ws3, df_cm, header_fill, header_font, thin_border)
        print(f"  Sheet 'Confusion_Matrices': {len(df_cm)} baris")
    else:
        ws3.cell(1, 1, "Belum ada data.")

    # ── Sheet 4: Batch/Epoch Sweep ────────────────────────────
    ws4 = wb.create_sheet("Batch_Epoch_Sweep")
    df_sweep = build_batch_epoch_df()
    if not df_sweep.empty:
        _write_df(ws4, df_sweep, header_fill, header_font, thin_border)
        print(f"  Sheet 'Batch_Epoch_Sweep': {len(df_sweep)} eksperimen")
    else:
        ws4.cell(1, 1, "Belum ada data. Jalankan run_experiments.py --only-sweep terlebih dahulu.")
        print("  Sheet 'Batch_Epoch_Sweep': kosong")

    # ── Sheet 5: YOLO Variants ────────────────────────────────
    ws5 = wb.create_sheet("YOLO_Variants")
    df_yolo = build_yolo_summary_df()
    if not df_yolo.empty:
        _write_df(ws5, df_yolo, header_fill, header_font, thin_border)
        print(f"  Sheet 'YOLO_Variants': {len(df_yolo)} run")
    else:
        ws5.cell(1, 1, "Belum ada data. Jalankan train_yolov12_cls.py atau run_experiments.py --only-yolo.")
        print("  Sheet 'YOLO_Variants': kosong")

    wb.save(out_path)
    print(f"\nFile Excel tersimpan: {out_path}")


if __name__ == "__main__":
    main()
