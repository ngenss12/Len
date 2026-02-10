import argparse
import json
import os
from typing import List

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from src.data import build_samples_from_df, load_baidu_folder, ID_TO_LABEL_ZH
from src.metrics import compute_metrics, compute_confusion, make_classification_report
from src.model import LSTMClassifier, LSTMClassifierNoPack
from src.utils import ensure_dir, get_device, set_seed


class SeqDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], y_list: List[int]):
        self.X = X_list
        self.y = y_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    # batch: list of (X(T,F), y)
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = xs[0].shape[1]
    x_pad = torch.zeros((len(xs), max_len, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0], :] = torch.from_numpy(x)
    y = torch.tensor(ys, dtype=torch.long)
    return x_pad, lengths, y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []
    all_prob = []
    for x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        logits = model(x, lengths)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        all_y.append(y.numpy())
        all_pred.append(pred)
        all_prob.append(prob)
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return y_true, y_pred, y_prob


def plot_history(history, out_path: str):
    epochs = [h["epoch"] for h in history]
    loss = [h["train_loss"] for h in history]
    acc = [h["accuracy"] for h in history]
    f1 = [h["f1_macro"] for h in history]

    plt.figure()
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, acc, label="val_accuracy")
    plt.plot(epochs, f1, label="val_f1_macro")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, labels: List[str], out_path: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_multiclass(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], out_path: str):
    # One-vs-rest ROC for each class + micro-average
    n_classes = y_prob.shape[1]
    y_true_oh = np.zeros((len(y_true), n_classes), dtype=np.int32)
    y_true_oh[np.arange(len(y_true)), y_true] = 1

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.4f})")
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{labels[i]} (AUC={roc_auc[i]:.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (OvR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="train", help="Folder train_dataset (as in AIS_Classification1/train)")
    ap.add_argument("--out_dir", default="outputs/lstm", help="Output dir for model + reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resample_mins", type=int, default=0, help="0=off, else resample every N minutes")
    ap.add_argument("--max_len", type=int, default=512, help="Max points per vessel (truncate if longer)")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--export_friendly", action="store_true", default=False,
                    help="Use ONNX/TRT-friendly LSTM (no pack). Recommended to retrain.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out_dir)

    # 1) Load raw CSV(s)
    traj_df, label_df = load_baidu_folder(args.train_dir)
    resample = args.resample_mins if args.resample_mins and args.resample_mins > 0 else None

    # 2) Build one sample per vessel_id
    samples = build_samples_from_df(
        traj_df,
        label_df=label_df,
        resample_mins=resample,
        max_len=args.max_len,
        require_label=True,
    )
    vessel_ids = [s.vessel_id for s in samples]
    y = [s.y for s in samples]
    X = [s.X for s in samples]

    # filter out any missing label (just in case)
    keep = [(xi, yi, vid) for xi, yi, vid in zip(X, y, vessel_ids) if yi is not None]
    X = [k[0] for k in keep]
    y = [int(k[1]) for k in keep]
    vessel_ids = [int(k[2]) for k in keep]

    print(f"Loaded {len(X)} vessel samples from {args.train_dir}")

    # 3) Train/Val split (stratified by class)
    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(idx, test_size=args.val_ratio, random_state=args.seed, stratify=y)

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    # 4) Fit scaler on training points (all timesteps)
    scaler = StandardScaler()
    train_points = np.concatenate(X_train, axis=0)
    scaler.fit(train_points)

    def transform_list(xs):
        out = []
        for a in xs:
            out.append(scaler.transform(a).astype(np.float32))
        return out

    X_train = transform_list(X_train)
    X_val = transform_list(X_val)

    # 5) Dataloaders
    dl_train = DataLoader(
        SeqDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dl_val = DataLoader(
        SeqDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 6) Model
    if args.export_friendly:
        if args.bidirectional:
            print("export_friendly=True: forcing bidirectional=False for ONNX/TRT stability.")
        model = LSTMClassifierNoPack(
            input_dim=4,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=False,
            num_classes=3,
        ).to(device)
    else:
        model = LSTMClassifier(
            input_dim=4,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            num_classes=3,
        ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    history = []
    best_dir = os.path.join(args.out_dir, "best")
    ensure_dir(best_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, lengths, yb in tqdm(dl_train, desc=f"epoch {epoch:03d}", leave=False):
            x = x.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)

            logits = model(x, lengths)
            loss = criterion(logits, yb)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else float("nan")

        # eval
        y_true, y_pred, y_prob = evaluate(model, dl_val, device)
        m = compute_metrics(y_true, y_pred, y_prob)

        row = {"epoch": epoch, "train_loss": train_loss, **m.to_dict()}
        history.append(row)

        print(
            f"[epoch {epoch:03d}] loss={train_loss:.4f} | "
            f"acc={m.accuracy*100:.2f}% | "
            f"prec(macro)={m.precision_macro*100:.2f}% | "
            f"recall(macro)={m.recall_macro*100:.2f}% | "
            f"f1(macro)={m.f1_macro*100:.2f}%"
            + (f" | auc(macro-ovr)={m.auc_macro_ovr:.4f}" if m.auc_macro_ovr is not None else "")
        )

        # save best
        if m.f1_macro > best_f1:
            best_f1 = m.f1_macro
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            joblib.dump(scaler, os.path.join(best_dir, "scaler.pkl"))

            names = [ID_TO_LABEL_ZH[i] for i in [0, 1, 2]]
            report = make_classification_report(y_true, y_pred, target_names=names)
            with open(os.path.join(best_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)

            cm = compute_confusion(y_true, y_pred)
            with open(os.path.join(best_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
                json.dump({"labels": names, "confusion_matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)

            with open(os.path.join(best_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"best_epoch": epoch, **m.to_dict()}, f, indent=2)

            # plots for the best epoch
            plot_confusion(cm, names, os.path.join(best_dir, "confusion_matrix.png"))
            plot_roc_multiclass(y_true, y_prob, names, os.path.join(best_dir, "roc.png"))

    # save training history + history plot
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, os.path.join(args.out_dir, "history.png"))

    print(f"Done. Best val F1(macro) = {best_f1*100:.2f}%")
    print(f"Model saved to: {best_dir}")
    print(f"Plots saved: {os.path.join(args.out_dir, 'history.png')} and {os.path.join(best_dir, 'roc.png')}")


if __name__ == "__main__":
    main()
