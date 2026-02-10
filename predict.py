import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import build_samples_from_df, load_baidu_folder, ID_TO_LABEL_ZH, label_to_int
from src.model import LSTMClassifier, LSTMClassifierNoPack
from src.utils import ensure_dir, get_device


class SeqDataset(Dataset):
    def __init__(self, X_list, ids):
        self.X = X_list
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ids[idx]


def collate_fn(batch):
    xs, ids = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = xs[0].shape[1]
    x_pad = torch.zeros((len(xs), max_len, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0], :] = torch.from_numpy(x)
    return x_pad, lengths, np.array(ids, dtype=np.int64)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="test_dataset", help="Folder test_dataset")
    ap.add_argument("--model_dir", default="outputs/lstm/best", help="Folder containing model.pt + scaler.pkl")
    ap.add_argument("--submit_template", default="submit_example.csv", help="Template for submission format")
    ap.add_argument("--out_csv", default="infer/submit.csv", help="Output submission csv path")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--resample_mins", type=int, default=0)
    ap.add_argument("--export_friendly", action="store_true", default=False,
                    help="Use ONNX/TRT-friendly LSTM (no pack). Must match training.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = get_device(args.device)
    ensure_dir(os.path.dirname(args.out_csv))

    # load model + scaler
    scaler = joblib.load(os.path.join(args.model_dir, "scaler.pkl"))
    if args.export_friendly:
        model = LSTMClassifierNoPack(
            input_dim=4,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            num_classes=3,
        )
    else:
        model = LSTMClassifier(
            input_dim=4,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            num_classes=3,
        )
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device))
    model.to(device).eval()

    # load test raw
    traj_df, label_df = load_baidu_folder(args.test_dir)
    resample = args.resample_mins if args.resample_mins and args.resample_mins > 0 else None

    samples = build_samples_from_df(
        traj_df,
        label_df=label_df,
        resample_mins=resample,
        max_len=args.max_len,
        require_label=False,
    )

    # transform
    X = [scaler.transform(s.X).astype(np.float32) for s in samples]
    ids = [s.vessel_id for s in samples]

    dl = DataLoader(SeqDataset(X, ids), batch_size=128, shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_pred = []
    all_prob = []

    for x, lengths, batch_ids in dl:
        x = x.to(device)
        lengths = lengths.to(device)
        logits = model(x, lengths)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)

        all_ids.append(batch_ids)
        all_pred.append(pred)
        all_prob.append(prob)

    all_ids = np.concatenate(all_ids)
    all_pred = np.concatenate(all_pred)
    all_prob = np.concatenate(all_prob)

    # build submission using template's column names
    tmpl = pd.read_csv(args.submit_template, encoding="utf-8-sig")
    id_col = tmpl.columns[0]
    type_col = tmpl.columns[1]

    out = pd.DataFrame({id_col: all_ids, type_col: [ID_TO_LABEL_ZH[int(i)] for i in all_pred]})

    # keep template ordering if possible
    if id_col in tmpl.columns:
        # If template contains a fixed list of IDs, align to it
        if tmpl[id_col].nunique() == len(tmpl) and len(tmpl) > 0:
            out = tmpl[[id_col]].merge(out, on=id_col, how="left")

    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved submission to: {args.out_csv}")
    # optional: also save probs
    prob_path = os.path.splitext(args.out_csv)[0] + "_proba.npy"
    np.save(prob_path, all_prob)
    print(f"Saved probabilities to: {prob_path}")


if __name__ == "__main__":
    main()
