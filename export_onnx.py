import argparse
import os

import torch

from src.model import LSTMClassifierNoPack
from src.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/lstm/best", help="Folder containing model.pt")
    ap.add_argument("--out_path", default="outputs/lstm/best/model.onnx", help="Output ONNX path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    device = get_device(args.device)

    model = LSTMClassifierNoPack(
        input_dim=4,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        num_classes=3,
    )
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device))
    model.to(device).eval()

    x = torch.randn(args.batch_size, args.max_len, 4, device=device)
    lengths = torch.full((args.batch_size,), args.max_len, dtype=torch.long, device=device)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.onnx.export(
        model,
        (x, lengths),
        args.out_path,
        input_names=["x", "lengths"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch", 1: "seq"},
            "lengths": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Saved ONNX to: {args.out_path}")


if __name__ == "__main__":
    main()
