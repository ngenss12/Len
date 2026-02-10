import argparse
import os
import numpy as np
import torch

from src.model import LSTMClassifierNoPack
from src.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/lstm/best", help="Folder containing model.pt")
    ap.add_argument("--onnx_path", default="outputs/lstm/best/model.onnx")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(
            "onnxruntime is not installed. Install it with `pip install onnxruntime` (or onnxruntime-gpu)."
        ) from e

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
    lengths = torch.randint(1, args.max_len + 1, (args.batch_size,), device=device)

    with torch.no_grad():
        pt_logits = model(x, lengths).cpu().numpy()

    sess = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(
        None,
        {
            "x": x.cpu().numpy().astype(np.float32),
            "lengths": lengths.cpu().numpy().astype(np.int64),
        },
    )[0]

    max_abs = float(np.max(np.abs(pt_logits - ort_logits)))
    print(f"Max abs diff: {max_abs:.6f}")


if __name__ == "__main__":
    main()
