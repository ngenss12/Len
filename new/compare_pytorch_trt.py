import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataload import FinalDataset
from model import ResNet50WithRT


CLASS_NAMES = ["Bulk Carrier", "Container Ship", "Fishing", "Tanker"]


def load_pytorch_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = ResNet50WithRT(rt_dim=8, num_classes=4, pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_trt_engine(engine_path: str):
    try:
        import tensorrt as trt
    except Exception as e:
        raise RuntimeError("TensorRT python package not found.") from e

    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    return trt, engine, context


def get_profile_max_batch(engine, input_name="image", profile_idx=0):
    try:
        _, _, max_shape = engine.get_tensor_profile_shape(input_name, profile_idx)
        return int(max_shape[0])
    except Exception:
        return 64


def trt_forward(trt, engine, context, stream, image: torch.Tensor, rt: torch.Tensor) -> torch.Tensor:
    ok_i = context.set_input_shape("image", tuple(image.shape))
    ok_r = context.set_input_shape("rt", tuple(rt.shape))
    if (ok_i is False) or (ok_r is False):
        raise RuntimeError(f"Input shapes outside TRT profile: image={tuple(image.shape)}, rt={tuple(rt.shape)}")

    out_name = "logits"
    out_shape = tuple(context.get_tensor_shape(out_name))
    out_np_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(out_name)))

    dtype_map = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
    }
    if out_np_dtype not in dtype_map:
        raise RuntimeError(f"Unsupported TensorRT output dtype: {out_np_dtype}")

    logits = torch.empty(out_shape, device="cuda", dtype=dtype_map[out_np_dtype])

    ok = False
    if hasattr(context, "set_tensor_address"):
        context.set_tensor_address("image", int(image.data_ptr()))
        context.set_tensor_address("rt", int(rt.data_ptr()))
        context.set_tensor_address(out_name, int(logits.data_ptr()))
        if hasattr(context, "execute_async_v3"):
            ok = context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
        else:
            ok = context.execute_v2([])
    else:
        bindings = [0] * engine.num_io_tensors
        bindings[engine.get_tensor_index("image")] = int(image.data_ptr())
        bindings[engine.get_tensor_index("rt")] = int(rt.data_ptr())
        bindings[engine.get_tensor_index(out_name)] = int(logits.data_ptr())
        ok = context.execute_v2(bindings)

    if not ok:
        raise RuntimeError("TensorRT execution failed.")
    return logits


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="final/test.csv")
    ap.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pth)")
    ap.add_argument("--engine_path", required=True, help="Path to TensorRT engine (.trt)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_json", default="infer/compare_pt_trt.json")
    ap.add_argument("--out_mismatch_csv", default="infer/compare_pt_trt_mismatch.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TRT comparison.")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    ds = FinalDataset(args.test_csv)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    pt_model = load_pytorch_model(args.checkpoint, device)
    trt, engine, context = load_trt_engine(args.engine_path)

    max_profile_batch = get_profile_max_batch(engine, input_name="image", profile_idx=0)
    if args.batch_size > max_profile_batch:
        print(
            f"Requested batch_size={args.batch_size} exceeds TRT profile max batch={max_profile_batch}. "
            f"Use <= {max_profile_batch}."
        )
        return

    stream = torch.cuda.Stream()

    all_y = []
    pt_pred = []
    trt_pred = []
    pt_prob = []
    trt_prob = []

    for batch in dl:
        image = batch["image"].to(device, non_blocking=True)
        rt = batch["rt"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        pt_logits = pt_model(image, rt)
        trt_logits = trt_forward(trt, engine, context, stream, image, rt)

        pt_p = torch.softmax(pt_logits.float(), dim=1).cpu().numpy()
        trt_p = torch.softmax(trt_logits.float(), dim=1).cpu().numpy()

        all_y.append(y.cpu().numpy())
        pt_pred.append(pt_p.argmax(axis=1))
        trt_pred.append(trt_p.argmax(axis=1))
        pt_prob.append(pt_p)
        trt_prob.append(trt_p)

    y_true = np.concatenate(all_y)
    y_pt = np.concatenate(pt_pred)
    y_trt = np.concatenate(trt_pred)
    p_pt = np.concatenate(pt_prob, axis=0)
    p_trt = np.concatenate(trt_prob, axis=0)

    agreement = float((y_pt == y_trt).mean())
    prob_diff = np.abs(p_pt - p_trt)

    report = {
        "num_samples": int(len(y_true)),
        "agreement_pt_vs_trt": agreement,
        "pt_accuracy": float(accuracy_score(y_true, y_pt)),
        "trt_accuracy": float(accuracy_score(y_true, y_trt)),
        "pt_f1_macro": float(f1_score(y_true, y_pt, average="macro", zero_division=0)),
        "trt_f1_macro": float(f1_score(y_true, y_trt, average="macro", zero_division=0)),
        "prob_max_abs_diff": float(prob_diff.max()),
        "prob_mean_abs_diff": float(prob_diff.mean()),
        "cm_true_vs_pt": confusion_matrix(y_true, y_pt, labels=[0, 1, 2, 3]).tolist(),
        "cm_true_vs_trt": confusion_matrix(y_true, y_trt, labels=[0, 1, 2, 3]).tolist(),
        "cm_pt_vs_trt": confusion_matrix(y_pt, y_trt, labels=[0, 1, 2, 3]).tolist(),
        "class_names": CLASS_NAMES,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    mismatch_idx = np.where(y_pt != y_trt)[0]
    mismatch_df = pd.DataFrame(
        {
            "index": mismatch_idx,
            "true_id": y_true[mismatch_idx],
            "true_label": [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in y_true[mismatch_idx]],
            "pt_pred_id": y_pt[mismatch_idx],
            "pt_pred_label": [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in y_pt[mismatch_idx]],
            "trt_pred_id": y_trt[mismatch_idx],
            "trt_pred_label": [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in y_trt[mismatch_idx]],
        }
    )
    mismatch_df.to_csv(args.out_mismatch_csv, index=False, encoding="utf-8-sig")

    print(f"Samples: {report['num_samples']}")
    print(f"PT vs TRT agreement: {report['agreement_pt_vs_trt']*100:.4f}%")
    print(f"PT acc/F1:  {report['pt_accuracy']*100:.2f}% / {report['pt_f1_macro']*100:.2f}%")
    print(f"TRT acc/F1: {report['trt_accuracy']*100:.2f}% / {report['trt_f1_macro']*100:.2f}%")
    print(f"Prob diff max/mean abs: {report['prob_max_abs_diff']:.8f} / {report['prob_mean_abs_diff']:.8f}")
    print(f"Saved report: {args.out_json}")
    print(f"Saved mismatches: {args.out_mismatch_csv}")


if __name__ == "__main__":
    main()
