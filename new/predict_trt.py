import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import tifffile as tiff


CLASS_NAMES = ["Bulk Carrier", "Container Ship", "Fishing", "Tanker"]


class FinalCsvDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        if "img_path" not in self.df.columns:
            raise KeyError("CSV must contain 'img_path' column")
        if "rt" not in self.df.columns:
            raise KeyError("CSV must contain 'rt' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["img_path"])
        img = tiff.imread(img_path).astype(np.float32)

        if img.ndim == 2:
            img = img[None, :, :]
        elif img.ndim == 3 and img.shape[-1] in (1, 2, 3, 4):
            img = np.transpose(img, (2, 0, 1))

        rt = row["rt"]
        if isinstance(rt, str):
            rt = ast.literal_eval(rt)
        rt = np.asarray(rt, dtype=np.float32)

        return {
            "image": torch.from_numpy(img).float(),
            "rt": torch.from_numpy(rt).float(),
            "img_path": img_path,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    rt = torch.stack([b["rt"] for b in batch], dim=0)
    paths = [b["img_path"] for b in batch]
    return images, rt, paths


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
    ap.add_argument("--engine_path", default="onnx/resnet50_with_rt.trt")
    ap.add_argument("--out_csv", default="infer/predictions_trt.csv")
    ap.add_argument("--out_proba", default="infer/predictions_trt_proba.npy")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT inference.")

    trt, engine, context = load_trt_engine(args.engine_path)
    stream = torch.cuda.Stream()

    max_profile_batch = get_profile_max_batch(engine, input_name="image", profile_idx=0)
    effective_batch = min(args.batch_size, max_profile_batch)
    if effective_batch != args.batch_size:
        print(
            f"Requested batch_size={args.batch_size} exceeds TensorRT profile max batch={max_profile_batch}. "
            f"Using batch_size={effective_batch}."
        )

    ds = FinalCsvDataset(args.test_csv)
    dl = DataLoader(ds, batch_size=effective_batch, shuffle=False, collate_fn=collate_fn)

    pred_rows = []
    all_prob = []

    for image, rt, paths in dl:
        image = image.to("cuda", non_blocking=True)
        rt = rt.to("cuda", non_blocking=True)

        logits = trt_forward(trt, engine, context, stream, image, rt)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        all_prob.append(probs)
        for i in range(len(paths)):
            pred_rows.append(
                {
                    "img_path": paths[i],
                    "img_id": Path(paths[i]).name,
                    "pred_id": int(pred[i]),
                    "pred_label": CLASS_NAMES[int(pred[i])] if int(pred[i]) < len(CLASS_NAMES) else str(int(pred[i])),
                    "confidence": float(conf[i]),
                }
            )

    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(pred_rows).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    np.save(args.out_proba, np.concatenate(all_prob, axis=0))

    print(f"Saved predictions: {args.out_csv}")
    print(f"Saved probabilities: {args.out_proba}")


if __name__ == "__main__":
    main()
