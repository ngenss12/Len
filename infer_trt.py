import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_path", default="outputs/lstm/best/model.trt")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=128)
    args = ap.parse_args()

    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except Exception as e:
        raise RuntimeError(
            "TensorRT/pycuda not found. Install TensorRT and pycuda, or use trtexec."
        ) from e

    logger = trt.Logger(trt.Logger.INFO)
    with open(args.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to load TensorRT engine.")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT context.")

    x = np.random.randn(args.batch_size, args.seq_len, 4).astype(np.float32)
    lengths = np.full((args.batch_size,), args.seq_len, dtype=np.int64)

    # Set shapes for dynamic inputs
    context.set_input_shape("x", x.shape)
    context.set_input_shape("lengths", lengths.shape)

    # Allocate buffers
    bindings = [None] * engine.num_io_tensors
    device_buffers = {}

    def alloc_and_bind(name, host_array):
        idx = engine.get_tensor_index(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(np.prod(host_array.shape))
        host = host_array.astype(dtype, copy=False)
        device = cuda.mem_alloc(host.nbytes)
        bindings[idx] = int(device)
        device_buffers[name] = (host, device)

    alloc_and_bind("x", x)
    alloc_and_bind("lengths", lengths)

    # Output
    out_name = "logits"
    out_shape = tuple(context.get_tensor_shape(out_name))
    out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))
    out_host = np.empty(out_shape, dtype=out_dtype)
    out_device = cuda.mem_alloc(out_host.nbytes)
    bindings[engine.get_tensor_index(out_name)] = int(out_device)

    # H2D
    for name in ["x", "lengths"]:
        host, device = device_buffers[name]
        cuda.memcpy_htod(device, host)

    # Execute
    context.execute_v2(bindings)

    # D2H
    cuda.memcpy_dtoh(out_host, out_device)
    print(f"Output logits shape: {out_host.shape}")


if __name__ == "__main__":
    main()
