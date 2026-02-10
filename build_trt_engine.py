import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_path", default="outputs/lstm/best/model.onnx")
    ap.add_argument("--engine_path", default="outputs/lstm/best/model.trt")
    ap.add_argument("--min_batch", type=int, default=1)
    ap.add_argument("--opt_batch", type=int, default=16)
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--min_len", type=int, default=32)
    ap.add_argument("--opt_len", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", default=False)
    args = ap.parse_args()

    try:
        import tensorrt as trt
    except Exception as e:
        raise RuntimeError(
            "TensorRT python package not found. Install TensorRT or use trtexec."
        ) from e

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            msgs = "\n".join([str(parser.get_error(i)) for i in range(parser.num_errors)])
            raise RuntimeError(f"Failed to parse ONNX:\n{msgs}")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    x_name = "x"
    lengths_name = "lengths"
    profile.set_shape(
        x_name,
        (args.min_batch, args.min_len, 4),
        (args.opt_batch, args.opt_len, 4),
        (args.max_batch, args.max_len, 4),
    )
    profile.set_shape(
        lengths_name,
        (args.min_batch,),
        (args.opt_batch,),
        (args.max_batch,),
    )
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    os.makedirs(os.path.dirname(args.engine_path), exist_ok=True)
    with open(args.engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"Saved TensorRT engine to: {args.engine_path}")


if __name__ == "__main__":
    main()
