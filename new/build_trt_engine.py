import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_path", default="onnx/resnet50_with_rt.onnx")
    ap.add_argument("--engine_path", default="onnx/resnet50_with_rt.trt")
    ap.add_argument("--min_batch", type=int, default=1)
    ap.add_argument("--opt_batch", type=int, default=16)
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--image_c", type=int, default=2)
    ap.add_argument("--image_h", type=int, default=64)
    ap.add_argument("--image_w", type=int, default=64)
    ap.add_argument("--rt_dim", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", default=False)
    args = ap.parse_args()

    try:
        import tensorrt as trt
    except Exception as e:
        raise RuntimeError("TensorRT python package not found.") from e

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            msgs = "\n".join([str(parser.get_error(i)) for i in range(parser.num_errors)])
            raise RuntimeError(f"Failed to parse ONNX:\n{msgs}")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    else:
        config.max_workspace_size = 1 << 30

    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "image",
        (args.min_batch, args.image_c, args.image_h, args.image_w),
        (args.opt_batch, args.image_c, args.image_h, args.image_w),
        (args.max_batch, args.image_c, args.image_h, args.image_w),
    )
    profile.set_shape(
        "rt",
        (args.min_batch, args.rt_dim),
        (args.opt_batch, args.rt_dim),
        (args.max_batch, args.rt_dim),
    )
    config.add_optimization_profile(profile)

    engine_path = Path(args.engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT serialized engine.")
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized_engine))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine.")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

    print(f"Saved TensorRT engine to: {engine_path}")


if __name__ == "__main__":
    main()
