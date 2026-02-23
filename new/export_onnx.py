import argparse
from pathlib import Path

import torch

from model import ResNet50WithRT

def load_model(checkpoint_path, device, rt_dim=8, num_classes=4, pretrained=False):
    model = ResNet50WithRT(rt_dim=rt_dim, num_classes=num_classes, pretrained=pretrained)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def export_onnx(
    output_path,
    checkpoint_path=None,
    device="cpu",
    batch_size=1,
    image_size=64,
    rt_dim=8,
    num_classes=4,
    dynamo=False,
):
    model = load_model(checkpoint_path, device, rt_dim=rt_dim, num_classes=num_classes)

    dummy_image = torch.randn(batch_size, 2, image_size, image_size, device=device, dtype=torch.float32)
    dummy_rt = torch.randn(batch_size, rt_dim, device=device, dtype=torch.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_image, dummy_rt),
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image", "rt"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "rt": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=dynamo,
    )


def main():
    parser = argparse.ArgumentParser(description="Export ResNet50WithRT to ONNX")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="onnx/resnet50_with_rt.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--rt-dim", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--dynamo", action="store_true", default=False, help="Use torch.export-based ONNX exporter")
    args = parser.parse_args()

    export_onnx(
        output_path=args.output,
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        rt_dim=args.rt_dim,
        num_classes=args.num_classes,
        dynamo=args.dynamo,
    )


if __name__ == "__main__":
    main()
