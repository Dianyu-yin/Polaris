"""在不使用 GPU 的情况下完整加载两份离线 torchvision ImageNet 权重。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deployable_convnext_densenet_hybrid.model import build_imagenet_model


def check_weights(weights_dir: Path) -> dict[str, object]:
    """这个函数在 CPU 构建双 backbone，并返回可审计的参数量与初始化 provenance。"""

    model, device = build_imagenet_model(weights_dir, device="cpu")
    return {
        "status": "PASS",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": model.initialization,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义离线权重 CPU preflight 的唯一输入目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    """这个函数运行 CPU preflight 并输出单行机器可读结果。"""

    print(json.dumps(check_weights(parse_args().weights_dir), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
