import hashlib
import re
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import torch
from torch import nn
from torchvision import models


CLASS_NAMES: Tuple[str, str, str] = ("high", "low", "middle")
IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGENET_WEIGHT_FILES = {
    "convnext_tiny": (
        "convnext_tiny-983f1562.pth",
        "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
    ),
    "densenet121": (
        "densenet121-a639ec97.pth",
        "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395",
    ),
}

_DENSENET_LEGACY_KEY_PATTERN = re.compile(
    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
)


def _sha256_file(path: Path) -> str:
    """这个函数流式计算初始化权重 hash，防止离线节点加载错文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_densenet_legacy_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """这个函数复现 torchvision 官方 DenseNet loader 的旧键名迁移，例如 norm.1→norm1。"""

    normalized = dict(state_dict)
    for old_key in list(normalized):
        match = _DENSENET_LEGACY_KEY_PATTERN.match(old_key)
        if match is None:
            continue
        new_key = match.group(1) + match.group(2)
        if new_key in normalized:
            raise ValueError(f"DenseNet key normalization collision: {old_key} -> {new_key}")
        normalized[new_key] = normalized.pop(old_key)
    return normalized


def _load_verified_imagenet_state_dict(weights_dir: Path, model_name: str) -> Mapping[str, torch.Tensor]:
    """这个函数只加载 hash 完全匹配的 torchvision 官方 ImageNet-1K 权重。"""

    filename, expected_hash = IMAGENET_WEIGHT_FILES[model_name]
    path = weights_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing offline ImageNet weight: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"ImageNet weight SHA256 mismatch for {path}: {actual_hash} != {expected_hash}"
        )
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"ImageNet weight did not contain a state dict: {path}")
    if model_name == "densenet121":
        return _normalize_densenet_legacy_keys(state_dict)
    return state_dict


class HybridConvNeXtDenseNet(nn.Module):
    """ConvNeXt-Tiny + DenseNet121 hybrid used for GCL PCE classification/regression."""

    def __init__(
        self,
        n_classes: int = len(CLASS_NAMES),
        imagenet_weights_dir: str | Path | None = None,
    ) -> None:
        super().__init__()

        convnext = models.convnext_tiny(weights=None)
        densenet = models.densenet121(weights=None)
        if imagenet_weights_dir is not None:
            resolved_dir = Path(imagenet_weights_dir).expanduser().resolve()
            convnext.load_state_dict(
                _load_verified_imagenet_state_dict(resolved_dir, "convnext_tiny"), strict=True
            )
            densenet.load_state_dict(
                _load_verified_imagenet_state_dict(resolved_dir, "densenet121"), strict=True
            )
        self.convnext_features = convnext.features
        self.convnext_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.densenet_features = densenet.features
        self.densenet_pool = nn.AdaptiveAvgPool2d((1, 1))

        combined_dim = 768 + 1024
        self.cls_head = nn.Linear(combined_dim, n_classes)
        self.reg_head = nn.Linear(combined_dim, 1)
        self.initialization = (
            {
                model_name: {
                    "filename": filename,
                    "sha256": expected_hash,
                }
                for model_name, (filename, expected_hash) in IMAGENET_WEIGHT_FILES.items()
            }
            if imagenet_weights_dir is not None
            else {"source": "none_or_external_checkpoint"}
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        convnext_feat = self.convnext_pool(self.convnext_features(x)).flatten(1)
        densenet_feat = self.densenet_pool(self.densenet_features(x)).flatten(1)
        features = torch.cat([convnext_feat, densenet_feat], dim=1)
        return self.cls_head(features), self.reg_head(features).squeeze(1)


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint did not contain a state dict: {checkpoint_path}")

    return state_dict


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    class_names: Iterable[str] = CLASS_NAMES,
) -> Tuple[HybridConvNeXtDenseNet, torch.device]:
    """Load the packaged hybrid model and return (model, device)."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if device is None or str(device).lower() == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    n_classes = len(tuple(class_names))
    model = HybridConvNeXtDenseNet(n_classes=n_classes)
    state_dict = _load_state_dict(checkpoint_path, resolved_device)
    model.load_state_dict(state_dict, strict=True)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def build_imagenet_model(
    weights_dir: str | Path,
    device: str | torch.device | None = None,
    class_names: Iterable[str] = CLASS_NAMES,
) -> Tuple[HybridConvNeXtDenseNet, torch.device]:
    """这个函数从本地已校验 ImageNet-1K 权重构建全新的双分支 CNN。"""

    if device is None or str(device).lower() == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    model = HybridConvNeXtDenseNet(
        n_classes=len(tuple(class_names)),
        imagenet_weights_dir=weights_dir,
    )
    model.to(resolved_device)
    return model, resolved_device
