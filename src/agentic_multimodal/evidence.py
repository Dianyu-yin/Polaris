"""Generate CNN predictions, CAM evidence, and tabular features.

This module keeps the packaged HybridConvNeXtDenseNet prediction unchanged and
adds an evidence layer that downstream defect detection and calibrator modules
can consume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from deployable_convnext_densenet_hybrid.model import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_default_weights() -> Path:
    """这个函数定位默认 CNN 权重，让项目源码和交付包源码快照都能直接运行。"""

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        project_weights = parent / "deployable_convnext_densenet_hybrid" / "weights" / "hybrid_best.pt"
        if project_weights.exists():
            return project_weights
        delivery_weights = parent / "without_qwen_convnext_densenet_hybrid" / "weights" / "hybrid_best.pt"
        if delivery_weights.exists():
            return delivery_weights
    return PROJECT_ROOT / "deployable_convnext_densenet_hybrid" / "weights" / "hybrid_best.pt"


DEFAULT_WEIGHTS = _resolve_default_weights()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def iter_images(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return supported image files from a single path or folder."""

    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_label_index(labels_csv: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional labels keyed by image filename for calibrator validation."""

    if labels_csv is None:
        return {}
    labels_path = Path(labels_csv).expanduser().resolve()
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_path}")

    index: dict[str, dict[str, Any]] = {}
    with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            filename = str(row.get("filename") or row.get("image") or row.get("path") or "").strip()
            if not filename:
                combined_path = str(row.get("combined_path") or row.get("class_path") or "").strip()
                filename = Path(combined_path.replace("\\", "/")).name if combined_path else ""
            if not filename:
                continue
            label: dict[str, Any] = {}
            if row.get("class"):
                label["true_class"] = row["class"]
            if row.get("efficiency"):
                label["true_efficiency"] = row["efficiency"]
            if row.get("original_id"):
                label["original_id"] = row["original_id"]
            index[Path(filename).name] = label
    return index


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """Load an image in the same RGB orientation used by the deployable predictor."""

    image = Image.open(image_path)
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a batched model tensor."""

    return PREPROCESS(image).unsqueeze(0)


@torch.no_grad()
def extract_hybrid_feature(
    image_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """这个函数提取冻结 ConvNeXt 与 DenseNet 的拼接特征，供视觉 RAG 做相似案例检索。"""

    image_tensor = preprocess_image(load_rgb_image(image_path)).to(device)
    convnext_feature = model.convnext_pool(model.convnext_features(image_tensor)).flatten(1)
    densenet_feature = model.densenet_pool(model.densenet_features(image_tensor)).flatten(1)
    feature = torch.cat([convnext_feature, densenet_feature], dim=1)
    feature = torch.nn.functional.normalize(feature, p=2, dim=1)
    return feature[0].detach().cpu().numpy().astype(np.float32)


def _entropy(probabilities: torch.Tensor) -> float:
    probs = probabilities.clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


def _top2_margin(probabilities: torch.Tensor) -> float:
    values = torch.topk(probabilities, k=min(2, probabilities.numel())).values
    if values.numel() < 2:
        return float(values[0].item())
    return float((values[0] - values[1]).item())


def _target_layers(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return {
        "convnext": model.convnext_features[-1],
        "densenet": model.densenet_features[-1],
    }


class _CamHook:
    def __init__(self, layer: torch.nn.Module) -> None:
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self._forward_handle = layer.register_forward_hook(self._save_activation)
        self._backward_handle = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.activation = output

    def _save_gradient(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradient = grad_output[0]

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: int,
) -> np.ndarray:
    """Generate a normalized Grad-CAM map for one class and one target layer."""

    hook = _CamHook(target_layer)
    try:
        model.zero_grad(set_to_none=True)
        x = image_tensor.detach().clone().requires_grad_(True)
        logits, _efficiency = model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        if hook.activation is None or hook.gradient is None:
            raise RuntimeError("CAM hooks did not capture activations and gradients")

        activation = hook.activation.detach()
        gradient = hook.gradient.detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        raw_cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.relu(raw_cam)
        if float(cam.max().detach().cpu().item()) == 0.0:
            cam = raw_cam.abs()
        cam = F.interpolate(cam, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu()
        cam_min = float(cam.min().item())
        cam_max = float(cam.max().item())
        if math.isclose(cam_max, cam_min):
            return np.zeros(IMAGE_SIZE, dtype=np.float32)
        return ((cam - cam_min) / (cam_max - cam_min + 1e-8)).numpy().astype(np.float32)
    finally:
        hook.close()
        model.zero_grad(set_to_none=True)


def _cam_stats(cam: np.ndarray) -> dict[str, float]:
    mask = cam >= 0.5
    return {
        "coverage": round(float(mask.mean()), 6),
        "mean": round(float(cam.mean()), 6),
        "peak": round(float(cam.max()), 6),
    }


def _cam_iou(left: np.ndarray, right: np.ndarray, threshold: float = 0.5) -> float:
    left_mask = left >= threshold
    right_mask = right >= threshold
    union = np.logical_or(left_mask, right_mask).sum()
    if union == 0:
        return 0.0
    return round(float(np.logical_and(left_mask, right_mask).sum() / union), 6)


def _save_cam_overlay(image: Image.Image, cam: np.ndarray, output_path: Path) -> None:
    base = image.resize(IMAGE_SIZE).convert("RGBA")
    alpha = np.clip(cam * 170, 0, 170).astype(np.uint8)
    heatmap = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 4), dtype=np.uint8)
    heatmap[..., 0] = 255
    heatmap[..., 1] = np.clip(cam * 190, 0, 190).astype(np.uint8)
    heatmap[..., 3] = alpha
    overlay = Image.alpha_composite(base, Image.fromarray(heatmap))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(output_path)


def _quality_flags(max_probability: float, margin: float, cams: dict[str, np.ndarray]) -> list[str]:
    flags: list[str] = []
    if max_probability < 0.6:
        flags.append("low_confidence")
    if margin < 0.15:
        flags.append("narrow_margin")

    if cams:
        coverages = [_cam_stats(cam)["coverage"] for cam in cams.values()]
        avg_coverage = sum(coverages) / len(coverages)
        if avg_coverage < 0.01:
            flags.append("tiny_cam")
        if avg_coverage > 0.5:
            flags.append("diffuse_cam")
    if "convnext" in cams and "densenet" in cams and _cam_iou(cams["convnext"], cams["densenet"]) < 0.1:
        flags.append("branch_cam_disagreement")
    return flags


def predict_with_evidence(
    image_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str | Path | None = None,
    generate_cam: bool = True,
) -> dict[str, Any]:
    """Run unchanged CNN prediction and attach evidence metadata."""

    image_path = Path(image_path).expanduser().resolve()
    image = load_rgb_image(image_path)
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits, efficiency = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        pred_idx = int(probabilities.argmax().item())
        predicted_class = CLASS_NAMES[pred_idx]
        predicted_efficiency = float(efficiency[0].detach().cpu().item())

    probability_values = {
        f"prob_{class_name}": round(float(probabilities[idx].item()), 6)
        for idx, class_name in enumerate(CLASS_NAMES)
    }
    max_probability = max(probability_values.values())
    margin = _top2_margin(probabilities)
    entropy = _entropy(probabilities)

    cam_paths: dict[str, str] = {}
    cam_stats: dict[str, dict[str, float]] = {}
    cams: dict[str, np.ndarray] = {}
    cam_error: str | None = None

    if generate_cam:
        try:
            for branch_name, layer in _target_layers(model).items():
                cam = generate_gradcam(model, image_tensor, layer, pred_idx)
                cams[branch_name] = cam
                cam_stats[branch_name] = _cam_stats(cam)
                if output_dir is not None:
                    cam_path = Path(output_dir) / "cam" / f"{image_path.stem}_{branch_name}_{predicted_class}.png"
                    _save_cam_overlay(image, cam, cam_path)
                    cam_paths[branch_name] = str(cam_path.resolve())

            if "convnext" in cams and "densenet" in cams:
                fused = np.maximum(cams["convnext"], cams["densenet"])
                cams["fused"] = fused
                cam_stats["fused"] = _cam_stats(fused)
                cam_stats["branch_cam_iou"] = {"value": _cam_iou(cams["convnext"], cams["densenet"])}
                if output_dir is not None:
                    fused_path = Path(output_dir) / "cam" / f"{image_path.stem}_fused_{predicted_class}.png"
                    _save_cam_overlay(image, fused, fused_path)
                    cam_paths["fused"] = str(fused_path.resolve())
        except Exception as exc:  # CAM should never break the base prediction path.
            cam_error = str(exc)

    flags = _quality_flags(max_probability=max_probability, margin=margin, cams=cams)
    if cam_error:
        flags.append("cam_generation_failed")

    result: dict[str, Any] = {
        "image": str(image_path),
        "predicted_class": predicted_class,
        "predicted_efficiency": round(predicted_efficiency, 4),
        "logits": [round(float(value), 6) for value in logits[0].detach().cpu().tolist()],
        **probability_values,
        "max_probability": round(float(max_probability), 6),
        "margin": round(float(margin), 6),
        "uncertainty_entropy": round(float(entropy), 6),
        "cam_paths": cam_paths,
        "cam_stats": cam_stats,
        "quality_flags": flags,
    }
    if cam_error:
        result["cam_error"] = cam_error
    return result


def evidence_to_feature_row(evidence: dict[str, Any]) -> dict[str, Any]:
    """Flatten an evidence object for calibrator-friendly CSV output."""

    cam_stats = evidence.get("cam_stats", {})
    branch_iou = cam_stats.get("branch_cam_iou", {}).get("value", "")
    row: dict[str, Any] = {
        "image": evidence.get("image", ""),
        "original_id": evidence.get("original_id", ""),
        "true_class": evidence.get("true_class", ""),
        "true_efficiency": evidence.get("true_efficiency", ""),
        "predicted_class": evidence.get("predicted_class", ""),
        "predicted_efficiency": evidence.get("predicted_efficiency", ""),
        "prob_high": evidence.get("prob_high", ""),
        "prob_low": evidence.get("prob_low", ""),
        "prob_middle": evidence.get("prob_middle", ""),
        "max_probability": evidence.get("max_probability", ""),
        "margin": evidence.get("margin", ""),
        "uncertainty_entropy": evidence.get("uncertainty_entropy", ""),
        "quality_flags": "|".join(evidence.get("quality_flags", [])),
        "branch_cam_iou": branch_iou,
    }
    for branch_name in ("convnext", "densenet", "fused"):
        stats = cam_stats.get(branch_name, {})
        row[f"{branch_name}_cam_coverage"] = stats.get("coverage", "")
        row[f"{branch_name}_cam_mean"] = stats.get("mean", "")
        row[f"{branch_name}_cam_peak"] = stats.get("peak", "")
    return row


def _write_csv(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_evidence_json(evidence: dict[str, Any], output_dir: Path) -> Path:
    image_path = Path(str(evidence["image"]))
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_path = evidence_dir / f"{image_path.stem}.json"
    output_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return output_path


def run_batch(
    image_paths: Iterable[Path],
    output_dir: Path,
    weights: Path = DEFAULT_WEIGHTS,
    device: str = "auto",
    generate_cam: bool = True,
    labels_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    model, resolved_device = load_model(weights, device=device)
    label_index = load_label_index(labels_csv)
    evidence_items: list[dict[str, Any]] = []
    for image_path in image_paths:
        evidence = predict_with_evidence(
            image_path=image_path,
            model=model,
            device=resolved_device,
            output_dir=output_dir,
            generate_cam=generate_cam,
        )
        evidence.update(label_index.get(Path(image_path).name, {}))
        _write_evidence_json(evidence, output_dir)
        evidence_items.append(evidence)
    return evidence_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GCL hybrid-model evidence packs.")
    parser.add_argument("input", type=Path, help="Image file or image folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("agentic_outputs"), help="Output directory.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to hybrid_best.pt.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda device.")
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively.")
    parser.add_argument("--no-cam", action="store_true", help="Skip CAM generation.")
    parser.add_argument("--labels-csv", type=Path, help="Optional CSV containing filename, class, and efficiency labels.")
    args = parser.parse_args()

    image_paths = iter_images(args.input, recursive=args.recursive)
    if not image_paths:
        raise SystemExit(f"No supported images found under: {args.input}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_items = run_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        weights=args.weights,
        device=args.device,
        generate_cam=not args.no_cam,
        labels_csv=args.labels_csv,
    )

    feature_rows = [evidence_to_feature_row(item) for item in evidence_items]
    _write_csv(feature_rows, output_dir / "evidence_features.csv")
    _write_csv(
        [
            {
                "image": item["image"],
                "predicted_class": item["predicted_class"],
                "predicted_efficiency": item["predicted_efficiency"],
                "prob_high": item["prob_high"],
                "prob_low": item["prob_low"],
                "prob_middle": item["prob_middle"],
            }
            for item in evidence_items
        ],
        output_dir / "predictions.csv",
    )
    print(f"Wrote {len(evidence_items)} evidence packs to {output_dir}")


if __name__ == "__main__":
    main()
