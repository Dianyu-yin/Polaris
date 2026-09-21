"""Unified CNN-only or Qwen-assisted inference entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from agentic_multimodal.defect_detection import (
    DEFAULT_QWEN_MODEL,
    analyze_image_defects,
    load_defect_specs,
    normalize_defect_name,
)
from agentic_multimodal.defect_fusion.model import (
    DefectEffectEncoder,
    DefectFusionPCEModel,
    SURVEY_SCHEMA_VERSION,
    TransformerDefectFusionPCEModel,
    build_default_defect_config,
)
from agentic_multimodal.evidence import (
    CLASS_NAMES,
    DEFAULT_WEIGHTS,
    IMAGE_EXTENSIONS,
    PREPROCESS,
    extract_hybrid_feature,
    load_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_delivery_root() -> Path:
    """这个函数定位双模型交付包目录，让源码快照和项目根目录都能找到 checkpoint。"""

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        has_without_qwen = (
            parent / "without_qwen_convnext_densenet_hybrid" / "weights" / "hybrid_best.pt"
        ).exists()
        has_with_qwen = (
            parent / "with_qwen_defect_fusion" / "checkpoint" / "defect_fusion_best.pt"
        ).exists()
        if has_without_qwen and has_with_qwen:
            return parent
    return PROJECT_ROOT / "deliverables" / "gcl_dual_model_delivery_20260701_1055"


DELIVERY_ROOT = _resolve_delivery_root()
DEFAULT_DELIVERY_WEIGHTS = DELIVERY_ROOT / "without_qwen_convnext_densenet_hybrid" / "weights" / "hybrid_best.pt"
DEFAULT_FUSION_CHECKPOINT = DELIVERY_ROOT / "with_qwen_defect_fusion" / "checkpoint" / "defect_fusion_best.pt"
_default_cache_env = os.environ.get("POLARIS_QWEN_CACHE")
DEFAULT_DEFECT_CACHE = Path(_default_cache_env).expanduser() if _default_cache_env else None


def _resolve_device(device_name: str) -> torch.device:
    """这个函数根据用户参数选择 CPU 或 CUDA 推理设备。"""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _iter_images(input_path: Path, recursive: bool = False) -> list[Path]:
    """这个函数把单张图片或文件夹输入整理成可推理的图片列表。"""

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


def _load_image_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    """这个函数用与训练一致的预处理方式把 EL 图片转成模型 tensor。"""

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return PREPROCESS(image).unsqueeze(0).to(device)


def _load_json_findings(path: Path) -> list[dict[str, Any]]:
    """这个函数读取缓存的 Qwen defect JSON，并兼容 findings/defects 两种字段。"""

    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        raw_items = data.get("findings") or data.get("defects") or []
    else:
        raw_items = []
    return raw_items if isinstance(raw_items, list) else []


def _load_json_payload(path: Path) -> dict[str, Any]:
    """这个函数读取完整的版本化 Qwen cache，供 schema 校验和 provenance 回传。"""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {"findings": data if isinstance(data, list) else []}


def _encode_findings(findings: list[dict[str, Any]], defect_to_id: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """这个函数把 defect/severity JSON 编码成 fusion 模型需要的 ID tensor。"""

    defect_ids: list[int] = []
    severity_ids: list[int] = []
    for item in findings:
        if isinstance(item, list) and len(item) >= 2:
            defect_name = normalize_defect_name(str(item[0]).strip())
            raw_severity = item[1]
        elif isinstance(item, dict):
            defect_name = normalize_defect_name(
                str(item.get("defect") or item.get("name") or item.get("label") or "").strip()
            )
            raw_severity = item.get("severity")
        else:
            continue

        defect_id = defect_to_id.get(defect_name)
        if defect_id is None:
            continue
        try:
            severity = int(raw_severity)
        except (TypeError, ValueError):
            continue
        if severity not in {1, 2, 3}:
            continue
        defect_ids.append(defect_id)
        severity_ids.append(severity)

    if not defect_ids:
        defect_ids = [0]
        severity_ids = [0]
    return torch.tensor([defect_ids], dtype=torch.long), torch.tensor([severity_ids], dtype=torch.long)


def _class_result(probabilities: torch.Tensor) -> dict[str, Any]:
    """这个函数把分类概率整理成可写入 JSON 的类别结果。"""

    probabilities = probabilities.detach().cpu()
    pred_idx = int(probabilities.argmax().item())
    result: dict[str, Any] = {"predicted_class": CLASS_NAMES[pred_idx]}
    for idx, class_name in enumerate(CLASS_NAMES):
        result[f"prob_{class_name}"] = round(float(probabilities[idx].item()), 6)
    return result


def _load_fusion_model(
    checkpoint_path: Path,
    weights_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """这个函数加载 residual Transformer 或 MLP defect-fusion checkpoint。"""

    checkpoint = torch.load(checkpoint_path.expanduser().resolve(), map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_config = checkpoint.get("defect_config", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_schema = checkpoint_config.get("schema_version") if isinstance(checkpoint_config, dict) else None
    if checkpoint_schema != SURVEY_SCHEMA_VERSION:
        raise ValueError(
            "Fusion checkpoint is not compatible with the survey_defect_v2 taxonomy. "
            "Train a new fusion head from the packaged base CNN weights instead of resuming the legacy checkpoint."
        )
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    cnn, _device = load_model(weights_path, device=str(device))
    defect_config = build_default_defect_config()
    defect_encoder = DefectEffectEncoder(
        defect_config=defect_config,
        embedding_dim=int(checkpoint_args.get("defect_embedding_dim", 16)),
        hidden_dim=int(checkpoint_args.get("defect_hidden_dim", 64)),
        output_dim=int(checkpoint_args.get("defect_output_dim", 64)),
        dropout=float(checkpoint_args.get("dropout", 0.2)),
    )

    architecture = str(checkpoint_args.get("fusion_architecture", "mlp"))
    residual_pce = not bool(checkpoint_args.get("no_residual_pce", False))
    if architecture == "transformer":
        model = TransformerDefectFusionPCEModel(
            cnn=cnn,
            defect_encoder=defect_encoder,
            transformer_dim=int(checkpoint_args.get("transformer_dim", 256)),
            transformer_heads=int(checkpoint_args.get("transformer_heads", 4)),
            transformer_layers=int(checkpoint_args.get("transformer_layers", 1)),
            fusion_hidden_dim=int(checkpoint_args.get("fusion_hidden_dim", 256)),
            dropout=float(checkpoint_args.get("dropout", 0.2)),
            residual_pce=residual_pce,
        )
    else:
        model = DefectFusionPCEModel(
            cnn=cnn,
            defect_encoder=defect_encoder,
            fusion_hidden_dim=int(checkpoint_args.get("fusion_hidden_dim", 256)),
            dropout=float(checkpoint_args.get("dropout", 0.2)),
            residual_pce=residual_pce,
        )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def _retrieve_expert_evidence(
    image_path: Path,
    fallback_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """这个函数用当前图像的冻结 CNN feature 检索相似专家案例，并返回可审计 evidence。"""

    store = getattr(args, "expert_store", None)
    if store is None:
        return {"retrieved_cases": [], "retrieval_status": "not_configured"}
    query_embedding = extract_hybrid_feature(image_path, fallback_model, device)
    hits = store.retrieve_by_vector(
        query_embedding,
        top_k=int(getattr(args, "expert_top_k", 3)),
        min_similarity=float(getattr(args, "expert_min_similarity", 0.0)),
        exclude_image_name=(image_path.name if bool(getattr(args, "exclude_current_expert_image", True)) else None),
    )
    hit_payloads = [hit.to_dict() if hasattr(hit, "to_dict") else dict(hit) for hit in hits]
    return {
        "retrieved_cases": hit_payloads,
        "index_metadata": store.get_index_metadata(),
        "retrieval_status": "matched" if hit_payloads else "no_reliable_match",
        "top_k": int(getattr(args, "expert_top_k", 3)),
        "min_similarity": float(getattr(args, "expert_min_similarity", 0.0)),
        "query_image": image_path.name,
        "excluded_current_image": bool(getattr(args, "exclude_current_expert_image", True)),
    }


def _resolve_qwen_findings(
    image_path: Path,
    args: argparse.Namespace,
    expert_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """这个函数决定使用缓存 defect JSON 还是实时调用 Qwen，并记录调用是否成功。"""

    cache_path = args.defects_cache_dir / f"{image_path.stem}.json" if args.defects_cache_dir else None
    if cache_path is not None and cache_path.exists() and not args.refresh_qwen:
        payload = _load_json_payload(cache_path)
        if payload.get("schema_version") == SURVEY_SCHEMA_VERSION:
            return {
                "qwen_status": "cache_hit",
                "qwen_called": False,
                "qwen_cache_path": str(cache_path.resolve()),
                "provider_error": payload.get("provider_error"),
                "findings": _load_json_findings(cache_path),
                "expert_evidence": payload.get("expert_evidence", {}),
                "cache_metadata": {
                    key: payload.get(key)
                    for key in (
                        "schema_version",
                        "prompt_version",
                        "prompt_hash",
                        "survey_config_hash",
                        "expert_qa_hash",
                        "retrieval_index_hash",
                        "source_original_id",
                    )
                },
            }

    expert_evidence = expert_evidence or getattr(args, "expert_evidence", None)
    result = analyze_image_defects(
        image_path=image_path,
        defects=load_defect_specs(),
        model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
        expert_evidence=expert_evidence,
        mock_response=getattr(args, "mock_response", None),
    )
    if result.get("provider_error") and args.require_qwen_success:
        raise RuntimeError(str(result["provider_error"]))
    return {
        "qwen_status": "api_success" if not result.get("provider_error") else "api_failed",
        "qwen_called": True,
        "qwen_cache_path": str(cache_path.resolve()) if cache_path is not None else None,
        "qwen_model": result.get("model"),
        "qwen_requested_model": result.get("requested_model"),
        "qwen_model_candidates": result.get("model_candidates", []),
        "qwen_model_errors": result.get("model_errors", []),
        "provider_error": result.get("provider_error"),
        "findings": result.get("findings", []),
        "expert_evidence": result.get("expert_evidence", expert_evidence or {}),
        "cache_metadata": {},
    }


@torch.no_grad()
def predict_cnn_only(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """这个函数执行不启用 Qwen 时的原始 HybridConvNeXtDenseNet 推理。"""

    image_tensor = _load_image_tensor(image_path, device)
    logits, pce = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    result = {
        "image": str(image_path.resolve()),
        "mode": "cnn_only",
        "qwen_enabled": False,
        "predicted_efficiency": round(float(pce[0].detach().cpu().item()), 4),
        **_class_result(probabilities),
    }
    if metadata:
        result.update(metadata)
    return result


@torch.no_grad()
def predict_with_qwen(
    image_path: Path,
    model: torch.nn.Module,
    fallback_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """这个函数执行启用 Qwen 时的缺陷解析和 defect-fusion 推理。"""

    expert_evidence = _retrieve_expert_evidence(image_path, fallback_model, device, args)
    qwen_result = _resolve_qwen_findings(image_path, args, expert_evidence=expert_evidence)
    if qwen_result["qwen_status"] == "api_failed":
        return predict_cnn_only(
            image_path=image_path,
            model=fallback_model,
            device=device,
            metadata={
                "mode": "cnn_only_fallback",
                "qwen_enabled": False,
                "qwen_requested": True,
                "fallback_reason": "qwen_api_failed",
                "qwen_status": qwen_result["qwen_status"],
                "qwen_called": qwen_result["qwen_called"],
                "qwen_cache_path": qwen_result["qwen_cache_path"],
                "qwen_model": qwen_result.get("qwen_model"),
                "qwen_requested_model": qwen_result.get("qwen_requested_model"),
                "qwen_model_candidates": qwen_result.get("qwen_model_candidates", []),
                "qwen_model_errors": qwen_result.get("qwen_model_errors", []),
                "provider_error": qwen_result["provider_error"],
                "defects": [],
                "expert_evidence": qwen_result.get("expert_evidence", {}),
                "cache_metadata": qwen_result.get("cache_metadata", {}),
            },
        )

    defect_config = build_default_defect_config()
    defect_ids, severity_ids = _encode_findings(qwen_result["findings"], defect_config.defect_to_id)
    output = model(
        images=_load_image_tensor(image_path, device),
        defect_ids=defect_ids.to(device),
        llm_severity_ids=severity_ids.to(device),
    )
    probabilities = output["class_probabilities"][0]
    return {
        "image": str(image_path.resolve()),
        "mode": "qwen_defect_fusion",
        "qwen_enabled": True,
        "qwen_requested": True,
        "qwen_status": qwen_result["qwen_status"],
        "qwen_called": qwen_result["qwen_called"],
        "qwen_cache_path": qwen_result["qwen_cache_path"],
        "qwen_model": qwen_result.get("qwen_model"),
        "qwen_requested_model": qwen_result.get("qwen_requested_model"),
        "qwen_model_candidates": qwen_result.get("qwen_model_candidates", []),
        "qwen_model_errors": qwen_result.get("qwen_model_errors", []),
        "provider_error": qwen_result["provider_error"],
        "defects": qwen_result["findings"],
        "expert_evidence": qwen_result.get("expert_evidence", {}),
        "cache_metadata": qwen_result.get("cache_metadata", {}),
        "predicted_efficiency": round(float(output["final_pce"][0].detach().cpu().item()), 4),
        "cnn_efficiency": round(float(output["cnn_pce"][0].detach().cpu().item()), 4),
        "pce_delta": round(float(output["pce_delta"][0].detach().cpu().item()), 4),
        **_class_result(probabilities),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-mode GCL inference with optional Qwen defect fusion.")
    parser.add_argument("input", type=Path, help="Image file or folder.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan folders.")
    parser.add_argument("--use-qwen", action="store_true", help="Enable Qwen defect fusion instead of CNN-only inference.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_DELIVERY_WEIGHTS if DEFAULT_DELIVERY_WEIGHTS.exists() else DEFAULT_WEIGHTS)
    parser.add_argument("--fusion-checkpoint", type=Path, default=DEFAULT_FUSION_CHECKPOINT)
    parser.add_argument("--defects-cache-dir", type=Path, default=DEFAULT_DEFECT_CACHE)
    parser.add_argument("--refresh-qwen", action="store_true", help="Call Qwen even when a cached defect JSON exists.")
    parser.add_argument("--require-qwen-success", action="store_true", help="Fail instead of degrading when the Qwen API call fails.")
    parser.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL))
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--mock-response", help="Offline smoke hook; parse this JSON instead of calling Qwen.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = _resolve_device(args.device)
    image_paths = _iter_images(args.input, recursive=args.recursive)
    if not image_paths:
        raise SystemExit(f"No supported images found under: {args.input}")

    if args.use_qwen:
        model = _load_fusion_model(args.fusion_checkpoint, args.weights, device)
        fallback_model, device = load_model(args.weights, device=str(device))
        results = [predict_with_qwen(path, model, fallback_model, device, args) for path in image_paths]
    else:
        model, device = load_model(args.weights, device=str(device))
        results = [predict_cnn_only(path, model, device) for path in image_paths]

    output_text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
