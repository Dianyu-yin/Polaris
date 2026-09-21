"""在锁定的旧协议 1,304-row validation fold 上评价一个正式 checkpoint。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ablation.experiment_contract import get_condition
from ablation.train_fixed import (
    CHECKPOINT_FORMAT_VERSION,
    EXPECTED_VAL_SHA256,
    _resolve_device,
    sha256_file,
    validate_mode_cache_manifest,
)
from agentic_multimodal.defect_fusion.dataset import (
    CLASS_LABEL_TO_ID,
    DefectFusionDataset,
    collate_defect_fusion_batch,
)
from agentic_multimodal.defect_fusion.model import (
    CnnOnlyHybridPCEModel,
    DefectEffectEncoder,
    DefectFusionPCEModel,
    TransformerDefectFusionPCEModel,
    build_default_defect_config,
)
from deployable_convnext_densenet_hybrid.model import CLASS_NAMES, HybridConvNeXtDenseNet


EXPECTED_VALIDATION_ROWS = 1304
EXPECTED_VALIDATION_GROUPS = 487
EXPECTED_CLASSIFICATION_SUPPORT = 1298


def metric_summary(predictions: list[float], targets: list[float]) -> dict[str, float]:
    """这个函数按预注册公式计算误差指标，并保留 PCE percentage-point 单位。"""

    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be non-empty and have equal length")
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    mae = sum(abs(error) for error in errors) / len(errors)
    mse = sum(error * error for error in errors) / len(errors)
    target_mean = sum(targets) / len(targets)
    sse = sum(error * error for error in errors)
    sst = sum((target - target_mean) ** 2 for target in targets)
    if sst <= 0:
        raise ValueError("R2 is undefined because test targets have zero variance")
    return {
        "mae": mae,
        "rmse": math.sqrt(mse),
        "r2": 1.0 - sse / sst,
        "bias": sum(errors) / len(errors),
        "median_absolute_error": float(statistics.median(abs(error) for error in errors)),
    }


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """这个函数加载 formal compact checkpoint，并拒绝未知格式或非 best artifact。"""

    path = path.expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dict: {path}")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    return checkpoint


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """这个函数重建 CNN/full fusion，并加载 fusion 微调后的完整模型状态。"""

    condition_payload = checkpoint.get("condition") or {}
    condition = get_condition(str(condition_payload.get("condition_id")))
    seed = int(checkpoint.get("seed", -1))
    stage = str(checkpoint.get("stage"))
    defect_config = build_default_defect_config()
    if stage == "cnn":
        state_dict = checkpoint.get("cnn_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("CNN checkpoint has no cnn_state_dict")
        cnn = HybridConvNeXtDenseNet()
        cnn.load_state_dict(state_dict, strict=True)
        cnn.to(device)
        model: torch.nn.Module = CnnOnlyHybridPCEModel(cnn=cnn).to(device)
        provenance = {"cnn_checkpoint": str(checkpoint_path), "cnn_checkpoint_sha256": sha256_file(checkpoint_path)}
    elif stage == "fusion":
        run_config = checkpoint.get("run_config") or {}
        initialization = ((run_config.get("model") or {}).get("initialization") or {})
        cnn = HybridConvNeXtDenseNet().to(device)
        defect_encoder = DefectEffectEncoder(
            defect_config=defect_config,
            embedding_dim=16,
            hidden_dim=64,
            output_dim=64,
            dropout=float(((run_config.get("training") or {}).get("dropout", 0.2))),
        )
        # 历史 run_config 的 dropout 位于 args；固定默认值仍锁定为用户确认的 0.2。
        dropout = float(((run_config.get("args") or {}).get("dropout", 0.2)))
        if condition.architecture == "transformer":
            model = TransformerDefectFusionPCEModel(
                cnn=cnn,
                defect_encoder=defect_encoder,
                transformer_dim=256,
                transformer_heads=4,
                transformer_layers=1,
                fusion_hidden_dim=256,
                dropout=dropout,
                residual_pce=condition.residual_pce,
                use_null_semantic_token=condition.use_null_semantic_token,
            ).to(device)
        elif condition.architecture == "mlp":
            model = DefectFusionPCEModel(
                cnn=cnn,
                defect_encoder=defect_encoder,
                fusion_hidden_dim=256,
                dropout=dropout,
                residual_pce=condition.residual_pce,
            ).to(device)
        else:
            raise ValueError(f"Unsupported fusion architecture: {condition.architecture}")
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict) or not any(
            str(key).startswith("cnn.") for key in model_state
        ):
            raise ValueError("Fusion checkpoint must contain the full fine-tuned model state")
        model.load_state_dict(model_state, strict=True)
        provenance = {
            "fusion_checkpoint": str(checkpoint_path),
            "fusion_checkpoint_sha256": sha256_file(checkpoint_path),
            "initial_cnn_checkpoint": initialization.get("checkpoint"),
            "initial_cnn_checkpoint_sha256": initialization.get("checkpoint_sha256"),
        }
    else:
        raise ValueError(f"Unsupported checkpoint stage: {stage}")
    model.eval()
    model.cnn.eval()
    return model, condition, provenance


def _build_validation_dataset(
    args: argparse.Namespace,
    condition: Any,
) -> tuple[DefectFusionDataset, Path | None]:
    """这个函数只读取锁定旧 validation CSV，并对语义条件要求完整 cache。"""

    defects_dir = None
    if condition.cache_name is not None:
        if args.cache_root is None:
            raise ValueError(f"Condition {condition.condition_id} requires --cache-root")
        defects_dir = args.cache_root.expanduser().resolve() / condition.cache_name
        if not defects_dir.is_dir():
            raise FileNotFoundError(f"Missing cache directory for {condition.condition_id}: {defects_dir}")
    dataset = DefectFusionDataset(
        labels_csv=args.val_csv,
        defect_to_id=build_default_defect_config().defect_to_id,
        image_root=args.image_root,
        defects_dir=defects_dir,
        cache_mode=condition.cache_mode,
    )
    groups = {sample.group_id for sample in dataset.samples}
    classification_support = sum(
        sample.class_label in CLASS_LABEL_TO_ID for sample in dataset.samples
    )
    if len(dataset) != EXPECTED_VALIDATION_ROWS or len(groups) != EXPECTED_VALIDATION_GROUPS:
        raise ValueError(
            "Validation must contain 1304 rows/487 groups, "
            f"got {len(dataset)}/{len(groups)}"
        )
    if classification_support != EXPECTED_CLASSIFICATION_SUPPORT:
        raise ValueError(
            f"Validation classification support {classification_support} != {EXPECTED_CLASSIFICATION_SUPPORT}"
        )
    return dataset, defects_dir


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """这个函数推理 1,304 个验证样本并输出逐行 prediction 与锁定指标。"""

    device = _resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model, condition, provenance = _build_model_from_checkpoint(
        checkpoint, checkpoint_path, device
    )
    val_path = args.val_csv.expanduser().resolve()
    val_hash = sha256_file(val_path)
    if val_hash != EXPECTED_VAL_SHA256:
        raise ValueError(f"Validation CSV SHA-256 drift: {val_hash} != {EXPECTED_VAL_SHA256}")
    dataset, defects_dir = _build_validation_dataset(args, condition)
    cache_manifest = (
        validate_mode_cache_manifest(defects_dir, condition)
        if defects_dir is not None
        else None
    )
    if cache_manifest is not None:
        training_cache = (((checkpoint.get("run_config") or {}).get("inputs") or {}).get("cache_manifest"))
        if not isinstance(training_cache, dict) or training_cache.get("sha256") != cache_manifest["sha256"]:
            raise ValueError("Evaluation cache manifest differs from the cache used for training")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_defect_fusion_batch,
    )
    predictions: list[float] = []
    targets: list[float] = []
    records: list[dict[str, Any]] = []
    class_correct = 0
    class_support = 0
    use_amp = bool(args.amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            defect_ids = batch["defect_ids"].to(device)
            severity_ids = batch["llm_severity_ids"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(
                    images=images,
                    defect_ids=defect_ids,
                    llm_severity_ids=severity_ids,
                )
            batch_predictions = output["final_pce"].detach().float().cpu().tolist()
            batch_targets = batch["target_pce"].float().tolist()
            batch_cnn = output["cnn_pce"].detach().float().cpu().tolist()
            batch_delta = output["pce_delta"].detach().float().cpu().tolist()
            batch_class = output["logits"].argmax(dim=1).detach().cpu().tolist()
            for index, prediction in enumerate(batch_predictions):
                target = float(batch_targets[index])
                signed_error = float(prediction) - target
                true_class = str(batch["class_labels"][index])
                true_class_index = CLASS_LABEL_TO_ID.get(true_class, -100)
                predicted_class_index = int(batch_class[index])
                scored = true_class_index >= 0
                if scored:
                    class_support += 1
                    class_correct += int(predicted_class_index == true_class_index)
                record = {
                    "condition": condition.condition_id,
                    "seed": int(checkpoint["seed"]),
                    "group_id": batch["group_ids"][index],
                    "filename": batch["filenames"][index],
                    "target_pce": target,
                    "predicted_pce": float(prediction),
                    "signed_error": signed_error,
                    "absolute_error": abs(signed_error),
                    "cnn_pce": float(batch_cnn[index]),
                    "pce_delta": float(batch_delta[index]),
                    "true_class": true_class,
                    "predicted_class": CLASS_NAMES[predicted_class_index],
                    "classification_scored": int(scored),
                }
                if not all(
                    math.isfinite(float(record[key]))
                    for key in ("target_pce", "predicted_pce", "signed_error", "absolute_error", "cnn_pce", "pce_delta")
                ):
                    raise FloatingPointError(f"Non-finite prediction record: {record}")
                records.append(record)
                predictions.append(float(prediction))
                targets.append(target)

    if len(records) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("Evaluation output must contain exactly 1304 rows")
    if len({row["group_id"] for row in records}) != EXPECTED_VALIDATION_GROUPS:
        raise ValueError("Evaluation output must contain exactly 487 original-image groups")
    if class_support != EXPECTED_CLASSIFICATION_SUPPORT:
        raise ValueError(f"Classification denominator drift: {class_support}")
    metrics = metric_summary(predictions, targets)
    metrics["classification_accuracy"] = class_correct / class_support
    metrics["classification_support"] = class_support
    metrics["n_validation_rows"] = len(records)
    metrics["n_validation_groups"] = len({row["group_id"] for row in records})

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "PASS",
        "condition": condition.condition_id,
        "condition_config": condition.to_dict(),
        "seed": int(checkpoint["seed"]),
        "n_validation_rows": len(records),
        "n_validation_groups": len({row["group_id"] for row in records}),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_best_val_mae": float(checkpoint["best_val_mae"]),
        "validation_csv": str(val_path),
        "validation_csv_sha256": val_hash,
        "defects_dir": str(defects_dir) if defects_dir else None,
        "cache_manifest": cache_manifest,
        "provenance": provenance,
        "metrics": metrics,
        "definitions": {
            "unit": "PCE percentage points",
            "mae": "mean(abs(predicted_pce - target_pce)); lower is better",
            "rmse": "sqrt(mean((predicted_pce - target_pce)^2)); lower is better",
            "r2": "1 - SSE/SST; higher is better",
            "bias": "mean(predicted_pce - target_pce); ideal is 0",
            "median_absolute_error": "median(abs(predicted_pce - target_pce)); lower is better",
            "classification_accuracy": "correct / 1298 non-rejected validation rows; higher is better",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义固定旧 validation evaluator 的命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """这个函数运行锁定 validation 评价并打印最终 metrics JSON。"""

    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
