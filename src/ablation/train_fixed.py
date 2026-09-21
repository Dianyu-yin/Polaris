"""按固定旧fold训练POLARIS CNN或六个fusion条件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ablation.cache_contract import validate_mode_cache_manifest
from ablation.experiment_contract import TRAINING_SEEDS, Condition, get_condition
from agentic_multimodal.defect_fusion.dataset import (
    DefectFusionDataset,
    collate_defect_fusion_batch,
)
from agentic_multimodal.defect_fusion.model import (
    CnnOnlyHybridPCEModel,
    DefectEffectEncoder,
    DefectFusionPCEModel,
    TransformerDefectFusionPCEModel,
    build_default_defect_config,
    configure_trainable_cnn_layers,
)
from agentic_multimodal.defect_fusion.train import (
    _run_epoch,
    _seed_training,
    _set_cnn_stage_trainable,
)
from deployable_convnext_densenet_hybrid.model import (
    HybridConvNeXtDenseNet,
    build_imagenet_model,
    load_model,
)


EXPECTED_SPLIT = {
    "train": {"rows": 5142, "groups": 1974, "rejected_groups": 25},
    "val": {"rows": 1304, "groups": 487, "rejected_groups": 6},
}
EXPECTED_TRAIN_SHA256 = "8f014367549bba8347b0a0a9c91292a65ea163e4873a9d2068759051ad646d8c"
EXPECTED_VAL_SHA256 = "be7af23822cbcd6a60f8829a84886519f9f85860352fc28b40dfb879de364dd4"
EXPECTED_BASE_CNN_SHA256 = "a38bce10ffe2b2e637cba75a10327d75b9779f06d23de4517b9fed2544bc1f60"
TRAINING_CONTRACT_VERSION = "polaris-legacy-ablation-training-v2"
CHECKPOINT_FORMAT_VERSION = "polaris-legacy-ablation-compact-v2"
BEST_RULE = "strictly lower fixed-validation row MAE; ties keep earliest epoch"


def sha256_file(path: Path) -> str:
    """这个函数流式计算 run 输入和 checkpoint hash，锁定每次训练的 provenance。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    """这个函数把命令行参数转换成 checkpoint/config 可序列化字典。"""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _resolve_device(name: str) -> torch.device:
    """这个函数解析 auto/cuda/cpu，并拒绝请求 CUDA 时静默退回 CPU。"""

    if name == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Formal training requires CUDA, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def _dataset_groups(dataset: DefectFusionDataset) -> set[str]:
    """这个函数返回 Dataset 中独立 original_id 集合。"""

    return {sample.group_id for sample in dataset.samples}


def _validate_fixed_datasets(
    train_dataset: DefectFusionDataset,
    val_dataset: DefectFusionDataset,
) -> dict[str, Any]:
    """这个函数验证旧fold数量、增强图口径、分类mask和组零交集。"""

    train_groups = _dataset_groups(train_dataset)
    val_groups = _dataset_groups(val_dataset)
    intersection = sorted(train_groups & val_groups)
    summary: dict[str, Any] = {
        "train": {
            "rows": len(train_dataset),
            "groups": len(train_groups),
            "rejected_groups": len(
                {sample.group_id for sample in train_dataset.samples if sample.class_label == "rejected"}
            ),
        },
        "val": {
            "rows": len(val_dataset),
            "groups": len(val_groups),
            "rejected_groups": len(
                {sample.group_id for sample in val_dataset.samples if sample.class_label == "rejected"}
            ),
            "classification_support": sum(
                sample.class_label in {"high", "low", "middle"}
                for sample in val_dataset.samples
            ),
        },
        "intersection": intersection,
    }
    for split_name in ("train", "val"):
        for key, expected in EXPECTED_SPLIT[split_name].items():
            if summary[split_name][key] != expected:
                raise ValueError(
                    f"{split_name} {key}={summary[split_name][key]} != contract {expected}"
                )
    if intersection:
        raise ValueError(f"train/val original_id leakage: {intersection[:20]}")
    if summary["val"]["classification_support"] != 1298:
        raise ValueError("Validation classification support must be 1298 after masking rejected")
    return summary


def _load_cnn_checkpoint(path: Path, device: torch.device) -> tuple[HybridConvNeXtDenseNet, dict[str, Any]]:
    """这个函数加载同 seed CNN best，并验证 compact checkpoint 的 stage 和 raw CNN state。"""

    path = path.expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("stage") != "cnn":
        raise ValueError(f"Not a formal CNN checkpoint: {path}")
    state_dict = checkpoint.get("cnn_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"CNN checkpoint has no cnn_state_dict: {path}")
    cnn = HybridConvNeXtDenseNet()
    cnn.load_state_dict(state_dict, strict=True)
    cnn.to(device)
    return cnn, checkpoint


def _build_datasets(args: argparse.Namespace, condition: Condition) -> tuple[DefectFusionDataset, DefectFusionDataset, dict[str, Any]]:
    """这个函数从显式 train/val CSV 建立两个 Dataset，绝不在训练器内重新划分。"""

    defect_config = build_default_defect_config()
    defects_dir = None
    cache_manifest = None
    if condition.cache_name is not None:
        if args.cache_root is None:
            raise ValueError(f"Condition {condition.condition_id} requires --cache-root")
        defects_dir = args.cache_root.expanduser().resolve() / condition.cache_name
        if not defects_dir.is_dir():
            raise FileNotFoundError(f"Missing cache directory for {condition.condition_id}: {defects_dir}")
        cache_manifest = validate_mode_cache_manifest(defects_dir, condition)

    common = {
        "defect_to_id": defect_config.defect_to_id,
        "image_root": args.image_root,
        "defects_dir": defects_dir,
        "cache_mode": condition.cache_mode,
    }
    train_dataset = DefectFusionDataset(labels_csv=args.train_csv, **common)
    val_dataset = DefectFusionDataset(labels_csv=args.val_csv, **common)
    return train_dataset, val_dataset, {
        "defect_config": defect_config,
        "defects_dir": defects_dir,
        "cache_manifest": cache_manifest,
        "split": _validate_fixed_datasets(train_dataset, val_dataset),
    }


def _build_model(
    args: argparse.Namespace,
    condition: Condition,
    defect_config: Any,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """这个函数按锁定协议构建模型，并把 task/ImageNet 初始化来源写入 provenance。"""

    if args.stage == "cnn":
        if args.cnn_initialization == "task_checkpoint":
            if args.base_cnn_checkpoint is None:
                raise ValueError("Task-checkpoint CNN stage requires --base-cnn-checkpoint")
            if args.imagenet_weights_dir is not None:
                raise ValueError("Task-checkpoint CNN stage must not also receive ImageNet weights")
            base_path = args.base_cnn_checkpoint.expanduser().resolve()
            base_hash = sha256_file(base_path)
            if base_hash != EXPECTED_BASE_CNN_SHA256:
                raise ValueError(
                    f"Base CNN SHA-256 drift: {base_hash} != {EXPECTED_BASE_CNN_SHA256}"
                )
            cnn, _ = load_model(base_path, device=device)
            initialization = {
                "source": "packaged task-trained hybrid_best.pt",
                "checkpoint": str(base_path),
                "checkpoint_sha256": base_hash,
                "hybrid_best_loaded": True,
            }
        elif args.cnn_initialization == "imagenet":
            if args.imagenet_weights_dir is None:
                raise ValueError("ImageNet CNN stage requires --imagenet-weights-dir")
            if args.base_cnn_checkpoint is not None:
                raise ValueError("ImageNet CNN stage must not receive --base-cnn-checkpoint")
            weights_dir = args.imagenet_weights_dir.expanduser().resolve()
            cnn, _ = build_imagenet_model(weights_dir=weights_dir, device=device)
            initialization = {
                "source": "verified torchvision ImageNet-1K V1 backbone weights",
                "weights_dir": str(weights_dir),
                "files": cnn.initialization,
                "hybrid_best_loaded": False,
                "task_heads": "new randomly initialized classification and regression heads",
            }
        else:
            raise ValueError(f"Unsupported CNN initialization: {args.cnn_initialization}")
        cnn_trainability = configure_trainable_cnn_layers(
            cnn,
            trainable_convnext_blocks=1,
            trainable_densenet_blocks=2,
            train_heads=True,
        )
        model: nn.Module = CnnOnlyHybridPCEModel(cnn=cnn).to(device)
    else:
        if args.cnn_checkpoint is None:
            raise ValueError("Fusion stage requires --cnn-checkpoint")
        cnn, cnn_checkpoint = _load_cnn_checkpoint(args.cnn_checkpoint, device)
        if int(cnn_checkpoint.get("seed", -1)) != int(args.seed):
            raise ValueError(
                f"CNN checkpoint seed {cnn_checkpoint.get('seed')} != fusion seed {args.seed}"
            )
        cnn_trainability = configure_trainable_cnn_layers(
            cnn,
            trainable_convnext_blocks=1,
            trainable_densenet_blocks=2,
            train_heads=True,
        )
        defect_encoder = DefectEffectEncoder(
            defect_config=defect_config,
            embedding_dim=16,
            hidden_dim=64,
            output_dim=64,
            dropout=args.dropout,
        )
        if condition.architecture == "transformer":
            model = TransformerDefectFusionPCEModel(
                cnn=cnn,
                defect_encoder=defect_encoder,
                transformer_dim=256,
                transformer_heads=4,
                transformer_layers=1,
                fusion_hidden_dim=256,
                dropout=args.dropout,
                residual_pce=condition.residual_pce,
                use_null_semantic_token=condition.use_null_semantic_token,
            ).to(device)
        elif condition.architecture == "mlp":
            model = DefectFusionPCEModel(
                cnn=cnn,
                defect_encoder=defect_encoder,
                fusion_hidden_dim=256,
                dropout=args.dropout,
                residual_pce=condition.residual_pce,
            ).to(device)
        else:
            raise ValueError(f"Unsupported fusion architecture: {condition.architecture}")
        initialization = {
            "source": "same-seed formal CNN best",
            "checkpoint": str(args.cnn_checkpoint.expanduser().resolve()),
            "checkpoint_sha256": sha256_file(args.cnn_checkpoint.expanduser().resolve()),
            "cnn_best_epoch": cnn_checkpoint.get("best_epoch"),
            "cnn_best_val_mae": cnn_checkpoint.get("best_val_mae"),
        }
    return model, {"cnn_trainability": cnn_trainability, "initialization": initialization}


def _parameter_summary(model: nn.Module) -> dict[str, int]:
    """这个函数报告总参数、可训练参数和可训练 CNN 参数，供 smoke 审计冻结状态。"""

    cnn_ids = {id(parameter) for parameter in model.cnn.parameters()}
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_cnn": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in cnn_ids
        ),
    }


def _compact_checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    condition: Condition,
    epoch: int,
    val_mae: float,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """这个函数保存不含optimizer/scaler的best，并保留fusion微调后的完整CNN。"""

    if args.stage == "cnn":
        state_key = "cnn_state_dict"
        state_dict = model.cnn.state_dict()
    else:
        state_key = "model_state_dict"
        state_dict = model.state_dict()
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": args.stage,
        "run_scope": getattr(args, "run_scope", "formal"),
        "condition": condition.to_dict(),
        "seed": int(args.seed),
        "best_epoch": int(epoch),
        "best_val_mae": float(val_mae),
        state_key: state_dict,
        "run_config": run_config,
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """这个函数先写同目录临时文件再替换，避免中断留下半个 checkpoint。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    """这个函数执行一个 condition×seed run，并以严格最小 validation MAE 选择最早 best。"""

    if args.run_scope == "formal" and int(args.seed) not in TRAINING_SEEDS:
        raise ValueError(f"Formal seed must be one of {TRAINING_SEEDS}, got {args.seed}")
    if args.run_scope == "formal" and args.cnn_initialization != "task_checkpoint":
        raise ValueError("Formal runs must retain task-trained hybrid_best.pt initialization")
    condition = get_condition(args.condition)
    if args.stage == "cnn" and condition.condition_id != "cnn_only":
        raise ValueError("CNN stage only accepts condition=cnn_only")
    if args.stage == "fusion" and condition.condition_id == "cnn_only":
        raise ValueError("Fusion stage cannot train condition=cnn_only")
    locked_values = {
        "epochs": (args.epochs, 20),
        "batch_size": (args.batch_size, 4),
        "lr": (args.lr, 1e-4),
        "cnn_lr": (args.cnn_lr, 1e-5),
        "weight_decay": (args.weight_decay, 1e-4),
        "classification_loss_weight": (args.classification_loss_weight, 0.2),
        "cnn_regression_loss_weight": (args.cnn_regression_loss_weight, 0.5),
        "freeze_cnn_epochs": (args.freeze_cnn_epochs, 3),
        "dropout": (args.dropout, 0.2),
    }
    drift = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in locked_values.items()
        if actual != expected
    }
    if drift:
        raise ValueError(f"Legacy formal training argument drift: {drift}")
    if args.stage == "fusion" and not condition.residual_pce and condition.delta_penalty_weight != 0:
        raise ValueError("Direct-PCE cannot use delta penalty")

    _seed_training(int(args.seed))
    device = _resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "COMPLETED.json"
    if completed_path.exists():
        raise FileExistsError(f"Run is already complete and will not be overwritten: {completed_path}")

    train_dataset, val_dataset, dataset_info = _build_datasets(args, condition)
    train_hash = sha256_file(args.train_csv.expanduser().resolve())
    val_hash = sha256_file(args.val_csv.expanduser().resolve())
    if train_hash != EXPECTED_TRAIN_SHA256 or val_hash != EXPECTED_VAL_SHA256:
        raise ValueError(
            "Fixed fold SHA-256 drift: "
            f"train={train_hash} expected={EXPECTED_TRAIN_SHA256}; "
            f"val={val_hash} expected={EXPECTED_VAL_SHA256}"
        )
    model, model_info = _build_model(
        args,
        condition,
        defect_config=dataset_info["defect_config"],
        device=device,
    )
    parameter_summary = _parameter_summary(model)
    if args.stage == "cnn" and parameter_summary["trainable_cnn"] != 16_456_708:
        raise ValueError(
            f"CNN trainable parameter count drift: {parameter_summary['trainable_cnn']} != 16456708"
        )
    if args.stage == "fusion" and parameter_summary["trainable_cnn"] != 16_456_708:
        raise ValueError("Fusion legacy contract requires the selected CNN parameters")

    cnn_ids = {id(parameter) for parameter in model.cnn.parameters()}
    selected_cnn_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in cnn_ids
    ]
    selected_fusion_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in cnn_ids
    ]
    if args.stage == "cnn":
        optimizer_parameters: Any = selected_cnn_parameters
    else:
        if not selected_fusion_parameters:
            raise ValueError("Fusion model has no trainable fusion parameters")
        optimizer_parameters = [
            {
                "params": selected_cnn_parameters,
                "lr": args.cnn_lr,
                "group_name": "cnn",
            },
            {
                "params": selected_fusion_parameters,
                "lr": args.lr,
                "group_name": "fusion",
            },
        ]
    if not selected_cnn_parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_defect_fusion_batch,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    regression_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_config = {
        "contract_version": TRAINING_CONTRACT_VERSION,
        "run_scope": args.run_scope,
        "stage": args.stage,
        "condition": condition.to_dict(),
        "seed": int(args.seed),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "inputs": {
            "train_csv": str(args.train_csv.expanduser().resolve()),
            "train_csv_sha256": train_hash,
            "val_csv": str(args.val_csv.expanduser().resolve()),
            "val_csv_sha256": val_hash,
            "image_root": str(args.image_root.expanduser().resolve()),
            "defects_dir": str(dataset_info["defects_dir"]) if dataset_info["defects_dir"] else None,
            "cache_manifest": dataset_info["cache_manifest"],
        },
        "fixed_split": dataset_info["split"],
        "model": {**model_info, "parameters": parameter_summary},
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "cnn_lr": args.cnn_lr if args.stage == "fusion" else None,
            "weight_decay": args.weight_decay,
            "groups": [
                {
                    "group_name": str(group.get("group_name", "cnn")),
                    "lr": float(group["lr"]),
                    "parameter_count": sum(parameter.numel() for parameter in group["params"]),
                }
                for group in optimizer.param_groups
            ],
        },
        "loss": {
            "pce": "MSE",
            "classification_weight": args.classification_loss_weight,
            "cnn_regression_loss_weight": (
                args.cnn_regression_loss_weight if args.stage == "fusion" else 0.0
            ),
            "delta_soft_limit": condition.delta_soft_limit,
            "delta_penalty_weight": condition.delta_penalty_weight,
            "rejected_class_target": -100,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "amp": use_amp,
            "freeze_cnn_epochs": args.freeze_cnn_epochs if args.stage == "fusion" else 0,
            "best_rule": BEST_RULE,
        },
        "args": _jsonable_args(args),
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    best_val_mae = math.inf
    best_epoch = 0
    best_path = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        cnn_frozen = args.stage == "fusion" and epoch <= args.freeze_cnn_epochs
        if args.stage == "fusion":
            _set_cnn_stage_trainable(selected_cnn_parameters, not cnn_frozen)
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            regression_criterion=regression_criterion,
            classification_criterion=classification_criterion,
            classification_loss_weight=args.classification_loss_weight,
            cnn_regression_loss_weight=(
                args.cnn_regression_loss_weight if args.stage == "fusion" else 0.0
            ),
            delta_penalty_weight=condition.delta_penalty_weight,
            delta_soft_limit=condition.delta_soft_limit,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
            log_every_steps=args.log_every_steps,
            epoch=epoch,
            freeze_cnn=cnn_frozen,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            regression_criterion=regression_criterion,
            classification_criterion=classification_criterion,
            classification_loss_weight=args.classification_loss_weight,
            cnn_regression_loss_weight=(
                args.cnn_regression_loss_weight if args.stage == "fusion" else 0.0
            ),
            delta_penalty_weight=condition.delta_penalty_weight,
            delta_soft_limit=condition.delta_soft_limit,
            optimizer=None,
            use_amp=use_amp,
        )
        if not math.isfinite(float(train_metrics["loss"])) or not math.isfinite(float(val_metrics["mae"])):
            raise FloatingPointError(f"Non-finite metric at epoch {epoch}")
        row = {
            "epoch": epoch,
            "cnn_frozen": cnn_frozen,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        current_val_mae = float(val_metrics["mae"])
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            best_epoch = epoch
            payload = _compact_checkpoint(
                model=model,
                args=args,
                condition=condition,
                epoch=epoch,
                val_mae=current_val_mae,
                run_config=run_config,
            )
            _atomic_torch_save(payload, best_path)
        history_payload = {
            "status": "RUNNING",
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "history": history,
        }
        (output_dir / "training_history.json").write_text(
            json.dumps(history_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "cnn_frozen": cnn_frozen,
                    "train_loss": train_metrics["loss"],
                    "train_mae": train_metrics["mae"],
                    "val_mae": current_val_mae,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if not best_path.is_file():
        raise RuntimeError("Training finished without a best checkpoint")
    result = {
        "status": "PASS",
        "run_scope": args.run_scope,
        "stage": args.stage,
        "condition": condition.condition_id,
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "epochs_completed": len(history),
    }
    (output_dir / "training_history.json").write_text(
        json.dumps({**result, "history": history}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义固定协议训练入口，禁止训练器内部重新 split。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cnn", "fusion"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-cnn-checkpoint", type=Path)
    parser.add_argument("--imagenet-weights-dir", type=Path)
    parser.add_argument("--cnn-checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-scope", choices=("formal", "exploratory"), default="formal")
    parser.add_argument(
        "--cnn-initialization",
        choices=("task_checkpoint", "imagenet"),
        default="task_checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cnn-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-loss-weight", type=float, default=0.2)
    parser.add_argument("--cnn-regression-loss-weight", type=float, default=0.5)
    parser.add_argument("--freeze-cnn-epochs", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=25)
    return parser.parse_args(argv)


def main() -> None:
    """这个函数运行单个正式训练并打印最终机器可读结果。"""

    print(json.dumps(train(parse_args()), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
