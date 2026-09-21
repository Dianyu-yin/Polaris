"""Train PCE regression with frozen-backbone CNN and learnable defect effects."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from agentic_multimodal.defect_fusion.dataset import DefectFusionDataset, collate_defect_fusion_batch
from agentic_multimodal.defect_fusion.model import (
    CnnOnlyHybridPCEModel,
    DefectEffectEncoder,
    DefectFusionPCEModel,
    TransformerDefectFusionPCEModel,
    build_default_defect_config,
    configure_trainable_cnn_layers,
)
from agentic_multimodal.evidence import DEFAULT_WEIGHTS
from deployable_convnext_densenet_hybrid.model import load_model


def _resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _seed_training(seed: int) -> None:
    """这个函数固定 Python、CPU 和 CUDA 随机种子，使重复训练的初始化与 batch 顺序可追溯。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    if n_items <= 1 or val_fraction <= 0:
        return indices, []
    val_count = max(1, int(round(n_items * val_fraction)))
    val_count = min(val_count, n_items - 1)
    return indices[val_count:], indices[:val_count]


def _split_indices_by_group(
    dataset: DefectFusionDataset,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if len(dataset) <= 1 or val_fraction <= 0:
        return list(range(len(dataset))), [], {
            "split_mode": "group",
            "leaked_groups": 0,
            "leaked_rows": 0,
        }

    groups: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(dataset.samples):
        group_id = sample.group_id or sample.image_path.stem
        item = groups.setdefault(
            group_id,
            {
                "indices": [],
                "class_label": sample.class_label or "unknown",
            },
        )
        item["indices"].append(index)

    groups_by_class: dict[str, list[str]] = defaultdict(list)
    class_row_counts: Counter[str] = Counter()
    for group_id, group in groups.items():
        class_label = str(group["class_label"])
        groups_by_class[class_label].append(group_id)
        class_row_counts[class_label] += len(group["indices"])

    rng = random.Random(seed)
    val_groups: set[str] = set()
    for class_label, group_ids in sorted(groups_by_class.items()):
        shuffled = list(group_ids)
        rng.shuffle(shuffled)
        target_rows = max(1, int(round(class_row_counts[class_label] * val_fraction)))
        selected_rows = 0
        for group_id in shuffled:
            if len(shuffled) > 1 and len(val_groups) + 1 == len(groups):
                break
            val_groups.add(group_id)
            selected_rows += len(groups[group_id]["indices"])
            if selected_rows >= target_rows:
                break

    train_indices: list[int] = []
    val_indices: list[int] = []
    train_groups: set[str] = set()
    actual_val_groups: set[str] = set()
    for group_id, group in groups.items():
        if group_id in val_groups:
            val_indices.extend(group["indices"])
            actual_val_groups.add(group_id)
        else:
            train_indices.extend(group["indices"])
            train_groups.add(group_id)

    if not train_indices and val_indices:
        moved_group = next(iter(actual_val_groups))
        train_indices.extend(groups[moved_group]["indices"])
        val_indices = [index for index in val_indices if index not in groups[moved_group]["indices"]]
        actual_val_groups.remove(moved_group)
        train_groups.add(moved_group)

    leaked_groups = train_groups & actual_val_groups
    split_summary = {
        "split_mode": "group",
        "group_key": "original_id",
        "n_groups": len(groups),
        "n_train_groups": len(train_groups),
        "n_val_groups": len(actual_val_groups),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "leaked_groups": len(leaked_groups),
        "leaked_rows": 0,
        "train_class_counts": dict(Counter(dataset.samples[index].class_label or "unknown" for index in train_indices)),
        "val_class_counts": dict(Counter(dataset.samples[index].class_label or "unknown" for index in val_indices)),
    }
    return train_indices, val_indices, split_summary


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "images": batch["images"].to(device),
        "target_pce": batch["target_pce"].to(device),
        "class_targets": batch["class_targets"].to(device),
        "defect_ids": batch["defect_ids"].to(device),
        "llm_severity_ids": batch["llm_severity_ids"].to(device),
    }


def _metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    if not predictions:
        return {"mae": math.nan, "rmse": math.nan}
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {"mae": mae, "rmse": rmse}


def _soft_delta_penalty(pce_delta: torch.Tensor, soft_limit: float) -> torch.Tensor:
    """这个函数只惩罚超过 soft limit 的 PCE 修正，阈值内修正保持完全自由。"""

    excess = torch.relu(pce_delta.abs() - float(soft_limit))
    return excess.square().mean()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    regression_criterion: nn.Module,
    classification_criterion: nn.Module,
    classification_loss_weight: float = 0.0,
    cnn_regression_loss_weight: float = 0.0,
    delta_penalty_weight: float = 0.0,
    delta_soft_limit: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    log_every_steps: int = 0,
    epoch: int = 0,
    freeze_cnn: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training:
        model.set_frozen_backbone_eval()
        if freeze_cnn:
            model.cnn.eval()

    total_loss = 0.0
    total_regression_loss = 0.0
    total_cnn_regression_loss = 0.0
    total_delta_penalty = 0.0
    total_classification_loss = 0.0
    predictions: list[float] = []
    targets: list[float] = []
    class_correct = 0
    class_total = 0

    for step, raw_batch in enumerate(loader, start=1):
        batch = _batch_to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(
                    images=batch["images"],
                    defect_ids=batch["defect_ids"],
                    llm_severity_ids=batch["llm_severity_ids"],
                )
                regression_loss = regression_criterion(output["final_pce"], batch["target_pce"])
                cnn_regression_loss = regression_criterion(output["cnn_pce"], batch["target_pce"])
                delta_penalty = _soft_delta_penalty(output["pce_delta"], delta_soft_limit)
                classification_loss = output["final_pce"].new_tensor(0.0)
                valid_class_targets = batch["class_targets"].ne(-100)
                if classification_loss_weight > 0 and valid_class_targets.any():
                    classification_loss = classification_criterion(output["logits"], batch["class_targets"])
                loss = (
                    regression_loss
                    + cnn_regression_loss_weight * cnn_regression_loss
                    + delta_penalty_weight * delta_penalty
                    + classification_loss_weight * classification_loss
                )
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * int(batch["target_pce"].numel())
        total_regression_loss += float(regression_loss.detach().cpu().item()) * int(batch["target_pce"].numel())
        total_cnn_regression_loss += float(cnn_regression_loss.detach().cpu().item()) * int(batch["target_pce"].numel())
        total_delta_penalty += float(delta_penalty.detach().cpu().item()) * int(batch["target_pce"].numel())
        total_classification_loss += float(classification_loss.detach().cpu().item()) * int(batch["target_pce"].numel())
        predictions.extend(output["final_pce"].detach().cpu().tolist())
        targets.extend(batch["target_pce"].detach().cpu().tolist())
        valid_class_targets = batch["class_targets"].ne(-100)
        if valid_class_targets.any():
            predicted_classes = output["logits"].argmax(dim=1)
            class_correct += int((predicted_classes[valid_class_targets] == batch["class_targets"][valid_class_targets]).sum().item())
            class_total += int(valid_class_targets.sum().item())
        if training and log_every_steps and step % log_every_steps == 0:
            partial_metrics = _metrics(predictions, targets)
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        "total_steps": len(loader),
                        "train_mae_so_far": partial_metrics["mae"],
                        "train_loss_so_far": total_loss / max(1, len(targets)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    metrics = _metrics(predictions, targets)
    metrics["loss"] = total_loss / max(1, len(targets))
    metrics["regression_loss"] = total_regression_loss / max(1, len(targets))
    metrics["cnn_regression_loss"] = total_cnn_regression_loss / max(1, len(targets))
    metrics["delta_penalty"] = total_delta_penalty / max(1, len(targets))
    metrics["classification_loss"] = total_classification_loss / max(1, len(targets))
    metrics["classification_accuracy"] = class_correct / class_total if class_total else math.nan
    metrics["n_samples"] = len(targets)
    return metrics


def _trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _load_resume_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | None,
    device: torch.device,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if checkpoint_path is None:
        return None, None
    resolved_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(resolved_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    metadata = {
        "path": str(resolved_path),
        "epoch": int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0,
        "val_mae": float(checkpoint.get("val_mae", math.inf)) if isinstance(checkpoint, dict) else math.inf,
        "has_optimizer_state": bool(isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict")),
        "has_scaler_state": bool(isinstance(checkpoint, dict) and checkpoint.get("scaler_state_dict")),
    }
    return metadata, checkpoint if isinstance(checkpoint, dict) else None


def _load_cnn_initialization(
    cnn: nn.Module,
    checkpoint_path: Path | None,
    device: torch.device,
) -> dict[str, Any] | None:
    """这个函数从 CNN-only 最佳 checkpoint 中提取 cnn.* 权重，作为融合模型稳定的视觉基线。"""

    if checkpoint_path is None:
        return None
    resolved_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(resolved_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"CNN initialization checkpoint did not contain a state dict: {resolved_path}")
    if any(str(key).startswith("cnn.") for key in state_dict):
        cnn_state_dict = {
            str(key)[len("cnn.") :]: value
            for key, value in state_dict.items()
            if str(key).startswith("cnn.")
        }
    else:
        cnn_state_dict = state_dict
    cnn.load_state_dict(cnn_state_dict, strict=True)
    return {
        "path": str(resolved_path),
        "epoch": int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0,
        "val_mae": float(checkpoint.get("val_mae", math.nan)) if isinstance(checkpoint, dict) else math.nan,
    }


def _set_cnn_stage_trainable(cnn_parameters: list[nn.Parameter], enabled: bool) -> None:
    """这个函数在 warmup 结束时恢复预先选定的 CNN 参数，避免冻结阶段主干发生任何更新。"""

    for parameter in cnn_parameters:
        parameter.requires_grad_(enabled)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.resume_checkpoint is not None and args.cnn_init_checkpoint is not None:
        raise ValueError("--resume-checkpoint and --cnn-init-checkpoint cannot be used together")
    for name in (
        "lr",
        "weight_decay",
        "cnn_regression_loss_weight",
        "delta_penalty_weight",
        "delta_soft_limit",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.cnn_lr is not None and float(args.cnn_lr) < 0.0:
        raise ValueError("--cnn-lr must be non-negative")
    if int(args.freeze_cnn_epochs) < 0:
        raise ValueError("--freeze-cnn-epochs must be non-negative")

    _seed_training(int(args.seed))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    defect_config = build_default_defect_config()
    dataset = DefectFusionDataset(
        labels_csv=args.labels_csv,
        defect_to_id=defect_config.defect_to_id,
        image_root=args.image_root,
        defects_dir=args.defects_dir,
        max_samples=args.max_samples,
    )
    if args.split_mode == "group":
        train_indices, val_indices, split_summary = _split_indices_by_group(dataset, args.val_fraction, args.seed)
    else:
        train_indices, val_indices = _split_indices(len(dataset), args.val_fraction, args.seed)
        train_groups = {dataset.samples[index].group_id for index in train_indices}
        val_groups = {dataset.samples[index].group_id for index in val_indices}
        leaked_groups = train_groups & val_groups
        leaked_rows = sum(
            1
            for sample in dataset.samples
            if sample.group_id in leaked_groups
        )
        split_summary = {
            "split_mode": "random",
            "group_key": "row_index",
            "n_groups": len(train_groups | val_groups),
            "n_train_groups": len(train_groups),
            "n_val_groups": len(val_groups),
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "leaked_groups": len(leaked_groups),
            "leaked_rows": leaked_rows,
            "train_class_counts": dict(Counter(dataset.samples[index].class_label or "unknown" for index in train_indices)),
            "val_class_counts": dict(Counter(dataset.samples[index].class_label or "unknown" for index in val_indices)),
        }
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_defect_fusion_batch,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_defect_fusion_batch,
        )
        if val_dataset is not None
        else None
    )

    cnn, _loaded_device = load_model(args.weights, device=device)
    cnn_initialization = _load_cnn_initialization(cnn, args.cnn_init_checkpoint, device)
    cnn_trainability = configure_trainable_cnn_layers(
        cnn,
        trainable_convnext_blocks=args.unfreeze_convnext_blocks,
        trainable_densenet_blocks=args.unfreeze_densenet_blocks,
        train_heads=not args.freeze_cnn_heads,
    )
    if args.fusion_architecture == "cnn_only":
        model = CnnOnlyHybridPCEModel(cnn=cnn).to(device)
    else:
        defect_encoder = DefectEffectEncoder(
            defect_config=defect_config,
            embedding_dim=args.defect_embedding_dim,
            hidden_dim=args.defect_hidden_dim,
            output_dim=args.defect_output_dim,
            dropout=args.dropout,
        )
    if args.fusion_architecture == "transformer":
        model = TransformerDefectFusionPCEModel(
            cnn=cnn,
            defect_encoder=defect_encoder,
            transformer_dim=args.transformer_dim,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            fusion_hidden_dim=args.fusion_hidden_dim,
            dropout=args.dropout,
            residual_pce=not args.no_residual_pce,
        ).to(device)
    elif args.fusion_architecture == "mlp":
        model = DefectFusionPCEModel(
            cnn=cnn,
            defect_encoder=defect_encoder,
            fusion_hidden_dim=args.fusion_hidden_dim,
            dropout=args.dropout,
            residual_pce=not args.no_residual_pce,
        ).to(device)
    resume_info, resume_checkpoint = _load_resume_checkpoint(model, args.resume_checkpoint, device)

    cnn_parameter_ids = {id(parameter) for parameter in model.cnn.parameters()}
    selected_cnn_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in cnn_parameter_ids
    ]
    selected_fusion_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in cnn_parameter_ids
    ]
    use_separate_learning_rates = args.cnn_lr is not None or args.freeze_cnn_epochs > 0
    if use_separate_learning_rates:
        parameter_groups: list[dict[str, Any]] = []
        if selected_cnn_parameters:
            parameter_groups.append(
                {
                    "params": selected_cnn_parameters,
                    "lr": float(args.cnn_lr if args.cnn_lr is not None else args.lr),
                    "group_name": "cnn",
                }
            )
        if selected_fusion_parameters:
            parameter_groups.append(
                {
                    "params": selected_fusion_parameters,
                    "lr": float(args.lr),
                    "group_name": "fusion",
                }
            )
        optimizer_parameters: Any = parameter_groups
    else:
        optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer_state_restored = False
    scaler_state_restored = False
    if resume_checkpoint and resume_checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
        optimizer_state_restored = True
    if resume_checkpoint and use_amp and resume_checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        scaler_state_restored = True
    if resume_info is not None:
        resume_info = {
            **resume_info,
            "optimizer_state_restored": optimizer_state_restored,
            "scaler_state_restored": scaler_state_restored,
        }
    regression_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    history: list[dict[str, Any]] = []
    best_val_mae = math.inf if resume_info is None else resume_info["val_mae"]
    best_epoch = 0 if resume_info is None else resume_info["epoch"]
    best_checkpoint_path = output_dir / "defect_fusion_best.pt"
    latest_checkpoint_path = output_dir / "defect_fusion_latest.pt"
    if resume_info is not None:
        resume_start_path = output_dir / "defect_fusion_resume_start.pt"
        shutil.copy2(Path(resume_info["path"]), resume_start_path)
        shutil.copy2(Path(resume_info["path"]), best_checkpoint_path)

    run_config = {
        "labels_csv": str(Path(args.labels_csv).expanduser().resolve()),
        "image_root": str(Path(args.image_root).expanduser().resolve()) if args.image_root else None,
        "defects_dir": str(Path(args.defects_dir).expanduser().resolve()) if args.defects_dir else None,
        "weights": str(Path(args.weights).expanduser().resolve()),
        "device": str(device),
        "n_samples": len(dataset),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "split": split_summary,
        "cnn_trainability": cnn_trainability,
        "cnn_initialization": cnn_initialization,
        "optimizer_groups": [
            {
                "group_name": group.get("group_name", "all"),
                "lr": group["lr"],
                "n_parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        ],
        "resume": resume_info,
        "model_parameters": _trainable_parameter_summary(model),
        "fusion_architecture": args.fusion_architecture,
        "defect_config": None
        if args.fusion_architecture == "cnn_only"
        else {
            "schema_version": "survey_defect_v2",
            "defect_names": list(defect_config.defect_names),
        },
        "args": _jsonable_args(args),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    epoch_offset = int(resume_info["epoch"]) if resume_info is not None else 0
    for local_epoch in range(1, args.epochs + 1):
        epoch = epoch_offset + local_epoch
        cnn_frozen = local_epoch <= int(args.freeze_cnn_epochs)
        _set_cnn_stage_trainable(selected_cnn_parameters, not cnn_frozen)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            regression_criterion,
            classification_criterion,
            classification_loss_weight=args.classification_loss_weight,
            cnn_regression_loss_weight=args.cnn_regression_loss_weight,
            delta_penalty_weight=args.delta_penalty_weight,
            delta_soft_limit=args.delta_soft_limit,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
            log_every_steps=args.log_every_steps,
            epoch=epoch,
            freeze_cnn=cnn_frozen,
        )
        val_metrics = (
            _run_epoch(
                model,
                val_loader,
                device,
                regression_criterion,
                classification_criterion,
                classification_loss_weight=args.classification_loss_weight,
                cnn_regression_loss_weight=args.cnn_regression_loss_weight,
                delta_penalty_weight=args.delta_penalty_weight,
                delta_soft_limit=args.delta_soft_limit,
                use_amp=use_amp,
            )
            if val_loader is not None
            else {}
        )
        row = {
            "epoch": epoch,
            "local_epoch": local_epoch,
            "cnn_frozen": cnn_frozen,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        current_val_mae = float(val_metrics.get("mae", train_metrics["mae"]))
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "defect_config": model.export_config(),
            "cnn_trainability": cnn_trainability,
            "epoch": epoch,
            "val_mae": current_val_mae,
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "args": _jsonable_args(args),
            "resume": resume_info,
        }
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            best_epoch = epoch
            checkpoint_payload["val_mae"] = best_val_mae
            checkpoint_payload["best_val_mae"] = best_val_mae
            checkpoint_payload["best_epoch"] = best_epoch
            torch.save(checkpoint_payload, best_checkpoint_path)
        torch.save(checkpoint_payload, latest_checkpoint_path)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "cnn_frozen": cnn_frozen,
                    "train_loss": train_metrics["loss"],
                    "train_mae": train_metrics["mae"],
                    "val_mae": val_metrics.get("mae"),
                    "best_val_mae": best_val_mae,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "history": history,
        "config": run_config,
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train defect-effect fusion for PCE regression.")
    parser.add_argument("--labels-csv", type=Path, default=Path("combined/all_images.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("combined"))
    parser.add_argument("--defects-dir", type=Path, help="Directory containing per-image defect JSON files.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--cnn-init-checkpoint",
        type=Path,
        help="Initialize the CNN submodule from a trained CNN-only checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("agentic_outputs/defect_fusion_train"))
    parser.add_argument("--resume-checkpoint", type=Path, help="Load an existing defect-fusion checkpoint before training.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cnn-lr", type=float, help="Optional lower learning rate for selected CNN parameters.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["group", "random"], default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, help="Optional smoke-test cap.")
    parser.add_argument("--defect-embedding-dim", type=int, default=16)
    parser.add_argument("--defect-hidden-dim", type=int, default=64)
    parser.add_argument("--defect-output-dim", type=int, default=64)
    parser.add_argument("--fusion-architecture", choices=["cnn_only", "mlp", "transformer"], default="transformer")
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--classification-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--cnn-regression-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight that anchors cnn_pce to the true PCE while fusion is trained.",
    )
    parser.add_argument(
        "--delta-soft-limit",
        type=float,
        default=0.0,
        help="Do not penalize |pce_delta| below this threshold.",
    )
    parser.add_argument(
        "--delta-penalty-weight",
        type=float,
        default=0.0,
        help="Weight for squared pce_delta excess above --delta-soft-limit.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze-convnext-blocks", type=int, default=1)
    parser.add_argument("--unfreeze-densenet-blocks", type=int, default=2)
    parser.add_argument("--freeze-cnn-heads", action="store_true")
    parser.add_argument(
        "--freeze-cnn-epochs",
        type=int,
        default=0,
        help="Freeze all otherwise-selected CNN parameters for the first N local epochs.",
    )
    parser.add_argument("--no-residual-pce", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--log-every-steps", type=int, default=25)
    return parser.parse_args(argv)


def main() -> None:
    summary = train(parse_args())
    print(json.dumps({"best_epoch": summary["best_epoch"], "best_val_mae": summary["best_val_mae"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
