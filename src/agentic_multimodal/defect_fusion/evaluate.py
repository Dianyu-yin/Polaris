"""Evaluate defect-fusion PCE checkpoints on the held-out split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Subset

from agentic_multimodal.defect_detection import normalize_defect_name
from agentic_multimodal.defect_fusion.dataset import (
    DefectFusionDataset,
    collate_defect_fusion_batch,
)
from agentic_multimodal.defect_fusion.model import (
    CnnOnlyHybridPCEModel,
    DefectEffectEncoder,
    DefectFusionPCEModel,
    SURVEY_SCHEMA_VERSION,
    TransformerDefectFusionPCEModel,
    build_default_defect_config,
    configure_trainable_cnn_layers,
)
from agentic_multimodal.defect_fusion.train import _split_indices, _split_indices_by_group
from agentic_multimodal.evidence import DEFAULT_WEIGHTS, load_model


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _metric_summary(predictions: list[float], targets: list[float]) -> dict[str, float]:
    if not predictions:
        return {"mae": math.nan, "rmse": math.nan, "bias": math.nan, "r2": math.nan}
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    abs_errors = [abs(error) for error in errors]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    bias = sum(errors) / len(errors)
    target_mean = sum(targets) / len(targets)
    ss_res = sum(error * error for error in errors)
    ss_tot = sum((target - target_mean) ** 2 for target in targets)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else math.nan
    sorted_abs = sorted(abs_errors)
    median_abs = sorted_abs[len(sorted_abs) // 2]
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "median_abs_error": median_abs,
        "max_abs_error": max(abs_errors),
    }


def _read_defects(defects_dir: Path | None, image_path: str) -> list[dict[str, Any]]:
    if defects_dir is None:
        return []
    defect_path = defects_dir / f"{Path(image_path).stem}.json"
    if not defect_path.exists():
        return []
    data = json.loads(defect_path.read_text(encoding="utf-8-sig"))
    raw_items = data.get("findings") or data.get("defects") or [] if isinstance(data, dict) else []
    return raw_items if isinstance(raw_items, list) else []


def _defect_text(defects: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in defects:
        if isinstance(item, dict):
            name = normalize_defect_name(str(item.get("defect") or item.get("name") or item.get("label") or "").strip())
            severity = item.get("severity")
        elif isinstance(item, list) and len(item) >= 2:
            name = normalize_defect_name(str(item[0]).strip())
            severity = item[1]
        else:
            continue
        if name:
            parts.append(f"{name}:{severity}")
    return "; ".join(parts) if parts else "none"


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for char in paragraph:
            trial = current + char
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def _make_contact_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    top_k: int,
    columns: int = 3,
) -> None:
    selected = rows[:top_k]
    if not selected:
        return

    thumb_w, thumb_h = 220, 220
    cell_w, cell_h = 360, 360
    title_h = 54
    rows_count = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (columns * cell_w, title_h + rows_count * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    text_font = _font(17)
    small_font = _font(15)

    draw.rectangle((0, 0, canvas.width, title_h), fill=(31, 39, 49))
    draw.text((18, 14), title, fill="white", font=title_font)

    for idx, row in enumerate(selected):
        col = idx % columns
        r = idx // columns
        x0 = col * cell_w
        y0 = title_h + r * cell_h
        draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline=(215, 215, 215))

        try:
            image = Image.open(row["image_path"])
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.contain(image, (thumb_w, thumb_h))
        except Exception:
            image = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
        px = x0 + (cell_w - image.width) // 2
        py = y0 + 12
        canvas.paste(image, (px, py))

        text_y = y0 + 245
        detail_lines = [
            str(row["filename"]),
            f"target={float(row['target_pce']):.3f} fusion={float(row['fusion_pce']):.3f}",
            f"abs_err={float(row['fusion_abs_error']):.3f} cnn={float(row['cnn_pce']):.3f}",
            f"defects: {row.get('defects_text', 'none')}",
        ]
        for line in detail_lines:
            for wrapped in _wrap_text(draw, line, small_font if line.startswith("defects:") else text_font, cell_w - 24):
                if text_y > y0 + cell_h - 22:
                    break
                draw.text((x0 + 14, text_y), wrapped, fill=(25, 25, 25), font=small_font if line.startswith("defects:") else text_font)
                text_y += 21
            if text_y > y0 + cell_h - 22:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _dedupe_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        group_id = str(row.get("group_id") or row.get("filename") or "")
        if group_id in seen:
            continue
        seen.add(group_id)
        selected.append(row)
    return selected


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_defect_config = checkpoint.get("defect_config", {}) if isinstance(checkpoint, dict) else {}
    declared_architecture = str(checkpoint_args.get("fusion_architecture", args.fusion_architecture))
    if declared_architecture != "cnn_only" and checkpoint_defect_config.get("schema_version") != SURVEY_SCHEMA_VERSION:
        raise ValueError("Evaluation checkpoint is not a survey_defect_v2 fusion checkpoint")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

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
        split_summary = {
            "split_mode": "random",
            "group_key": "row_index",
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "leaked_groups": len(leaked_groups),
            "leaked_rows": sum(1 for sample in dataset.samples if sample.group_id in leaked_groups),
        }

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_defect_fusion_batch,
    )

    cnn, _ = load_model(args.weights, device=device)
    configure_trainable_cnn_layers(
        cnn,
        trainable_convnext_blocks=int(checkpoint_args.get("unfreeze_convnext_blocks", 1)),
        trainable_densenet_blocks=int(checkpoint_args.get("unfreeze_densenet_blocks", 2)),
        train_heads=not bool(checkpoint_args.get("freeze_cnn_heads", False)),
    )
    defect_encoder = DefectEffectEncoder(
        defect_config=defect_config,
        embedding_dim=int(checkpoint_args.get("defect_embedding_dim", 16)),
        hidden_dim=int(checkpoint_args.get("defect_hidden_dim", 64)),
        output_dim=int(checkpoint_args.get("defect_output_dim", 64)),
        dropout=float(checkpoint_args.get("dropout", 0.2)),
    )
    fusion_architecture = declared_architecture
    if fusion_architecture == "cnn_only":
        model = CnnOnlyHybridPCEModel(cnn=cnn)
    elif fusion_architecture == "transformer":
        model = TransformerDefectFusionPCEModel(
            cnn=cnn,
            defect_encoder=defect_encoder,
            transformer_dim=int(checkpoint_args.get("transformer_dim", 256)),
            transformer_heads=int(checkpoint_args.get("transformer_heads", 4)),
            transformer_layers=int(checkpoint_args.get("transformer_layers", 1)),
            fusion_hidden_dim=int(checkpoint_args.get("fusion_hidden_dim", 256)),
            dropout=float(checkpoint_args.get("dropout", 0.2)),
            residual_pce=bool(
                checkpoint_defect_config.get(
                    "residual_pce",
                    not bool(checkpoint_args.get("no_residual_pce", False)),
                )
            ),
        )
    else:
        model = DefectFusionPCEModel(
            cnn=cnn,
            defect_encoder=defect_encoder,
            fusion_hidden_dim=int(checkpoint_args.get("fusion_hidden_dim", 256)),
            dropout=float(checkpoint_args.get("dropout", 0.2)),
            residual_pce=not bool(checkpoint_args.get("no_residual_pce", False)),
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    fusion_predictions: list[float] = []
    cnn_predictions: list[float] = []
    targets: list[float] = []
    class_predictions: list[int] = []
    class_targets: list[int] = []
    defects_dir = args.defects_dir.expanduser().resolve() if args.defects_dir is not None else None

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            defect_ids = batch["defect_ids"].to(device)
            severity_ids = batch["llm_severity_ids"].to(device)
            target_pce = batch["target_pce"].to(device)
            output = model(images=images, defect_ids=defect_ids, llm_severity_ids=severity_ids)

            final_pce = output["final_pce"].detach().cpu().tolist()
            cnn_pce = output["cnn_pce"].detach().cpu().tolist()
            pce_delta = output["pce_delta"].detach().cpu().tolist()
            class_indices = output["class_probabilities"].argmax(dim=1).detach().cpu().tolist()
            target_class_indices = batch["class_targets"].detach().cpu().tolist()
            target_values = target_pce.detach().cpu().tolist()

            for idx, filename in enumerate(batch["filenames"]):
                defects = _read_defects(defects_dir, batch["image_paths"][idx])
                record = {
                    "filename": filename,
                    "image_path": batch["image_paths"][idx],
                    "group_id": batch["group_ids"][idx],
                    "class_label": batch["class_labels"][idx],
                    "target_pce": float(target_values[idx]),
                    "fusion_pce": float(final_pce[idx]),
                    "cnn_pce": float(cnn_pce[idx]),
                    "pce_delta": float(pce_delta[idx]),
                    "fusion_error": float(final_pce[idx] - target_values[idx]),
                    "cnn_error": float(cnn_pce[idx] - target_values[idx]),
                    "fusion_abs_error": abs(float(final_pce[idx] - target_values[idx])),
                    "cnn_abs_error": abs(float(cnn_pce[idx] - target_values[idx])),
                    "predicted_class_index": int(class_indices[idx]),
                    "true_class_index": int(target_class_indices[idx]),
                    "defects_text": _defect_text(defects),
                }
                records.append(record)
                fusion_predictions.append(record["fusion_pce"])
                cnn_predictions.append(record["cnn_pce"])
                targets.append(record["target_pce"])
                if int(target_class_indices[idx]) >= 0:
                    class_predictions.append(int(class_indices[idx]))
                    class_targets.append(int(target_class_indices[idx]))

    records_sorted = sorted(records, key=lambda row: row["fusion_abs_error"])
    fieldnames = list(records_sorted[0].keys()) if records_sorted else []
    for csv_name, rows in (
        ("predictions.csv", records),
        ("validation_examples_sorted.csv", records_sorted),
    ):
        with (output_dir / csv_name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _make_contact_sheet(
        records_sorted,
        output_dir / "validation_good_examples.png",
        "Validation good examples: lowest fusion absolute error",
        args.top_k,
    )
    _make_contact_sheet(
        list(reversed(records_sorted)),
        output_dir / "validation_bad_examples.png",
        "Validation bad examples: highest fusion absolute error",
        args.top_k,
    )
    good_unique = _dedupe_by_group(records_sorted)
    bad_unique = _dedupe_by_group(list(reversed(records_sorted)))
    _make_contact_sheet(
        good_unique,
        output_dir / "validation_good_examples_unique_groups.png",
        "Validation good examples: unique original_id groups",
        args.top_k,
    )
    _make_contact_sheet(
        bad_unique,
        output_dir / "validation_bad_examples_unique_groups.png",
        "Validation bad examples: unique original_id groups",
        args.top_k,
    )

    train_groups = {dataset.samples[index].group_id for index in train_indices}
    val_groups = {dataset.samples[index].group_id for index in val_indices}
    leaked_groups = train_groups & val_groups
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "output_dir": str(output_dir),
        "note": "Validation uses cached defect JSON files and does not call a VLM/API live.",
        "split": {
            **split_summary,
            "verified_leaked_groups": len(leaked_groups),
            "verified_leaked_rows": sum(1 for sample in dataset.samples if sample.group_id in leaked_groups),
        },
        "checkpoint_metadata": {
            "epoch": int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0,
            "val_mae": float(checkpoint.get("val_mae", math.nan)) if isinstance(checkpoint, dict) else math.nan,
            "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))) if isinstance(checkpoint, dict) else 0,
            "best_val_mae": float(checkpoint.get("best_val_mae", checkpoint.get("val_mae", math.nan))) if isinstance(checkpoint, dict) else math.nan,
            "fusion_architecture": fusion_architecture,
        },
        "n_records": len(records),
        "fusion": _metric_summary(fusion_predictions, targets),
        "cnn": _metric_summary(cnn_predictions, targets),
        "classification": {
            "accuracy": (
                sum(prediction == target for prediction, target in zip(class_predictions, class_targets))
                / len(class_targets)
                if class_targets
                else math.nan
            ),
            "support": len(class_targets),
        },
        "args": _jsonable_args(args),
    }
    (output_dir / "validation_eval.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "validation_examples_summary.json").write_text(
        json.dumps(
            {
                "good": records_sorted[: args.top_k],
                "bad": list(reversed(records_sorted))[: args.top_k],
                "good_unique_groups": good_unique[: args.top_k],
                "bad_unique_groups": bad_unique[: args.top_k],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a defect-fusion PCE checkpoint.")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--defects-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["group", "random"], default="group")
    parser.add_argument("--fusion-architecture", choices=["cnn_only", "mlp", "transformer"], default="mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def main() -> None:
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
