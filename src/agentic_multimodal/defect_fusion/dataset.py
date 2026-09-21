"""Datasets for trainable defect-effect PCE fusion."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset

from agentic_multimodal.defect_detection import normalize_defect_name
from agentic_multimodal.evidence import PREPROCESS


CLASS_LABEL_TO_ID = {"high": 0, "low": 1, "middle": 2}


@dataclass(frozen=True)
class DefectFusionSample:
    image_path: Path
    efficiency: float
    filename: str
    group_id: str
    class_label: str
    defect_ids: list[int]
    llm_severity_ids: list[int]
    split: str
    is_canonical: bool


def _load_rgb_image(image_path: Path) -> Image.Image:
    """这个函数读取 EL 图像，并允许 PIL 恢复仅缺少尾部字节的可解码 JPEG。"""

    previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(image_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting


def _resolve_image_path(row: dict[str, str], labels_csv: Path, image_root: Path | None) -> Path | None:
    """这个函数优先解析可移植相对路径，并兼容历史 Windows 绝对路径和增强图目录。"""

    filename = str(row.get("filename") or row.get("image") or row.get("path") or "").strip()
    relative_path = str(row.get("relative_path") or "").strip()
    raw_combined_path = str(row.get("combined_path") or row.get("class_path") or "").strip()

    candidates: list[Path] = []
    if image_root is not None and relative_path:
        candidates.append(image_root / Path(relative_path.replace("\\", "/")))
    if raw_combined_path:
        candidates.append(Path(raw_combined_path))
    if image_root is not None and filename:
        candidates.append(image_root / filename)
        if filename.lower().startswith("aug_"):
            candidates.append(image_root / "augmented_classes" / filename)
    if filename:
        candidates.append(labels_csv.parent / filename)
        candidates.append(Path(filename))

    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def _read_efficiency(row: dict[str, str]) -> float | None:
    for key in ("efficiency", "Label", "label", "true_efficiency"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _load_defect_findings(defect_json_path: Path, required: bool = False) -> list[dict[str, Any]]:
    """这个函数读取结构化 Qwen cache；正式语义条件缺文件或坏 JSON 时立即失败。"""

    if not defect_json_path.exists():
        if required:
            raise FileNotFoundError(f"Required Qwen cache is missing: {defect_json_path}")
        return []
    data = json.loads(defect_json_path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        if "findings" in data:
            raw_items = data["findings"]
        elif "defects" in data:
            raw_items = data["defects"]
        elif required:
            raise ValueError(f"Required Qwen cache has no findings/defects list: {defect_json_path}")
        else:
            raw_items = []
    else:
        if required:
            raise ValueError(f"Required Qwen cache must be a JSON object/list: {defect_json_path}")
        return []
    if not isinstance(raw_items, list):
        if required:
            raise ValueError(f"Required Qwen findings must be a list: {defect_json_path}")
        return []
    return raw_items


def _resolve_defect_cache_path(defects_dir: Path, group_id: str, image_stem: str) -> Path:
    """这个函数优先使用 original_id 组级 cache，并兼容旧的逐图 stem cache。"""

    group_path = defects_dir / f"{group_id}.json"
    if group_path.is_file():
        return group_path
    return defects_dir / f"{image_stem}.json"


def _encode_findings(
    findings: list[dict[str, Any]],
    defect_to_id: dict[str, int],
) -> tuple[list[int], list[int]]:
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
    return defect_ids, severity_ids


class DefectFusionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        labels_csv: str | Path,
        defect_to_id: dict[str, int],
        image_root: str | Path | None = None,
        defects_dir: str | Path | None = None,
        transform: Callable[[Image.Image], torch.Tensor] = PREPROCESS,
        max_samples: int | None = None,
        cache_mode: str = "optional",
        expected_split: str | None = None,
        require_canonical: bool = False,
    ) -> None:
        """这个数据集从固定 CSV 读取图像，并按 original_id 复用同一份语义 cache。"""

        if cache_mode not in {"none", "optional", "required"}:
            raise ValueError("cache_mode must be one of: none, optional, required")
        if cache_mode == "required" and defects_dir is None:
            raise ValueError("cache_mode=required requires defects_dir")
        self.labels_csv = Path(labels_csv).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve() if image_root else None
        self.defects_dir = Path(defects_dir).expanduser().resolve() if defects_dir else None
        self.transform = transform
        self.cache_mode = cache_mode
        self.expected_split = expected_split
        self.require_canonical = require_canonical
        self.samples: list[DefectFusionSample] = []

        with self.labels_csv.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                efficiency = _read_efficiency(row)
                image_path = _resolve_image_path(row, self.labels_csv, self.image_root)
                if efficiency is None or image_path is None:
                    continue
                filename = image_path.name
                group_id = str(row.get("original_id") or image_path.stem).strip() or image_path.stem
                class_label = str(row.get("class") or row.get("class_label") or "").strip()
                split = str(row.get("split") or "").strip()
                is_canonical = str(row.get("is_canonical") or "0").strip() in {"1", "true", "True"}
                if expected_split is not None and split != expected_split:
                    raise ValueError(
                        f"CSV {self.labels_csv} contains split={split!r}, expected {expected_split!r}"
                    )
                if require_canonical and not is_canonical:
                    raise ValueError(
                        f"CSV {self.labels_csv} contains non-canonical row: {filename} ({group_id})"
                    )
                findings: list[dict[str, Any]] = []
                if self.cache_mode != "none" and self.defects_dir is not None:
                    defect_path = _resolve_defect_cache_path(
                        self.defects_dir, group_id=group_id, image_stem=image_path.stem
                    )
                    findings = _load_defect_findings(
                        defect_path, required=self.cache_mode == "required"
                    )
                defect_ids, severity_ids = _encode_findings(findings, defect_to_id)
                self.samples.append(
                    DefectFusionSample(
                        image_path=image_path,
                        efficiency=efficiency,
                        filename=filename,
                        group_id=group_id,
                        class_label=class_label,
                        defect_ids=defect_ids,
                        llm_severity_ids=severity_ids,
                        split=split,
                        is_canonical=is_canonical,
                    )
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        if not self.samples:
            raise ValueError(f"No usable training samples found in {self.labels_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self.transform(_load_rgb_image(sample.image_path))
        return {
            "image": image,
            "target_pce": torch.tensor(sample.efficiency, dtype=torch.float32),
            "defect_ids": torch.tensor(sample.defect_ids, dtype=torch.long),
            "llm_severity_ids": torch.tensor(sample.llm_severity_ids, dtype=torch.long),
            "filename": sample.filename,
            "image_path": str(sample.image_path),
            "group_id": sample.group_id,
            "class_label": sample.class_label,
            "split": sample.split,
            "is_canonical": sample.is_canonical,
        }


def collate_defect_fusion_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """这个函数把样本列表整理成 batch，并把类别标签编码成训练可用的 class target。"""

    images = torch.stack([item["image"] for item in items], dim=0)
    targets = torch.stack([item["target_pce"] for item in items], dim=0)
    class_targets = torch.tensor(
        [CLASS_LABEL_TO_ID.get(str(item["class_label"]).strip(), -100) for item in items],
        dtype=torch.long,
    )
    max_defects = max(1, max(int(item["defect_ids"].numel()) for item in items))

    defect_ids = torch.zeros((len(items), max_defects), dtype=torch.long)
    llm_severity_ids = torch.zeros((len(items), max_defects), dtype=torch.long)
    for row_idx, item in enumerate(items):
        n_defects = int(item["defect_ids"].numel())
        if n_defects == 0:
            continue
        defect_ids[row_idx, :n_defects] = item["defect_ids"]
        llm_severity_ids[row_idx, :n_defects] = item["llm_severity_ids"]

    return {
        "images": images,
        "target_pce": targets,
        "class_targets": class_targets,
        "defect_ids": defect_ids,
        "llm_severity_ids": llm_severity_ids,
        "filenames": [item["filename"] for item in items],
        "image_paths": [item["image_path"] for item in items],
        "group_ids": [item["group_id"] for item in items],
        "class_labels": [item["class_label"] for item in items],
        "splits": [item["split"] for item in items],
        "is_canonical": [bool(item["is_canonical"]) for item in items],
    }
