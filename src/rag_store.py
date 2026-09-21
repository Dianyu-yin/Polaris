"""专家问卷 v2 的图像案例聚合与视觉检索。

本模块只使用问卷前四个字段，并将同一 ``image_name`` 的多份回答聚合为
一个可审计案例。检索向量必须由调用方注入的视觉 ``feature_extractor``
产生；这里不再加载文本 embedding 模型，也不再把聊天文本当作 RAG query。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


CACHE_FILENAME = "expert_visual_index_v2.json"
CACHE_PATH = CACHE_FILENAME  # 保留旧常量名，默认路径实际按 source 文件所在目录解析。
CACHE_FORMAT_VERSION = "expert_visual_index_v2"
SURVEY_SCHEMA_VERSION = "survey_four_fields_v2"
AGGREGATION_VERSION = "image_level_votes_v2"
RETRIEVAL_VERSION = "l2_cosine_v2"

SEVERITY_VALUES = (0, 1, 2, 3)
DEFECT_VALUES = (
    "dark_spots",
    "vertical_stripe_distribution",
    "dispersed_tiny_bright_spots",
    "regional_luminescence_difference",
)
LOCATION_VALUES = ("top", "left", "right", "bottom", "center")
UNIFORMITY_VALUES = ("uniform", "slightly_nonuniform", "highly_nonuniform")

REQUIRED_COLUMNS = (
    "image_name",
    "severity_overall",
    "primary_defect_type",
    "defect_location",
    "emission_uniformity",
)


class QuestionnaireValidationError(ValueError):
    """表示问卷必填字段缺失或不属于 v2 严格枚举。"""


@dataclass(frozen=True)
class VisualRetrievalHit:
    """保存一个视觉相似专家案例及其可复现的 cosine similarity。"""

    score: float
    case: dict[str, Any]

    @property
    def image_name(self) -> str:
        """返回专家案例对应的原始图像名。"""

        return str(self.case["image_name"])

    @property
    def case_id(self) -> str:
        """返回聚合案例的稳定标识。"""

        return str(self.case["case_id"])

    @property
    def support(self) -> int:
        """返回该图像收到的专家回答数。"""

        return int(self.case["support"])

    @property
    def similarity(self) -> float:
        """以语义更明确的名称暴露 cosine score。"""

        return self.score

    def to_dict(self) -> dict[str, Any]:
        """生成可直接写入 provenance/expert_evidence 的 JSON 结构。"""

        return {
            "case_id": self.case_id,
            "image_name": self.image_name,
            "similarity": self.score,
            "support": self.support,
            "expert_case": copy.deepcopy(self.case),
        }


# 兼容旧导入名称，但其数据语义已经是视觉案例而不是 question/answer。
RetrievalHit = VisualRetrievalHit


_SEVERITY_ALIASES: dict[int, tuple[str, ...]] = {
    0: ("no defect",),
    1: ("mild (localized, limited impact on overall emission)",),
    2: ("moderate (clearly visible, affects part of the device)",),
    3: ("severe (large area or strongly affects overall emission)",),
}
_SEVERITY_NAMES = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
_DEFECT_ALIASES: dict[str, tuple[str, ...]] = {
    "dark_spots": ("dark spot(s)",),
    "vertical_stripe_distribution": (
        "vertical stripe distribution with alternating light and dark areas",
    ),
    "dispersed_tiny_bright_spots": ("dispersed tiny bright spots",),
    "regional_luminescence_difference": (
        "regional-level differences in luminescence intensity",
    ),
}
_LOCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "top": ("the top part of the image",),
    "left": ("the left part of the image",),
    "right": ("the right part of the image",),
    "bottom": ("the bottom part of the image",),
    "center": ("the center of the image",),
}
_UNIFORMITY_ALIASES: dict[str, tuple[str, ...]] = {
    "uniform": ("uniform",),
    "slightly_nonuniform": ("slightly non-uniform",),
    "highly_nonuniform": ("highly non-uniform",),
}


def _clean_text(value: Any, field_name: str) -> str:
    """统一 Unicode、大小写和空白，并拒绝空的问卷必填值。"""

    is_nan = isinstance(value, (float, np.floating)) and math.isnan(float(value))
    if value is None or (not isinstance(value, (list, tuple, set)) and is_nan):
        raise QuestionnaireValidationError(f"{field_name} 不能为空")
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        raise QuestionnaireValidationError(f"{field_name} 不能为空")
    return text


def _normalize_scalar(
    value: Any,
    *,
    field_name: str,
    canonical_values: Sequence[Any],
    aliases: Mapping[Any, Sequence[str]],
) -> Any:
    """把单选题答案严格映射到 canonical enum，未知文本立即报错。"""

    text = _clean_text(value, field_name)
    for canonical in canonical_values:
        if text == str(canonical).lower():
            return canonical
        for alias in aliases.get(canonical, ()):
            alias_text = _clean_text(alias, field_name)
            if text == alias_text or text.startswith(f"{alias_text} "):
                return canonical
    raise QuestionnaireValidationError(
        f"{field_name} 的值不属于 v2 枚举: {value!r}"
    )


def normalize_severity(value: Any) -> int:
    """将严重程度规范为 none=0、mild=1、moderate=2、severe=3。"""

    if isinstance(value, (int, np.integer)) and int(value) in SEVERITY_VALUES:
        return int(value)
    cleaned_value = _clean_text(value, "severity_overall")
    if cleaned_value in _SEVERITY_NAMES:
        return _SEVERITY_NAMES[cleaned_value]
    return int(
        _normalize_scalar(
            value,
            field_name="severity_overall",
            canonical_values=SEVERITY_VALUES,
            aliases=_SEVERITY_ALIASES,
        )
    )


def normalize_defect(value: Any) -> str:
    """将主要缺陷严格规范为问卷定义的四类 taxonomy。"""

    return str(
        _normalize_scalar(
            value,
            field_name="primary_defect_type",
            canonical_values=DEFECT_VALUES,
            aliases=_DEFECT_ALIASES,
        )
    )


def normalize_locations(value: Any) -> list[str]:
    """将复选位置拆分、去重并按固定枚举顺序返回。"""

    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        raw_parts = _clean_text(value, "defect_location").split("|")
    normalized: set[str] = set()
    for raw_part in raw_parts:
        location = _normalize_scalar(
            raw_part,
            field_name="defect_location",
            canonical_values=LOCATION_VALUES,
            aliases=_LOCATION_ALIASES,
        )
        normalized.add(str(location))
    if not normalized:
        raise QuestionnaireValidationError("defect_location 不能为空")
    return [location for location in LOCATION_VALUES if location in normalized]


def normalize_uniformity(value: Any) -> str:
    """将发光均匀性严格规范为 uniform、slightly 或 highly nonuniform。"""

    return str(
        _normalize_scalar(
            value,
            field_name="emission_uniformity",
            canonical_values=UNIFORMITY_VALUES,
            aliases=_UNIFORMITY_ALIASES,
        )
    )


def normalize_questionnaire_row(
    row: Mapping[str, Any], *, source_row: int | None = None
) -> dict[str, Any]:
    """只读取前四个有效问卷字段，并附上可回查源行的规范化记录。"""

    image_name = str(row.get("image_name", "")).strip()
    if not image_name or image_name.lower() == "nan":
        raise QuestionnaireValidationError("image_name 不能为空")
    try:
        severity = normalize_severity(row.get("severity_overall"))
        defect = normalize_defect(row.get("primary_defect_type"))
        locations = normalize_locations(row.get("defect_location"))
        uniformity = normalize_uniformity(row.get("emission_uniformity"))
    except QuestionnaireValidationError as error:
        prefix = f"源数据第 {source_row} 行: " if source_row is not None else ""
        raise QuestionnaireValidationError(f"{prefix}{error}") from error

    reference_payload = {
        "image_name": image_name,
        "severity_overall": str(row.get("severity_overall")),
        "primary_defect_type": str(row.get("primary_defect_type")),
        "defect_location": str(row.get("defect_location")),
        "emission_uniformity": str(row.get("emission_uniformity")),
    }
    reference_id = hashlib.sha256(
        json.dumps(reference_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "image_name": image_name,
        "severity": severity,
        "defect": defect,
        "locations": locations,
        "uniformity": uniformity,
        "raw_reference": {
            "reference_id": reference_id,
            "source_row": source_row,
            **reference_payload,
        },
    }


def _empty_votes(values: Iterable[Any]) -> dict[Any, int]:
    """按 canonical 顺序创建包含零票类别的完整 vote 字典。"""

    return {value: 0 for value in values}


def _votes_and_ratios(
    values: Sequence[Any], canonical_values: Sequence[Any], support: int
) -> tuple[dict[Any, int], dict[Any, float]]:
    """统计每个 canonical 值的票数及相对于回答人数的比例。"""

    votes = _empty_votes(canonical_values)
    votes.update(Counter(values))
    ratios = {key: count / support for key, count in votes.items()}
    return votes, ratios


def _majority(votes: Mapping[Any, int], canonical_values: Sequence[Any]) -> Any:
    """按固定枚举顺序稳定解决平票并返回多数结果。"""

    return max(canonical_values, key=lambda value: votes[value])


def _pairwise_location_agreement(location_sets: Sequence[set[str]]) -> float:
    """用回答两两 Jaccard 均值衡量多标签位置的一致度。"""

    if len(location_sets) <= 1:
        return 1.0
    scores: list[float] = []
    for left_index, left in enumerate(location_sets[:-1]):
        for right in location_sets[left_index + 1 :]:
            union = left | right
            scores.append(1.0 if not union else len(left & right) / len(union))
    return float(np.mean(scores))


def aggregate_expert_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把规范化回答按 image_name 聚合为 votes、ratios 和分歧完整的案例。"""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image_name"]), []).append(row)

    cases: list[dict[str, Any]] = []
    for image_name in sorted(grouped, key=str.casefold):
        image_rows = grouped[image_name]
        support = len(image_rows)
        defects = [str(row["defect"]) for row in image_rows]
        severities = [int(row["severity"]) for row in image_rows]
        uniformities = [str(row["uniformity"]) for row in image_rows]
        location_sets = [set(row["locations"]) for row in image_rows]
        flat_locations = [location for values in location_sets for location in values]

        defect_votes, defect_ratios = _votes_and_ratios(
            defects, DEFECT_VALUES, support
        )
        severity_votes_int, severity_ratios_int = _votes_and_ratios(
            severities, SEVERITY_VALUES, support
        )
        uniformity_votes, uniformity_ratios = _votes_and_ratios(
            uniformities, UNIFORMITY_VALUES, support
        )
        location_votes, location_ratios = _votes_and_ratios(
            flat_locations, LOCATION_VALUES, support
        )

        defect_agreement = max(defect_ratios.values())
        severity_agreement = max(severity_ratios_int.values())
        uniformity_agreement = max(uniformity_ratios.values())
        location_agreement = _pairwise_location_agreement(location_sets)
        agreement = {
            "defect": defect_agreement,
            "severity": severity_agreement,
            "uniformity": uniformity_agreement,
            "location": location_agreement,
        }
        agreement["overall"] = float(np.mean(list(agreement.values())))
        disagreement = {key: 1.0 - value for key, value in agreement.items()}

        # 严重度字典固定使用字符串键，确保首次 build 与 JSON load 后结构一致。
        severity_votes = {str(key): value for key, value in severity_votes_int.items()}
        severity_ratios = {
            str(key): value for key, value in severity_ratios_int.items()
        }
        case_id = "expert-image:" + hashlib.sha256(
            image_name.casefold().encode("utf-8")
        ).hexdigest()[:16]
        cases.append(
            {
                "case_id": case_id,
                "image_name": image_name,
                "support": support,
                "defect_votes": defect_votes,
                "defect_ratios": defect_ratios,
                "primary_defect": _majority(defect_votes, DEFECT_VALUES),
                "severity_votes": severity_votes,
                "severity_ratios": severity_ratios,
                "severity_mean": float(np.mean(severities)),
                "severity_median": float(np.median(severities)),
                "severity_majority": int(
                    _majority(severity_votes_int, SEVERITY_VALUES)
                ),
                "uniformity_votes": uniformity_votes,
                "uniformity_ratios": uniformity_ratios,
                "uniformity_majority": _majority(
                    uniformity_votes, UNIFORMITY_VALUES
                ),
                "location_votes": location_votes,
                "location_ratios": location_ratios,
                "agreement": agreement,
                "disagreement": disagreement,
                "raw_references": [
                    copy.deepcopy(row["raw_reference"]) for row in image_rows
                ],
            }
        )
    return cases


def _file_hash(path: str | os.PathLike[str]) -> str:
    """流式计算文件 SHA-256，避免把大 checkpoint 一次性读入内存。"""

    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _l2_normalize(vector: Any, *, context: str) -> np.ndarray:
    """校验并 L2 normalize 单个视觉 embedding。"""

    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{context} 的 feature 必须是非空有限向量")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"{context} 的 feature 不能是零向量")
    return array / norm


def _image_key(value: str | os.PathLike[str]) -> str:
    """将完整路径和文件名统一为大小写无关的排除键。"""

    return Path(str(value).replace("\\", "/")).name.casefold()


class ExpertQAStore:
    """聚合专家问卷并维护由冻结 CNN feature 构建的视觉案例索引。"""

    def __init__(
        self,
        source_path: str | os.PathLike[str],
        cache_path: str | os.PathLike[str] | None = None,
        *,
        survey_schema_version: str = SURVEY_SCHEMA_VERSION,
        aggregation_version: str = AGGREGATION_VERSION,
        retrieval_version: str = RETRIEVAL_VERSION,
    ) -> None:
        """绑定数据源与版本；相对 cache 路径始终相对数据源目录解析。"""

        self.source_path = str(Path(source_path).expanduser().resolve())
        source_parent = Path(self.source_path).parent
        if cache_path is None:
            resolved_cache = source_parent / CACHE_FILENAME
        else:
            resolved_cache = Path(cache_path).expanduser()
            if not resolved_cache.is_absolute():
                resolved_cache = source_parent / resolved_cache
        self.cache_path = str(resolved_cache.resolve())
        self.survey_schema_version = survey_schema_version
        self.aggregation_version = aggregation_version
        self.retrieval_version = retrieval_version

        self.cases: list[dict[str, Any]] = []
        self.image_names: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.index_metadata: dict[str, Any] = {}
        # 仅供旧 UI 的 len(store.questions) 状态显示；文本 retrieve 已被禁用。
        self.questions: list[str] = []
        self.answers: list[str] = []

    def _read_csv_with_encoding_fallback(self) -> list[dict[str, Any]]:
        """这个函数用标准库读取 CSV，避免 retrieval 作业额外依赖 pandas。"""

        encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                with open(self.source_path, "r", encoding=encoding, newline="") as stream:
                    return [dict(row) for row in csv.DictReader(stream)]
            except UnicodeError as error:
                last_error = error
        raise ValueError(
            f"无法用 {encodings} 解码 {self.source_path}: {last_error}"
        )

    def _read_raw_table(self) -> list[dict[str, Any]]:
        """读取 CSV/Excel 并把列名规范成小写，拒绝缺少前四问字段的数据。"""

        extension = Path(self.source_path).suffix.lower()
        if extension == ".csv":
            records = self._read_csv_with_encoding_fallback()
        elif extension in {".xlsx", ".xls"}:
            try:
                import pandas as pd
            except ImportError as error:
                raise RuntimeError("Excel expert QA input requires optional pandas") from error
            records = pd.read_excel(self.source_path).to_dict(orient="records")
        else:
            raise ValueError(f"不支持的数据文件类型: {extension}")
        normalized = [
            {str(column).strip().lower(): value for column, value in record.items()}
            for record in records
        ]
        columns = set(normalized[0]) if normalized else set()
        missing = set(REQUIRED_COLUMNS) - columns
        if missing:
            raise QuestionnaireValidationError(
                f"问卷缺少 v2 必填列: {sorted(missing)}"
            )
        return normalized

    def load_expert_cases(self) -> list[dict[str, Any]]:
        """严格规范化源问卷，并返回按图像聚合后的独立案例副本。"""

        records = self._read_raw_table()
        normalized_rows = [
            normalize_questionnaire_row(row, source_row=source_row)
            for source_row, row in enumerate(records, start=2)
        ]
        return aggregate_expert_cases(normalized_rows)

    def _resolve_checkpoint_hash(
        self,
        checkpoint_path: str | os.PathLike[str] | None,
        checkpoint_hash: str | None,
    ) -> str:
        """优先验证显式 hash，否则从 checkpoint 文件计算内容 hash。"""

        if checkpoint_hash is not None:
            normalized_hash = checkpoint_hash.strip()
            if not normalized_hash:
                raise ValueError("checkpoint_hash 不能为空")
            return normalized_hash
        if checkpoint_path is None:
            raise ValueError(
                "视觉索引必须提供 checkpoint_hash 或 checkpoint_path，"
                "以防错误复用旧 CNN feature"
            )
        return _file_hash(Path(checkpoint_path).expanduser().resolve())

    def _expected_metadata(
        self,
        *,
        checkpoint_hash: str,
        survey_config_path: str | os.PathLike[str] | None,
    ) -> dict[str, Any]:
        """生成 cache 命中时必须逐项一致的版本与内容 hash。"""

        survey_config_hash = (
            _file_hash(Path(survey_config_path).expanduser().resolve())
            if survey_config_path is not None
            else None
        )
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "source_hash": _file_hash(self.source_path),
            "survey_schema_version": self.survey_schema_version,
            "survey_config_hash": survey_config_hash,
            "checkpoint_hash": checkpoint_hash,
            "aggregation_version": self.aggregation_version,
            "retrieval_version": self.retrieval_version,
        }

    def _set_index_state(
        self,
        *,
        cases: list[dict[str, Any]],
        embeddings: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """统一设置内存索引，并维护旧 UI 只读计数属性。"""

        self.cases = cases
        self.image_names = [str(case["image_name"]) for case in cases]
        self.embeddings = embeddings
        self.index_metadata = metadata
        self.questions = list(self.image_names)
        self.answers = []

    def _try_load_cache(self, expected_metadata: dict[str, Any]) -> bool:
        """仅在 metadata 和向量完整性全部通过时加载视觉 cache。"""

        cache_file = Path(self.cache_path)
        if not cache_file.exists():
            return False
        try:
            with cache_file.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
            if payload.get("metadata") != expected_metadata:
                return False
            cases = payload["cases"]
            image_names = payload["image_names"]
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if (
                not isinstance(cases, list)
                or [case.get("image_name") for case in cases] != image_names
                or embeddings.ndim != 2
                or embeddings.shape[0] != len(cases)
                or embeddings.shape[1] == 0
                or not np.all(np.isfinite(embeddings))
                or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
            ):
                return False
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False
        self._set_index_state(
            cases=cases, embeddings=embeddings, metadata=expected_metadata
        )
        return True

    def _resolve_image_source(
        self,
        image_name: str,
        *,
        image_root: str | os.PathLike[str] | None,
        image_resolver: Callable[[str], Any] | None,
    ) -> Any:
        """通过注入 resolver 或 portable image_root 找到专家案例图像。"""

        if image_resolver is not None:
            source = image_resolver(image_name)
            if source is None:
                raise FileNotFoundError(f"image_resolver 未找到 {image_name}")
            return source

        root = (
            Path(image_root).expanduser().resolve()
            if image_root is not None
            else Path(self.source_path).parent
        )
        direct_path = root / image_name
        if direct_path.is_file():
            return str(direct_path)
        matches = [path for path in root.rglob(image_name) if path.is_file()]
        if len(matches) == 1:
            return str(matches[0])
        if not matches:
            raise FileNotFoundError(f"在 {root} 下找不到专家图像 {image_name}")
        raise ValueError(f"在 {root} 下发现多个同名专家图像 {image_name}: {matches}")

    def _write_cache(
        self,
        *,
        cases: list[dict[str, Any]],
        embeddings: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """通过同目录临时文件原子写入视觉索引，避免半写 cache。"""

        cache_file = Path(self.cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        payload = {
            "metadata": metadata,
            "cases": cases,
            "image_names": [case["image_name"] for case in cases],
            "embeddings": embeddings.tolist(),
        }
        with temporary_file.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        os.replace(temporary_file, cache_file)

    def build_or_load_index(
        self,
        feature_extractor: Callable[[Any], Any] | None = None,
        checkpoint_hash: str | None = None,
        *,
        image_root: str | os.PathLike[str] | None = None,
        image_resolver: Callable[[str], Any] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        survey_config_path: str | os.PathLike[str] | None = None,
        force_rebuild: bool = False,
    ) -> str:
        """校验版本 cache；未命中时用注入的 CNN extractor 构建 L2 索引。"""

        resolved_checkpoint_hash = self._resolve_checkpoint_hash(
            checkpoint_path, checkpoint_hash
        )
        metadata = self._expected_metadata(
            checkpoint_hash=resolved_checkpoint_hash,
            survey_config_path=survey_config_path,
        )
        if not force_rebuild and self._try_load_cache(metadata):
            return "loaded"
        if feature_extractor is None:
            raise RuntimeError(
                "视觉 cache 缺失或已失效，必须注入 feature_extractor 重新构建"
            )

        cases = self.load_expert_cases()
        vectors: list[np.ndarray] = []
        expected_dimension: int | None = None
        for case in cases:
            image_name = str(case["image_name"])
            image_source = self._resolve_image_source(
                image_name,
                image_root=image_root,
                image_resolver=image_resolver,
            )
            vector = _l2_normalize(
                feature_extractor(image_source), context=f"专家图像 {image_name}"
            )
            if expected_dimension is None:
                expected_dimension = vector.size
            elif vector.size != expected_dimension:
                raise ValueError(
                    f"专家图像 {image_name} 的 feature 维度 {vector.size} "
                    f"与预期 {expected_dimension} 不一致"
                )
            vectors.append(vector)
        if not vectors:
            raise ValueError("问卷中没有可构建视觉索引的专家案例")
        embeddings = np.stack(vectors).astype(np.float32, copy=False)
        self._write_cache(cases=cases, embeddings=embeddings, metadata=metadata)
        self._set_index_state(
            cases=cases, embeddings=embeddings, metadata=metadata
        )
        return "built"

    def build_or_load_visual_index(self, *args: Any, **kwargs: Any) -> str:
        """提供语义明确的 v2 名称，并转发到兼容的 build_or_load_index。"""

        return self.build_or_load_index(*args, **kwargs)

    def get_index_metadata(self) -> dict[str, Any]:
        """返回可写入下游 provenance 的索引 hash/version 元数据副本。"""

        if not self.index_metadata:
            raise RuntimeError("视觉索引尚未构建或加载")
        return copy.deepcopy(self.index_metadata)

    def retrieve_by_embedding(
        self,
        query_embedding: Any,
        *,
        top_k: int = 3,
        min_similarity: float = 0.0,
        exclude_image: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    ) -> list[VisualRetrievalHit]:
        """用 L2 cosine 检索 top-k，并支持阈值和排除当前图像防泄漏。"""

        if self.embeddings is None or not self.cases:
            raise RuntimeError("视觉索引尚未构建，请先调用 build_or_load_visual_index()")
        if not isinstance(top_k, (int, np.integer)) or int(top_k) <= 0:
            raise ValueError("top_k 必须是正整数")
        if not math.isfinite(float(min_similarity)) or not -1.0 <= float(min_similarity) <= 1.0:
            raise ValueError("min_similarity 必须是 [-1, 1] 内的有限数")

        query = _l2_normalize(query_embedding, context="query image")
        if query.size != self.embeddings.shape[1]:
            raise ValueError(
                f"query feature 维度 {query.size} 与索引维度 {self.embeddings.shape[1]} 不一致"
            )
        if exclude_image is None:
            excluded_keys: set[str] = set()
        elif isinstance(exclude_image, (str, os.PathLike)):
            excluded_keys = {_image_key(exclude_image)}
        else:
            excluded_keys = {_image_key(value) for value in exclude_image}

        scores = self.embeddings @ query
        ranked_indices = np.argsort(-scores, kind="stable")
        hits: list[VisualRetrievalHit] = []
        for index in ranked_indices:
            score = float(scores[index])
            case = self.cases[int(index)]
            if score < float(min_similarity):
                continue
            if _image_key(case["image_name"]) in excluded_keys:
                continue
            hits.append(
                VisualRetrievalHit(score=score, case=copy.deepcopy(case))
            )
            if len(hits) >= int(top_k):
                break
        return hits

    def retrieve_by_image(
        self,
        query_image: Any,
        feature_extractor: Callable[[Any], Any],
        *,
        top_k: int = 3,
        min_similarity: float = 0.0,
        exclude_image: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    ) -> list[VisualRetrievalHit]:
        """用同一注入 extractor 提取 query image feature 后执行视觉检索。"""

        return self.retrieve_by_embedding(
            feature_extractor(query_image),
            top_k=top_k,
            min_similarity=min_similarity,
            exclude_image=exclude_image,
        )

    def retrieve_by_vector(
        self,
        query_vector: Any,
        top_k: int = 3,
        min_similarity: float = 0.0,
        exclude_image_name: str
        | os.PathLike[str]
        | Iterable[str | os.PathLike[str]]
        | None = None,
    ) -> list[VisualRetrievalHit]:
        """以主线易集成参数名转发 query vector、阈值与同图排除规则。"""

        return self.retrieve_by_embedding(
            query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            exclude_image=exclude_image_name,
        )

    def retrieve(
        self,
        query: Any,
        top_k: int = 3,
        **kwargs: Any,
    ) -> list[VisualRetrievalHit]:
        """兼容旧方法名，但明确拒绝聊天文本，避免它冒充图像 RAG。"""

        if isinstance(query, (str, os.PathLike)):
            raise TypeError(
                "v2 已移除聊天文本 RAG；请提取当前图像 feature 后调用 "
                "retrieve_by_embedding()，或调用 retrieve_by_image()"
            )
        return self.retrieve_by_embedding(query, top_k=top_k, **kwargs)


# 新代码可采用更准确的名称；旧代码仍可导入 ExpertQAStore。
ExpertVisualCaseStore = ExpertQAStore


def format_context_block(hits: Sequence[VisualRetrievalHit]) -> str:
    """把程序生成的视觉案例证据格式化为 Qwen prompt 可用的 JSON 区块。"""

    if not hits:
        return ""
    evidence = []
    for hit in hits:
        case = hit.case
        evidence.append(
            {
                "case_id": hit.case_id,
                "image_name": hit.image_name,
                "similarity": hit.score,
                "support": hit.support,
                "defect_votes": case["defect_votes"],
                "defect_ratios": case["defect_ratios"],
                "severity_votes": case["severity_votes"],
                "severity_ratios": case["severity_ratios"],
                "severity_mean": case["severity_mean"],
                "severity_median": case["severity_median"],
                "uniformity_votes": case["uniformity_votes"],
                "uniformity_ratios": case["uniformity_ratios"],
                "location_votes": case["location_votes"],
                "location_ratios": case["location_ratios"],
                "agreement": case["agreement"],
                "disagreement": case["disagreement"],
            }
        )
    return (
        "VISUALLY_SIMILAR_AGGREGATED_EXPERT_CASES\n"
        "这些案例仅是 prior/evidence；必须优先依据当前 EL 图像的视觉特征。\n"
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )
