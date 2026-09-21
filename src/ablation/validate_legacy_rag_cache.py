"""验证 legacy_full_rag 适配目录的逐字节来源、schema 与声明边界。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "polaris-legacy-rag-adapter-v1"
EVIDENCE_MODE = "legacy_full_rag"
EXPECTED_GROUPS = 2461
EXPECTED_SPLIT_COUNTS = {"train": 1723, "val": 369, "test": 369}
EXPECTED_CONFLICT_GROUP = "20240327-38-80-AE"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "1a7a5879c25d48bb9e562ca9f90d24de0893175ed4273866a03fb6f26a2ea4a1"
)
EXPECTED_GROUP_SPLIT_SHA256 = (
    "35123b831a622ea0123f89198bc3e6019cbd68d97444d7d0201dafa3424715a0"
)
EXPECTED_DATA_MANIFEST_SHA256 = (
    "85c465c020d973c6b5c888a34efa1f3a400c1cc180dc283a623f5b3f2e01cbae"
)
EXPECTED_SPLIT_ASSIGNMENT_SHA256 = (
    "d93062a751ef1a885b21c6cee932ff3bd9f0d22f9ca335b4ec6f599b0c7b649e"
)
EXPECTED_GROUP_IDS_SHA256 = (
    "9c9d96cad5e81c49f730bb843ba0c82c2f14334a7b7ba8fdceaa7535a8609ca9"
)
EXPECTED_SOURCE_MAPPING_SHA256 = (
    "775d883e82c63e14e536787c153ace5d72769dc2d1b77532703c42526412ef4a"
)
EXPECTED_CACHE_FILES_SHA256 = (
    "df5bdcdd5761c0a9596b1e0b1b28702135183a088c491c7dfc8cc3b0e15af102"
)
EXPECTED_SOURCE_METADATA = {
    "schema_version": "survey_defect_v2",
    "prompt_version": "survey_visual_rag_prompt_v2",
    "prompt_hash": "348b0577d37608ee2be359be5094cf36e0c2d800e6bf9a2c860466787a49dce1",
    "survey_config_hash": "a53d377f7c932749eff0bdb90c301b6c8dad2a44b2d6d147a78af0861193a641",
    "expert_qa_hash": "bb0b74860aa7141abc09eb1cb4e57e6c0727517e23f632bd16fa67c87f4f953b",
    "retrieval_index_hash": "9487f25ef8ce737e3c3f15920d0f3bfbae75ef6c8d2cab916fa76d4f30fdea08",
    "cnn_checkpoint_hash": "a38bce10ffe2b2e637cba75a10327d75b9779f06d23de4517b9fed2544bc1f60",
}
EXPECTED_FINDING_COUNTS = {"0": 3, "1": 2458}
EXPECTED_EVIDENCE_COUNTS = {"0": 258, "1": 83, "2": 87, "3": 2033}
DEFECT_CODES = {
    "dark_spots",
    "vertical_stripe_distribution",
    "dispersed_tiny_bright_spots",
    "regional_luminescence_difference",
}
LOCATIONS = {"top", "left", "right", "bottom", "center"}
UNIFORMITIES = {"uniform", "slightly_nonuniform", "highly_nonuniform"}
FINDING_KEYS = {"defect", "severity", "reason", "locations", "emission_uniformity"}
UNRECORDED_PROVENANCE_FIELDS = {
    "provider_response_model",
    "qwen_base_url",
    "temperature",
    "fallback_models_allowed",
}
SECRET_FIELD_PATTERN = re.compile(
    r"(?:api[_-]?key|secret|password|authorization|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{8,}"),
    re.compile(
        r"(?i)\b(?:api[_ -]?key|secret|password|authorization|access[_ -]?token)"
        r"\s*[:=]\s*[\"']?[^\s,}\"']{6,}"
    ),
)


class LegacyRagValidationError(ValueError):
    """这个异常类统一表示历史 RAG 适配合同或字节来源验证失败。"""


def _require(condition: bool, message: str) -> None:
    """这个函数把布尔合同失败转换成带上下文的验证异常。"""

    if not condition:
        raise LegacyRagValidationError(message)


def sha256_file(path: Path) -> str:
    """这个函数流式计算文件 SHA-256，避免把大 payload 一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """这个函数用稳定 JSON 序列化计算集合与映射的聚合 SHA-256。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    """这个函数读取顶层必须为 object 的 UTF-8 JSON 文件。"""

    _require(path.is_file(), f"缺少 JSON 文件: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyRagValidationError(f"无法读取 JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON 顶层必须是 object: {path}")
    return value


def _scan_secrets(value: Any, context: str = "root") -> None:
    """这个函数递归拒绝输出 payload 中的 credential 字段或疑似密钥值。"""

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if SECRET_FIELD_PATTERN.search(key_text):
                safe_audit = key_text == "secret_values_persisted" and item is False
                _require(safe_audit, f"{context} 含 credential 字段: {key_text}")
            _scan_secrets(item, f"{context}.{key_text}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_secrets(item, f"{context}[{index}]")
        return
    if isinstance(value, str):
        _require(
            not any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS),
            f"{context} 含疑似 API key/secret 值",
        )


def _validate_finding(finding: Any, context: str) -> None:
    """这个函数验证历史 primary finding 没有偏离锁定的四类问卷 schema。"""

    _require(isinstance(finding, dict), f"{context} finding 必须是 object")
    _require(set(finding) == FINDING_KEYS, f"{context} finding keys 漂移")
    _require(finding.get("defect") in DEFECT_CODES, f"{context} defect code 非法")
    severity = finding.get("severity")
    _require(type(severity) is int and severity in {1, 2, 3}, f"{context} severity 非法")
    reason = finding.get("reason")
    _require(isinstance(reason, str) and reason.strip(), f"{context} reason 必须非空")
    locations = finding.get("locations")
    _require(isinstance(locations, list) and locations, f"{context} locations 必须非空")
    _require(
        all(type(location) is str and location in LOCATIONS for location in locations),
        f"{context} location enum 非法",
    )
    _require(len(locations) == len(set(locations)), f"{context} locations 含重复值")
    _require(
        finding.get("emission_uniformity") in UNIFORMITIES,
        f"{context} emission_uniformity 非法",
    )


def _validate_payload(
    payload: dict[str, Any],
    group_id: str,
    *,
    canonical_filename: str | None,
    context: str,
) -> tuple[int, int]:
    """这个函数验证单组历史 payload 身份、findings、检索证据与 provenance 缺失。"""

    _scan_secrets(payload, context)
    _require(payload.get("source_original_id") == group_id, f"{context} group_id 不匹配")
    for key, expected in EXPECTED_SOURCE_METADATA.items():
        _require(payload.get(key) == expected, f"{context} {key} 漂移")
    _require(payload.get("qwen_status") == "api_success", f"{context} qwen_status 漂移")
    _require(payload.get("provider") == "qwen", f"{context} provider 记录值漂移")
    _require(payload.get("model") == "qwen", f"{context} model 记录值漂移")
    _require(payload.get("requested_model") == "qwen", f"{context} requested_model 漂移")
    _require(payload.get("provider_error") is None, f"{context} provider_error 非空")
    for field in UNRECORDED_PROVENANCE_FIELDS:
        _require(field not in payload, f"{context} 不得补造历史未记录字段 {field}")

    image = payload.get("image")
    _require(isinstance(image, str) and image.strip(), f"{context} image 非法")
    if canonical_filename is not None:
        _require(image == canonical_filename, f"{context} image 与 canonical filename 不匹配")
    _require(payload.get("defect_source_image") == image, f"{context} defect source 漂移")
    _require(payload.get("defect_reused_from_source") is False, f"{context} reused 标记漂移")

    findings = payload.get("findings")
    _require(isinstance(findings, list) and len(findings) <= 1, f"{context} findings 非法")
    for finding in findings:
        _validate_finding(finding, context)
    _require(payload.get("defects") == findings, f"{context} defects 与 findings 不一致")
    _require(
        payload.get("structured_output") == {"defects": findings},
        f"{context} structured_output 与 findings 不一致",
    )

    expert_evidence = payload.get("expert_evidence")
    _require(isinstance(expert_evidence, dict), f"{context} expert_evidence 缺失")
    retrieved = expert_evidence.get("retrieved_cases")
    _require(isinstance(retrieved, list) and len(retrieved) <= 3, f"{context} evidence 数量非法")
    _require(expert_evidence.get("query_image") == image, f"{context} query image 漂移")
    _require(expert_evidence.get("top_k") == 3, f"{context} top_k 漂移")
    minimum = expert_evidence.get("min_similarity")
    _require(
        type(minimum) in (int, float) and float(minimum) == 0.75,
        f"{context} min_similarity 漂移",
    )
    _require(
        expert_evidence.get("excluded_current_image") is True,
        f"{context} 未记录 current-image exclusion",
    )
    expected_status = "matched" if retrieved else "no_reliable_match"
    _require(
        expert_evidence.get("retrieval_status") == expected_status,
        f"{context} retrieval_status 与 evidence 数量不一致",
    )
    case_ids: list[str] = []
    for index, item in enumerate(retrieved, start=1):
        _require(isinstance(item, dict), f"{context} evidence[{index}] 必须是 object")
        case_id = item.get("case_id")
        case_image = item.get("image_name")
        similarity = item.get("similarity")
        _require(isinstance(case_id, str) and case_id, f"{context} evidence case_id 非法")
        _require(isinstance(case_image, str) and case_image, f"{context} evidence image 非法")
        _require(case_image != image, f"{context} evidence 泄漏 current image")
        _require(
            type(similarity) in (int, float)
            and math.isfinite(float(similarity))
            and float(similarity) >= 0.75,
            f"{context} evidence similarity 非法",
        )
        case_ids.append(case_id)
    _require(len(case_ids) == len(set(case_ids)), f"{context} evidence case_id 重复")

    provenance = payload.get("provenance")
    _require(isinstance(provenance, dict), f"{context} provenance 缺失")
    _require(
        provenance.get("expert_evidence_count") == len(retrieved),
        f"{context} evidence count provenance 不一致",
    )
    return len(findings), len(retrieved)


def _read_group_split(path: Path) -> list[dict[str, str]]:
    """这个函数读取冻结 group split，并验证 2461 个 canonical group 的唯一身份。"""

    _require(path.is_file(), f"缺少 group split: {path}")
    _require(sha256_file(path) == EXPECTED_GROUP_SPLIT_SHA256, "group_split SHA-256 漂移")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == EXPECTED_GROUPS, f"group split 行数 {len(rows)} != {EXPECTED_GROUPS}")
    required = {"original_id", "split", "canonical_filename"}
    _require(rows and required.issubset(rows[0]), "group split 缺少必要列")
    by_id: dict[str, dict[str, str]] = {}
    split_counts: Counter[str] = Counter()
    canonical_names: set[str] = set()
    for row in rows:
        group_id = row["original_id"].strip()
        canonical = row["canonical_filename"].strip()
        split = row["split"].strip()
        _require(group_id and group_id not in by_id, f"group_id 为空或重复: {group_id!r}")
        _require("/" not in group_id and "\\" not in group_id, f"group_id 非安全文件名: {group_id}")
        _require(canonical and Path(canonical).name == canonical, f"canonical filename 非法: {canonical}")
        _require(canonical not in canonical_names, f"canonical filename 重复: {canonical}")
        _require(split in EXPECTED_SPLIT_COUNTS, f"未知 split: {split}")
        _require(group_id != EXPECTED_CONFLICT_GROUP, "冻结 split 仍包含冲突组")
        by_id[group_id] = {**row, "original_id": group_id, "canonical_filename": canonical}
        canonical_names.add(canonical)
        split_counts[split] += 1
    normalized_split_counts = {
        split: split_counts.get(split, 0) for split in EXPECTED_SPLIT_COUNTS
    }
    _require(
        normalized_split_counts == EXPECTED_SPLIT_COUNTS,
        f"split group counts 漂移: {normalized_split_counts}",
    )
    _require(
        stable_json_hash(sorted(by_id)) == EXPECTED_GROUP_IDS_SHA256,
        "group ID 集合 SHA-256 漂移",
    )
    return [by_id[group_id] for group_id in sorted(by_id)]


def _validate_data_manifest(path: Path) -> dict[str, Any]:
    """这个函数把 group split 绑定到已确认的 cleaned-data manifest 与冲突排除。"""

    _require(sha256_file(path) == EXPECTED_DATA_MANIFEST_SHA256, "data_manifest SHA-256 漂移")
    manifest = _read_json_object(path)
    _require(manifest.get("status") == "PASS", "data manifest status 不是 PASS")
    _require(
        manifest.get("contract_version") == "polaris-ablation-split-v1",
        "data manifest contract version 漂移",
    )
    _require(manifest.get("assignment_sha256") == EXPECTED_SPLIT_ASSIGNMENT_SHA256, "split assignment SHA-256 漂移")
    _require(manifest.get("conflict_group_excluded") == EXPECTED_CONFLICT_GROUP, "冲突排除记录漂移")
    _require((manifest.get("clean") or {}).get("groups") == EXPECTED_GROUPS, "clean group 数量漂移")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict), "data manifest 缺少 artifacts")
    group_artifact = artifacts.get("group_split.csv")
    _require(isinstance(group_artifact, dict), "data manifest 缺少 group_split artifact")
    _require(group_artifact.get("sha256") == EXPECTED_GROUP_SPLIT_SHA256, "manifest 内 group_split SHA-256 漂移")
    splits = manifest.get("splits")
    _require(isinstance(splits, dict), "data manifest 缺少 splits")
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        _require((splits.get(split) or {}).get("groups") == expected, f"{split} group 数量漂移")
    return manifest


def _validate_source_manifest(path: Path) -> dict[str, Any]:
    """这个函数锁定唯一获准复用的历史 cache manifest，而不推断缺失 provider 信息。"""

    _require(
        sha256_file(path) == EXPECTED_SOURCE_MANIFEST_SHA256,
        "历史 cache_manifest SHA-256 漂移",
    )
    manifest = _read_json_object(path)
    for key, expected in EXPECTED_SOURCE_METADATA.items():
        _require(manifest.get(key) == expected, f"历史 manifest {key} 漂移")
    _require(manifest.get("all_groups") == 2462, "历史 manifest all_groups 漂移")
    _require(manifest.get("selected_groups") == 2462, "历史 manifest selected_groups 漂移")
    _require(manifest.get("completed_groups") == 2462, "历史 manifest completed_groups 漂移")
    _require(manifest.get("failed_groups") == 0, "历史 manifest 含失败组")
    _require(manifest.get("reuse_original_id") is True, "历史 manifest reuse_original_id 漂移")
    _require(manifest.get("mock_mode") is False, "历史 manifest 不是正式 API 运行")
    return manifest


def inspect_legacy_source(
    source_cache_dir: Path,
    group_split: Path,
    data_manifest: Path,
) -> dict[str, Any]:
    """这个函数只读审计旧 cache，并生成 canonical group 到源字节的唯一映射。"""

    source_cache_dir = source_cache_dir.expanduser().resolve()
    group_split = group_split.expanduser().resolve()
    data_manifest = data_manifest.expanduser().resolve()
    _require(source_cache_dir.is_dir(), f"历史 cache 目录不存在: {source_cache_dir}")
    _validate_data_manifest(data_manifest)
    rows = _read_group_split(group_split)
    _validate_source_manifest(source_cache_dir / "cache_manifest.json")

    records: list[dict[str, Any]] = []
    source_mapping: dict[str, dict[str, str]] = {}
    cache_hashes: dict[str, str] = {}
    finding_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    source_names: set[str] = set()
    for row in rows:
        group_id = row["original_id"]
        canonical = row["canonical_filename"]
        source_name = f"{Path(canonical).stem}.json"
        source_path = source_cache_dir / source_name
        _require(source_path.is_file() and not source_path.is_symlink(), f"缺少普通源 payload: {source_path}")
        _require(source_name not in source_names, f"多个 group 映射到同一源 payload: {source_name}")
        payload = _read_json_object(source_path)
        finding_count, evidence_count = _validate_payload(
            payload,
            group_id,
            canonical_filename=canonical,
            context=f"source/{source_name}",
        )
        source_sha256 = sha256_file(source_path)
        source_mapping[group_id] = {
            "source_filename": source_name,
            "source_sha256": source_sha256,
            "canonical_filename": canonical,
        }
        cache_hashes[group_id] = source_sha256
        finding_counts[str(finding_count)] += 1
        evidence_counts[str(evidence_count)] += 1
        source_names.add(source_name)
        records.append(
            {
                "group_id": group_id,
                "canonical_filename": canonical,
                "source_path": source_path,
                "source_filename": source_name,
                "source_sha256": source_sha256,
            }
        )

    normalized_findings = {key: finding_counts.get(key, 0) for key in ("0", "1")}
    normalized_evidence = {key: evidence_counts.get(key, 0) for key in ("0", "1", "2", "3")}
    _require(normalized_findings == EXPECTED_FINDING_COUNTS, f"finding count 分布漂移: {normalized_findings}")
    _require(normalized_evidence == EXPECTED_EVIDENCE_COUNTS, f"evidence count 分布漂移: {normalized_evidence}")
    _require(
        stable_json_hash(source_mapping) == EXPECTED_SOURCE_MAPPING_SHA256,
        "canonical source mapping SHA-256 漂移",
    )
    _require(
        stable_json_hash(cache_hashes) == EXPECTED_CACHE_FILES_SHA256,
        "canonical payload aggregate SHA-256 漂移",
    )
    return {
        "records": records,
        "source_mapping": source_mapping,
        "cache_hashes": cache_hashes,
        "finding_counts": normalized_findings,
        "evidence_counts": normalized_evidence,
    }


def make_adapter_manifest() -> dict[str, Any]:
    """这个函数构造唯一正式 adapter manifest，并显式保留历史 provenance 缺口。"""

    return {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "evidence_mode": EVIDENCE_MODE,
        "groups_expected": EXPECTED_GROUPS,
        "groups_cached": EXPECTED_GROUPS,
        "failures": [],
        "group_ids_sha256": EXPECTED_GROUP_IDS_SHA256,
        "source_mapping_sha256": EXPECTED_SOURCE_MAPPING_SHA256,
        "cache_files_sha256": EXPECTED_CACHE_FILES_SHA256,
        "copy_contract": {
            "group_payload_copy": "byte_for_byte",
            "destination_filename": "source_original_id.json",
            "canonical_groups_only": True,
        },
        "source_cache": {
            "manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            **EXPECTED_SOURCE_METADATA,
        },
        "current_split": {
            "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
            "group_split_sha256": EXPECTED_GROUP_SPLIT_SHA256,
            "assignment_sha256": EXPECTED_SPLIT_ASSIGNMENT_SHA256,
            "conflict_group_excluded": EXPECTED_CONFLICT_GROUP,
            "split_group_counts": EXPECTED_SPLIT_COUNTS,
        },
        "payload_summary": {
            "findings_per_group": EXPECTED_FINDING_COUNTS,
            "expert_evidence_per_group": EXPECTED_EVIDENCE_COUNTS,
            "qwen_status_recorded_values": ["api_success"],
            "provider_recorded_values": ["qwen"],
            "requested_model_recorded_values": ["qwen"],
            "payload_model_recorded_values": ["qwen"],
        },
        "historical_provenance_limits": {
            "provider_response_model_verified": False,
            "endpoint_recorded": False,
            "temperature_recorded": False,
            "fallback_policy_recorded": False,
            "retrieval_encoder_training_groups_recorded": False,
            "statement": (
                "Historical payloads do not record provider response model, endpoint, "
                "temperature, fallback policy, or retrieval-encoder training groups; "
                "the adapter does not infer them."
            ),
        },
        "credential_audit": {
            "adapter_payloads_scanned": EXPECTED_GROUPS,
            "copied_source_code": False,
            "secret_values_persisted": False,
        },
    }


def validate_legacy_rag_cache_for_training(cache_dir: Path) -> dict[str, Any]:
    """这个函数仅凭适配目录验证训练可用性、聚合 hash 和诚实 provenance 声明。"""

    cache_dir = cache_dir.expanduser().resolve()
    _require(cache_dir.is_dir(), f"legacy cache 目录不存在: {cache_dir}")
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = _read_json_object(manifest_path)
    _scan_secrets(manifest, "cache_manifest")
    _require(manifest == make_adapter_manifest(), "legacy adapter manifest 内容漂移")

    entries = list(cache_dir.iterdir())
    unexpected = [
        path.name
        for path in entries
        if path.name != "cache_manifest.json"
        and (not path.is_file() or path.is_symlink() or path.suffix.lower() != ".json")
    ]
    _require(not unexpected, f"legacy cache 含非合同文件: {sorted(unexpected)[:10]}")
    payload_paths = {
        path.stem: path
        for path in entries
        if path.name != "cache_manifest.json" and path.is_file() and path.suffix.lower() == ".json"
    }
    _require(len(payload_paths) == EXPECTED_GROUPS, f"legacy payload 数量 {len(payload_paths)} != {EXPECTED_GROUPS}")
    _require(
        stable_json_hash(sorted(payload_paths)) == EXPECTED_GROUP_IDS_SHA256,
        "legacy payload group ID 集合漂移",
    )

    cache_hashes: dict[str, str] = {}
    finding_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    for group_id in sorted(payload_paths):
        path = payload_paths[group_id]
        payload = _read_json_object(path)
        finding_count, evidence_count = _validate_payload(
            payload,
            group_id,
            canonical_filename=None,
            context=f"adapter/{path.name}",
        )
        cache_hashes[group_id] = sha256_file(path)
        finding_counts[str(finding_count)] += 1
        evidence_counts[str(evidence_count)] += 1
    normalized_findings = {key: finding_counts.get(key, 0) for key in ("0", "1")}
    normalized_evidence = {key: evidence_counts.get(key, 0) for key in ("0", "1", "2", "3")}
    _require(normalized_findings == EXPECTED_FINDING_COUNTS, "adapter finding count 分布漂移")
    _require(normalized_evidence == EXPECTED_EVIDENCE_COUNTS, "adapter evidence count 分布漂移")
    aggregate_hash = stable_json_hash(cache_hashes)
    _require(aggregate_hash == EXPECTED_CACHE_FILES_SHA256, "adapter payload aggregate SHA-256 漂移")
    return {
        "status": "PASS",
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "contract_version": CONTRACT_VERSION,
        "evidence_mode": EVIDENCE_MODE,
        "groups_cached": EXPECTED_GROUPS,
        "cache_files_sha256": aggregate_hash,
        "historical_provenance_limits": manifest["historical_provenance_limits"],
    }


def validate_legacy_rag_cache(
    cache_dir: Path,
    source_cache_dir: Path,
    group_split: Path,
    data_manifest: Path,
) -> dict[str, Any]:
    """这个函数完成正式全验证，并逐组证明 adapter payload 与获准源文件字节相同。"""

    inspection = inspect_legacy_source(source_cache_dir, group_split, data_manifest)
    result = validate_legacy_rag_cache_for_training(cache_dir)
    cache_dir = cache_dir.expanduser().resolve()
    for record in inspection["records"]:
        destination = cache_dir / f"{record['group_id']}.json"
        _require(destination.is_file(), f"缺少 adapter payload: {destination}")
        _require(
            destination.read_bytes() == record["source_path"].read_bytes(),
            f"adapter payload 不是逐字节复制: {record['group_id']}",
        )
    return {
        **result,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "source_mapping_sha256": EXPECTED_SOURCE_MAPPING_SHA256,
        "byte_identical_groups": EXPECTED_GROUPS,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数声明 dedicated validator 的四个只读输入。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--group-split", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """这个函数执行 validator，并用单行 JSON 与退出码报告 PASS/FAIL。"""

    try:
        args = parse_args(argv)
        result = validate_legacy_rag_cache(
            args.cache_dir,
            args.source_cache_dir,
            args.group_split,
            args.data_manifest,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
