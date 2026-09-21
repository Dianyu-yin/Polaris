"""按 cache contract_version 分流验证新 Qwen cache 与历史 RAG adapter。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ablation.experiment_contract import Condition
from ablation.validate_legacy_rag_cache import (
    CONTRACT_VERSION as LEGACY_CONTRACT_VERSION,
    EVIDENCE_MODE as LEGACY_EVIDENCE_MODE,
    sha256_file,
    validate_legacy_rag_cache_for_training,
)


QWEN_CONTRACT_VERSION = "polaris-qwen-cache-v2"
QWEN_CACHE_NAMES = {"image_only", "random_3"}
EXPECTED_CACHE_GROUPS = 2461
EXPECTED_QWEN_BASE_URL = "https://models.sjtu.edu.cn/api/v1"
EXPECTED_PROMPT_HASH = "348b0577d37608ee2be359be5094cf36e0c2d800e6bf9a2c860466787a49dce1"
EXPECTED_RANDOM_SELECTION_AGGREGATE_SHA256 = (
    "9d6452434127cdd890f7383c2332cad8b80606523ff7c145e49f4792ea51d677"
)


class CacheContractError(ValueError):
    """这个异常类表示训练条件和落盘 cache contract 不兼容。"""


def _require(condition: bool, message: str) -> None:
    """这个函数把 manifest 合同失败转换成统一异常。"""

    if not condition:
        raise CacheContractError(message)


def _read_manifest(path: Path) -> dict[str, Any]:
    """这个函数读取 cache manifest，并拒绝缺失、坏 JSON 或非 object 顶层。"""

    if not path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheContractError(f"Cannot read cache manifest {path}: {exc}") from exc
    _require(isinstance(manifest, dict), f"Cache manifest must be an object: {path}")
    return manifest


def _is_sha256(value: object) -> bool:
    """这个函数验证字段是否为规范的 64 位小写 SHA-256 十六进制字符串。"""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_qwen_v2_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    condition: Condition,
) -> dict[str, Any]:
    """这个函数验证新 Qwen 两模式合同，尤其要求 temperature 字段确实被省略。"""

    _require(condition.cache_name in QWEN_CACHE_NAMES, f"Qwen v2 不支持 cache {condition.cache_name!r}")
    expected = {
        "status": "PASS",
        "contract_version": QWEN_CONTRACT_VERSION,
        "evidence_mode": condition.cache_name,
        "schema_version": "survey_defect_v2",
        "prompt_version": "survey_visual_rag_prompt_v2",
        "prompt_template_sha256": EXPECTED_PROMPT_HASH,
        "requested_model": "qwen",
        "actual_models": ["qwen"],
        "provider_response_models": ["qwen"],
        "model_sources": ["provider_response"],
        "model_candidates": ["qwen"],
        "qwen_base_url": EXPECTED_QWEN_BASE_URL,
        "temperature_policy": "omitted",
        "random_seed": 20260827 if condition.cache_name == "random_3" else None,
        "random_selection_aggregate_sha256": (
            EXPECTED_RANDOM_SELECTION_AGGREGATE_SHA256
            if condition.cache_name == "random_3"
            else None
        ),
        "groups_expected": EXPECTED_CACHE_GROUPS,
        "groups_cached": EXPECTED_CACHE_GROUPS,
        "failures": [],
        "api_key_persisted": False,
        "fallback_models_allowed": False,
    }
    drift = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    _require(not drift, f"Qwen v2 manifest contract drift for {condition.condition_id}: {drift}")
    _require("temperature" not in manifest, "Qwen v2 manifest 不得补写 temperature")
    _require(
        _is_sha256(manifest.get("query_image_files_sha256")),
        "Qwen v2 manifest 缺少规范的 query_image_files_sha256",
    )
    status_counts = manifest.get("status_counts")
    _require(
        isinstance(status_counts, dict)
        and set(status_counts) == {"written", "skipped_valid", "failed"}
        and all(
            isinstance(status_counts.get(key), int)
            and not isinstance(status_counts.get(key), bool)
            and status_counts[key] >= 0
            for key in ("written", "skipped_valid", "failed")
        )
        and status_counts["failed"] == 0
        and status_counts["written"] + status_counts["skipped_valid"]
        == EXPECTED_CACHE_GROUPS,
        "Qwen v2 manifest status_counts 非法",
    )
    credential_audit = manifest.get("credential_audit")
    _require(isinstance(credential_audit, dict), "Qwen v2 manifest 缺少 credential_audit")
    _require(
        credential_audit.get("secret_values_persisted") is False,
        "Qwen v2 credential audit 未证明 secret_values_persisted=false",
    )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "contract_version": QWEN_CONTRACT_VERSION,
        "evidence_mode": condition.cache_name,
        "groups_cached": EXPECTED_CACHE_GROUPS,
        "requested_model": "qwen",
        "provider_response_models": ["qwen"],
        "qwen_base_url": EXPECTED_QWEN_BASE_URL,
        "temperature_policy": "omitted",
        "query_image_files_sha256": manifest["query_image_files_sha256"],
        "random_selection_aggregate_sha256": manifest[
            "random_selection_aggregate_sha256"
        ],
        "status_counts": status_counts,
        "fallback_models_allowed": False,
    }


def validate_mode_cache_manifest(
    defects_dir: Path,
    condition: Condition,
) -> dict[str, Any]:
    """这个函数按 manifest 版本分流，阻止 legacy provenance 被包装成新 Qwen 证明。"""

    defects_dir = defects_dir.expanduser().resolve()
    manifest_path = defects_dir / "cache_manifest.json"
    manifest = _read_manifest(manifest_path)
    contract_version = manifest.get("contract_version")
    if contract_version == LEGACY_CONTRACT_VERSION:
        _require(
            condition.cache_name == LEGACY_EVIDENCE_MODE,
            f"legacy adapter 不能用于 cache {condition.cache_name!r}",
        )
        return validate_legacy_rag_cache_for_training(defects_dir)
    if contract_version == QWEN_CONTRACT_VERSION:
        return _validate_qwen_v2_manifest(manifest_path, manifest, condition)
    raise CacheContractError(
        f"Unsupported cache contract_version {contract_version!r} for {condition.condition_id}"
    )
