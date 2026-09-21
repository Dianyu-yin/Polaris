"""锁定 POLARIS 正式七条件，以及经确认的扩展消融条件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


TRAINING_SEEDS = (42, 123, 3407, 2026, 2027)


@dataclass(frozen=True)
class Condition:
    """这个不可变配置记录一个消融条件唯一允许变化的组件。"""

    condition_id: str
    display_name: str
    panel: str
    architecture: str
    cache_name: str | None
    residual_pce: bool
    use_null_semantic_token: bool
    delta_soft_limit: float
    delta_penalty_weight: float

    @property
    def cache_mode(self) -> str:
        """这个属性把有语义/无语义条件映射成 Dataset 的 cache 合同。"""

        return "none" if self.cache_name is None else "required"

    def to_dict(self) -> dict[str, object]:
        """这个函数把不可变条件转换为可写入 run config 的普通字典。"""

        return {**asdict(self), "cache_mode": self.cache_mode}


CONDITIONS = (
    Condition(
        condition_id="cnn_only",
        display_name="CNN-only",
        panel="knowledge",
        architecture="cnn_only",
        cache_name=None,
        residual_pce=False,
        use_null_semantic_token=False,
        delta_soft_limit=0.0,
        delta_penalty_weight=0.0,
    ),
    Condition(
        condition_id="null_semantic_token",
        display_name="Null semantic token",
        panel="knowledge",
        architecture="transformer",
        cache_name=None,
        residual_pce=True,
        use_null_semantic_token=True,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
    Condition(
        condition_id="qwen_image_only",
        display_name="Qwen image-only",
        panel="knowledge",
        architecture="transformer",
        cache_name="image_only",
        residual_pce=True,
        use_null_semantic_token=False,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
    Condition(
        condition_id="random_3_expert_evidence",
        display_name="Random-3 expert evidence",
        panel="knowledge",
        architecture="transformer",
        cache_name="random_3",
        residual_pce=True,
        use_null_semantic_token=False,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
    Condition(
        condition_id="full_polaris_legacy_thresholded_rag",
        display_name="Full POLARIS (legacy thresholded RAG)",
        panel="knowledge",
        architecture="transformer",
        cache_name="legacy_full_rag",
        residual_pce=True,
        use_null_semantic_token=False,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
    Condition(
        condition_id="full_rag_mlp_residual",
        display_name="Full RAG + MLP residual",
        panel="architecture",
        architecture="mlp",
        cache_name="legacy_full_rag",
        residual_pce=True,
        use_null_semantic_token=False,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
    Condition(
        condition_id="full_rag_transformer_direct",
        display_name="Full RAG + Transformer direct-PCE",
        panel="architecture",
        architecture="transformer",
        cache_name="legacy_full_rag",
        residual_pce=False,
        use_null_semantic_token=False,
        delta_soft_limit=0.0,
        delta_penalty_weight=0.0,
    ),
)
CONDITION_BY_ID = {condition.condition_id: condition for condition in CONDITIONS}


# 这三个扩展条件只把既有 MLP residual 的知识来源改成 Null、image-only 或 Random-3。
MLP_KNOWLEDGE_CONDITIONS = (
    Condition(
        condition_id="mlp_null_semantic_token",
        display_name="MLP + Null semantic feature",
        panel="mlp_knowledge",
        architecture="mlp",
        cache_name=None,
        residual_pce=True,
        use_null_semantic_token=True,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
    Condition(
        condition_id="mlp_qwen_image_only",
        display_name="MLP + Qwen image-only",
        panel="mlp_knowledge",
        architecture="mlp",
        cache_name="image_only",
        residual_pce=True,
        use_null_semantic_token=False,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
    Condition(
        condition_id="mlp_random_3_expert_evidence",
        display_name="MLP + Random-3 expert evidence",
        panel="mlp_knowledge",
        architecture="mlp",
        cache_name="random_3",
        residual_pce=True,
        use_null_semantic_token=False,
        delta_soft_limit=3.0,
        delta_penalty_weight=0.01,
    ),
)
MLP_KNOWLEDGE_CONDITION_BY_ID = {
    condition.condition_id: condition for condition in MLP_KNOWLEDGE_CONDITIONS
}
ALL_CONDITION_BY_ID = {**CONDITION_BY_ID, **MLP_KNOWLEDGE_CONDITION_BY_ID}


def get_condition(condition_id: str) -> Condition:
    """这个函数按稳定 ID 返回条件，未知 ID 立即失败以避免跑错实验。"""

    try:
        return ALL_CONDITION_BY_ID[condition_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown condition {condition_id!r}; expected one of {sorted(ALL_CONDITION_BY_ID)}"
        ) from error


def validate_contract() -> None:
    """这个函数检查正式七条件不变，并验证 MLP 知识扩展只改变 cache 来源。"""

    if len(CONDITIONS) != 7 or len(CONDITION_BY_ID) != 7:
        raise ValueError("Experiment contract must contain exactly seven unique conditions")
    direct = get_condition("full_rag_transformer_direct")
    if direct.residual_pce or direct.delta_penalty_weight != 0.0:
        raise ValueError("Direct-PCE condition cannot use residual anchoring or delta penalty")
    null = get_condition("null_semantic_token")
    if null.cache_name is not None or not null.use_null_semantic_token:
        raise ValueError("Null semantic condition must use one learnable null token and no cache")
    legacy_ids = {
        "full_polaris_legacy_thresholded_rag",
        "full_rag_mlp_residual",
        "full_rag_transformer_direct",
    }
    if {get_condition(condition_id).cache_name for condition_id in legacy_ids} != {
        "legacy_full_rag"
    }:
        raise ValueError("All three Full-RAG architecture branches must share legacy_full_rag")
    if set(CONDITION_BY_ID) & set(MLP_KNOWLEDGE_CONDITION_BY_ID):
        raise ValueError("MLP knowledge extension condition IDs must not overwrite formal IDs")
    if len(MLP_KNOWLEDGE_CONDITIONS) != 3:
        raise ValueError("MLP knowledge extension must contain exactly three retraining conditions")
    if {
        (condition.architecture, condition.residual_pce, condition.delta_soft_limit, condition.delta_penalty_weight)
        for condition in MLP_KNOWLEDGE_CONDITIONS
    } != {("mlp", True, 3.0, 0.01)}:
        raise ValueError("MLP knowledge conditions may differ only in their semantic source")
    if {condition.cache_name for condition in MLP_KNOWLEDGE_CONDITIONS} != {
        None,
        "image_only",
        "random_3",
    }:
        raise ValueError("MLP knowledge conditions must be Null, image_only, and random_3")


validate_contract()
