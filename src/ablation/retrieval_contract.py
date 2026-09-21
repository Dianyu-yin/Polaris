"""定义不依赖 torch 的 POLARIS shuffled Top-3 抽样合同。"""

from __future__ import annotations

import hashlib
import random


EXPECTED_EXPERT_CASES = 19
SHUFFLE_SEED = 20260827


def _stable_group_seed(group_id: str, seed: int = SHUFFLE_SEED) -> int:
    """这个函数为每个 original_id 派生固定 seed，使结果不受并行和遍历顺序影响。"""

    digest = hashlib.sha256(f"polaris-shuffled-top3-v1:{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def choose_shuffled_case_ids(
    group_id: str,
    all_case_ids: list[str],
    true_top3_case_ids: list[str],
    seed: int = SHUFFLE_SEED,
) -> tuple[list[str], list[str]]:
    """这个函数排除真实 Top-3 后，从剩余16个案例无放回固定抽取3个。"""

    if len(all_case_ids) != EXPECTED_EXPERT_CASES or len(set(all_case_ids)) != EXPECTED_EXPERT_CASES:
        raise ValueError("Expected exactly 19 unique expert case IDs")
    if len(true_top3_case_ids) != 3 or len(set(true_top3_case_ids)) != 3:
        raise ValueError("true_top3_case_ids must contain exactly three unique IDs")
    pool = sorted(set(all_case_ids) - set(true_top3_case_ids))
    if len(pool) != 16:
        raise ValueError(f"Shuffled candidate pool must contain 16 cases, got {len(pool)}")
    selected = random.Random(_stable_group_seed(group_id, seed)).sample(pool, 3)
    return selected, pool

