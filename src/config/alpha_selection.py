"""Effective-alpha selection loading helpers.

正式研究的 alpha universe 以 ``reports/alpha_ic_analysis/effective_alphas.json``
為唯一入口；舊常數清單只保留給歷史相容，不再作為靜默 fallback。
"""

from __future__ import annotations

import json
from pathlib import Path


EFFECTIVE_ALPHAS_PATH = Path("reports/alpha_ic_analysis/effective_alphas.json")


def load_effective_alpha_ids(
    path: str | Path = EFFECTIVE_ALPHAS_PATH,
    *,
    required: bool = True,
) -> list[str] | None:
    """Load the TEJ IS-only effective alpha list.

    Args:
        path: JSON artifact path.
        required: If True, missing/empty files raise RuntimeError. If False,
            return None so tests or explicit all-alpha experiments can continue.
    """
    target = Path(path)
    if not target.exists():
        if required:
            raise RuntimeError(
                f"缺少 effective alpha 清單：{target}。正式研究必須先執行 "
                "python scripts/run_is_oos_validation.py --data-source tej "
                "--train-end 2024-06-30"
            )
        return None

    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    alphas = data.get("effective_alphas") or data.get("all_alphas") or []
    if not alphas:
        if required:
            raise RuntimeError(f"effective alpha 清單為空：{target}")
        return None
    return [str(a) for a in alphas]
