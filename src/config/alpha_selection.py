"""Effective-alpha selection loading helpers.

正式研究的 alpha universe 以 ``reports/alpha_ic_analysis/effective_alphas.json``
為唯一入口；舊常數清單只保留給歷史相容，不再作為靜默 fallback。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config.constants import WQ101_INDCLASS_OR_CAP_ALPHA_IDS, WQ101_PURE_PRICE_ALPHA_IDS


EFFECTIVE_ALPHAS_PATH = Path("reports/alpha_ic_analysis/effective_alphas.json")


def exclude_indclass_cap_alpha_ids(alpha_ids: list[str] | None) -> list[str] | None:
    """排除需要 placeholder ``indclass`` 或 ``cap`` 輸入的 WQ101 alpha。

    TEJ OHLCV parquet 尚未接入真實產業分類或市值 panel。在這些資料接上前，
    這個 helper 提供 reviewer-facing 實驗用的保守 ablation universe。
    """
    if alpha_ids is None:
        return list(WQ101_PURE_PRICE_ALPHA_IDS)
    blocked = set(WQ101_INDCLASS_OR_CAP_ALPHA_IDS)
    return [str(a) for a in alpha_ids if str(a) not in blocked]


def load_effective_alpha_ids(
    path: str | Path = EFFECTIVE_ALPHAS_PATH,
    *,
    required: bool = True,
    exclude_indclass_cap: bool = False,
) -> list[str] | None:
    """讀取 TEJ IS-only effective alpha 清單。

    Args:
        path: JSON artifact 路徑。
        required: 若為 True，檔案不存在或清單為空時丟出 RuntimeError。若為 False，
            回傳 None，讓測試或明確指定的全 alpha 實驗可以繼續。
        exclude_indclass_cap: 若為 True，排除需要 placeholder ``indclass`` 或
            ``cap`` 輸入的 alpha。
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
    loaded = [str(a) for a in alphas]
    if exclude_indclass_cap:
        loaded = exclude_indclass_cap_alpha_ids(loaded) or []
    return loaded
