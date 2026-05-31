"""Static IS alpha selector。

這個 selector 將既有 `effective_alphas.json` 清單包成 point-in-time 決策。
行為目標是與 legacy effective-alpha filter 等價，並額外輸出可審計 snapshot。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.alpha_selection.base import (
    AlphaSelectionSnapshot,
    SelectorContext,
    build_snapshot,
    stable_hash,
)
from src.config.alpha_selection import (
    EFFECTIVE_ALPHAS_PATH,
    load_effective_alpha_ids,
)
from src.config.constants import (
    WQ101_ALL_ALPHA_IDS,
    WQ101_INDCLASS_OR_CAP_ALPHA_IDS,
)


@dataclass(frozen=True)
class StaticISSelector:
    """重現目前 TEJ IS-only effective alpha 清單的 selector。"""

    effective_alphas_path: str | Path = EFFECTIVE_ALPHAS_PATH
    alpha_ids: tuple[str, ...] | None = None
    skip_effective_filter: bool = False
    exclude_indclass_cap: bool = False
    selector_version: str = "static_is_v1"

    @property
    def config_hash(self) -> str:
        return stable_hash(
            {
                "selector": "static_is",
                "effective_alphas_path": str(self.effective_alphas_path),
                "alpha_ids": list(self.alpha_ids) if self.alpha_ids else None,
                "skip_effective_filter": self.skip_effective_filter,
                "exclude_indclass_cap": self.exclude_indclass_cap,
                "selector_version": self.selector_version,
            }
        )

    def selected_alpha_ids(self) -> list[str]:
        """回傳與 legacy `_resolve_alpha_ids_for_run` 等價的 feature list。"""
        if self.skip_effective_filter:
            resolved = list(self.alpha_ids) if self.alpha_ids else list(WQ101_ALL_ALPHA_IDS)
        else:
            effective = load_effective_alpha_ids(
                self.effective_alphas_path,
                required=True,
            ) or []
            if self.alpha_ids:
                requested = {str(a) for a in self.alpha_ids}
                resolved = [a for a in effective if a in requested]
            else:
                resolved = effective

        if self.exclude_indclass_cap:
            blocked = set(WQ101_INDCLASS_OR_CAP_ALPHA_IDS)
            resolved = [a for a in resolved if a not in blocked]
        return [str(a) for a in resolved]

    def select(self, context: SelectorContext) -> AlphaSelectionSnapshot:
        selected = set(self.selected_alpha_ids())
        requested = {str(a) for a in self.alpha_ids} if self.alpha_ids else None
        blocked = set(WQ101_INDCLASS_OR_CAP_ALPHA_IDS)

        if self.skip_effective_filter:
            candidate_ids = list(self.alpha_ids) if self.alpha_ids else list(WQ101_ALL_ALPHA_IDS)
            effective_ids = set(candidate_ids)
        else:
            effective_list = load_effective_alpha_ids(
                self.effective_alphas_path,
                required=True,
            ) or []
            effective_ids = set(effective_list)
            candidate_ids = list(effective_list)
            if requested:
                extra_requested = sorted(requested - effective_ids)
                candidate_ids.extend(extra_requested)

        rows = []
        n_selected = len(selected)
        seen: set[str] = set()
        ordered_candidate_ids: list[str] = []
        for candidate in candidate_ids:
            aid = str(candidate)
            if aid not in seen:
                ordered_candidate_ids.append(aid)
                seen.add(aid)

        for aid in ordered_candidate_ids:
            reason = None
            is_selected = aid in selected
            if requested is not None and aid not in requested:
                reason = "not_requested"
            if not self.skip_effective_filter and aid not in effective_ids:
                reason = "not_in_effective_list"
            if self.exclude_indclass_cap and aid in blocked:
                reason = "requires_indclass_or_cap"
            weight = (1.0 / n_selected) if is_selected and n_selected else 0.0
            rows.append(
                {
                    "alpha_id": aid,
                    "selected": bool(is_selected),
                    "weight": weight,
                    "score": 1.0 if is_selected else 0.0,
                    "coverage": None,
                    "rolling_rank_ic": None,
                    "stability": None,
                    "drift_score": None,
                    "turnover_penalty": None,
                    "excluded_reason": reason,
                }
            )

        scores = pd.DataFrame(rows)
        return build_snapshot(
            context=context,
            selector_name="static_is",
            selector_version=self.selector_version,
            scores=scores,
        )
