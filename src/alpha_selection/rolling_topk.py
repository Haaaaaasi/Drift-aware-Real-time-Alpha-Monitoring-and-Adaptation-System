"""Point-in-time rolling top-k alpha selector。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.alpha_selection.base import (
    AlphaSelectionSnapshot,
    SelectorContext,
    build_snapshot,
    stable_hash,
)


@dataclass(frozen=True)
class RollingTopKSelector:
    """用成熟 label 的 rolling rank IC 動態挑選 alpha。

    這個 selector 的防線是 `label_available_at <= as_of_date`。即使 alpha / label
    panel 已經包含未來資料，只要 label 尚未 mature，就不能進入 ranking window。
    """

    candidate_alpha_ids: tuple[str, ...]
    top_k: int = 30
    window_days: int = 252
    min_coverage: float = 0.20
    min_observations: int = 1000
    stability_penalty: float = 0.0
    score_metric: str = "abs_rank_ic"
    selector_version: str = "rolling_topk_v1"
    base_alpha_ids: tuple[str, ...] | None = None
    admission_enabled: bool = False
    admission_max_promoted: int = 4
    admission_min_score: float = 0.03
    admission_min_coverage: float | None = None
    admission_min_observations: int | None = None
    admission_subwindows: int = 3
    admission_min_subwindow_passes: int = 2
    admission_subwindow_min_abs_ic: float = 0.01
    admission_max_abs_corr_to_live: float | None = 0.98

    @property
    def config_hash(self) -> str:
        return stable_hash(
            {
                "selector": "rolling_topk",
                "candidate_alpha_ids": list(self.candidate_alpha_ids),
                "top_k": self.top_k,
                "window_days": self.window_days,
                "min_coverage": self.min_coverage,
                "min_observations": self.min_observations,
                "stability_penalty": self.stability_penalty,
                "score_metric": self.score_metric,
                "selector_version": self.selector_version,
                "base_alpha_ids": list(self.base_alpha_ids or []),
                "admission_enabled": self.admission_enabled,
                "admission_max_promoted": self.admission_max_promoted,
                "admission_min_score": self.admission_min_score,
                "admission_min_coverage": self.admission_min_coverage,
                "admission_min_observations": self.admission_min_observations,
                "admission_subwindows": self.admission_subwindows,
                "admission_min_subwindow_passes": self.admission_min_subwindow_passes,
                "admission_subwindow_min_abs_ic": self.admission_subwindow_min_abs_ic,
                "admission_max_abs_corr_to_live": self.admission_max_abs_corr_to_live,
            }
        )

    def select(
        self,
        context: SelectorContext,
        *,
        alpha_panel: pd.DataFrame,
        labels: pd.Series,
        label_available_at: pd.Series,
        previous_selected_alpha_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> AlphaSelectionSnapshot:
        if self.top_k <= 0:
            raise ValueError("rolling_topk top_k 必須大於 0")
        if not self.candidate_alpha_ids:
            raise ValueError("rolling_topk candidate_alpha_ids 不可為空")
        if not 0.0 <= self.stability_penalty <= 1.0:
            raise ValueError("rolling_topk stability_penalty 必須介於 0 與 1 之間")
        if self.admission_enabled:
            if not self.base_alpha_ids:
                raise ValueError("admission gate 需要 base_alpha_ids 作為 incumbent live universe")
            if self.admission_max_promoted < 0:
                raise ValueError("admission_max_promoted 不可為負")
            if self.admission_subwindows <= 0:
                raise ValueError("admission_subwindows 必須大於 0")
            if self.admission_min_subwindow_passes < 0:
                raise ValueError("admission_min_subwindow_passes 不可為負")
            if not 0.0 <= self.admission_subwindow_min_abs_ic <= 1.0:
                raise ValueError("admission_subwindow_min_abs_ic 必須介於 0 與 1 之間")
            if (
                self.admission_max_abs_corr_to_live is not None
                and not 0.0 <= self.admission_max_abs_corr_to_live <= 1.0
            ):
                raise ValueError("admission_max_abs_corr_to_live 必須介於 0 與 1 之間")

        scores = self._score_alphas(
            context=context,
            alpha_panel=alpha_panel,
            labels=labels,
            label_available_at=label_available_at,
            previous_selected_alpha_ids=previous_selected_alpha_ids,
        )
        return build_snapshot(
            context=context,
            selector_name="rolling_topk",
            selector_version=self.selector_version,
            scores=scores,
        )

    def _score_alphas(
        self,
        *,
        context: SelectorContext,
        alpha_panel: pd.DataFrame,
        labels: pd.Series,
        label_available_at: pd.Series,
        previous_selected_alpha_ids: list[str] | tuple[str, ...] | set[str] | None,
    ) -> pd.DataFrame:
        as_of = pd.Timestamp(context.as_of_date)
        score_start = as_of - pd.Timedelta(days=self.window_days)
        if context.train_window_start is not None:
            score_start = max(score_start, pd.Timestamp(context.train_window_start))
        score_end = (
            pd.Timestamp(context.train_window_end)
            if context.train_window_end is not None
            else pd.Timestamp(context.label_cutoff)
        )

        label_dates = labels.index.get_level_values("tradetime")
        mature_mask = (
            (pd.to_datetime(label_available_at) <= as_of)
            & (label_dates >= score_start)
            & (label_dates <= score_end)
        )
        y = labels[mature_mask].dropna()

        candidate_order = {aid: idx for idx, aid in enumerate(self.candidate_alpha_ids)}
        previous_selected = (
            {str(a) for a in previous_selected_alpha_ids}
            if previous_selected_alpha_ids is not None
            else None
        )
        rows: list[dict] = []
        if y.empty:
            for aid in self.candidate_alpha_ids:
                rows.append(
                    self._empty_score_row(
                        aid,
                        "insufficient_history",
                        candidate_order[aid],
                        previous_selected,
                    )
                )
            return pd.DataFrame(rows)

        panel = alpha_panel[
            (alpha_panel["alpha_id"].isin(self.candidate_alpha_ids))
            & (alpha_panel["tradetime"] >= score_start)
            & (alpha_panel["tradetime"] <= score_end)
        ]
        if panel.empty:
            for aid in self.candidate_alpha_ids:
                rows.append(
                    self._empty_score_row(
                        aid,
                        "insufficient_history",
                        candidate_order[aid],
                        previous_selected,
                    )
                )
            return pd.DataFrame(rows)

        wide = panel.pivot_table(
            index=["security_id", "tradetime"],
            columns="alpha_id",
            values="alpha_value",
        )
        common = wide.index.intersection(y.index)
        wide = wide.loc[common]
        y = y.loc[common]
        expected_n = max(len(y), 1)

        for aid in self.candidate_alpha_ids:
            if aid not in wide.columns:
                rows.append(
                    self._empty_score_row(
                        aid,
                        "insufficient_history",
                        candidate_order[aid],
                        previous_selected,
                    )
                )
                continue
            x = wide[aid]
            valid = x.notna() & y.notna()
            n_obs = int(valid.sum())
            coverage = float(n_obs / expected_n)
            if n_obs < self.min_observations:
                rows.append(
                    self._score_row(
                        aid,
                        rank_ic=np.nan,
                        n_observations=n_obs,
                        coverage=coverage,
                        selected=False,
                        weight=0.0,
                        excluded_reason="insufficient_history",
                        order=candidate_order[aid],
                        previously_selected=previous_selected,
                    )
                )
                continue
            if coverage < self.min_coverage:
                rows.append(
                    self._score_row(
                        aid,
                        rank_ic=np.nan,
                        n_observations=n_obs,
                        coverage=coverage,
                        selected=False,
                        weight=0.0,
                        excluded_reason="low_coverage",
                        order=candidate_order[aid],
                        previously_selected=previous_selected,
                    )
                )
                continue
            if x[valid].nunique(dropna=True) < 2 or y[valid].nunique(dropna=True) < 2:
                rows.append(
                    self._score_row(
                        aid,
                        rank_ic=np.nan,
                        n_observations=n_obs,
                        coverage=coverage,
                        selected=False,
                        weight=0.0,
                        excluded_reason="insufficient_variance",
                        order=candidate_order[aid],
                        previously_selected=previous_selected,
                    )
                )
                continue

            rank_ic = float(x[valid].corr(y[valid], method="spearman"))
            rows.append(
                self._score_row(
                    aid,
                    rank_ic=rank_ic,
                    n_observations=n_obs,
                    coverage=coverage,
                    selected=False,
                    weight=0.0,
                    excluded_reason=None,
                    order=candidate_order[aid],
                    previously_selected=previous_selected,
                )
            )

        score_df = pd.DataFrame(rows)
        score_df = self._apply_admission_gate(
            score_df=score_df,
            wide=wide,
            y=y,
            previous_selected=previous_selected,
        )

        valid_scores = score_df["score"].replace([np.inf, -np.inf], np.nan).notna()
        selectable = score_df["admission_status"].isin(["live", "admitted", "open"])
        eligible = score_df[valid_scores & selectable].sort_values(
            ["score", "coverage", "_candidate_order"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selected_ids = set(eligible.head(self.top_k)["alpha_id"].astype(str))
        n_selected = len(selected_ids)
        weights = {aid: (1.0 / n_selected if n_selected else 0.0) for aid in selected_ids}

        score_df["selected"] = score_df["alpha_id"].isin(selected_ids)
        score_df["weight"] = score_df["alpha_id"].map(weights).fillna(0.0)
        score_df.loc[
            (~score_df["selected"]) & score_df["excluded_reason"].isna(),
            "excluded_reason",
        ] = "not_top_k"
        score_df = score_df.sort_values(
            ["selected", "score", "coverage", "_candidate_order"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return score_df.drop(columns=["_candidate_order"])

    def _apply_admission_gate(
        self,
        *,
        score_df: pd.DataFrame,
        wide: pd.DataFrame,
        y: pd.Series,
        previous_selected: set[str] | None,
    ) -> pd.DataFrame:
        """標記 quarantine alpha 是否能進入本次 live selector 候選池。

        Gate 只影響新增 alpha；incumbent live alpha 仍依原本 rolling_topk 分數競爭。
        這樣可以避免 all_valid 一次全量進 live selector，同時保留 point-in-time 升級路徑。
        """
        score_df = score_df.copy()
        base_ids = {str(a) for a in (self.base_alpha_ids or [])}
        candidate_ids = {str(a) for a in self.candidate_alpha_ids}
        quarantine_ids = candidate_ids - base_ids

        score_df["alpha_pool"] = np.where(
            score_df["alpha_id"].isin(base_ids),
            "live",
            np.where(score_df["alpha_id"].isin(quarantine_ids), "quarantine", "candidate"),
        )
        score_df["admission_status"] = "open"
        score_df["admission_score"] = score_df["score"]
        score_df["admission_reason"] = None
        score_df["admission_subwindow_pass_count"] = None
        score_df["max_abs_corr_to_live"] = None

        if not self.admission_enabled:
            return score_df

        score_df.loc[score_df["alpha_pool"] == "live", "admission_status"] = "live"
        score_df.loc[score_df["alpha_pool"] == "live", "admission_reason"] = "incumbent_live"

        if not quarantine_ids:
            return score_df

        min_coverage = (
            self.min_coverage
            if self.admission_min_coverage is None
            else self.admission_min_coverage
        )
        min_observations = (
            self.min_observations
            if self.admission_min_observations is None
            else self.admission_min_observations
        )

        gate_rows: list[dict] = []
        for idx, row in score_df[score_df["alpha_pool"] == "quarantine"].iterrows():
            aid = str(row["alpha_id"])
            reasons: list[str] = []
            if pd.isna(row["score"]):
                reasons.append(str(row["excluded_reason"] or "invalid_score"))
            if int(row["n_observations"]) < min_observations:
                reasons.append("admission_insufficient_history")
            if float(row["coverage"]) < min_coverage:
                reasons.append("admission_low_coverage")
            if pd.isna(row["score"]) or float(row["score"]) < self.admission_min_score:
                reasons.append("admission_low_score")

            subwindow_passes = self._subwindow_pass_count(aid, wide=wide, y=y)
            score_df.at[idx, "admission_subwindow_pass_count"] = subwindow_passes
            if subwindow_passes < self.admission_min_subwindow_passes:
                reasons.append("admission_unstable_subwindows")

            max_corr = self._max_abs_corr_to_live(aid, wide=wide, base_ids=base_ids)
            score_df.at[idx, "max_abs_corr_to_live"] = max_corr
            if (
                self.admission_max_abs_corr_to_live is not None
                and pd.notna(max_corr)
                and float(max_corr) > self.admission_max_abs_corr_to_live
            ):
                reasons.append("admission_redundant_family")

            gate_rows.append(
                {
                    "idx": idx,
                    "alpha_id": aid,
                    "score": float(row["score"]) if pd.notna(row["score"]) else np.nan,
                    "coverage": float(row["coverage"]),
                    "subwindow_passes": subwindow_passes,
                    "max_corr": max_corr,
                    "reasons": reasons,
                }
            )

        passed = [r for r in gate_rows if not r["reasons"]]
        passed = sorted(
            passed,
            key=lambda r: (
                np.nan_to_num(r["score"], nan=-np.inf),
                r["coverage"],
                r["subwindow_passes"],
            ),
            reverse=True,
        )
        admitted_idxs = {r["idx"] for r in passed[: self.admission_max_promoted]}

        for r in gate_rows:
            idx = r["idx"]
            if idx in admitted_idxs:
                score_df.at[idx, "admission_status"] = "admitted"
                score_df.at[idx, "admission_reason"] = "passed_admission_gate"
                continue
            if r["reasons"]:
                reason = ";".join(dict.fromkeys(r["reasons"]))
            else:
                reason = "admission_capacity"
            score_df.at[idx, "admission_status"] = "quarantine"
            score_df.at[idx, "admission_reason"] = reason
            score_df.at[idx, "excluded_reason"] = reason

        return score_df

    def _subwindow_pass_count(self, alpha_id: str, *, wide: pd.DataFrame, y: pd.Series) -> int:
        if alpha_id not in wide.columns or y.empty:
            return 0
        dates = pd.Index(pd.to_datetime(y.index.get_level_values("tradetime")).unique()).sort_values()
        if dates.empty:
            return 0
        windows = [pd.Index(chunk) for chunk in np.array_split(dates.to_numpy(), self.admission_subwindows)]
        passes = 0
        x = wide[alpha_id]
        y_dates = pd.to_datetime(y.index.get_level_values("tradetime"))
        for window_dates in windows:
            if len(window_dates) == 0:
                continue
            mask = pd.Index(y_dates).isin(window_dates)
            xx = x[mask]
            yy = y[mask]
            valid = xx.notna() & yy.notna()
            if int(valid.sum()) < 3:
                continue
            if xx[valid].nunique(dropna=True) < 2 or yy[valid].nunique(dropna=True) < 2:
                continue
            rank_ic = float(xx[valid].corr(yy[valid], method="spearman"))
            if pd.notna(rank_ic) and abs(rank_ic) >= self.admission_subwindow_min_abs_ic:
                passes += 1
        return passes

    def _max_abs_corr_to_live(
        self,
        alpha_id: str,
        *,
        wide: pd.DataFrame,
        base_ids: set[str],
    ) -> float:
        live_cols = [aid for aid in base_ids if aid in wide.columns and aid != alpha_id]
        if alpha_id not in wide.columns or not live_cols:
            return np.nan
        x = wide[alpha_id]
        corr_values: list[float] = []
        for live_id in live_cols:
            y = wide[live_id]
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 3:
                continue
            if x[valid].nunique(dropna=True) < 2 or y[valid].nunique(dropna=True) < 2:
                continue
            corr = float(x[valid].corr(y[valid], method="spearman"))
            if pd.notna(corr):
                corr_values.append(abs(corr))
        if not corr_values:
            return np.nan
        return float(max(corr_values))

    def _empty_score_row(
        self,
        alpha_id: str,
        reason: str,
        order: int,
        previous_selected: set[str] | None = None,
    ) -> dict:
        return self._score_row(
            alpha_id,
            rank_ic=np.nan,
            n_observations=0,
            coverage=0.0,
            selected=False,
            weight=0.0,
            excluded_reason=reason,
            order=order,
            previously_selected=previous_selected,
        )

    def _score_row(
        self,
        alpha_id: str,
        *,
        rank_ic: float,
        n_observations: int,
        coverage: float,
        selected: bool,
        weight: float,
        excluded_reason: str | None,
        order: int,
        previously_selected: set[str] | None,
    ) -> dict:
        stability = None if previously_selected is None else float(alpha_id in previously_selected)
        if np.isnan(rank_ic):
            raw_score = np.nan
            score = np.nan
            turnover_penalty = None
        elif self.score_metric == "abs_rank_ic":
            raw_score = abs(rank_ic) * coverage
            if previously_selected is not None and alpha_id not in previously_selected:
                score = raw_score * (1.0 - self.stability_penalty)
                turnover_penalty = raw_score - score
            else:
                score = raw_score
                turnover_penalty = 0.0
        else:
            raise ValueError(f"不支援的 rolling_topk score_metric: {self.score_metric!r}")
        return {
            "alpha_id": str(alpha_id),
            "selected": bool(selected),
            "weight": float(weight),
            "raw_score": raw_score,
            "score": score,
            "n_observations": int(n_observations),
            "coverage": float(coverage),
            "rolling_rank_ic": rank_ic,
            "stability": stability,
            "drift_score": None,
            "turnover_penalty": turnover_penalty,
            "alpha_pool": None,
            "admission_status": None,
            "admission_score": None,
            "admission_reason": None,
            "admission_subwindow_pass_count": None,
            "max_abs_corr_to_live": None,
            "excluded_reason": excluded_reason,
            "_candidate_order": int(order),
        }
