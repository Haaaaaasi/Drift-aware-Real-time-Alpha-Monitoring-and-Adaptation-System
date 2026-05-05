"""Layer 10 — Policy 3: Recurring concept pool and ECPF-like model reuse.

When drift is detected, instead of always retraining from scratch, search a pool of
historical regime-model pairs for a similar past concept and reuse that model.

Similarity 評分（Phase B-1 重構）
-------------------------------
原始 raw cosine 在 5 維非標準化 fingerprint 上，``volume_ratio``（量綱 0.5–2.0）
會主導全部 score，導致任意兩 regime 的 cosine 都 > 0.98——pool 形同失能（每次
都 hit 同一筆 initial entry）。

新評分：

    score = exp(-d_zscored / distance_scale)  *  staleness_factor  *  perf_gate

* ``d_mean``：mean-form z-scored distance = sqrt(Σ((diff/scale)²) / N)；
  語意為「平均每維 std 差距」，與維度數無關（Phase C Option C）。
  pool size ≥ 3 時用 pool std 作 scale；否則用 ``SCALE_PRIORS`` cold-start 兜底。
* ``staleness_factor = exp(-age_days / tau_days)``：太老的 entry 自然衰減
* ``perf_gate``：若 pool entry 的 holdout rank_ic 低於下限直接歸零

Score 落在 (0, 1]，語意：d=0→1、d=1（平均每維 1 std）→0.61、d=2→0.37。
``similarity_threshold`` 預設 0.5，相當於允許平均每維 ~0.69 std 的差異。
加減 fingerprint 維度不需重調 threshold 或 distance_scale。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from src.adaptation.model_registry import ModelRegistryManager
from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.config.constants import AdaptationPolicy

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-dim scale priors（cold-start 用）— Phase B-2 擴充至 15 維
# ---------------------------------------------------------------------------
# 來源：實測 2024 全年 fingerprint 觀察值的 typical-magnitude 估計。
# Pool size ≥ 3 時改用實際 pool std；這只是 cold-start fallback。
#
# 三組維度：
# * 5 個 base dims（Phase A）
# * 4 個 cross-sectional 結構：cs_return_std / skew / kurt / tail_spread
# * 3 個時序結構：vol_of_vol_5d / vol_of_vol_20d / cvar_5pct
# * 3 個 alpha-side：alpha_ic_mean / alpha_ic_std / alpha_ic_pos_fraction
SCALE_PRIORS: dict[str, float] = {
    # -- base
    "volatility": 0.01,
    "autocorrelation": 0.10,
    "avg_cross_correlation": 0.15,
    "trend_strength": 0.001,
    "volume_ratio": 0.30,
    # -- cross-sectional dispersion / shape
    "cs_return_std": 0.015,
    "cs_return_skew": 0.5,
    "cs_return_kurt": 2.0,
    "cs_tail_spread": 0.025,
    # -- temporal structure
    "vol_of_vol_5d": 0.005,
    "vol_of_vol_20d": 0.003,
    "cvar_5pct": 0.020,
    # -- alpha-side（Group 3 透過 alpha_ic_stats 注入；無資料時 fallback 0）
    "alpha_ic_mean": 0.02,
    "alpha_ic_std": 0.03,
    "alpha_ic_pos_fraction": 0.20,
}


def _compute_pool_scales(
    pool: pd.DataFrame,
    keys: list[str],
    min_entries: int = 3,
) -> dict[str, float]:
    """每個 fingerprint 維度的 scale（用作 z-score 分母）。

    Pool entry 數 < ``min_entries`` 時以 ``SCALE_PRIORS`` 兜底，避免少樣本下
    std 估計失準。Pool 維度 std 過小（< 1e-4，等同「pool 內這個維度幾乎沒變化」）
    亦改用 prior 避免被噪聲放大。
    """
    if len(pool) < min_entries:
        return {k: SCALE_PRIORS.get(k, 1.0) for k in keys}
    fps = [
        json.loads(x) if isinstance(x, str) else (x or {})
        for x in pool["fingerprint"].tolist()
    ]
    arr = np.array([[float(fp.get(k, 0.0)) for k in keys] for fp in fps], dtype=float)
    pool_std = arr.std(axis=0, ddof=1)
    out: dict[str, float] = {}
    for i, k in enumerate(keys):
        if pool_std[i] > 1e-4:
            out[k] = float(pool_std[i])
        else:
            out[k] = SCALE_PRIORS.get(k, 1.0)
    return out


def _standardized_distance(
    current_fp: dict[str, float],
    hist_fp: dict[str, float],
    scales: dict[str, float],
) -> float:
    """Mean-form z-scored distance（Option C，維度無關）。

    每維度先除以 scale 再算 Euclidean，最後除以 sqrt(N)，使結果等價於
    「平均每維度的標準化差距」。語意：d=1 表示平均每維差 1 std，與維度數無關。
    未來新增/移除 fingerprint 維度不需重調 distance_scale 或 threshold。
    """
    keys = list(current_fp.keys())
    n = max(len(keys), 1)
    diffs = np.array(
        [
            (float(current_fp[k]) - float(hist_fp.get(k, 0.0))) / max(scales.get(k, 1.0), 1e-8)
            for k in keys
        ],
        dtype=float,
    )
    return float(np.linalg.norm(diffs) / np.sqrt(n))


def _distance_to_similarity(d: float, distance_scale: float = 2.0) -> float:
    """Map [0, ∞) Euclidean distance → (0, 1] similarity via exp decay。

    參考點（``distance_scale=2``）：
    * d = 0 → 1.0      （完全一致）
    * d = 1 → 0.61     （某一維 ~1 std 差或多維小差）
    * d = 2 → 0.37     （某一維 2 std 差或多維中差）
    * d = 4 → 0.135    （明顯不同 regime）
    """
    return float(np.exp(-d / max(distance_scale, 1e-6)))


def _staleness_factor(
    detected_at: datetime,
    now: datetime,
    tau_days: float = 180.0,
) -> float:
    """以 ``exp(-age_days / tau_days)`` 衰減過老的 pool entry。

    ``tau_days=180`` 大約是「一季半之後權重降到 ~0.37」。
    """
    if detected_at is None:
        return 1.0
    age_days = (now - detected_at).total_seconds() / 86400.0
    return float(np.exp(-max(age_days, 0.0) / max(tau_days, 1e-6)))


def _performance_gate(
    perf_summary: dict | None,
    min_rank_ic: float = 0.0,
) -> float:
    """若 entry 的 holdout rank_ic 低於下限直接歸零（不重用爛模型）。"""
    if not perf_summary:
        return 1.0
    val = perf_summary.get("rank_ic")
    if val is None:
        val = perf_summary.get("ic", 0.0)
    try:
        return 1.0 if float(val) >= min_rank_ic else 0.0
    except (TypeError, ValueError):
        return 1.0


# ---------------------------------------------------------------------------
# Phase B-2 — Alpha-side fingerprint helper
# ---------------------------------------------------------------------------

def compute_alpha_ic_stats(
    alpha_panel: pd.DataFrame,
    fwd_returns: pd.Series,
    label_available_at: pd.Series,
    t: pd.Timestamp,
    *,
    window_days: int = 60,
    purge_days: int = 5,
    horizon_days: int = 5,
) -> dict[str, float]:
    """計算最近 ``window_days`` 個（成熟標籤）日內每支 alpha 的 rank IC 的橫截面統計。

    僅使用「``label_available_at <= t``」的成熟標籤窗口，避免 look-ahead bias。

    Args:
        alpha_panel: long format ``[security_id, tradetime, alpha_id, alpha_value]``。
        fwd_returns: forward returns，``MultiIndex(security_id, tradetime)``。
        label_available_at: 與 ``fwd_returns`` 同 index 的 ``Timestamp``，標籤成熟日。
        t: 當前評估時點（trigger time）。
        window_days, purge_days, horizon_days: 評估窗口為
            ``[t - window - purge - horizon, t - purge - horizon]``。

    Returns:
        dict with ``alpha_ic_mean`` / ``alpha_ic_std`` / ``alpha_ic_pos_fraction`` /
        ``n_alphas``，無資料時各值為 0.0。
    """
    end_cut = t - pd.Timedelta(days=purge_days + horizon_days)
    start_cut = end_cut - pd.Timedelta(days=window_days)

    out = {
        "alpha_ic_mean": 0.0,
        "alpha_ic_std": 0.0,
        "alpha_ic_pos_fraction": 0.0,
        "n_alphas": 0,
    }

    if alpha_panel is None or alpha_panel.empty or fwd_returns is None or fwd_returns.empty:
        return out

    try:
        ap_mask = (alpha_panel["tradetime"] > start_cut) & (alpha_panel["tradetime"] <= end_cut)
        ap = alpha_panel.loc[ap_mask, ["security_id", "tradetime", "alpha_id", "alpha_value"]]
        if ap.empty:
            return out

        tradetime_idx = label_available_at.index.get_level_values("tradetime")
        fwd_mask = (
            (label_available_at <= t)
            & (tradetime_idx > start_cut)
            & (tradetime_idx <= end_cut)
        )
        fwd = fwd_returns[fwd_mask]
        if fwd.empty:
            return out

        fwd_df = fwd.reset_index()
        # 統一欄位名以方便 merge
        fwd_df.columns = ["security_id", "tradetime", "fwd_ret"]
        merged = ap.merge(fwd_df, on=["security_id", "tradetime"], how="inner")
        if merged.empty:
            return out

        # 每天每 alpha 的截面 spearman corr，然後對 alpha 取時間平均
        def _daily_corr(group: pd.DataFrame) -> float:
            if len(group) < 5:
                return np.nan
            return group["alpha_value"].corr(group["fwd_ret"], method="spearman")

        daily = merged.groupby(["alpha_id", "tradetime"]).apply(_daily_corr).reset_index(name="ic")
        per_alpha = daily.groupby("alpha_id")["ic"].mean()
        per_alpha = per_alpha[np.isfinite(per_alpha)]
        if per_alpha.empty:
            return out

        out["alpha_ic_mean"] = float(per_alpha.mean())
        out["alpha_ic_std"] = float(per_alpha.std(ddof=1)) if len(per_alpha) > 1 else 0.0
        out["alpha_ic_pos_fraction"] = float((per_alpha > 0).mean())
        out["n_alphas"] = int(len(per_alpha))
        return out
    except Exception as exc:
        logger.warning("compute_alpha_ic_stats_failed", error=str(exc))
        return out


class RecurringConceptPool:
    """Manage a pool of (regime_fingerprint, model) pairs for concept reuse.

    Regime fingerprint = vector of market features:
    (volatility_level, return_autocorr, avg_corr, trend_strength, volume_ratio)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        *,
        distance_scale: float = 2.0,
        staleness_tau_days: float = 180.0,
        min_rank_ic: float = 0.0,
    ) -> None:
        """
        Args:
            similarity_threshold: 0~1 之間的最低 score 門檻。Phase B-1 之後的 score
                等價於「regime 平均距離 ~ 1.4 std 內」，預設 0.5；舊 cosine 0.8 對應到
                新公式約等於 ``exp(-0.45) = 0.64``，建議由 caller 視 pool 規模調整。
            distance_scale: ``exp(-d / distance_scale)`` 的衰減速率。
            staleness_tau_days: pool entry 衰減半衰期（指數）。
            min_rank_ic: ``performance_gate`` 的最低 holdout rank_ic（避免重用爛模型）。
        """
        self._threshold = similarity_threshold
        self._distance_scale = distance_scale
        self._staleness_tau_days = staleness_tau_days
        self._min_rank_ic = min_rank_ic
        self._registry = ModelRegistryManager()

    def compute_regime_fingerprint(
        self,
        market_data: pd.DataFrame,
        alpha_ic_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute a 15-dimensional regime fingerprint.

        Args:
            market_data: Recent standardized_bars [security_id, tradetime, close, vol, ...].
            alpha_ic_stats: 可選的 alpha-side 統計（caller 預先以最近成熟標籤算好），
                需含 ``alpha_ic_mean`` / ``alpha_ic_std`` / ``alpha_ic_pos_fraction``。
                未提供時這 3 維 fallback 0.0。

        Note:
            All output values are guaranteed finite (NaN/Inf are replaced with 0.0)，
            因為 PostgreSQL JSONB 拒絕 ``NaN``、且舊 cosine 相似度遇 NaN 會退回 0
            造成 silent pool miss。NaN 主要來源：constant returns 時 ``autocorr``、
            少於 2 檔股票或同質報酬時 cross-correlation。
        """
        def _safe(x: float, default: float = 0.0) -> float:
            x = float(x)
            return x if np.isfinite(x) else default

        # ------------- base 5 dims（Phase A）-------------
        returns = market_data.groupby("security_id")["close"].pct_change()
        vol = _safe(returns.std())
        autocorr = _safe(returns.autocorr(lag=1)) if len(returns) > 10 else 0.0
        avg_return = _safe(returns.mean())
        volume_ratio = _safe(
            market_data["vol"].tail(5).mean()
            / max(market_data["vol"].mean(), 1e-8)
        )

        # Cross-asset correlation
        if market_data["security_id"].nunique() > 1:
            pivot_ret = market_data.pivot_table(
                index="tradetime", columns="security_id", values="close"
            ).pct_change()
            corr_mat = pivot_ret.corr().values
            if corr_mat.shape[0] > 1:
                tri = corr_mat[np.triu_indices_from(corr_mat, k=1)]
                tri = tri[np.isfinite(tri)]
                avg_corr = _safe(tri.mean()) if len(tri) > 0 else 0.0
            else:
                avg_corr = 0.0
        else:
            avg_corr = 0.0

        # ------------- Group 1: cross-sectional dispersion / shape -------------
        # 對每一個交易日做橫斷面 returns 分布的 moment / tail spread，
        # 再跨日取均值，讓單日噪聲被平滑。
        cs_std = 0.0
        cs_skew = 0.0
        cs_kurt = 0.0
        cs_tail_spread = 0.0
        if market_data["security_id"].nunique() > 1:
            try:
                ret_panel = market_data.pivot_table(
                    index="tradetime", columns="security_id", values="close"
                ).pct_change().iloc[1:]  # 第 1 列 NaN 去掉
                if not ret_panel.empty:
                    cs_std = _safe(ret_panel.std(axis=1).mean())
                    cs_skew = _safe(ret_panel.skew(axis=1).mean())
                    cs_kurt = _safe(ret_panel.kurt(axis=1).mean())
                    # Tail spread：每日 top10% 平均 - bottom10% 平均，再跨日均值
                    def _row_tail_spread(row: pd.Series) -> float:
                        vals = row.dropna().values
                        if len(vals) < 10:
                            return np.nan
                        q = max(1, int(round(len(vals) * 0.1)))
                        sorted_vals = np.sort(vals)
                        return float(sorted_vals[-q:].mean() - sorted_vals[:q].mean())
                    daily_spread = ret_panel.apply(_row_tail_spread, axis=1)
                    cs_tail_spread = _safe(daily_spread.mean())
            except Exception:
                pass  # 任一 moment 計算失敗就維持 0.0

        # ------------- Group 2: temporal structure -------------
        # vol_of_vol：先在每日截面做 std，得單一時間序列 daily_vol；再算 daily_vol 自身
        # 的 5d / 20d rolling std，再對整段時間取均值（兜底）。
        vol_of_vol_5d = 0.0
        vol_of_vol_20d = 0.0
        cvar_5pct = 0.0
        try:
            if market_data["security_id"].nunique() > 1:
                ret_panel = market_data.pivot_table(
                    index="tradetime", columns="security_id", values="close"
                ).pct_change()
                daily_vol = ret_panel.std(axis=1).dropna()
            else:
                daily_vol = returns.dropna()
            if len(daily_vol) >= 5:
                vol_of_vol_5d = _safe(daily_vol.rolling(5).std().mean())
            if len(daily_vol) >= 20:
                vol_of_vol_20d = _safe(daily_vol.rolling(20).std().mean())
            ret_flat = returns.dropna()
            if len(ret_flat) >= 20:
                # CVaR_5%：底端 5% 平均（負值越大代表 tail risk 越重）
                threshold = ret_flat.quantile(0.05)
                cvar_5pct = _safe(ret_flat[ret_flat <= threshold].mean())
        except Exception:
            pass

        # ------------- Group 3: alpha-side（caller 注入）-------------
        if alpha_ic_stats:
            alpha_ic_mean = _safe(alpha_ic_stats.get("alpha_ic_mean", 0.0))
            alpha_ic_std = _safe(alpha_ic_stats.get("alpha_ic_std", 0.0))
            alpha_ic_pos_fraction = _safe(alpha_ic_stats.get("alpha_ic_pos_fraction", 0.0))
        else:
            alpha_ic_mean = 0.0
            alpha_ic_std = 0.0
            alpha_ic_pos_fraction = 0.0

        fingerprint = {
            # base
            "volatility": vol,
            "autocorrelation": autocorr,
            "avg_cross_correlation": avg_corr,
            "trend_strength": avg_return,
            "volume_ratio": volume_ratio,
            # cross-sectional
            "cs_return_std": cs_std,
            "cs_return_skew": cs_skew,
            "cs_return_kurt": cs_kurt,
            "cs_tail_spread": cs_tail_spread,
            # temporal
            "vol_of_vol_5d": vol_of_vol_5d,
            "vol_of_vol_20d": vol_of_vol_20d,
            "cvar_5pct": cvar_5pct,
            # alpha-side
            "alpha_ic_mean": alpha_ic_mean,
            "alpha_ic_std": alpha_ic_std,
            "alpha_ic_pos_fraction": alpha_ic_pos_fraction,
        }
        return fingerprint

    def find_similar_regimes(
        self,
        current_fp: dict[str, float],
        since: Optional[datetime] = None,
        top_k: int = 1,
        return_best_seen: bool = False,
    ) -> list[tuple[str, float]] | tuple[list[tuple[str, float]], float]:
        """Search the regime pool and return up to ``top_k`` candidates ≥ threshold.

        Phase B-1 評分公式：
            score = exp(-d_zscored / distance_scale) * staleness * perf_gate

        Phase B-3：``perf_gate`` 優先使用 ``last_evaluated_ic``（最近 shadow eval），
        無則 fallback 到 ``performance_summary.rank_ic``（訓練時 holdout）。

        Args:
            current_fp: 目前 regime fingerprint。
            since: 若提供，只搜尋 ``detected_at >= since`` 的 entry（限制本次 session）。
            top_k: 至多回傳幾筆。預設 1（與舊 ``find_similar_regime`` 相容）。

        Returns:
            ``[(regime_id, score), ...]``，依 score 由大到小排序，
            僅包含 score ≥ ``self._threshold`` 的 entry。pool 空 / 全部低於門檻時回 ``[]``。
        """
        conn = get_pg_connection()
        try:
            if since is not None:
                pool = pd.read_sql(
                    "SELECT * FROM regime_pool WHERE detected_at >= %s",
                    conn,
                    params=[since],
                )
            else:
                pool = pd.read_sql("SELECT * FROM regime_pool", conn)
        finally:
            conn.close()

        if pool.empty:
            return []

        keys = list(current_fp.keys())
        scales = _compute_pool_scales(pool, keys)
        now = datetime.utcnow()

        scored: list[tuple[str, float, dict[str, float]]] = []

        for _, row in pool.iterrows():
            hist_fp = (
                json.loads(row["fingerprint"])
                if isinstance(row["fingerprint"], str)
                else row["fingerprint"]
            )
            d = _standardized_distance(current_fp, hist_fp, scales)
            raw_sim = _distance_to_similarity(d, distance_scale=self._distance_scale)

            det_at = row["detected_at"]
            # 處理 TIMESTAMPTZ vs naive：統一轉成 naive UTC 比較（pool 內 detected_at
            # 由 add_to_pool 寫入 datetime.utcnow()，Postgres 會加 +00 tz；下行剝掉
            # tzinfo 後 (now - det_at) 才有意義）。
            if det_at is not None and getattr(det_at, "tzinfo", None) is not None:
                det_at = det_at.replace(tzinfo=None)
            stale = _staleness_factor(det_at, now, tau_days=self._staleness_tau_days)

            # Phase B-3：last_evaluated_ic 優先；NULL 時 fallback holdout perf
            last_ic_raw = row.get("last_evaluated_ic")
            try:
                last_ic = float(last_ic_raw) if last_ic_raw is not None else None
                if last_ic is not None and not np.isfinite(last_ic):
                    last_ic = None
            except (TypeError, ValueError):
                last_ic = None
            if last_ic is not None:
                # 直接 gate：last_evaluated_ic >= min_rank_ic
                perf = 1.0 if last_ic >= self._min_rank_ic else 0.0
            else:
                perf_summary = row.get("performance_summary")
                if isinstance(perf_summary, str):
                    try:
                        perf_summary = json.loads(perf_summary)
                    except Exception:
                        perf_summary = None
                perf = _performance_gate(perf_summary, min_rank_ic=self._min_rank_ic)

            score = raw_sim * stale * perf
            scored.append((
                row["regime_id"],
                score,
                {
                    "distance_zscored": d,
                    "raw_sim": raw_sim,
                    "staleness": stale,
                    "perf_gate": perf,
                    "last_eval_ic": last_ic if last_ic is not None else float("nan"),
                },
            ))

        # 由高分排序，取符合 threshold 的前 top_k
        # 額外要求 score > 0：避免 perf_gate 殺成 0 的 entry 在 threshold=0 時偷渡進結果
        scored.sort(key=lambda x: x[1], reverse=True)
        passed = [(rid, s) for (rid, s, _) in scored if s > 0 and s >= self._threshold][:top_k]
        # Phase B-3 診斷：始終追蹤最高分（即便低於 threshold），供 retrain_log / log 顯示
        best_seen = scored[0][1] if scored else 0.0
        if passed:
            top_breakdown = scored[0][2]
            logger.info(
                "similar_regimes_found",
                top_score=round(passed[0][1], 4),
                n_passed=len(passed),
                pool_size=len(pool),
                **{k: round(v, 4) for k, v in top_breakdown.items() if not (
                    isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                )},
            )
        elif scored:
            top_id, top_score, top_breakdown = scored[0]
            logger.info(
                "similar_regime_below_threshold",
                best_id=top_id,
                best_score=round(top_score, 4),
                threshold=self._threshold,
                pool_size=len(pool),
                **{k: round(v, 4) for k, v in top_breakdown.items() if not (
                    isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                )},
            )
        if return_best_seen:
            return passed, float(best_seen)
        return passed

    def find_similar_regime(
        self,
        current_fp: dict[str, float],
        since: Optional[datetime] = None,
    ) -> tuple[str | None, float]:
        """單一候選版（Phase B-1 介面），保留以維持向後相容。

        Returns:
            ``(regime_id, score)`` if 通過 threshold；否則 ``(None, best_score_seen)``。
        """
        results = self.find_similar_regimes(current_fp, since=since, top_k=1)
        if results:
            return results[0]
        # 為了與舊介面相容：即使沒通過 threshold 也回傳「最高 score」供呼叫端紀錄
        # 重新查 pool 算最高分
        conn = get_pg_connection()
        try:
            if since is not None:
                pool = pd.read_sql(
                    "SELECT * FROM regime_pool WHERE detected_at >= %s",
                    conn, params=[since],
                )
            else:
                pool = pd.read_sql("SELECT * FROM regime_pool", conn)
        finally:
            conn.close()
        if pool.empty:
            return None, 0.0
        keys = list(current_fp.keys())
        scales = _compute_pool_scales(pool, keys)
        now = datetime.utcnow()
        best = 0.0
        for _, row in pool.iterrows():
            hist_fp = (
                json.loads(row["fingerprint"])
                if isinstance(row["fingerprint"], str)
                else row["fingerprint"]
            )
            d = _standardized_distance(current_fp, hist_fp, scales)
            raw_sim = _distance_to_similarity(d, distance_scale=self._distance_scale)
            det_at = row["detected_at"]
            if det_at is not None and getattr(det_at, "tzinfo", None) is not None:
                det_at = det_at.replace(tzinfo=None)
            stale = _staleness_factor(det_at, now, tau_days=self._staleness_tau_days)
            score = raw_sim * stale  # 不含 perf_gate（為了讓 caller 能看到「結構相似度」）
            best = max(best, score)
        return None, best

    def add_to_pool(
        self,
        fingerprint: dict[str, float],
        model_id: str,
        alpha_weights: dict[str, float],
        performance_summary: dict[str, float],
    ) -> str:
        """Add a new regime-model pair to the pool."""
        regime_id = f"regime_{uuid4().hex[:8]}"
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO regime_pool
                        (regime_id, detected_at, fingerprint, associated_model_id,
                         associated_alpha_weights, performance_summary)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        regime_id, datetime.utcnow(),
                        json.dumps(fingerprint), model_id,
                        json.dumps(alpha_weights), json.dumps(performance_summary),
                    ),
                )
            conn.commit()
            logger.info("regime_added_to_pool", regime_id=regime_id, model_id=model_id)
            return regime_id
        finally:
            conn.close()

    def get_regime_model(self, regime_id: str) -> dict | None:
        """Retrieve the model/weights associated with a regime."""
        conn = get_pg_connection()
        try:
            df = pd.read_sql(
                "SELECT * FROM regime_pool WHERE regime_id = %s",
                conn,
                params=[regime_id],
            )
            return df.iloc[0].to_dict() if not df.empty else None
        finally:
            conn.close()

    def record_reuse(self, regime_id: str) -> None:
        """Increment the reuse counter for a regime."""
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE regime_pool SET times_reused = times_reused + 1, "
                    "last_reused_at = %s WHERE regime_id = %s",
                    (datetime.utcnow(), regime_id),
                )
            conn.commit()
        finally:
            conn.close()

    def update_last_evaluated_ic(self, regime_id: str, ic: float) -> None:
        """寫入最近一次 shadow-eval 對該 regime 的 IC（Phase B-3 quality feedback）。

        無論 reuse 是否被選中，每次 evaluator 計算過該候選的 IC 都應更新——這形成一條
        持續的 quality 紀錄，後續可用於 ``_performance_gate`` 動態調整門檻。
        """
        if not np.isfinite(ic):
            return
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE regime_pool SET last_evaluated_ic = %s, last_evaluated_at = %s "
                    "WHERE regime_id = %s",
                    (float(ic), datetime.utcnow(), regime_id),
                )
            conn.commit()
        finally:
            conn.close()
