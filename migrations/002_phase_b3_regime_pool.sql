-- ============================================================
-- Phase B-3 — regime_pool quality feedback loop
-- ============================================================
-- 新增 ``last_evaluated_ic`` 欄位儲存最近一次 shadow-eval 對該 regime
-- 對應 model 算出來的 IC（NULL = 尚未被 shadow 評估過）。
--
-- 用途：
-- * 多候選 reuse 時，evaluator 可以比較最新表現
-- * 後續可加入 ``_performance_gate`` 作為動態品質下限
--
-- 對舊資料完全相容：所有既有 row 自動為 NULL，pool 程式碼會 fallback
-- 到 ``performance_summary.rank_ic`` 維持原行為。
-- ============================================================

ALTER TABLE regime_pool
    ADD COLUMN IF NOT EXISTS last_evaluated_ic NUMERIC;

ALTER TABLE regime_pool
    ADD COLUMN IF NOT EXISTS last_evaluated_at TIMESTAMPTZ;

COMMENT ON COLUMN regime_pool.last_evaluated_ic IS
    'Most-recent shadow-evaluation IC of the associated model on a recent matured window. NULL until first reuse evaluation.';
COMMENT ON COLUMN regime_pool.last_evaluated_at IS
    'Timestamp of the most recent shadow-evaluation update.';
