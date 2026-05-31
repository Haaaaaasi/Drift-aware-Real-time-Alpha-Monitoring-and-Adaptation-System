---
name: Alpha cache bar alignment / terminal exposure diagnostic（2026-05-13）
description: terminal exposure diagnostic 抓到 alpha cache 未對齊實際 bar keys，導致下市前股票在 OOS 後仍被選入持倉；已修補並啟動 cache-aligned frozen validation。
type: project
originSessionId: codex-2026-05-13
---

# Alpha cache bar alignment / terminal exposure diagnostic（2026-05-13）

## 發現

原本的 terminal exposure diagnostic 是為了檢查 `next_return = NaN -> 0` 是否會低估下市或資料終端風險；但它先抓到更嚴重的資料對齊漏洞：

- 舊 frozen OOS 結果中，`scheduled_20` 會持有已在 OOS 前結束交易的股票，例如 1592。
- 根因不是單純的「下市日報酬歸 0」假設，而是 alpha cache 讀取路徑只用 `tradetime` / `alpha_id` 篩選，沒有再與當次 bars 的 `(security_id, tradetime)` 實際 key 對齊。
- 因此，TEJ bars 已沒有該股票交易列時，cache 中的 stale alpha row 仍可能被模型選入 portfolio；後續 `next_return` 取不到值而被視為 0。
- 這會把不存在的持倉變成「零報酬凍結資產」，污染 OOS 績效與 terminal exposure 判斷。

## 修補

`src/alpha_engine/alpha_cache.py` 已新增 `_align_to_bar_keys(alpha_panel, bars)`：

- 每次 cache slice / cache hit / recompute 後，都把 alpha rows inner join 到當次 bars 的 `(security_id, tradetime)`。
- 若 cache rows 被裁掉，logger 會輸出 `alpha_panel_aligned_to_bars`，包含 rows_before / rows_after。
- 新增 unit test：`tests/unit/test_alpha_cache.py::test_cache_slice_aligns_to_requested_bar_keys`，確認 cache 中不存在於 bars 的股票日期會被剔除。

驗證：

- `pytest tests/unit/test_alpha_cache.py tests/unit/test_placebo_signal.py tests/unit/test_exit_discipline.py -q`：14 passed。
- TEJ cache smoke：1592 在 OOS 期間 stale alpha rows 已降為 0，1592 alpha 最大日期回到 2022-01-26。

## 對既有結果的影響

所有修補前的 tail cleanup frozen OOS 結果都應視為 suspect，不可拿來做正式 claim。之後只使用 run tag 含 `cachealign` 的 frozen OOS 結果。

已完成的 cache-aligned frozen preliminary：

- `next_vwap`：`scheduled_20` cum +22.337%、Sharpe 0.587；`none` +1.680%、Sharpe 0.164；`ew_buy_hold_universe` +4.178%、Sharpe 0.220。
- `next_open`：`scheduled_20` cum +36.606%、Sharpe 0.771；`none` +11.649%、Sharpe 0.361；`ew_buy_hold_universe` +10.618%、Sharpe 0.403。
- terminal exposure diagnostic：`scheduled_20` / `none` 在 `next_vwap` 與 `next_open` 的 true terminal exposure 皆為 0；剩下 missing rows 來自 dataset end boundary。`ew_buy_hold_universe` 有小量 true terminal exposure，max daily 約 0.006737。

## 仍在執行

過夜 workflow：

`reports/adaptation_ab/frozen_oos_validation_20260512/`

流程順序：

1. cache-aligned frozen `next_vwap`
2. cache-aligned frozen `next_open`
3. terminal exposure diagnostic
4. placebo shuffled signal：`next_vwap` 30 seeds、`next_open` 10 seeds
5. benchmark sensitivity：same cadence / liquidity-filtered EW benchmark

placebo 與 benchmark sensitivity 完成前，scheduled_20 的正式 claim 仍不可寫成強結論。

## 最終結果（2026-05-13 07:09 完成）

輸出目錄：

`reports/adaptation_ab/frozen_oos_validation_20260512/`

### Terminal exposure

- `scheduled_20` / `none` 在 `next_vwap` 與 `next_open` 的 true terminal exposure 都是 0。
- 剩下 missing rows 都是 dataset end boundary，因此「策略績效靠下市或缺報酬歸 0」的疑慮可關掉。
- `ew_buy_hold_universe` 有 7 檔 true terminal stocks，max daily true terminal exposure 約 0.006737，屬 benchmark 端的小量殘餘敏感度。

### Placebo shuffled signal

- `next_vwap`：real cum +22.337%、Sharpe 0.587；30 個 shuffled placebo 平均 cum -2.752%、Sharpe -0.003，95th percentile 分別只有 +7.143% / 0.306。
- `next_open`：real cum +36.606%、Sharpe 0.771；10 個 shuffled placebo 平均 cum +3.274%、Sharpe 0.183，95th percentile 分別只有 +13.281% / 0.469。
- 結論：pipeline 不是自帶正報酬；真實 signal 的 return / Sharpe 明顯高於 shuffled signal null。但 real drawdown 仍比 placebo 深，不能 claim 風險面優越。

### Benchmark sensitivity

- `next_vwap`：scheduled_20 +22.337%、Sharpe 0.587；same-cadence EW +4.298%、Sharpe 0.224；liquidity-filtered EW +19.585%、Sharpe 0.563。
- `next_open`：scheduled_20 +36.606%、Sharpe 0.771；same-cadence EW +10.834%、Sharpe 0.410；liquidity-filtered EW +27.291%、Sharpe 0.671。
- 結論：scheduled_20 對原始 EW 與 same-cadence EW 的 claim 可以寫得比較強；但對 liquidity-filtered EW 只能寫成小幅改善，尤其 `next_vwap` 幾乎接近，且 scheduled_20 drawdown 較深。

### Reviewer-facing claim

可寫：

`在 TEJ survivorship-correct OOS、移除 indclass/cap alpha、T+1 open/VWAP execution、cache-aligned alpha rows、terminal exposure cleared、shuffled-signal placebo passed 的設定下，scheduled_20 的真實 signal 優於 no-adapt、原始 EW 與 same-cadence EW；相對 liquidity-filtered EW 仍有正向但較弱的增益。`

不可寫：

`scheduled_20 已穩健顯著擊敗所有 benchmark` 或 `風險調整後全面優於 benchmark`。目前 drawdown 較深，且 liquidity-filtered EW 已吃掉大量 edge。

## 正式接手文件同步

2026-05-13 已同步更新 `AGENTS.md` 與 `CLAUDE.md`：

- 將「目前進度」更新到 frozen OOS baseline。
- 移除主文中 Phase A 高累積報酬表作為正式結論的呈現，改標為 deprecated 歷史。
- 明確指定後續 model_pool 必須以 `scheduled_20` incumbent baseline 作為主要比較對象。
- 新增 alpha cache 必須對齊當次 bars keys 的不可違反規則。
- 更新下市處理規則，說明 terminal exposure diagnostic 已確認 `scheduled_20` / `none` true terminal exposure 為 0。
