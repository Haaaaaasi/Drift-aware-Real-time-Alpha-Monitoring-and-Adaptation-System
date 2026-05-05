# WP9 訊號診斷摘要

- 期間：2022-06-01 到 2024-12-31
- 資料源：tej
- effective alphas：64
- XGBoost 訓練樣本：343901，features：64
- 最佳 zero-cost portfolio：`xgb_daily_topk_zero_cost`，avg gross 24.932 bps/day，cum 351.16%

## 5 日訊號 IC

- simple signed IC ensemble：0.0504
- XGBoost：0.0124

## 需要解讀的紅旗

- 若 5 日 IC 為正但 daily top-k gross 為負，代表 horizon/portfolio 對齊有問題。
- 若 simple ensemble 勝過 XGBoost，代表 meta model 可能把薄訊號過擬合壞。
- 若 simple decile top-bottom spread 也不明顯，才比較能說 alpha 本身在 TEJ OOS 太弱。