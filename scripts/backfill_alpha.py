"""以 DolphinDB 批次計算並寫入 alpha_features 表。

Usage:
    # MVP v1 預設 15 個近似
    python scripts/backfill_alpha.py --start 2022-01-01 --end 2024-12-31

    # 全 101 個 WQ101 alpha（含 indclass 類——使用 hash-based proxy）
    python scripts/backfill_alpha.py --start 2022-01-01 --end 2024-12-31 --alpha-set all

    # 僅純量價 83 個（跳過 indclass 類）
    python scripts/backfill_alpha.py --start 2022-01-01 --end 2024-12-31 --alpha-set pure
"""

from __future__ import annotations

import argparse
from datetime import date

from src.alpha_engine.batch_compute import BatchAlphaComputer
from src.common.db import get_dolphindb
from src.common.logging import get_logger, setup_logging
from src.config.constants import (
    MVP_V1_ALPHA_IDS,
    WQ101_ALL_ALPHA_IDS,
    WQ101_PURE_PRICE_ALPHA_IDS,
)

setup_logging()
logger = get_logger("backfill_alpha")


def _alpha_set_to_ids(name: str) -> list[int]:
    mapping = {
        "mvp_v1": MVP_V1_ALPHA_IDS,
        "all": WQ101_ALL_ALPHA_IDS,
        "pure": WQ101_PURE_PRICE_ALPHA_IDS,
    }
    if name not in mapping:
        raise ValueError(f"--alpha-set 必須是 mvp_v1/all/pure，收到 {name!r}")
    return [int(a.replace("wq", "").lstrip("0")) for a in mapping[name]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument(
        "--alpha-set", choices=["mvp_v1", "all", "pure"], default="mvp_v1",
        help="mvp_v1=15 個預設 / all=101 個全量 / pure=83 個純量價",
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="寫入前先 DELETE alpha_features（避免上次失敗殘留 / 重跑重複）",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    alpha_ids = _alpha_set_to_ids(args.alpha_set)
    print(f"即將計算 {len(alpha_ids)} 個 alpha（{args.alpha_set}），期間 {start} → {end}")

    if args.truncate:
        client = get_dolphindb()
        try:
            print("Truncating dfs://darams_alpha/alpha_features ...")
            client.run('delete from loadTable("dfs://darams_alpha", "alpha_features")')
            cnt = client.run('exec count(*) from loadTable("dfs://darams_alpha", "alpha_features")')
            print(f"Truncate 完成，殘留列數 = {int(cnt)}")
        finally:
            try:
                client.close()
            except Exception:
                pass

    computer = BatchAlphaComputer()
    try:
        result = computer.compute(start_date=start, end_date=end, alpha_ids=alpha_ids)
        logger.info("backfill_complete", result=result.to_dict(orient="records"))
        print("\n=== 完成 ===")
        print(result.to_string(index=False))
    except Exception as e:
        logger.error("backfill_failed", error=str(e))
        print(f"Backfill failed: {e}")
        print("Ensure DolphinDB is running and wq101alpha module is installed.")
        raise
    finally:
        computer.close()


if __name__ == "__main__":
    main()
