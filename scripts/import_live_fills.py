"""匯入 broker/manual fills CSV 並重建 live account PnL。"""

from __future__ import annotations

import argparse
import json

from src.live.execution_service import LiveExecutionService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="成交明細 CSV")
    parser.add_argument("--account-id", default="paper_main")
    parser.add_argument("--price-source", default="broker_import")
    parser.add_argument("--adjustment-mode", default="raw")
    args = parser.parse_args()

    result = LiveExecutionService(account_id=args.account_id).import_fills_csv(
        csv_path=args.csv,
        price_source=args.price_source,
        adjustment_mode=args.adjustment_mode,
    )
    print(json.dumps(result.account_snapshot, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
