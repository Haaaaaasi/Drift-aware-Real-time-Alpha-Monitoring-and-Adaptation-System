"""DARAMS — Main entry point.

Usage:
    python main.py api        # Start FastAPI server
    python main.py backtest   # Run MVP v1 backtest with synthetic data
    python main.py monitor    # Run monitoring pipeline
    python main.py adapt      # Run adaptation pipeline
"""

from __future__ import annotations

import sys

from src.common.logging import setup_logging

setup_logging()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "api":
        import uvicorn
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

    elif command == "backtest":
        from pipelines.daily_batch_pipeline import run_backtest
        from datetime import date
        start = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2023, 1, 1)
        end = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2024, 12, 31)
        run_backtest(start, end, use_synthetic=True)

    elif command == "monitor":
        from pipelines.monitoring_pipeline import run_monitoring
        run_monitoring()

    elif command == "adapt":
        from pipelines.adaptation_pipeline import run_adaptation
        run_adaptation()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
