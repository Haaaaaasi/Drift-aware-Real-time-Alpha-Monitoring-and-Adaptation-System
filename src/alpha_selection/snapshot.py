"""Alpha selection snapshot artifact writer。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alpha_selection.base import AlphaSelectionSnapshot


def write_selection_artifacts(
    snapshots: list[AlphaSelectionSnapshot],
    run_dir: str | Path,
) -> dict[str, Path | None]:
    """輸出兩層 snapshot artifact。

    `alpha_selection_snapshots.csv` 是 event-level metadata；
    `alpha_scores.csv` 與 `alpha_weights_by_date.csv` 是 per-alpha 明細。
    """
    run_path = Path(run_dir)
    snapshot_path = run_path / "alpha_selection_snapshots.csv"
    scores_path = run_path / "alpha_scores.csv"
    weights_path = run_path / "alpha_weights_by_date.csv"

    if not snapshots:
        return {
            "snapshots_path": None,
            "scores_path": None,
            "weights_path": None,
        }

    pd.DataFrame([s.event for s in snapshots]).to_csv(snapshot_path, index=False)
    scores = pd.concat([s.scores for s in snapshots], ignore_index=True)
    scores.to_csv(scores_path, index=False)
    weights = scores[scores["selected"]][
        ["as_of_date", "snapshot_hash", "alpha_id", "weight"]
    ].copy()
    weights.to_csv(weights_path, index=False)

    return {
        "snapshots_path": snapshot_path,
        "scores_path": scores_path,
        "weights_path": weights_path,
    }
