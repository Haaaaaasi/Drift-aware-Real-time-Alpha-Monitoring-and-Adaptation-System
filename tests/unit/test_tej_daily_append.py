from __future__ import annotations

import pandas as pd

from src.ingestion.tej_daily_append import append_tej_daily_files


def _write_tej_csv(path, rows: list[tuple[str, str, float, float, float, float, int]]) -> None:
    df = pd.DataFrame(
        rows,
        columns=["證券", "年月日", "開盤價", "最高價", "最低價", "收盤價", "成交量千股"],
    )
    df.to_csv(path, sep="\t", encoding="utf-16-le", index=False)


def _existing_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": ["1101", "2330"],
            "datetime": pd.to_datetime(["2026-04-30", "2026-04-30"]),
            "open": [40.0, 800.0],
            "high": [41.0, 810.0],
            "low": [39.5, 795.0],
            "close": [40.5, 805.0],
            "volume": [1_000_000, 2_000_000],
        }
    )


def test_append_tej_daily_adds_new_date_and_updates_universe(tmp_path) -> None:
    output = tmp_path / "tw_stocks_tej.parquet"
    universe = tmp_path / "tw_stocks_tej_universe.parquet"
    _existing_bars().to_parquet(output, index=False)

    incoming = tmp_path / "tej_20260501.csv"
    _write_tej_csv(
        incoming,
        [
            ("1101 台泥", "20260501", 41.0, 42.0, 40.0, 41.5, 1200),
            ("2330 台積電", "20260501", 806.0, 820.0, 805.0, 818.0, 2500),
            ("ABCD 測試", "20260501", 1.0, 1.0, 1.0, 1.0, 1),
        ],
    )

    result = append_tej_daily_files(
        [incoming],
        output_path=output,
        universe_output_path=universe,
        backup_dir=None,
    )

    bars = pd.read_parquet(output)
    bounds = pd.read_parquet(universe)
    assert result.added_keys == 2
    assert result.overlap_keys == 0
    assert result.output_max_date == "2026-05-01"
    assert len(bars) == 4
    assert set(bars["security_id"]) == {"1101", "2330"}
    assert len(bounds) == 2


def test_append_tej_daily_overlap_uses_new_values(tmp_path) -> None:
    output = tmp_path / "tw_stocks_tej.parquet"
    universe = tmp_path / "tw_stocks_tej_universe.parquet"
    _existing_bars().to_parquet(output, index=False)

    incoming = tmp_path / "tej_fix.csv"
    _write_tej_csv(
        incoming,
        [("2330 台積電", "20260430", 801.0, 812.0, 799.0, 811.0, 3000)],
    )

    result = append_tej_daily_files(
        [incoming],
        output_path=output,
        universe_output_path=universe,
        backup_dir=None,
    )

    bars = pd.read_parquet(output).set_index(["security_id", "datetime"])
    assert result.added_keys == 0
    assert result.overlap_keys == 1
    assert bars.loc[("2330", pd.Timestamp("2026-04-30")), "close"] == 811.0
    assert bars.loc[("2330", pd.Timestamp("2026-04-30")), "volume"] == 3_000_000


def test_append_tej_daily_dry_run_does_not_write(tmp_path) -> None:
    output = tmp_path / "tw_stocks_tej.parquet"
    universe = tmp_path / "tw_stocks_tej_universe.parquet"
    _existing_bars().to_parquet(output, index=False)

    incoming = tmp_path / "tej_20260501.csv"
    _write_tej_csv(
        incoming,
        [("1101 台泥", "20260501", 41.0, 42.0, 40.0, 41.5, 1200)],
    )

    result = append_tej_daily_files(
        [incoming],
        output_path=output,
        universe_output_path=universe,
        backup_dir=None,
        dry_run=True,
    )

    bars = pd.read_parquet(output)
    assert result.dry_run is True
    assert result.added_keys == 1
    assert bars["datetime"].max() == pd.Timestamp("2026-04-30")
    assert not universe.exists()
