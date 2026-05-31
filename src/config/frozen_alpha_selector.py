"""Frozen alpha selector config loader.

本模組只負責把 reviewer-facing frozen YAML 轉成 pipeline 參數覆寫值。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"frozen config section {name!r} must be a mapping")
    return value


def _scheduled_retrain_every(policy: str | None) -> int | None:
    if not policy:
        return None
    prefix = "scheduled_"
    if policy.startswith(prefix):
        try:
            return int(policy[len(prefix):])
        except ValueError as exc:
            raise ValueError(f"invalid scheduled policy: {policy!r}") from exc
    return None


@dataclass(frozen=True)
class FrozenAlphaSelectorSpec:
    """Parsed frozen selector specification."""

    path: Path
    data: dict[str, Any]
    file_hash: str

    @property
    def frozen_selector_id(self) -> str:
        return str(self.data.get("frozen_selector_id", ""))

    def execution_price(self, mode: str = "primary") -> str:
        execution = _section(self.data, "execution")
        primary = str(execution.get("primary_price", "next_vwap"))
        secondary = str(execution.get("secondary_price", "next_open"))
        normalized = str(mode or "primary").lower()
        if normalized in {"primary", "primary_price"}:
            return primary
        if normalized in {"secondary", "secondary_price"}:
            return secondary
        if normalized in {primary, secondary}:
            return normalized
        raise ValueError(
            f"frozen_execution must be primary, secondary, {primary!r}, or {secondary!r}; "
            f"got {mode!r}"
        )

    def simulation_overrides(self, execution_mode: str = "primary") -> dict[str, Any]:
        data = _section(self.data, "data")
        universe = _section(self.data, "alpha_universe")
        selector = _section(self.data, "selector")
        model = _section(self.data, "model")
        portfolio = _section(self.data, "portfolio")
        costs = _section(self.data, "costs")
        labels = _section(self.data, "labels")
        retrain_every = _scheduled_retrain_every(str(model.get("adaptation_policy", "")))

        overrides: dict[str, Any] = {
            "csv_path": Path(str(data["bars_path"])),
            "data_source": str(data.get("data_source", "tej")),
            "allow_yfinance": False,
            "selector": str(selector["name"]),
            "selector_alpha_top_k": int(selector["alpha_top_k"]),
            "selector_window_days": int(selector["ranking_window_days"]),
            "selector_min_coverage": float(selector["min_coverage"]),
            "selector_min_observations": int(selector["min_observations"]),
            "selector_stability_penalty": float(selector["stability_penalty"]),
            "selector_admission_gate": bool(selector.get("admission_gate_enabled", False)),
            "alpha_ids": None,
            "skip_effective_filter": False,
            "exclude_indclass_cap_alphas": bool(
                universe.get("exclude_indclass_cap_alphas", True)
            ),
            "top_k": int(portfolio["top_k"]),
            "portfolio_method": str(portfolio["method"]),
            "rebalance_every": int(portfolio["rebalance_every"]),
            "entry_rank": int(portfolio["entry_rank"]),
            "exit_rank": int(portfolio["exit_rank"]),
            "max_turnover": float(portfolio["max_turnover"]),
            "min_holding_days": int(portfolio["min_holding_days"]),
            "tail_cleanup_weight": float(portfolio.get("tail_cleanup_weight", 0.0)),
            "objective": str(portfolio.get("objective", "net_return_proxy")),
            "execution_price": self.execution_price(execution_mode),
            "commission_rate": float(costs["commission_rate_per_side"]),
            "tax_rate": float(costs["tax_rate_sell_side"]),
            "slippage_bps": float(costs["slippage_bps_per_side"]),
            "round_trip_cost_pct": None,
            "horizon_days": int(labels["horizon_days"]),
            "purge_days": int(labels["purge_days"]),
        }
        if retrain_every is not None:
            overrides["retrain_every"] = retrain_every
        if "train_window_days" in model:
            overrides["train_window_days"] = int(model["train_window_days"])
        return overrides

    def metadata(self, execution_mode: str = "primary") -> dict[str, Any]:
        return {
            "frozen_config_path": str(self.path),
            "frozen_config_hash": self.file_hash,
            "frozen_selector_id": self.frozen_selector_id,
            "frozen_execution": execution_mode,
            "frozen_execution_price": self.execution_price(execution_mode),
        }


def load_frozen_alpha_selector(path: str | Path) -> FrozenAlphaSelectorSpec:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"frozen alpha selector config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"frozen alpha selector config must be a mapping: {p}")
    return FrozenAlphaSelectorSpec(path=p, data=data, file_hash=_file_sha256(p))
