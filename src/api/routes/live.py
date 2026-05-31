"""Live daily operating layer API routes."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.live.order_units import add_order_unit_columns, expand_to_shioaji_order_rows

router = APIRouter()

RECOMMENDATION_STATUSES = {"PENDING", "APPROVED", "EXPORTED", "EXECUTED", "CANCELED"}
EXPORT_DIR = Path("reports/live_exports")
TEJ_NAME_RE = re.compile(r"^(\d{4})\s+(.+)$")


class RecommendationStatusRequest(BaseModel):
    """買賣建議狀態更新請求。"""

    run_id: str | None = None
    security_ids: list[str] | None = None
    actions: list[str] | None = None
    from_status: str | None = "PENDING"
    status: Literal["PENDING", "APPROVED", "CANCELED"]
    official_only: bool = True


class RecommendationExportRequest(BaseModel):
    """買賣建議匯出請求。"""

    run_id: str | None = None
    statuses: list[str] = Field(default_factory=lambda: ["APPROVED"])
    actions: list[str] | None = None
    mark_exported: bool = True
    official_only: bool = True


class PaperFillRequest(BaseModel):
    """Paper execution request for approved recommendations."""

    run_id: str | None = None
    account_id: str = "paper_main"
    price_source: str = "paper_next_vwap"
    adjustment_mode: str = "raw"
    official_only: bool = True


@router.get("/runs")
def list_live_runs(
    limit: int = Query(30, ge=1, le=500),
    official_only: bool = Query(True),
):
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        where = "WHERE is_official = true" if official_only else ""
        df = pd.read_sql(
            """
            SELECT run_id, as_of_date, mode, run_purpose, is_official,
                   status, data_source, data_max_date, data_lag_days,
                   data_freshness_status,
                   production_model_id, feature_columns_hash, n_feature_alphas,
                   retrain_action, selector_snapshot_hash,
                   diagnostic_selector_snapshot_hash, run_started_at, run_finished_at
            FROM daily_live_runs
            {where}
            ORDER BY as_of_date DESC, run_started_at DESC
            LIMIT %s
            """.format(where=where),
            conn,
            params=[limit],
        )
        return df.to_dict(orient="records")
    finally:
        conn.close()


@router.post("/execution/paper-fill")
def paper_fill_trade_recommendations(request: PaperFillRequest):
    """將 approved recommendations 轉成 paper fills 並重建 account PnL。"""

    from src.common.db import get_pg_connection
    from src.live.execution_service import LiveExecutionService

    conn = get_pg_connection()
    try:
        run_id = request.run_id or _latest_run_id(conn, official_only=request.official_only)
    finally:
        conn.close()
    if run_id is None:
        raise HTTPException(status_code=404, detail="No live run found")

    result = LiveExecutionService(account_id=request.account_id).paper_fill_run(
        run_id=run_id,
        price_source=request.price_source,
        adjustment_mode=request.adjustment_mode,
    )
    return {
        "run_id": run_id,
        "account_id": request.account_id,
        "positions": len(result.positions),
        "executed_recommendations": len(result.executed_recommendation_ids),
        "account_snapshot": _jsonable_row(result.account_snapshot),
    }


@router.get("/account/snapshot/latest")
def get_latest_account_snapshot(
    account_id: str = Query("paper_main"),
    official_only: bool = Query(True),
):
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        join_run = "JOIN daily_live_runs r ON r.run_id = s.run_id" if official_only else ""
        where_official = "AND r.is_official = true" if official_only else ""
        df = pd.read_sql(
            """
            SELECT s.*
            FROM live_account_snapshots s
            {join_run}
            WHERE s.account_id = %s
              {where_official}
            ORDER BY s.as_of_date DESC, s.snapshot_time DESC
            LIMIT 1
            """.format(join_run=join_run, where_official=where_official),
            conn,
            params=[account_id],
        )
        if df.empty:
            return {
                "account_id": account_id,
                "status": "UNAVAILABLE",
                "message": "No account snapshot found",
            }
        row = df.iloc[0].to_dict()
        row["status"] = "AVAILABLE"
        return _jsonable_row(row)
    finally:
        conn.close()


@router.get("/console")
def get_live_console_state(
    limit: int = Query(20, ge=1, le=100),
    official_only: bool = Query(True),
):
    """回傳 Web console 需要的最新 live 狀態。"""

    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        where_run = "WHERE is_official = true" if official_only else ""
        latest_run = pd.read_sql(
            """
            SELECT *
            FROM daily_live_runs
            {where_run}
            ORDER BY as_of_date DESC, run_started_at DESC
            LIMIT 1
            """.format(where_run=where_run),
            conn,
        )
        runs = pd.read_sql(
            """
            SELECT run_id, as_of_date, mode, run_purpose, is_official, status,
                   data_source, data_max_date, data_lag_days,
                   data_freshness_status, production_model_id,
                   n_feature_alphas, retrain_action, run_started_at,
                   run_finished_at
            FROM daily_live_runs
            {where_run}
            ORDER BY as_of_date DESC, run_started_at DESC
            LIMIT %s
            """.format(where_run=where_run),
            conn,
            params=[limit],
        )
        server_now = datetime.utcnow()
        if latest_run.empty:
            return {
                "server_time": server_now.isoformat() + "Z",
                "run": None,
                "performance": None,
                "recommendation_summary": [],
                "recommendations": [],
                "holdings": [],
                "production_alphas": [],
                "diagnostic_alphas": [],
                "runs": _records(runs),
            }

        run = latest_run.iloc[0].to_dict()
        run = _with_live_freshness(run, server_now=server_now)
        run_id = run["run_id"]
        recommendation_summary = pd.read_sql(
            """
            SELECT action, count(*) AS n,
                   sum(abs(delta_weight)) AS gross_delta_weight
            FROM trade_recommendations
            WHERE run_id = %s
            GROUP BY action
            ORDER BY action
            """,
            conn,
            params=[run_id],
        )
        recommendations = pd.read_sql(
            """
            SELECT security_id, action, current_weight, target_weight,
                   delta_weight, current_shares, target_shares, delta_shares,
                   last_price, signal_score, rank, reason, status
            FROM trade_recommendations
            WHERE run_id = %s
            ORDER BY
                CASE action
                    WHEN 'BUY' THEN 1
                    WHEN 'INCREASE' THEN 2
                    WHEN 'REDUCE' THEN 3
                    WHEN 'SELL' THEN 4
                    ELSE 5
                END,
                abs(delta_weight) DESC
            LIMIT 200
            """,
            conn,
            params=[run_id],
        )
        holdings = pd.read_sql(
            """
            SELECT security_id, target_weight, target_shares, last_price,
                   market_value, unrealized_pnl, signal_score, rank,
                   holding_days, reason
            FROM portfolio_snapshots
            WHERE run_id = %s
            ORDER BY target_weight DESC, security_id
            LIMIT 200
            """,
            conn,
            params=[run_id],
        )
        recommendations = _attach_security_names(recommendations)
        recommendations = add_order_unit_columns(recommendations)
        holdings = _attach_security_names(holdings)
        performance = _load_console_performance(conn, run)
        production_alphas = _load_console_alphas(conn, run_id, "production")
        diagnostic_alphas = _load_console_alphas(conn, run_id, "diagnostic")
        return {
            "server_time": server_now.isoformat() + "Z",
            "run": _jsonable_row(run),
            "performance": _jsonable_row(performance),
            "recommendation_summary": _records(recommendation_summary),
            "recommendations": _records(recommendations),
            "holdings": _records(holdings),
            "production_alphas": _records(production_alphas),
            "diagnostic_alphas": _records(diagnostic_alphas),
            "runs": _records(runs),
        }
    finally:
        conn.close()


@router.get("/state/current")
def get_current_live_state(official_only: bool = Query(True)):
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        where = "WHERE is_official = true" if official_only else ""
        run = pd.read_sql(
            """
            SELECT *
            FROM daily_live_runs
            {where}
            ORDER BY as_of_date DESC, run_started_at DESC
            LIMIT 1
            """.format(where=where),
            conn,
        )
        if run.empty:
            return {"message": "No live run found"}
        run_id = run.iloc[0]["run_id"]
        holdings = pd.read_sql(
            """
            SELECT security_id, target_weight, target_shares, last_price,
                   market_value, signal_score, rank, holding_days, reason
            FROM portfolio_snapshots
            WHERE run_id = %s
            ORDER BY target_weight DESC
            """,
            conn,
            params=[run_id],
        )
        recommendations = pd.read_sql(
            """
            SELECT action, count(*) AS n,
                   sum(abs(delta_weight)) AS gross_delta_weight
            FROM trade_recommendations
            WHERE run_id = %s
            GROUP BY action
            ORDER BY action
            """,
            conn,
            params=[run_id],
        )
        return {
            "run": run.iloc[0].to_dict(),
            "holdings": holdings.to_dict(orient="records"),
            "recommendation_summary": recommendations.to_dict(orient="records"),
        }
    finally:
        conn.close()


@router.get("/recommendations/latest")
def get_latest_trade_recommendations(
    actions: str | None = Query(None, description="Comma separated actions, e.g. BUY,SELL"),
    official_only: bool = Query(True),
):
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        where_run = "WHERE is_official = true" if official_only else ""
        run = pd.read_sql(
            """
            SELECT run_id
            FROM daily_live_runs
            {where_run}
            ORDER BY as_of_date DESC, run_started_at DESC
            LIMIT 1
            """.format(where_run=where_run),
            conn,
        )
        if run.empty:
            return []
        params: list = [run.iloc[0]["run_id"]]
        where = "run_id = %s"
        if actions:
            requested = [a.strip().upper() for a in actions.split(",") if a.strip()]
            where += " AND action = ANY(%s)"
            params.append(requested)
        df = pd.read_sql(
            f"""
            SELECT *
            FROM trade_recommendations
            WHERE {where}
            ORDER BY
                CASE action
                    WHEN 'BUY' THEN 1
                    WHEN 'INCREASE' THEN 2
                    WHEN 'REDUCE' THEN 3
                    WHEN 'SELL' THEN 4
                    ELSE 5
                END,
                abs(delta_weight) DESC
            """,
            conn,
            params=params,
        )
        df = _attach_security_names(df)
        df = add_order_unit_columns(df)
        return df.to_dict(orient="records")
    finally:
        conn.close()


@router.post("/recommendations/status")
def update_trade_recommendation_status(request: RecommendationStatusRequest):
    """更新 latest 或指定 run 的 trade recommendation status。"""

    from src.common.db import get_pg_connection

    target_status = request.status.upper()
    if target_status not in {"PENDING", "APPROVED", "CANCELED"}:
        raise HTTPException(status_code=400, detail=f"Invalid target status: {request.status}")
    from_status = request.from_status.upper() if request.from_status else None
    if from_status is not None and from_status not in RECOMMENDATION_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid from_status: {request.from_status}")
    actions = _normalize_optional_list(request.actions)
    security_ids = _normalize_optional_list(request.security_ids)

    conn = get_pg_connection()
    try:
        run_id = request.run_id or _latest_run_id(conn, official_only=request.official_only)
        if run_id is None:
            raise HTTPException(status_code=404, detail="No live run found")

        clauses = ["run_id = %s"]
        params: list = [run_id]
        if from_status is not None:
            clauses.append("status = %s")
            params.append(from_status)
        if actions:
            clauses.append("action = ANY(%s)")
            params.append(actions)
        if security_ids:
            clauses.append("security_id = ANY(%s)")
            params.append(security_ids)

        sql = f"""
            UPDATE trade_recommendations
            SET status = %s, updated_at = now()
            WHERE {' AND '.join(clauses)}
            RETURNING recommendation_id, security_id, action, status
        """
        with conn.cursor() as cur:
            cur.execute(sql, [target_status, *params])
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
        conn.commit()
        updated = [dict(zip(columns, row)) for row in rows]
        return {
            "run_id": run_id,
            "target_status": target_status,
            "updated_count": len(updated),
            "updated": updated,
        }
    finally:
        conn.close()


@router.post("/recommendations/export")
def export_trade_recommendations(request: RecommendationExportRequest):
    """匯出 approved trade recommendations 成 CSV，並可標記為 EXPORTED。"""

    import pandas as pd
    from src.common.db import get_pg_connection

    statuses = [status.upper() for status in request.statuses]
    invalid = [status for status in statuses if status not in RECOMMENDATION_STATUSES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid statuses: {invalid}")
    actions = _normalize_optional_list(request.actions)

    conn = get_pg_connection()
    try:
        run_id = request.run_id or _latest_run_id(conn, official_only=request.official_only)
        if run_id is None:
            raise HTTPException(status_code=404, detail="No live run found")

        clauses = ["run_id = %s", "status = ANY(%s)", "action <> 'HOLD'"]
        params: list = [run_id, statuses]
        if actions:
            clauses.append("action = ANY(%s)")
            params.append(actions)
        df = pd.read_sql(
            f"""
            SELECT recommendation_id, run_id, as_of_date, security_id, action,
                   current_weight, target_weight, delta_weight,
                   current_shares, target_shares, delta_shares, last_price,
                   signal_score, rank, reason, status
            FROM trade_recommendations
            WHERE {' AND '.join(clauses)}
            ORDER BY
                CASE action
                    WHEN 'BUY' THEN 1
                    WHEN 'INCREASE' THEN 2
                    WHEN 'REDUCE' THEN 3
                    WHEN 'SELL' THEN 4
                    ELSE 5
                END,
                abs(delta_weight) DESC
            """,
            conn,
            params=params,
        )
        if df.empty:
            raise HTTPException(status_code=400, detail="No matching recommendations to export")

        df = _attach_security_names(df)
        export_df = _build_export_frame(df)
        if export_df.empty:
            raise HTTPException(status_code=400, detail="No non-zero share orders to export")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_path = EXPORT_DIR / f"trade_recommendations_{run_id}_{stamp}.csv"
        export_df.to_csv(export_path, index=False, encoding="utf-8-sig")

        marked_count = 0
        if request.mark_exported:
            ids = df["recommendation_id"].astype(int).tolist()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE trade_recommendations
                    SET status = 'EXPORTED', updated_at = now()
                    WHERE recommendation_id = ANY(%s)
                      AND status = ANY(%s)
                    """,
                    [ids, statuses],
                )
                marked_count = cur.rowcount
            conn.commit()

        return {
            "run_id": run_id,
            "export_path": str(export_path.as_posix()),
            "rows": len(export_df),
            "marked_exported": marked_count,
            "download_url": f"/api/v1/live/recommendations/export/file?path={export_path.as_posix()}",
            "gross_delta_weight": float(df["delta_weight"].abs().sum()),
        }
    finally:
        conn.close()


@router.get("/recommendations/export/file")
def download_trade_recommendation_export(path: str = Query(...)):
    """下載由 export endpoint 產生的 CSV。"""

    export_path = Path(path)
    allowed_root = EXPORT_DIR.resolve()
    try:
        resolved = export_path.resolve()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Export file not found") from exc
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise HTTPException(status_code=400, detail="Export path is outside reports/live_exports")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    return FileResponse(
        resolved,
        media_type="text/csv",
        filename=resolved.name,
    )


@router.get("/holdings/latest")
def get_latest_holdings(official_only: bool = Query(True)):
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        where_run = "WHERE is_official = true" if official_only else ""
        df = pd.read_sql(
            """
            SELECT ps.*
            FROM portfolio_snapshots ps
            JOIN (
                SELECT run_id
                FROM daily_live_runs
                {where_run}
                ORDER BY as_of_date DESC, run_started_at DESC
                LIMIT 1
            ) r ON r.run_id = ps.run_id
            ORDER BY ps.target_weight DESC
            """.format(where_run=where_run),
            conn,
        )
        df = _attach_security_names(df)
        return df.to_dict(orient="records")
    finally:
        conn.close()


@router.get("/alpha/latest")
def get_latest_alpha_config(
    role: str = Query("production"),
    official_only: bool = Query(True),
):
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        where_run = "WHERE is_official = true" if official_only else ""
        snapshot = pd.read_sql(
            """
            SELECT s.*
            FROM alpha_selection_snapshots s
            JOIN (
                SELECT run_id
                FROM daily_live_runs
                {where_run}
                ORDER BY as_of_date DESC, run_started_at DESC
                LIMIT 1
            ) r ON r.run_id = s.run_id
            WHERE s.snapshot_role = %s
            ORDER BY s.created_at DESC
            LIMIT 1
            """.format(where_run=where_run),
            conn,
            params=[role],
        )
        if snapshot.empty:
            return {"message": f"No {role} alpha snapshot found"}
        row = snapshot.iloc[0]
        scores = pd.read_sql(
            """
            SELECT alpha_id, selected, weight, score, rolling_rank_ic,
                   coverage, stability, excluded_reason
            FROM alpha_selection_scores
            WHERE run_id = %s
              AND snapshot_hash = %s
              AND snapshot_role = %s
            ORDER BY selected DESC, score DESC NULLS LAST, alpha_id
            """,
            conn,
            params=[row["run_id"], row["snapshot_hash"], role],
        )
        return {
            "snapshot": row.to_dict(),
            "scores": scores.to_dict(orient="records"),
        }
    finally:
        conn.close()


def _load_console_alphas(conn, run_id: str, role: str):
    import pandas as pd

    return pd.read_sql(
        """
        SELECT alpha_id, selected, weight, score, rolling_rank_ic,
               coverage, stability, excluded_reason
        FROM alpha_selection_scores
        WHERE run_id = %s
          AND snapshot_role = %s
        ORDER BY selected DESC, score DESC NULLS LAST, alpha_id
        LIMIT 200
        """,
        conn,
        params=[run_id, role],
    )


def _load_console_performance(conn, run: dict) -> dict:
    """計算目前 live console 可防守的累積報酬資訊。"""

    import pandas as pd

    run_id = str(run["run_id"])
    account_id = "paper_main"
    try:
        account = pd.read_sql(
            """
            SELECT *
            FROM live_account_snapshots
            WHERE account_id = %s
              AND (run_id = %s OR as_of_date <= %s)
            ORDER BY
                CASE WHEN run_id = %s THEN 0 ELSE 1 END,
                as_of_date DESC,
                snapshot_time DESC
            LIMIT 1
            """,
            conn,
            params=[account_id, run_id, run.get("as_of_date"), run_id],
        )
        if not account.empty:
            row = account.iloc[0].to_dict()
            official_window = pd.read_sql(
                """
                SELECT min(as_of_date) AS start_as_of,
                       max(as_of_date) AS latest_as_of,
                       count(*) AS n_official_runs
                FROM daily_live_runs
                WHERE is_official = true
                """,
                conn,
            ).iloc[0].to_dict()
            return {
                "status": "AVAILABLE",
                "basis": "live_account_snapshots",
                "account_id": account_id,
                "cumulative_return": _clean_number(row.get("cumulative_return")),
                "daily_return": _clean_number(row.get("daily_return")),
                "capital": None,
                "cash": _clean_number(row.get("cash")),
                "realized_pnl": _clean_number(row.get("realized_pnl")),
                "unrealized_pnl": _clean_number(row.get("unrealized_pnl")),
                "market_value": _clean_number(row.get("market_value")),
                "total_equity": _clean_number(row.get("total_equity")),
                "price_source": row.get("price_source"),
                "adjustment_mode": row.get("adjustment_mode"),
                "n_positions": None,
                "pnl_rows": None,
                "start_as_of": official_window.get("start_as_of"),
                "latest_as_of": official_window.get("latest_as_of"),
                "n_official_runs": int(official_window.get("n_official_runs") or 0),
                "message": "以 live_account_snapshots 的 account equity curve 計算。",
            }
    except Exception:
        # 尚未套用 006 migration 時維持舊版 defensive fallback。
        pass

    metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
    capital = metadata.get("capital")
    if capital is None:
        capital = 0.0
    capital = float(capital or 0.0)
    summary = pd.read_sql(
        """
        SELECT count(*) AS n_positions,
               count(unrealized_pnl) AS pnl_rows,
               sum(unrealized_pnl) AS unrealized_pnl,
               sum(market_value) AS market_value
        FROM portfolio_snapshots
        WHERE run_id = %s
        """,
        conn,
        params=[run_id],
    ).iloc[0].to_dict()
    official_window = pd.read_sql(
        """
        SELECT min(as_of_date) AS start_as_of,
               max(as_of_date) AS latest_as_of,
               count(*) AS n_official_runs
        FROM daily_live_runs
        WHERE is_official = true
        """,
        conn,
    ).iloc[0].to_dict()
    pnl_rows = int(summary.get("pnl_rows") or 0)
    unrealized_pnl = summary.get("unrealized_pnl")
    if pnl_rows > 0 and capital > 0 and unrealized_pnl is not None:
        cumulative_return = float(unrealized_pnl) / capital
        status = "AVAILABLE"
        basis = "unrealized_pnl_over_capital"
        message = "以 portfolio_snapshots.unrealized_pnl 除以 live capital 計算。"
    else:
        estimate = _load_target_mtm_estimate(conn, run, capital)
        if estimate is not None:
            return {
                **estimate,
                "start_as_of": official_window.get("start_as_of"),
                "latest_as_of": official_window.get("latest_as_of"),
                "n_official_runs": int(official_window.get("n_official_runs") or 0),
            }
        cumulative_return = None
        status = "UNAVAILABLE"
        basis = "execution_pnl_not_available"
        message = "尚未寫入成交或未實現損益，不能計算真實 live 累積報酬率。"
    return {
        "status": status,
        "basis": basis,
        "cumulative_return": cumulative_return,
        "capital": capital,
        "unrealized_pnl": None if unrealized_pnl is None else float(unrealized_pnl),
        "market_value": _clean_number(summary.get("market_value")),
        "n_positions": int(summary.get("n_positions") or 0),
        "pnl_rows": pnl_rows,
        "start_as_of": official_window.get("start_as_of"),
        "latest_as_of": official_window.get("latest_as_of"),
        "n_official_runs": int(official_window.get("n_official_runs") or 0),
        "message": message,
    }


def _load_target_mtm_estimate(conn, run: dict, capital: float) -> dict | None:
    """用上一筆 official target holdings 與本次 as-of 價格估算尚未成交回饋的 M2M。"""

    import pandas as pd

    if capital <= 0:
        return None
    run_id = str(run["run_id"])
    as_of_date = run.get("as_of_date")
    previous_run = pd.read_sql(
        """
        SELECT run_id, as_of_date
        FROM daily_live_runs
        WHERE is_official = true
          AND as_of_date < %s
        ORDER BY as_of_date DESC, run_finished_at DESC
        LIMIT 1
        """,
        conn,
        params=[as_of_date],
    )
    if previous_run.empty:
        return None

    previous = pd.read_sql(
        """
        SELECT security_id, target_weight, target_shares, last_price
        FROM portfolio_snapshots
        WHERE run_id = %s
        """,
        conn,
        params=[str(previous_run.iloc[0]["run_id"])],
    )
    if previous.empty:
        return None

    recommendation_prices = pd.read_sql(
        """
        SELECT security_id, last_price
        FROM trade_recommendations
        WHERE run_id = %s
        """,
        conn,
        params=[run_id],
    )
    snapshot_prices = pd.read_sql(
        """
        SELECT security_id, last_price
        FROM portfolio_snapshots
        WHERE run_id = %s
        """,
        conn,
        params=[run_id],
    )
    current_prices = pd.concat(
        [recommendation_prices, snapshot_prices],
        ignore_index=True,
    ).dropna(subset=["last_price"])
    if current_prices.empty:
        return None
    current_prices = current_prices.drop_duplicates("security_id", keep="first")
    current_prices = current_prices.rename(columns={"last_price": "current_price"})

    merged = previous.merge(current_prices, on="security_id", how="left")
    missing_prices = int(merged["current_price"].isna().sum())
    merged = merged.dropna(subset=["current_price", "last_price"])
    if merged.empty:
        return None

    target_weight = pd.to_numeric(merged["target_weight"])
    target_shares = pd.to_numeric(merged["target_shares"])
    entry_price = pd.to_numeric(merged["last_price"])
    current_price = pd.to_numeric(merged["current_price"])
    position_returns = current_price / entry_price - 1.0
    weighted_target_return = float((target_weight * position_returns).sum())
    initial_cost = float((target_shares * entry_price).sum())
    end_market_value = float((target_shares * current_price).sum())
    cash_residual = float(capital - initial_cost)
    total_equity = end_market_value + cash_residual
    cumulative_return = float(total_equity / capital - 1.0)
    message = (
        "尚未寫入成交 / account snapshot；此為上一筆 official target holdings "
        "以本次 as-of 價格估算的 M2M，非真實 execution PnL。"
    )
    if missing_prices:
        message += f" 有 {missing_prices} 檔缺少本次價格，已排除。"
    return {
        "status": "ESTIMATED",
        "basis": "target_portfolio_mark_to_market",
        "cumulative_return": cumulative_return,
        "weighted_target_return": weighted_target_return,
        "capital": capital,
        "unrealized_pnl": float(total_equity - capital),
        "market_value": end_market_value,
        "total_equity": total_equity,
        "cash_residual": cash_residual,
        "n_positions": int(len(merged)),
        "pnl_rows": 0,
        "previous_as_of": previous_run.iloc[0]["as_of_date"],
        "previous_run_id": str(previous_run.iloc[0]["run_id"]),
        "message": message,
    }


def _latest_run_id(conn, *, official_only: bool) -> str | None:
    import pandas as pd

    where = "WHERE is_official = true" if official_only else ""
    df = pd.read_sql(
        """
        SELECT run_id
        FROM daily_live_runs
        {where}
        ORDER BY as_of_date DESC, run_started_at DESC
        LIMIT 1
        """.format(where=where),
        conn,
    )
    if df.empty:
        return None
    return str(df.iloc[0]["run_id"])


def _normalize_optional_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    cleaned = [str(value).strip().upper() for value in values if str(value).strip()]
    return cleaned or None


def _build_export_frame(df):
    import pandas as pd

    export = expand_to_shioaji_order_rows(df)
    export_columns = [
        "run_id",
        "as_of_date",
        "security_id",
        "security_name",
        "order_side",
        "quantity",
        "share_quantity",
        "quantity_unit",
        "shioaji_order_lot",
        "shioaji_quantity",
        "shioaji_quantity_unit",
        "last_price",
        "notional",
        "action",
        "current_weight",
        "target_weight",
        "delta_weight",
        "current_shares",
        "target_shares",
        "delta_shares",
        "signal_score",
        "rank",
        "reason",
        "status",
    ]
    if export.empty:
        return pd.DataFrame(columns=export_columns)
    if "security_name" not in export.columns:
        export["security_name"] = None
    export["order_side"] = export["action"].map(
        {
            "BUY": "BUY",
            "INCREASE": "BUY",
            "SELL": "SELL",
            "REDUCE": "SELL",
        }
    ).fillna(export["action"])
    export["quantity"] = export["share_quantity"].fillna(0).astype(int)
    export["notional"] = export["share_quantity"] * export["last_price"].fillna(0.0)
    export = export[export_columns]
    return export


def _clean_number(value):
    import pandas as pd

    if pd.isna(value):
        return None
    return float(value)


def _attach_security_names(df):
    if df.empty or "security_id" not in df.columns:
        return df
    out = df.copy()
    name_map = _security_name_map()
    out["security_name"] = out["security_id"].astype(str).map(name_map)
    return out


@lru_cache(maxsize=1)
def _security_name_map() -> dict[str, str]:
    """從 TEJ 原始檔與 security_master 建立股號到中文名稱的 lookup。"""

    names: dict[str, str] = {}
    names.update(_security_names_from_db())
    names.update(_security_names_from_tej_csv())
    return names


def _security_names_from_tej_csv() -> dict[str, str]:
    import pandas as pd

    names: dict[str, str] = {}
    for path in sorted(Path(".").glob("OHLSV*.csv")):
        try:
            raw = pd.read_csv(path, encoding="utf-16-le", sep="\t", usecols=[0])
        except Exception:
            continue
        first_col = raw.iloc[:, 0].dropna().astype(str)
        for value in first_col.drop_duplicates():
            match = TEJ_NAME_RE.match(value.strip())
            if match:
                security_id, security_name = match.groups()
                names[security_id] = security_name.strip()
    return names


def _security_names_from_db() -> dict[str, str]:
    import pandas as pd
    from src.common.db import get_pg_connection

    try:
        conn = get_pg_connection()
    except Exception:
        return {}
    try:
        df = pd.read_sql(
            "SELECT security_id, name FROM security_master",
            conn,
        )
    except Exception:
        return {}
    finally:
        conn.close()
    if df.empty:
        return {}
    return {
        str(row["security_id"]): str(row["name"])
        for _, row in df.dropna(subset=["security_id", "name"]).iterrows()
    }


def _with_live_freshness(row: dict, *, server_now: datetime) -> dict:
    """為 console 即時計算資料落後天數，避免沿用 run 產生當天寫入的靜態 lag。"""

    import pandas as pd

    out = dict(row)
    as_of = out.get("as_of_date")
    data_max = out.get("data_max_date") or as_of
    as_of_day = pd.Timestamp(as_of).date() if as_of is not None else None
    data_max_day = pd.Timestamp(data_max).date() if data_max is not None else as_of_day
    reference_day = data_max_day or as_of_day
    if reference_day is None:
        out["live_data_lag_days"] = None
        out["live_data_freshness_status"] = "UNKNOWN"
        return out

    lag_days = max(0, (server_now.date() - reference_day).days)
    if lag_days <= 3:
        status = "FRESH"
    else:
        status = "STALE"
    out["live_data_lag_days"] = lag_days
    out["live_data_freshness_status"] = status
    out["stored_data_lag_days"] = out.get("data_lag_days")
    out["stored_data_freshness_status"] = out.get("data_freshness_status")
    return out


def _records(df):
    return [_jsonable_row(row) for row in df.to_dict(orient="records")]


def _jsonable_row(row: dict):
    from datetime import date, datetime
    from decimal import Decimal
    import pandas as pd

    out = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            out[key] = value
        elif isinstance(value, pd.Timestamp):
            out[key] = value.isoformat()
        elif isinstance(value, (datetime, date)):
            out[key] = value.isoformat()
        elif isinstance(value, Decimal):
            out[key] = float(value)
        elif pd.isna(value):
            out[key] = None
        else:
            out[key] = value
    return out
