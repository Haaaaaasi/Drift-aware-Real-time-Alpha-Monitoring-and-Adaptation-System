"""Layer 3 — Alpha registry management."""

from __future__ import annotations

import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger

logger = get_logger(__name__)


class AlphaRegistry:
    """Query and manage the alpha_registry table in PostgreSQL."""

    def get_active_alphas(self, category: str | None = None) -> pd.DataFrame:
        """Retrieve all active alpha metadata."""
        conn = get_pg_connection()
        try:
            query = "SELECT * FROM alpha_registry WHERE is_active = TRUE"
            if category:
                query += f" AND category = '{category}'"
            query += " ORDER BY alpha_id"
            return pd.read_sql(query, conn)
        finally:
            conn.close()

    def get_alpha_by_id(self, alpha_id: str) -> dict | None:
        conn = get_pg_connection()
        try:
            df = pd.read_sql(
                "SELECT * FROM alpha_registry WHERE alpha_id = %s",
                conn,
                params=[alpha_id],
            )
            return df.iloc[0].to_dict() if not df.empty else None
        finally:
            conn.close()

    def get_mvp_alphas(self) -> list[str]:
        """Return MVP v1 alpha IDs that don't require industry/cap."""
        conn = get_pg_connection()
        try:
            df = pd.read_sql(
                "SELECT alpha_id FROM alpha_registry "
                "WHERE is_active = TRUE AND requires_industry = FALSE AND requires_cap = FALSE "
                "ORDER BY alpha_id",
                conn,
            )
            return df["alpha_id"].tolist()
        finally:
            conn.close()
