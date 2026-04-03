"""Layer 3 — DolphinDB connection wrapper for alpha computation."""

from __future__ import annotations

import dolphindb as ddb
import pandas as pd

from src.common.logging import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


class DolphinDBClient:
    """Manage DolphinDB sessions and execute alpha computation scripts."""

    def __init__(self) -> None:
        self._settings = get_settings().dolphindb
        self._session: ddb.Session | None = None

    @property
    def session(self) -> ddb.Session:
        if self._session is None or not self._session.isConnected:
            self._session = ddb.Session()
            self._session.connect(
                self._settings.host,
                self._settings.port,
                self._settings.user,
                self._settings.password,
            )
            logger.info("dolphindb_connected")
        return self._session

    def run(self, script: str) -> object:
        """Execute a DolphinDB script and return the result."""
        return self.session.run(script)

    def upload(self, data: dict[str, pd.DataFrame]) -> None:
        """Upload DataFrames to DolphinDB session variables."""
        self.session.upload(data)

    def load_module(self, module_name: str) -> None:
        """Load a DolphinDB module (e.g., wq101alpha)."""
        self.run(f"use {module_name}")
        logger.info("module_loaded", module=module_name)

    def query_table(
        self,
        db_path: str,
        table_name: str,
        where_clause: str = "",
        columns: str = "*",
    ) -> pd.DataFrame:
        """Query a DolphinDB partitioned table and return as DataFrame."""
        script = f't = loadTable("{db_path}", "{table_name}")\n'
        script += f"select {columns} from t"
        if where_clause:
            script += f" where {where_clause}"
        return self.run(script)

    def close(self) -> None:
        if self._session and self._session.isConnected:
            self._session.close()
            logger.info("dolphindb_disconnected")
