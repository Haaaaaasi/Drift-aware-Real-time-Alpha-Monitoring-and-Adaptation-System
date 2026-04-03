"""Layer 3 — Streaming alpha computation manager (Python side).

TODO (MVP v2): Full implementation with DolphinDB streamEngineParser.
"""

from __future__ import annotations

from src.alpha_engine.dolphindb_client import DolphinDBClient
from src.common.logging import get_logger

logger = get_logger(__name__)


class StreamAlphaComputer:
    """Manage streaming alpha computation via DolphinDB streamEngineParser.

    MVP v2: This will subscribe to standardized_stream and push results
    to alpha_output_stream via the DAG of streaming engines.
    """

    def __init__(self) -> None:
        self._client = DolphinDBClient()
        self._engine_active = False

    def setup_engine(self) -> None:
        """Initialize the streaming alpha engine in DolphinDB."""
        self._client.run('run "/modules/alpha_stream.dos"')
        self._client.run("setupMVPAlphaStream()")
        self._engine_active = True
        logger.info("stream_alpha_engine_started")

    def stop_engine(self) -> None:
        """Stop the streaming engine and unsubscribe."""
        if self._engine_active:
            self._client.run(
                'unsubscribeTable(tableName="standardized_stream", '
                'actionName="darams_alpha_calc")'
            )
            self._engine_active = False
            logger.info("stream_alpha_engine_stopped")

    def push_bars(self, bars_dict: dict) -> None:
        """Push new standardized bars to the DolphinDB stream table."""
        self._client.upload({"new_bars": bars_dict})
        self._client.run("""
            st = objByName("standardized_stream")
            st.append!(new_bars)
        """)

    def close(self) -> None:
        self.stop_engine()
        self._client.close()
