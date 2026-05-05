"""Layer 3 — Streaming alpha computation manager (Python side).

Manages the DolphinDB streamEngineParser lifecycle and provides utilities
for replaying historical data through the streaming engine bar-by-bar.

Usage (real mode, requires DolphinDB):
    computer = StreamAlphaComputer()
    computer.setup_engine()
    results = computer.replay_dataframe(bars_df)
    computer.close()
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import pandas as pd

from src.alpha_engine.dolphindb_client import DolphinDBClient
from src.common.logging import get_logger

logger = get_logger(__name__)


class StreamAlphaComputer:
    """Manage streaming alpha computation via DolphinDB streamEngineParser.

    Subscribes to ``standardized_stream`` and collects results from
    ``alpha_output_stream``.  A Python-side handler accumulates emitted
    rows so callers can retrieve them via :meth:`collect_output`.
    """

    def __init__(self) -> None:
        self._client = DolphinDBClient()
        self._engine_active = False
        self._output_buffer: deque[dict] = deque()

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def setup_engine(self) -> None:
        """Initialize the streaming alpha engine in DolphinDB."""
        self._client.run('run "/modules/alpha_stream.dos"')
        self._client.run("setupMVPAlphaStream()")
        self._engine_active = True
        logger.info("stream_alpha_engine_started")

    def subscribe_output(self, callback: Callable[[pd.DataFrame], None] | None = None) -> None:
        """Subscribe Python to DolphinDB alpha_output_stream.

        Rows emitted by the streaming engine are appended to the internal
        ``_output_buffer``.  An optional ``callback`` is also called for
        each batch.
        """
        def _handler(msg: pd.DataFrame) -> None:
            for _, row in msg.iterrows():
                self._output_buffer.append(row.to_dict())
            if callback is not None:
                callback(msg)

        self._client.run(
            'subscribeTable('
            '    tableName="alpha_output_stream",'
            '    actionName="darams_py_collector",'
            '    offset=-1,'
            '    handler=__handler__,'
            '    msgAsTable=true'
            ')'
        )
        # Upload the Python handler into the DolphinDB session so the
        # subscribeTable call can reference it.  We use the ddb Python
        # session's native subscription API instead.
        self._client.session.subscribe(
            host=self._client._settings.host,
            port=self._client._settings.port,
            handler=_handler,
            tableName="alpha_output_stream",
            actionName="darams_py_collector",
            offset=-1,
            resub=True,
            msgAsTable=True,
        )
        logger.info("stream_output_subscribed")

    def stop_engine(self) -> None:
        """Stop the streaming engine and unsubscribe."""
        if self._engine_active:
            try:
                self._client.run(
                    'unsubscribeTable(tableName="standardized_stream",'
                    ' actionName="darams_alpha_calc")'
                )
            except Exception:
                pass
            try:
                self._client.session.unsubscribe(
                    host=self._client._settings.host,
                    port=self._client._settings.port,
                    tableName="alpha_output_stream",
                    actionName="darams_py_collector",
                )
            except Exception:
                pass
            self._engine_active = False
            logger.info("stream_alpha_engine_stopped")

    def close(self) -> None:
        self.stop_engine()
        self._client.close()

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def push_bar_df(self, bar_df: pd.DataFrame) -> None:
        """Push one day's standardized bars into the DolphinDB stream table.

        ``bar_df`` must have columns matching ``standardized_stream``:
        security_id, tradetime, open, high, low, close, vol, vwap, cap, indclass
        """
        self._client.upload({"_push_bars": bar_df})
        self._client.run(
            'objByName("standardized_stream").append!(_push_bars)'
        )

    def replay_dataframe(
        self,
        bars_df: pd.DataFrame,
        delay_ms: int = 0,
    ) -> list[dict]:
        """Replay all bars in ``bars_df`` day-by-day into the stream engine.

        Returns accumulated output rows after all bars have been pushed.

        Parameters
        ----------
        bars_df:
            Full OHLCV panel in long format with a ``tradetime`` column.
        delay_ms:
            Optional inter-bar delay in milliseconds (useful for demos).
        """
        self._output_buffer.clear()
        dates = sorted(bars_df["tradetime"].unique())
        logger.info("replay_starting", n_bars=len(bars_df), n_dates=len(dates))

        for date in dates:
            day_bars = bars_df[bars_df["tradetime"] == date]
            self.push_bar_df(day_bars)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

        # Give the streaming engine a moment to process the last batch
        time.sleep(0.5)
        result = list(self._output_buffer)
        logger.info("replay_complete", output_rows=len(result))
        return result

    # ------------------------------------------------------------------
    # Output collection
    # ------------------------------------------------------------------

    def collect_output(
        self,
        n_expected: int | None = None,
        timeout_sec: float = 10.0,
    ) -> list[dict]:
        """Drain the output buffer, optionally waiting for n_expected rows.

        Blocks until ``n_expected`` rows arrive or ``timeout_sec`` elapses.
        """
        if n_expected is None:
            return list(self._output_buffer)

        deadline = time.time() + timeout_sec
        while len(self._output_buffer) < n_expected and time.time() < deadline:
            time.sleep(0.05)
        return list(self._output_buffer)

    def clear_output(self) -> None:
        self._output_buffer.clear()
