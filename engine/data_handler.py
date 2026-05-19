"""
data_handler.py — Data Handler for the backtesting engine.

Responsibilities:
  - Load market data from Parquet files via the market-data-pipeline
  - Convert pipeline records into engine MarketEvents
  - Enforce strict chronological ordering across all symbols
  - Provide a clean iterator interface to the engine loop

The DataHandler is the sole point of contact between the external
market-data-pipeline and this engine.  No other module loads data.

Parquet schema expected (produced by the pipeline's normalization layer):
  timestamp (datetime64[UTC]), symbol (str), open, high, low,
  close (float64), volume (float64)

Usage:
    handler = DataHandler(symbols=["AAPL", "MSFT"], data_dir="data/")
    for event in handler:
        queue.put(event)
"""

import sys
import os
from typing import List, Iterator

import pandas as pd

from engine.events import MarketEvent, DividendEvent

# ---------------------------------------------------------------------------
# Path bootstrap — allow the engine repo to locate the pipeline package
# even if it has not been installed as a dependency.
# ---------------------------------------------------------------------------
_PIPELINE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "Backtester-Oriented-Market-Data-Pipeline")
)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from market_data.storage import (  # type: ignore[import-not-found]  # noqa: E402
    load_from_parquet,
    get_available_symbols,
    load_dividends_from_parquet,
)


class DataHandler:
    """
    Loads Parquet data for one or more symbols and streams engine MarketEvents
    in strict chronological order.

    Args:
        symbols  : List of ticker symbols to load.  If empty, all symbols
                   found in data_dir are used.
        data_dir : Directory containing <symbol>.parquet files.

    Raises:
        ValueError       : If no data is available for any requested symbol.
        FileNotFoundError: Propagated from the storage layer when a specific
                           symbol file is missing.
    """

    def __init__(
        self,
        symbols:           List[str],
        data_dir:          str  = "data/",
        include_dividends: bool = True,
    ) -> None:
        self._data_dir          = data_dir
        self._symbols           = symbols if symbols else get_available_symbols(data_dir)
        self._include_dividends = include_dividends

        if not self._symbols:
            raise ValueError(
                f"No symbols provided and no Parquet files found in '{data_dir}'."
            )

        self._df:     pd.DataFrame = self._load_and_merge()
        self._div_df: pd.DataFrame = self._load_dividends() if include_dividends else pd.DataFrame()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        """Symbols that were successfully loaded."""
        return list(self._df["symbol"].unique())

    @property
    def bar_count(self) -> int:
        """Total number of MarketEvents (bars) — excludes dividend events."""
        return len(self._df)

    @property
    def dividend_count(self) -> int:
        """Total number of dividend events in the stream."""
        return len(self._div_df) if not self._div_df.empty else 0

    def stream(self) -> Iterator:
        """
        Yield MarketEvents and DividendEvents in strict timestamp order.

        Uses a merge of two sorted sequences — O(n) not O(n log n).
        Dividend events on the same timestamp as a market bar are yielded
        AFTER the market bar so the portfolio has current prices first.
        """
        import heapq

        market_iter = (
            (row.timestamp, 0, MarketEvent(
                timestamp=row.timestamp.to_pydatetime(),
                symbol=row.symbol,
                price=float(row.close),
                volume=float(row.volume),
            ))
            for row in self._df.itertuples(index=False)
        )

        if self._include_dividends and not self._div_df.empty:
            div_iter = (
                (row.timestamp, 1, DividendEvent(
                    timestamp=row.timestamp.to_pydatetime(),
                    symbol=row.symbol,
                    dividend_per_share=float(row.dividend_per_share),
                ))
                for row in self._div_df.itertuples(index=False)
            )
            for _, _, event in heapq.merge(market_iter, div_iter, key=lambda x: (x[0], x[1])):
                yield event
        else:
            for _, _, event in market_iter:
                yield event

    def __iter__(self) -> Iterator:
        return self.stream()

    def __repr__(self) -> str:
        return (
            f"DataHandler(symbols={self._symbols}, "
            f"data_dir='{self._data_dir}', bars={self.bar_count})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_merge(self) -> pd.DataFrame:
        """
        Load each symbol from Parquet, merge into a single DataFrame,
        and sort strictly by timestamp.

        Symbols that are missing from disk are warned and skipped.
        If no symbols load successfully, raises ValueError.
        """
        frames: List[pd.DataFrame] = []

        for symbol in self._symbols:
            try:
                df = load_from_parquet(symbol, self._data_dir)
                frames.append(df)
            except FileNotFoundError:
                print(f"[DataHandler] Warning: no Parquet file for '{symbol}' — skipping.")

        if not frames:
            raise ValueError(
                "DataHandler could not load data for any of the requested symbols."
            )

        merged = pd.concat(frames, ignore_index=True)

        # Guarantee deterministic, chronological order
        merged = merged.sort_values(by=["timestamp", "symbol"], kind="mergesort")
        merged = merged.reset_index(drop=True)

        # Validate required columns are present
        required = {"timestamp", "symbol", "close", "volume"}
        missing  = required - set(merged.columns)
        if missing:
            raise ValueError(
                f"Parquet data is missing required columns: {missing}"
            )

        return merged

    def _load_dividends(self) -> pd.DataFrame:
        """
        Load dividend data for all symbols that have it.
        Symbols without dividend files are silently skipped.
        """
        frames = []
        for symbol in self._symbols:
            try:
                df = load_dividends_from_parquet(symbol, self._data_dir)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                pass  # dividend data is optional — never crash here

        if not frames:
            return pd.DataFrame(columns=["timestamp", "symbol", "dividend_per_share"])

        merged = pd.concat(frames, ignore_index=True)
        return merged.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
