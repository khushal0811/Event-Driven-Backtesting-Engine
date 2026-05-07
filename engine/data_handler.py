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

from engine.events import MarketEvent

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

from market_data.storage import load_from_parquet, get_available_symbols  # noqa: E402


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
        symbols: List[str],
        data_dir: str = "data/",
    ) -> None:
        self._data_dir = data_dir
        self._symbols  = symbols if symbols else get_available_symbols(data_dir)

        if not self._symbols:
            raise ValueError(
                f"No symbols provided and no Parquet files found in '{data_dir}'."
            )

        self._df: pd.DataFrame = self._load_and_merge()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        """Symbols that were successfully loaded."""
        return list(self._df["symbol"].unique())

    @property
    def bar_count(self) -> int:
        """Total number of MarketEvents that will be emitted."""
        return len(self._df)

    def stream(self) -> Iterator[MarketEvent]:
        """
        Yield MarketEvents in strict timestamp order.

        This is the primary method consumed by the engine loop.
        Each row in the merged, sorted DataFrame becomes one MarketEvent.
        """
        for row in self._df.itertuples(index=False):
            yield MarketEvent(
                timestamp=row.timestamp.to_pydatetime(),
                symbol=row.symbol,
                price=float(row.close),
                volume=float(row.volume),
            )

    def __iter__(self) -> Iterator[MarketEvent]:
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
