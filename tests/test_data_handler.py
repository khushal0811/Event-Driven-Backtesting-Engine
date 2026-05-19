"""
Group 2 — DataHandler (T07–T14)

Data loading, chronological ordering, and dividend interleaving.
All tests in this group require real Parquet data on disk.
"""

import pytest
from engine.data_handler import DataHandler
from engine.events import MarketEvent, DividendEvent


# -----------------------------------------------------------------------
# T07 — DataHandler loads a single symbol correctly
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT07:
    def test_single_symbol_loads(self, data_dir, has_aapl_data):
        handler = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        assert "AAPL" in handler.symbols
        assert handler.bar_count > 0


# -----------------------------------------------------------------------
# T08 — DataHandler loads multiple symbols correctly
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT08:
    def test_multi_symbol_loads(self, data_dir, has_aapl_data, has_msft_data):
        single = DataHandler(symbols=["AAPL"], data_dir=data_dir)
        multi  = DataHandler(symbols=["AAPL", "MSFT"], data_dir=data_dir)
        assert "AAPL" in multi.symbols and "MSFT" in multi.symbols
        assert multi.bar_count > single.bar_count


# -----------------------------------------------------------------------
# T09 — Events come out in strict chronological order
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT09:
    def test_chronological_order(self, data_dir, has_aapl_data, has_msft_data):
        handler = DataHandler(symbols=["AAPL", "MSFT"], data_dir=data_dir)
        events = list(handler.stream())
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp, (
                f"Out of order at index {i}: "
                f"{events[i-1].timestamp} > {events[i].timestamp}"
            )


# -----------------------------------------------------------------------
# T10 — DividendEvents appear in the stream when dividends exist
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT10:
    def test_dividend_events_present(self, data_dir, has_aapl_data):
        handler = DataHandler(
            symbols=["AAPL"], data_dir=data_dir, include_dividends=True
        )
        events = list(handler.stream())
        divs = [e for e in events if isinstance(e, DividendEvent)]
        assert len(divs) > 0, "AAPL should have at least one dividend"
        for d in divs:
            assert d.dividend_per_share > 0
            assert d.timestamp is not None


# -----------------------------------------------------------------------
# T11 — DividendEvents come AFTER MarketEvents on the same timestamp
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT11:
    def test_dividend_after_market_on_same_timestamp(self, data_dir, has_aapl_data):
        handler = DataHandler(
            symbols=["AAPL"], data_dir=data_dir, include_dividends=True
        )
        events = list(handler.stream())

        # Group events by timestamp
        from collections import defaultdict
        by_ts = defaultdict(list)
        for e in events:
            by_ts[e.timestamp].append(e)

        for ts, group in by_ts.items():
            types = [type(e).__name__ for e in group]
            if "MarketEvent" in types and "DividendEvent" in types:
                # Find indices
                market_idx = next(i for i, e in enumerate(group) if isinstance(e, MarketEvent))
                div_idx    = next(i for i, e in enumerate(group) if isinstance(e, DividendEvent))
                assert market_idx < div_idx, (
                    f"At {ts}: MarketEvent (idx={market_idx}) must come before "
                    f"DividendEvent (idx={div_idx})"
                )


# -----------------------------------------------------------------------
# T12 — DataHandler with include_dividends=False has zero DividendEvents
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT12:
    def test_no_dividends_when_disabled(self, data_dir, has_aapl_data):
        handler = DataHandler(
            symbols=["AAPL"], data_dir=data_dir, include_dividends=False
        )
        events = list(handler.stream())
        divs = [e for e in events if isinstance(e, DividendEvent)]
        assert len(divs) == 0


# -----------------------------------------------------------------------
# T13 — DataHandler raises error for missing symbol
# -----------------------------------------------------------------------
class TestT13:
    def test_missing_symbol_raises(self, data_dir):
        with pytest.raises((FileNotFoundError, ValueError)):
            DataHandler(symbols=["FAKESYMBOL999"], data_dir=data_dir)


# -----------------------------------------------------------------------
# T14 — DataHandler bar_count only counts MarketEvents not DividendEvents
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT14:
    def test_bar_count_excludes_dividends(self, data_dir, has_aapl_data):
        handler = DataHandler(
            symbols=["AAPL"], data_dir=data_dir, include_dividends=True
        )
        events = list(handler.stream())
        market_count = sum(1 for e in events if isinstance(e, MarketEvent))
        div_count    = sum(1 for e in events if isinstance(e, DividendEvent))

        assert handler.bar_count == market_count
        assert len(events) == market_count + div_count
        assert handler.dividend_count == div_count
