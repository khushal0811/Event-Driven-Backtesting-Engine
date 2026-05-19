"""
Group 3 — Portfolio (T15–T28)

Cash, positions, dividends, equity curve — the financial heart of the engine.
"""

import pytest
from datetime import datetime, timedelta

from engine.portfolio import Portfolio, PortfolioSnapshot
from engine.events import (
    FillEvent, MarketEvent, OrderSide, DividendEvent,
)


def _market(symbol, price, ts):
    return MarketEvent(timestamp=ts, symbol=symbol, price=price, volume=10000.0)


def _fill(symbol, side, qty, price, ts):
    return FillEvent(
        symbol=symbol, side=side, quantity=qty,
        fill_price=price, timestamp=ts,
    )


def _div(symbol, dps, ts):
    return DividendEvent(timestamp=ts, symbol=symbol, dividend_per_share=dps)


NOW = datetime(2024, 6, 1, 10, 0, 0)


# -----------------------------------------------------------------------
# T15 — Portfolio starts with correct cash and zero positions
# -----------------------------------------------------------------------
class TestT15:
    def test_initial_state(self):
        p = Portfolio(initial_cash=100_000.0)
        assert p.cash == 100_000.0
        assert p.positions == {}
        assert p.holdings_value == 0.0
        assert p.total_value == 100_000.0
        assert p.total_dividend_income == 0.0
        assert p.bar_history == []
        assert p.history == []


# -----------------------------------------------------------------------
# T16 — Portfolio rejects zero or negative initial cash
# -----------------------------------------------------------------------
class TestT16:
    def test_zero_cash_rejected(self):
        with pytest.raises(ValueError):
            Portfolio(initial_cash=0)

    def test_negative_cash_rejected(self):
        with pytest.raises(ValueError):
            Portfolio(initial_cash=-50_000)


# -----------------------------------------------------------------------
# T17 — BUY fill reduces cash and increases position correctly
# -----------------------------------------------------------------------
class TestT17:
    def test_buy_fill(self):
        p = Portfolio(initial_cash=100_000.0)
        p.on_fill_event(_fill("AAPL", OrderSide.BUY, 100, 150.0, NOW))
        assert p.cash == 85_000.0
        assert p.positions["AAPL"] == 100


# -----------------------------------------------------------------------
# T18 — SELL fill increases cash and decreases position correctly
# -----------------------------------------------------------------------
class TestT18:
    def test_sell_fill(self):
        p = Portfolio(initial_cash=100_000.0)
        p.on_fill_event(_fill("AAPL", OrderSide.BUY, 100, 150.0, NOW))
        p.on_fill_event(_fill("AAPL", OrderSide.SELL, 100, 160.0, NOW))
        assert p.cash == 101_000.0
        assert p.positions["AAPL"] == 0


# -----------------------------------------------------------------------
# T19 — Holdings value updates correctly on market price change
# -----------------------------------------------------------------------
class TestT19:
    def test_holdings_update(self):
        p = Portfolio(initial_cash=100_000.0)
        p.on_fill_event(_fill("AAPL", OrderSide.BUY, 100, 150.0, NOW))
        p.update_market_value(_market("AAPL", 200.0, NOW))
        assert p.holdings_value == 20_000.0
        assert p.total_value == 85_000.0 + 20_000.0


# -----------------------------------------------------------------------
# T20 — bar_history records one snapshot per MarketEvent
# -----------------------------------------------------------------------
class TestT20:
    def test_bar_history_count(self):
        p = Portfolio(initial_cash=100_000.0)
        for i in range(5):
            ts = NOW + timedelta(days=i)
            p.update_market_value(_market("AAPL", 150.0 + i, ts))
        assert len(p.bar_history) == 5


# -----------------------------------------------------------------------
# T21 — fill_history records only on fills not on every bar
# -----------------------------------------------------------------------
class TestT21:
    def test_fill_history_count(self):
        p = Portfolio(initial_cash=100_000.0)
        # 10 market events, 2 fills mixed in
        for i in range(10):
            ts = NOW + timedelta(days=i)
            p.update_market_value(_market("AAPL", 150.0 + i, ts))
            if i in (3, 7):
                p.on_fill_event(_fill("AAPL", OrderSide.BUY, 10, 150.0 + i, ts))
        assert len(p.history) == 2


# -----------------------------------------------------------------------
# T22 — Dividend income is credited correctly for long positions
# -----------------------------------------------------------------------
class TestT22:
    def test_dividend_credited(self):
        p = Portfolio(initial_cash=100_000.0)
        p.on_fill_event(_fill("MSFT", OrderSide.BUY, 200, 300.0, NOW))
        cash_before = p.cash
        p.on_dividend_event(_div("MSFT", 0.75, NOW))
        assert p.cash == cash_before + 200 * 0.75
        assert p.total_dividend_income == 150.0


# -----------------------------------------------------------------------
# T23 — Dividend income is NOT credited when position is zero
# -----------------------------------------------------------------------
class TestT23:
    def test_no_dividend_for_zero_position(self):
        p = Portfolio(initial_cash=100_000.0)
        cash_before = p.cash
        p.on_dividend_event(_div("AAPL", 0.50, NOW))
        assert p.cash == cash_before
        assert p.total_dividend_income == 0.0


# -----------------------------------------------------------------------
# T24 — Dividend income is NOT credited for short positions
# -----------------------------------------------------------------------
class TestT24:
    def test_no_dividend_for_short(self):
        p = Portfolio(initial_cash=100_000.0)
        p.on_fill_event(_fill("AAPL", OrderSide.SELL, 100, 150.0, NOW))
        cash_before = p.cash
        p.on_dividend_event(_div("AAPL", 0.50, NOW))
        assert p.cash == cash_before
        assert p.total_dividend_income == 0.0


# -----------------------------------------------------------------------
# T25 — Multiple symbols tracked independently
# -----------------------------------------------------------------------
class TestT25:
    def test_independent_symbols(self):
        p = Portfolio(initial_cash=200_000.0)
        p.on_fill_event(_fill("AAPL", OrderSide.BUY, 100, 150.0, NOW))
        p.on_fill_event(_fill("MSFT", OrderSide.BUY, 50, 300.0, NOW))
        assert p.positions["AAPL"] == 100
        assert p.positions["MSFT"] == 50

        p.update_market_value(_market("AAPL", 160.0, NOW))
        p.update_market_value(_market("MSFT", 310.0, NOW))
        assert p.holdings_value == (100 * 160.0) + (50 * 310.0)


# -----------------------------------------------------------------------
# T26 — Equity snapshots in bar_history have correct total_value
# -----------------------------------------------------------------------
class TestT26:
    def test_snapshot_total_value(self):
        p = Portfolio(initial_cash=100_000.0)
        p.on_fill_event(_fill("AAPL", OrderSide.BUY, 100, 150.0, NOW))
        p.update_market_value(_market("AAPL", 200.0, NOW))
        snap = p.bar_history[-1]
        assert snap.total_value == 85_000.0 + (100 * 200.0)


# -----------------------------------------------------------------------
# T27 — Portfolio total_value is always cash plus holdings
# -----------------------------------------------------------------------
class TestT27:
    def test_identity_holds(self):
        p = Portfolio(initial_cash=100_000.0)

        # After buy
        p.on_fill_event(_fill("AAPL", OrderSide.BUY, 50, 150.0, NOW))
        p.update_market_value(_market("AAPL", 155.0, NOW))
        assert p.total_value == p.cash + p.holdings_value

        # After sell
        p.on_fill_event(_fill("AAPL", OrderSide.SELL, 50, 155.0, NOW))
        p.update_market_value(_market("AAPL", 160.0, NOW))
        assert p.total_value == p.cash + p.holdings_value

        # After dividend
        p.on_fill_event(_fill("AAPL", OrderSide.BUY, 100, 160.0, NOW))
        p.on_dividend_event(_div("AAPL", 0.50, NOW))
        p.update_market_value(_market("AAPL", 162.0, NOW))
        assert p.total_value == p.cash + p.holdings_value


# -----------------------------------------------------------------------
# T28 — Portfolio handles symbols with no price data gracefully
# -----------------------------------------------------------------------
class TestT28:
    def test_no_price_data_no_crash(self):
        p = Portfolio(initial_cash=100_000.0)
        # Manually inject a position for a symbol we never sent a MarketEvent for
        p._positions["AAPL"] = 100
        # Should not crash — return 0 for unknown price
        assert p.holdings_value == 0.0
