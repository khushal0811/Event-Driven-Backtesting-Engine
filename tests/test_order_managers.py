"""
Group 4 — OrderManagers (T29–T38)

All three sizing methods: Fixed, Percentage, Risk-Based.
"""

import pytest
from datetime import datetime

from engine.events import (
    SignalEvent, SignalType, OrderEvent, OrderSide, MarketEvent,
)
from engine.event_queue import EventQueue
from engine.order_manager import (
    FixedSizeOrderManager,
    PercentageOrderManager,
    RiskBasedOrderManager,
)
from engine.portfolio import Portfolio


NOW = datetime(2024, 6, 1, 10, 0, 0)


def _collect_orders(queue: EventQueue):
    """Drain all OrderEvents from the queue."""
    orders = []
    while not queue.empty():
        e = queue.get()
        if isinstance(e, OrderEvent):
            orders.append(e)
    return orders


# -----------------------------------------------------------------------
# T29 — FixedSizeOrderManager emits correct quantity
# -----------------------------------------------------------------------
class TestT29:
    def test_fixed_size_quantity(self):
        om = FixedSizeOrderManager(quantity=100)
        q  = EventQueue()
        om.on_signal_event(
            SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q
        )
        orders = _collect_orders(q)
        assert len(orders) == 1
        assert orders[0].quantity == 100
        assert orders[0].side == OrderSide.BUY
        assert orders[0].symbol == "AAPL"


# -----------------------------------------------------------------------
# T30 — FixedSizeOrderManager suppresses duplicate direction
# -----------------------------------------------------------------------
class TestT30:
    def test_suppresses_duplicate_buy(self):
        om = FixedSizeOrderManager(quantity=100)
        q  = EventQueue()
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
        orders = _collect_orders(q)
        assert len(orders) == 1


# -----------------------------------------------------------------------
# T31 — FixedSizeOrderManager allows direction reversal
# -----------------------------------------------------------------------
class TestT31:
    def test_direction_reversal(self):
        om = FixedSizeOrderManager(quantity=100)
        q  = EventQueue()
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.SELL), q)
        orders = _collect_orders(q)
        assert len(orders) == 2
        assert orders[0].side == OrderSide.BUY
        assert orders[1].side == OrderSide.SELL


# -----------------------------------------------------------------------
# T32 — PercentageOrderManager sizes correctly
# -----------------------------------------------------------------------
class TestT32:
    def test_percentage_sizing(self):
        p  = Portfolio(initial_cash=100_000.0)
        p.update_market_value(MarketEvent(timestamp=NOW, symbol="AAPL", price=200.0, volume=1000.0))
        om = PercentageOrderManager(percent_of_equity=0.10, portfolio=p)
        q  = EventQueue()
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
        orders = _collect_orders(q)
        assert len(orders) == 1
        # floor((100,000 * 0.10) / 200) = 50
        assert orders[0].quantity == 50


# -----------------------------------------------------------------------
# T33 — PercentageOrderManager updates quantity as equity grows
# -----------------------------------------------------------------------
class TestT33:
    def test_percentage_scales_with_equity(self):
        p  = Portfolio(initial_cash=120_000.0)
        p.update_market_value(MarketEvent(timestamp=NOW, symbol="AAPL", price=200.0, volume=1000.0))
        om = PercentageOrderManager(percent_of_equity=0.10, portfolio=p)
        q  = EventQueue()
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
        orders = _collect_orders(q)
        # floor((120,000 * 0.10) / 200) = 60
        assert orders[0].quantity == 60


# -----------------------------------------------------------------------
# T34 — PercentageOrderManager emits nothing if price is unknown
# -----------------------------------------------------------------------
class TestT34:
    def test_no_order_without_price(self):
        p  = Portfolio(initial_cash=100_000.0)
        om = PercentageOrderManager(percent_of_equity=0.10, portfolio=p)
        q  = EventQueue()
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
        orders = _collect_orders(q)
        assert len(orders) == 0


# -----------------------------------------------------------------------
# T35 — RiskBasedOrderManager sizes correctly
# -----------------------------------------------------------------------
class TestT35:
    def test_risk_based_sizing(self):
        p  = Portfolio(initial_cash=100_000.0)
        p.update_market_value(MarketEvent(timestamp=NOW, symbol="AAPL", price=150.0, volume=1000.0))
        om = RiskBasedOrderManager(risk_per_trade=0.02, stop_fraction=0.02, portfolio=p)
        q  = EventQueue()
        om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
        orders = _collect_orders(q)
        assert len(orders) == 1
        # risk = 100,000 * 0.02 = 2,000
        # stop = 150 * 0.02 = 3.00
        # qty  = floor(2000 / 3.00) = 666
        assert orders[0].quantity == 666


# -----------------------------------------------------------------------
# T36 — RiskBasedOrderManager rejects invalid risk values
# -----------------------------------------------------------------------
class TestT36:
    def test_zero_risk_rejected(self):
        p = Portfolio(initial_cash=100_000.0)
        with pytest.raises(ValueError):
            RiskBasedOrderManager(risk_per_trade=0.0, stop_fraction=0.02, portfolio=p)

    def test_excessive_risk_rejected(self):
        p = Portfolio(initial_cash=100_000.0)
        with pytest.raises(ValueError):
            RiskBasedOrderManager(risk_per_trade=0.50, stop_fraction=0.02, portfolio=p)


# -----------------------------------------------------------------------
# T37 — PercentageOrderManager rejects invalid percent values
# -----------------------------------------------------------------------
class TestT37:
    def test_zero_percent_rejected(self):
        p = Portfolio(initial_cash=100_000.0)
        with pytest.raises(ValueError):
            PercentageOrderManager(percent_of_equity=0.0, portfolio=p)

    def test_over_100_rejected(self):
        p = Portfolio(initial_cash=100_000.0)
        with pytest.raises(ValueError):
            PercentageOrderManager(percent_of_equity=1.5, portfolio=p)


# -----------------------------------------------------------------------
# T38 — All three OrderManagers suppress same-direction duplicates
# -----------------------------------------------------------------------
class TestT38:
    def test_all_managers_suppress_duplicates(self):
        p = Portfolio(initial_cash=100_000.0)
        p.update_market_value(MarketEvent(timestamp=NOW, symbol="AAPL", price=150.0, volume=1000.0))

        managers = [
            FixedSizeOrderManager(quantity=100),
            PercentageOrderManager(percent_of_equity=0.10, portfolio=p),
            RiskBasedOrderManager(risk_per_trade=0.02, stop_fraction=0.02, portfolio=p),
        ]

        for om in managers:
            q = EventQueue()
            om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
            om.on_signal_event(SignalEvent(symbol="AAPL", signal_type=SignalType.BUY), q)
            orders = _collect_orders(q)
            assert len(orders) == 1, f"{om.__class__.__name__} emitted duplicate orders"
