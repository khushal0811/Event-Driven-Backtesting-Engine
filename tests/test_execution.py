"""
Group 6 — ExecutionEngine (T59–T63)

Fill pricing, stale price protection, slippage hook, multi-symbol isolation.
"""

import pytest
from datetime import datetime

from engine.events import (
    MarketEvent, OrderEvent, FillEvent, OrderSide, OrderType,
)
from engine.event_queue import EventQueue
from engine.execution import SimulatedExecutionEngine


NOW = datetime(2024, 6, 1, 10, 0, 0)


def _order(symbol, side, qty):
    return OrderEvent(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET)


def _market(symbol, price):
    return MarketEvent(timestamp=NOW, symbol=symbol, price=price, volume=10000.0)


def _collect_fills(queue):
    fills = []
    while not queue.empty():
        e = queue.get()
        if isinstance(e, FillEvent):
            fills.append(e)
    return fills


# -----------------------------------------------------------------------
# T59 — SimulatedExecutionEngine fills at last known price
# -----------------------------------------------------------------------
class TestT59:
    def test_fill_at_last_price(self):
        ee = SimulatedExecutionEngine()
        ee.update_price(_market("AAPL", 150.0))
        q = EventQueue()
        ee.on_order_event(_order("AAPL", OrderSide.BUY, 100), q)
        fills = _collect_fills(q)
        assert len(fills) == 1
        assert fills[0].fill_price == 150.0
        assert fills[0].quantity == 100
        assert fills[0].side == OrderSide.BUY


# -----------------------------------------------------------------------
# T60 — ExecutionEngine drops order if no price available
# -----------------------------------------------------------------------
class TestT60:
    def test_no_fill_without_price(self):
        ee = SimulatedExecutionEngine()
        q  = EventQueue()
        ee.on_order_event(_order("AAPL", OrderSide.BUY, 100), q)
        fills = _collect_fills(q)
        assert len(fills) == 0


# -----------------------------------------------------------------------
# T61 — ExecutionEngine price updates correctly between bars
# -----------------------------------------------------------------------
class TestT61:
    def test_price_updates(self):
        ee = SimulatedExecutionEngine()
        ee.update_price(_market("AAPL", 150.0))
        ee.update_price(_market("AAPL", 175.0))
        q = EventQueue()
        ee.on_order_event(_order("AAPL", OrderSide.BUY, 100), q)
        fills = _collect_fills(q)
        assert fills[0].fill_price == 175.0


# -----------------------------------------------------------------------
# T62 — Slippage model works via _get_fill_price override
# -----------------------------------------------------------------------
class TestT62:
    def test_slippage_hookpoint(self):
        class SlippageEngine(SimulatedExecutionEngine):
            def _get_fill_price(self, symbol):
                base = super()._get_fill_price(symbol)
                return base * 1.001 if base else None

        ee = SlippageEngine()
        ee.update_price(_market("AAPL", 150.0))
        q = EventQueue()
        ee.on_order_event(_order("AAPL", OrderSide.BUY, 100), q)
        fills = _collect_fills(q)
        assert abs(fills[0].fill_price - 150.15) < 0.01


# -----------------------------------------------------------------------
# T63 — ExecutionEngine tracks multiple symbols independently
# -----------------------------------------------------------------------
class TestT63:
    def test_multi_symbol_isolation(self):
        ee = SimulatedExecutionEngine()
        ee.update_price(_market("AAPL", 150.0))
        ee.update_price(_market("MSFT", 300.0))

        q = EventQueue()
        ee.on_order_event(_order("AAPL", OrderSide.BUY, 100), q)
        ee.on_order_event(_order("MSFT", OrderSide.BUY, 50), q)
        fills = _collect_fills(q)

        aapl_fill = next(f for f in fills if f.symbol == "AAPL")
        msft_fill = next(f for f in fills if f.symbol == "MSFT")
        assert aapl_fill.fill_price == 150.0
        assert msft_fill.fill_price == 300.0
