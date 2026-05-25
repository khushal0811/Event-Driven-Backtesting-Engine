"""
portfolio.py — Portfolio management for the backtesting engine.

Tracks:
  - Cash balance (updated on every FillEvent)
  - Positions per symbol (net quantity, positive = long)
  - Holdings value (positions × current market price)
  - Total portfolio value history (equity curve snapshots)

The Portfolio is the single source of truth for P&L state.
No other module modifies cash or positions.

Usage (engine loop):
    portfolio.update_market_value(market_event)   # on every MarketEvent
    portfolio.on_fill_event(fill_event)           # on every FillEvent
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from engine.events import FillEvent, MarketEvent, OrderSide, DividendEvent


# ---------------------------------------------------------------------------
# Snapshot — one row of the equity curve
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    """Immutable record of portfolio state at a point in time."""
    timestamp:      datetime
    cash:           float
    holdings_value: float
    total_value:    float


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Maintains cash, positions, and a running equity curve.

    Args:
        initial_cash : Starting capital in dollars. Must be positive.

    Raises:
        ValueError: If initial_cash <= 0.
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError(
                f"initial_cash must be positive, got {initial_cash}."
            )

        self._cash:     float             = initial_cash
        self._initial_cash: float         = initial_cash

        # Net quantity per symbol  (positive = long, 0 = flat)
        self._positions: Dict[str, int]   = {}

        # Last known market price per symbol (set by update_market_value)
        self._prices: Dict[str, float]    = {}

        # Equity curve — one snapshot per fill (trades only)
        self._fill_history: List[PortfolioSnapshot] = []

        # Continuous equity curve — one snapshot per market bar
        self._bar_history: List[PortfolioSnapshot] = []

        # Cumulative dividend income received during the backtest
        self._total_dividend_income: float = 0.0

        # Trade tracking
        self._completed_trades: List[dict] = []
        self._open_trades: Dict[str, List[dict]] = {}

        self._total_commission_paid: float = 0.0
        self._commission_model:  str   = "flat"
        self._commission_value:  float = 0.0
        self._slippage_bps:      float = 0.0

    def configure_costs(
        self,
        commission_model:  str,
        commission_value:  float,
        slippage_bps:      float,
    ) -> None:
        self._commission_model  = commission_model
        self._commission_value  = commission_value
        self._slippage_bps      = slippage_bps

    def _calculate_commission(self, qty: int, price: float) -> float:
        notional = qty * price
        if self._commission_model == "flat":
            return self._commission_value
        elif self._commission_model == "per_share":
            return qty * self._commission_value
        elif self._commission_model == "percentage":
            return notional * self._commission_value
        return 0.0

    # ------------------------------------------------------------------
    # Engine loop hooks
    # ------------------------------------------------------------------

    def update_market_value(self, event: MarketEvent) -> None:
        """
        Record the latest market price for a symbol and snapshot equity.

        Called on every MarketEvent bar — produces a continuous equity curve
        regardless of whether any trades occurred on that bar.
        """
        self._prices[event.symbol] = event.price

        # Record equity snapshot every bar for continuous chart rendering
        snapshot = PortfolioSnapshot(
            timestamp      = event.timestamp,
            cash           = self._cash,
            holdings_value = self.holdings_value,
            total_value    = self.total_value,
        )
        self._bar_history.append(snapshot)

    def update_last_price(self, symbol: str, price: float) -> None:
        """Update last known price without creating a snapshot."""
        self._prices[symbol] = price

    def record_bar_snapshot(self, timestamp: datetime) -> None:
        """Record equity snapshot for the given timestamp (usually once per consolidated bar)."""
        snapshot = PortfolioSnapshot(
            timestamp      = timestamp,
            cash           = self._cash,
            holdings_value = self.holdings_value,
            total_value    = self.total_value,
        )
        self._bar_history.append(snapshot)

    def on_fill_event(self, event: FillEvent) -> None:
        """
        Apply a fill to cash and positions, then record an equity snapshot.

        BUY  → cash decreases, position increases
        SELL → cash increases, position decreases
        """
        symbol   = event.symbol
        qty      = event.quantity
        price    = event.fill_price
        cost     = qty * price

        current_qty = self._positions.get(symbol, 0)

        commission = self._calculate_commission(qty, price)

        # Apply slippage to fill price (increases cost on buy, decreases on sell)
        slippage_amount = price * (self._slippage_bps / 10000)

        if event.side == OrderSide.BUY:
            total_cost = cost + commission + (qty * slippage_amount)
            if self._cash < total_cost:
                print(f"[Portfolio] WARNING: Insufficient cash to execute BUY of {qty} {symbol} at {price:.2f}. "
                      f"Required: {total_cost:.2f}, Available: {self._cash:.2f}. Skipping fill.")
                return
            self._cash -= total_cost
            self._total_commission_paid += commission
            self._positions[symbol] = current_qty + qty
        else:  # SELL
            # ── Long-only safety net: never go short ──
            if current_qty <= 0:
                print(f"[Portfolio] BLOCKED: Cannot sell {qty} {symbol} — no position held.")
                return
            if qty > current_qty:
                print(f"[Portfolio] CAPPED: Sell qty {qty} exceeds held {current_qty} for {symbol}. Capping to {current_qty}.")
                qty = current_qty
                cost = qty * price
                commission = self._calculate_commission(qty, price)

            net_proceeds = cost - commission - (qty * slippage_amount)
            self._cash += net_proceeds
            self._total_commission_paid += commission
            self._positions[symbol] = current_qty - qty

        # FIFO Trade matching
        if symbol not in self._open_trades:
            self._open_trades[symbol] = []

        trade_qty = qty
        
        if current_qty == 0:
            # Opening a new position
            self._open_trades[symbol].append({
                "price": price,
                "qty": trade_qty,
                "timestamp": event.timestamp,
                "side": event.side
            })
        elif current_qty > 0:
            # Currently long
            if event.side == OrderSide.BUY:
                # Adding to long
                self._open_trades[symbol].append({
                    "price": price,
                    "qty": trade_qty,
                    "timestamp": event.timestamp,
                    "side": event.side
                })
            else:
                # Reducing or reversing long
                while trade_qty > 0 and self._open_trades[symbol]:
                    entry = self._open_trades[symbol][0]
                    close_qty = min(trade_qty, entry["qty"])
                    pnl = close_qty * (price - entry["price"])
                    ret = (price - entry["price"]) / entry["price"] if entry["price"] > 0 else 0.0
                    self._completed_trades.append({
                        "symbol": symbol,
                        "pnl": pnl,
                        "return": ret,
                        "qty": close_qty,
                        "entry_price": entry["price"],
                        "exit_price": price,
                        "entry_time": entry["timestamp"],
                        "exit_time": event.timestamp,
                        "direction": "long"
                    })
                    trade_qty -= close_qty
                    entry["qty"] -= close_qty
                    if entry["qty"] <= 0:
                        self._open_trades[symbol].pop(0)
                if trade_qty > 0:
                    # Reversed to short
                    self._open_trades[symbol].append({
                        "price": price,
                        "qty": trade_qty,
                        "timestamp": event.timestamp,
                        "side": event.side
                    })
        else:
            # Currently short
            if event.side == OrderSide.SELL:
                # Adding to short
                self._open_trades[symbol].append({
                    "price": price,
                    "qty": trade_qty,
                    "timestamp": event.timestamp,
                    "side": event.side
                })
            else:
                # Reducing or reversing short
                while trade_qty > 0 and self._open_trades[symbol]:
                    entry = self._open_trades[symbol][0]
                    close_qty = min(trade_qty, entry["qty"])
                    pnl = close_qty * (entry["price"] - price)
                    ret = (entry["price"] - price) / entry["price"] if entry["price"] > 0 else 0.0
                    self._completed_trades.append({
                        "symbol": symbol,
                        "pnl": pnl,
                        "return": ret,
                        "qty": close_qty,
                        "entry_price": entry["price"],
                        "exit_price": price,
                        "entry_time": entry["timestamp"],
                        "exit_time": event.timestamp,
                        "direction": "short"
                    })
                    trade_qty -= close_qty
                    entry["qty"] -= close_qty
                    if entry["qty"] <= 0:
                        self._open_trades[symbol].pop(0)
                if trade_qty > 0:
                    # Reversed to long
                    self._open_trades[symbol].append({
                        "price": price,
                        "qty": trade_qty,
                        "timestamp": event.timestamp,
                        "side": event.side
                    })

        # Record snapshot
        snapshot = PortfolioSnapshot(
            timestamp      = event.timestamp,
            cash           = self._cash,
            holdings_value = self.holdings_value,
            total_value    = self.total_value,
        )
        self._fill_history.append(snapshot)

    def on_dividend_event(self, event: "DividendEvent") -> None:
        """
        Credit or debit dividend cash adjustments.
        - Long position (qty > 0): receive dividend income.
        - Short position (qty < 0): pay dividend liability (cash decreases).
        """
        symbol = event.symbol
        qty    = self._positions.get(symbol, 0)

        if qty == 0:
            return

        adjustment = qty * event.dividend_per_share
        self._cash += adjustment
        if qty > 0:
            self._total_dividend_income += adjustment

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def holdings_value(self) -> float:
        """
        Current market value of all open positions.

        Uses the last known price per symbol from update_market_value().
        Symbols with no price record are valued at 0 (conservative).
        """
        total = 0.0
        for symbol, qty in self._positions.items():
            price  = self._prices.get(symbol, 0.0)
            total += qty * price
        return total

    @property
    def total_value(self) -> float:
        """Cash + holdings value — the total portfolio NAV."""
        return self._cash + self.holdings_value

    @property
    def positions(self) -> Dict[str, int]:
        """Snapshot of current net positions (symbol → net quantity)."""
        return dict(self._positions)

    @property
    def history(self) -> List[PortfolioSnapshot]:
        """
        Trade-level equity curve — one snapshot per fill.
        Kept for backwards compatibility with existing metrics code.
        """
        return list(self._fill_history)

    @property
    def bar_history(self) -> List[PortfolioSnapshot]:
        """Continuous equity curve — one snapshot per market bar."""
        return list(self._bar_history)

    @property
    def total_dividend_income(self) -> float:
        """Total dividend income received during the backtest."""
        return self._total_dividend_income

    @property
    def completed_trades(self) -> List[dict]:
        """List of completed round-trip trades."""
        return list(self._completed_trades)

    @property
    def initial_cash(self) -> float:
        """Starting capital."""
        return self._initial_cash

    @property
    def total_commission_paid(self) -> float:
        return self._total_commission_paid

    def position(self, symbol: str) -> int:
        """Return net quantity held for a symbol (0 if flat)."""
        return self._positions.get(symbol, 0)

    def last_price(self, symbol: str) -> Optional[float]:
        """Return the last market price seen for a symbol, or None."""
        return self._prices.get(symbol)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Portfolio("
            f"cash={self._cash:.2f}, "
            f"holdings={self.holdings_value:.2f}, "
            f"total={self.total_value:.2f})"
        )
