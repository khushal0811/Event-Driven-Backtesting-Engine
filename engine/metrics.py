"""
metrics.py — Performance metrics for the backtesting engine.

Consumes the Portfolio's equity curve (list of PortfolioSnapshots) and
computes standard performance metrics used in quantitative finance.

Computed metrics:
  - Total return         : (final_value - initial_value) / initial_value
  - Annualised Sharpe    : mean(daily_returns) / std(daily_returns) * sqrt(252)
  - Max drawdown         : largest peak-to-trough decline in portfolio value
  - CAGR                 : compound annual growth rate
  - Volatility           : annualised standard deviation of returns
  - Win rate             : fraction of profitable trades
  - Alpha                : strategy return minus benchmark return

Usage:
    metrics = compute_metrics(
        bar_history=portfolio.bar_history,
        fill_history=portfolio.history,
        initial_cash=100_000,
    )
    print(metrics)
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from engine.portfolio import PortfolioSnapshot


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MetricsResult:
    """
    Immutable summary of backtest performance.

    Attributes:
        total_return              : Net return as a decimal (0.15 = +15%).
        sharpe_ratio              : Annualised Sharpe ratio (risk-free rate = 0).
                                    None if fewer than 2 equity snapshots exist.
        max_drawdown              : Largest peak-to-trough drawdown as a positive
                                    decimal (0.10 = -10% from peak). 0.0 if flat.
        initial_value             : Starting portfolio NAV.
        final_value               : Ending portfolio NAV.
        total_snapshots           : Number of equity curve data points used.
        cagr                      : Compound annual growth rate.
        volatility                : Annualised standard deviation of returns.
        win_rate                  : Fraction of profitable trades.
        total_trades              : Number of completed round-trip trades.
        avg_trade_return          : Average return per trade.
        total_dividend_income     : Total dividends received in dollars.
        price_return              : Return excluding dividends.
        total_return_with_dividends : Return including dividend income.
        benchmark_return          : Benchmark (e.g. SPY) total return.
        alpha                     : Strategy return minus benchmark return.
    """
    # Existing fields — unchanged
    total_return:    float
    sharpe_ratio:    Optional[float]
    max_drawdown:    float
    initial_value:   float
    final_value:     float
    total_snapshots: int

    # New fields
    cagr:                      Optional[float] = None
    volatility:                Optional[float] = None
    win_rate:                  Optional[float] = None
    total_trades:              int = 0
    avg_trade_return:          Optional[float] = None
    total_dividend_income:     float = 0.0
    price_return:              float = 0.0
    total_return_with_dividends: float = 0.0
    benchmark_return:          Optional[float] = None
    alpha:                     Optional[float] = None

    def __str__(self) -> str:
        sharpe    = f"{self.sharpe_ratio:.4f}"  if self.sharpe_ratio is not None else "N/A"
        cagr      = f"{self.cagr*100:+.2f}%"   if self.cagr is not None else "N/A"
        vol       = f"{self.volatility*100:.2f}%" if self.volatility is not None else "N/A"
        win       = f"{self.win_rate*100:.1f}%"   if self.win_rate is not None else "N/A"
        bench     = f"{self.benchmark_return*100:+.2f}%" if self.benchmark_return is not None else "N/A"
        alpha_str = f"{self.alpha*100:+.2f}%"   if self.alpha is not None else "N/A"
        return (
            f"\n{'='*44}\n"
            f"  Backtest Performance Summary\n"
            f"{'='*44}\n"
            f"  Initial Value        : ${self.initial_value:>12,.2f}\n"
            f"  Final Value          : ${self.final_value:>12,.2f}\n"
            f"  Price Return         : {self.price_return * 100:>+12.2f}%\n"
            f"  Dividend Income      : ${self.total_dividend_income:>12,.2f}\n"
            f"  Total Return         : {self.total_return_with_dividends * 100:>+12.2f}%\n"
            f"  CAGR                 : {cagr:>13}\n"
            f"  Sharpe Ratio         : {sharpe:>13}\n"
            f"  Max Drawdown         : {self.max_drawdown * 100:>12.2f}%\n"
            f"  Volatility (Ann.)    : {vol:>13}\n"
            f"  Win Rate             : {win:>13}\n"
            f"  Total Trades         : {self.total_trades:>13}\n"
            f"  Benchmark Return     : {bench:>13}\n"
            f"  Alpha                : {alpha_str:>13}\n"
            f"{'='*44}"
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_metrics(
    bar_history:      List[PortfolioSnapshot],
    fill_history:     List[PortfolioSnapshot],
    initial_cash:     float,
    start_date:       Optional[datetime] = None,
    end_date:         Optional[datetime] = None,
    dividend_income:  float = 0.0,
    benchmark_return: Optional[float] = None,
    trades:           Optional[List[dict]] = None,
    interval:         str = "1d",
) -> MetricsResult:
    """
    Compute performance metrics from equity curve data.

    Args:
        bar_history      : Every-bar PortfolioSnapshots (continuous equity curve).
        fill_history     : Fill-only PortfolioSnapshots (trade-level equity curve).
        initial_cash     : The portfolio's starting capital.
        start_date       : Backtest start date (for CAGR calculation).
        end_date         : Backtest end date (for CAGR calculation).
        dividend_income  : Total dividend income in dollars.
        benchmark_return : Benchmark total return (optional).
        trades           : List of trade dicts with 'pnl' and 'return' keys (optional).
        interval         : Bar interval (e.g. "1d", "1h", "15m", "1m").

    Returns:
        MetricsResult with all computed metrics.
    """
    # Use bar_history for continuous metrics, fall back to fill_history
    history = bar_history if bar_history else fill_history

    if not history:
        return MetricsResult(
            total_return    = 0.0,
            sharpe_ratio    = None,
            max_drawdown    = 0.0,
            initial_value   = initial_cash,
            final_value     = initial_cash,
            total_snapshots = 0,
        )

    equity_curve = [snap.total_value for snap in history]
    final_value  = equity_curve[-1]

    # Derive dates from history if not provided
    if start_date is None and history:
        start_date = history[0].timestamp
    if end_date is None and history:
        end_date = history[-1].timestamp

    # Core metrics
    total_ret    = _total_return(initial_cash, final_value)
    price_ret    = _total_return(initial_cash, final_value - dividend_income)
    total_w_div  = _total_return(initial_cash, final_value)

    # Alpha
    alpha = None
    if benchmark_return is not None:
        alpha = total_w_div - benchmark_return

    return MetricsResult(
        total_return              = total_ret,
        sharpe_ratio              = _sharpe_ratio(equity_curve, interval),
        max_drawdown              = _max_drawdown(equity_curve),
        initial_value             = initial_cash,
        final_value               = final_value,
        total_snapshots           = len(history),
        cagr                      = _cagr(initial_cash, final_value, start_date, end_date),
        volatility                = _annualised_volatility(equity_curve, interval),
        win_rate                  = _win_rate(trades),
        total_trades              = len(trades) if trades else len(fill_history),
        avg_trade_return          = _avg_trade_return(trades),
        total_dividend_income     = dividend_income,
        price_return              = price_ret,
        total_return_with_dividends = total_w_div,
        benchmark_return          = benchmark_return,
        alpha                     = alpha,
    )


# ---------------------------------------------------------------------------
# Internal calculation helpers
# ---------------------------------------------------------------------------

def _total_return(initial: float, final: float) -> float:
    """(final - initial) / initial"""
    if initial == 0.0:
        return 0.0
    return (final - initial) / initial


def _get_annualization_factor(interval: str) -> float:
    """Return number of bar periods in a trading year for the given interval."""
    factor_map = {
        "1m":  252.0 * 390.0,
        "2m":  252.0 * 195.0,
        "5m":  252.0 * 78.0,
        "15m": 252.0 * 26.0,
        "30m": 252.0 * 13.0,
        "1h":  252.0 * 7.0,
        "1d":  252.0,
    }
    return factor_map.get(interval, 252.0)


def _sharpe_ratio(equity_curve: List[float], interval: str = "1d") -> Optional[float]:
    """
    Annualised Sharpe ratio (risk-free rate = 0).

    Uses period-over-period returns from the equity curve snapshots.
    Returns None if there are fewer than 2 data points.
    """
    if len(equity_curve) < 2:
        return None

    returns: List[float] = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        curr = equity_curve[i]
        if prev == 0.0:
            returns.append(0.0)
        else:
            returns.append((curr - prev) / prev)

    n    = len(returns)
    mean = sum(returns) / n

    # Population std dev (ddof=0) — sufficient for backtesting context
    variance = sum((r - mean) ** 2 for r in returns) / n
    std      = math.sqrt(variance)

    if std == 0.0:
        return None  # flat returns — Sharpe undefined

    ann_factor = _get_annualization_factor(interval)
    return (mean / std) * math.sqrt(ann_factor)


def _max_drawdown(equity_curve: List[float]) -> float:
    """
    Maximum peak-to-trough drawdown as a positive decimal.

    Scans the equity curve once, tracking the running peak.
    Returns 0.0 if the curve never declines.
    """
    peak        = equity_curve[0]
    max_dd      = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def _cagr(
    initial: float, final: float,
    start_date: Optional[datetime], end_date: Optional[datetime]
) -> Optional[float]:
    """Compound Annual Growth Rate."""
    if not start_date or not end_date or initial <= 0:
        return None
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return None
    return (final / initial) ** (1.0 / years) - 1.0


def _annualised_volatility(equity_curve: List[float], interval: str = "1d") -> Optional[float]:
    """Annualised standard deviation of period returns."""
    if len(equity_curve) < 2:
        return None
    returns = [
        (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
        for i in range(1, len(equity_curve))
        if equity_curve[i-1] != 0.0
    ]
    if len(returns) < 2:
        return None
    n    = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)  # sample std
    
    ann_factor = _get_annualization_factor(interval)
    return math.sqrt(variance) * math.sqrt(ann_factor)


def _win_rate(trades: Optional[List[dict]]) -> Optional[float]:
    """Fraction of trades with positive P&L."""
    if not trades:
        return None
    winners = [t for t in trades if t.get("pnl", 0) > 0]
    return len(winners) / len(trades)


def _avg_trade_return(trades: Optional[List[dict]]) -> Optional[float]:
    """Average return per completed round-trip trade."""
    if not trades:
        return None
    returns = [t.get("return", 0.0) for t in trades]
    return sum(returns) / len(returns)
