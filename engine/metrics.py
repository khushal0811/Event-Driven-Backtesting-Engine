"""
metrics.py — Performance metrics for the backtesting engine.

Consumes the Portfolio's equity curve (list of PortfolioSnapshots) and
computes standard performance metrics used in quantitative finance.

Computed metrics:
  - Total return         : (final_value - initial_value) / initial_value
  - Annualised Sharpe    : mean(daily_returns) / std(daily_returns) * sqrt(252)
  - Max drawdown         : largest peak-to-trough decline in portfolio value

Usage:
    metrics = compute_metrics(portfolio.history, initial_cash=100_000)
    print(metrics)
"""

import math
from dataclasses import dataclass
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
        total_return    : Net return as a decimal (0.15 = +15%).
        sharpe_ratio    : Annualised Sharpe ratio (risk-free rate = 0).
                          None if fewer than 2 equity snapshots exist.
        max_drawdown    : Largest peak-to-trough drawdown as a positive
                          decimal (0.10 = -10% from peak). 0.0 if flat.
        initial_value   : Starting portfolio NAV.
        final_value     : Ending portfolio NAV.
        total_snapshots : Number of equity curve data points used.
    """
    total_return:    float
    sharpe_ratio:    Optional[float]
    max_drawdown:    float
    initial_value:   float
    final_value:     float
    total_snapshots: int

    def __str__(self) -> str:
        sharpe = f"{self.sharpe_ratio:.4f}" if self.sharpe_ratio is not None else "N/A"
        return (
            f"\n{'='*40}\n"
            f"  Backtest Performance Summary\n"
            f"{'='*40}\n"
            f"  Initial Value   : ${self.initial_value:>12,.2f}\n"
            f"  Final Value     : ${self.final_value:>12,.2f}\n"
            f"  Total Return    : {self.total_return * 100:>+.2f}%\n"
            f"  Sharpe Ratio    : {sharpe:>12}\n"
            f"  Max Drawdown    : {self.max_drawdown * 100:>.2f}%\n"
            f"  Equity Points   : {self.total_snapshots:>12}\n"
            f"{'='*40}"
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_metrics(
    history:       List[PortfolioSnapshot],
    initial_cash:  float,
) -> MetricsResult:
    """
    Compute performance metrics from an equity curve.

    Args:
        history      : List of PortfolioSnapshot objects from Portfolio.history.
        initial_cash : The portfolio's starting capital (Portfolio.initial_cash).

    Returns:
        MetricsResult with all computed metrics.

    Notes:
        - Sharpe ratio uses risk-free rate = 0 and annualises by sqrt(252).
        - If fewer than 2 snapshots exist, sharpe_ratio is None.
        - max_drawdown is expressed as a positive number (magnitude of decline).
    """
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

    return MetricsResult(
        total_return    = _total_return(initial_cash, final_value),
        sharpe_ratio    = _sharpe_ratio(equity_curve),
        max_drawdown    = _max_drawdown(equity_curve),
        initial_value   = initial_cash,
        final_value     = final_value,
        total_snapshots = len(history),
    )


# ---------------------------------------------------------------------------
# Internal calculation helpers
# ---------------------------------------------------------------------------

def _total_return(initial: float, final: float) -> float:
    """(final - initial) / initial"""
    if initial == 0.0:
        return 0.0
    return (final - initial) / initial


def _sharpe_ratio(equity_curve: List[float]) -> Optional[float]:
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

    return (mean / std) * math.sqrt(252)


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
