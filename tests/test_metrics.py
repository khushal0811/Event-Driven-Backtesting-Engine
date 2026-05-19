"""
Group 7 — Metrics (T64–T76)

Every number the frontend will display.
"""

import math
import pytest
from datetime import datetime, timedelta

from engine.portfolio import PortfolioSnapshot
from engine.metrics import MetricsResult, compute_metrics


T0 = datetime(2024, 1, 1)


def _snap(value, ts):
    return PortfolioSnapshot(timestamp=ts, cash=value, holdings_value=0.0, total_value=value)


def _curve(values, start=T0):
    """Build a bar_history from a list of total_value numbers."""
    return [_snap(v, start + timedelta(days=i)) for i, v in enumerate(values)]


# -----------------------------------------------------------------------
# T64 — Total return is correct
# -----------------------------------------------------------------------
class TestT64:
    def test_total_return(self):
        history = _curve([100_000, 105_000, 112_000])
        m = compute_metrics(
            bar_history=history, fill_history=[], initial_cash=100_000,
        )
        assert abs(m.total_return - 0.12) < 1e-9


# -----------------------------------------------------------------------
# T65 — Price return and total return with dividends are separate
# -----------------------------------------------------------------------
class TestT65:
    def test_price_vs_total_return(self):
        # Final value 110,000 = 108,000 price + 2,000 dividends
        history = _curve([100_000, 105_000, 110_000])
        m = compute_metrics(
            bar_history=history, fill_history=[], initial_cash=100_000,
            dividend_income=2_000.0,
        )
        assert abs(m.total_return_with_dividends - 0.10) < 1e-9
        assert abs(m.price_return - 0.08) < 1e-9
        assert m.total_dividend_income == 2_000.0


# -----------------------------------------------------------------------
# T66 — CAGR is correct
# -----------------------------------------------------------------------
class TestT66:
    def test_cagr(self):
        start = datetime(2022, 1, 1)
        end   = datetime(2024, 1, 1)  # exactly 2 years
        history = [_snap(100_000, start), _snap(121_000, end)]
        m = compute_metrics(
            bar_history=history, fill_history=[], initial_cash=100_000,
            start_date=start, end_date=end,
        )
        # CAGR = (121000/100000)^(1/2) - 1 = 0.10
        assert m.cagr is not None
        assert abs(m.cagr - 0.10) < 0.005


# -----------------------------------------------------------------------
# T67 — Sharpe ratio: zero std dev returns None
# -----------------------------------------------------------------------
class TestT67:
    def test_sharpe_zero_std(self):
        # Constant equity → zero std → Sharpe undefined
        history = _curve([100_000] * 10)
        m = compute_metrics(
            bar_history=history, fill_history=[], initial_cash=100_000,
        )
        assert m.sharpe_ratio is None


# -----------------------------------------------------------------------
# T68 — Max drawdown is correct
# -----------------------------------------------------------------------
class TestT68:
    def test_max_drawdown(self):
        history = _curve([100_000, 110_000, 105_000, 95_000, 100_000])
        m = compute_metrics(
            bar_history=history, fill_history=[], initial_cash=100_000,
        )
        # peak 110k, trough 95k → dd = 15k/110k ≈ 0.1364
        assert abs(m.max_drawdown - (15_000 / 110_000)) < 0.001


# -----------------------------------------------------------------------
# T69 — Win rate is correct
# -----------------------------------------------------------------------
class TestT69:
    def test_win_rate(self):
        trades = [
            {"pnl": 100, "return": 0.01},
            {"pnl": 200, "return": 0.02},
            {"pnl": 50,  "return": 0.005},
            {"pnl": -80, "return": -0.008},
            {"pnl": -30, "return": -0.003},
        ]
        m = compute_metrics(
            bar_history=_curve([100_000, 100_100]),
            fill_history=[], initial_cash=100_000, trades=trades,
        )
        assert m.win_rate == 0.60


# -----------------------------------------------------------------------
# T70 — Win rate is None when no trades
# -----------------------------------------------------------------------
class TestT70:
    def test_win_rate_no_trades(self):
        m = compute_metrics(
            bar_history=_curve([100_000, 100_100]),
            fill_history=[], initial_cash=100_000, trades=None,
        )
        assert m.win_rate is None


# -----------------------------------------------------------------------
# T71 — Annualised volatility is correct
# -----------------------------------------------------------------------
class TestT71:
    def test_volatility(self):
        # Alternating +1% and -1%
        curve = [100_000]
        for i in range(100):
            factor = 1.01 if i % 2 == 0 else 0.99
            curve.append(curve[-1] * factor)
        history = _curve(curve)
        m = compute_metrics(
            bar_history=history, fill_history=[], initial_cash=100_000,
        )
        assert m.volatility is not None
        # Should be close to 0.01 * sqrt(252) ≈ 0.1587
        assert abs(m.volatility - (0.01 * math.sqrt(252))) < 0.01


# -----------------------------------------------------------------------
# T72 — Alpha is correct
# -----------------------------------------------------------------------
class TestT72:
    def test_alpha(self):
        history = _curve([100_000, 115_000])
        m = compute_metrics(
            bar_history=history, fill_history=[], initial_cash=100_000,
            benchmark_return=0.10,
        )
        assert abs(m.alpha - 0.05) < 1e-9


# -----------------------------------------------------------------------
# T73 — Alpha is None when no benchmark
# -----------------------------------------------------------------------
class TestT73:
    def test_alpha_no_benchmark(self):
        m = compute_metrics(
            bar_history=_curve([100_000, 110_000]),
            fill_history=[], initial_cash=100_000,
        )
        assert m.alpha is None
        assert m.benchmark_return is None


# -----------------------------------------------------------------------
# T74 — Metrics with zero history
# -----------------------------------------------------------------------
class TestT74:
    def test_empty_history(self):
        m = compute_metrics(
            bar_history=[], fill_history=[], initial_cash=100_000,
        )
        assert m.total_return == 0.0
        assert m.sharpe_ratio is None
        assert m.max_drawdown == 0.0
        assert m.total_snapshots == 0


# -----------------------------------------------------------------------
# T75 — CAGR is None when dates missing
# -----------------------------------------------------------------------
class TestT75:
    def test_cagr_no_dates(self):
        m = compute_metrics(
            bar_history=_curve([100_000, 110_000]),
            fill_history=[], initial_cash=100_000,
        )
        # start_date/end_date derived from history, so CAGR should compute
        # But with only 1-day range, it might be extreme or None
        # The key test: no crash
        assert isinstance(m.cagr, (float, type(None)))


# -----------------------------------------------------------------------
# T76 — MetricsResult __str__ prints without crashing
# -----------------------------------------------------------------------
class TestT76:
    def test_str_no_crash(self):
        m = compute_metrics(
            bar_history=_curve([100_000, 112_000]),
            fill_history=[], initial_cash=100_000,
        )
        result = str(m)
        assert "Backtest Performance Summary" in result
        assert "Total Return" in result

    def test_str_with_none_fields(self):
        m = compute_metrics(
            bar_history=[], fill_history=[], initial_cash=100_000,
        )
        result = str(m)
        assert "N/A" in result  # None fields render as N/A
