"""
Group 9 — Config (T89–T95)

Validation and factory functions.
"""

import pytest
from datetime import date

from engine.config import BacktestConfig, StrategyConfig
from engine.metrics import MetricsResult


# -----------------------------------------------------------------------
# T89 — BacktestConfig validates correctly
# -----------------------------------------------------------------------
class TestT89:
    def test_valid_config(self):
        cfg = BacktestConfig(
            symbols=["AAPL"],
            strategy=StrategyConfig(type="momentum", parameters={"lookback": 20}),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
        )
        cfg.validate()  # Should not raise


# -----------------------------------------------------------------------
# T90 — BacktestConfig rejects end date before start date
# -----------------------------------------------------------------------
class TestT90:
    def test_end_before_start(self):
        cfg = BacktestConfig(
            symbols=["AAPL"],
            strategy=StrategyConfig(type="momentum"),
            start_date=date(2024, 6, 1),
            end_date=date(2022, 1, 1),
        )
        with pytest.raises(ValueError):
            cfg.validate()


# -----------------------------------------------------------------------
# T91 — BacktestConfig rejects empty symbols list
# -----------------------------------------------------------------------
class TestT91:
    def test_empty_symbols(self):
        cfg = BacktestConfig(
            symbols=[],
            strategy=StrategyConfig(type="momentum"),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
        )
        with pytest.raises(ValueError):
            cfg.validate()


# -----------------------------------------------------------------------
# T92 — BacktestConfig rejects invalid risk_per_trade
# -----------------------------------------------------------------------
class TestT92:
    def test_zero_risk(self):
        cfg = BacktestConfig(
            symbols=["AAPL"],
            strategy=StrategyConfig(type="momentum"),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
            risk_per_trade=0.0,
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_excessive_risk(self):
        cfg = BacktestConfig(
            symbols=["AAPL"],
            strategy=StrategyConfig(type="momentum"),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
            risk_per_trade=0.50,
        )
        with pytest.raises(ValueError):
            cfg.validate()


# -----------------------------------------------------------------------
# T93 — BacktestConfig rejects invalid position_sizing
# -----------------------------------------------------------------------
class TestT93:
    def test_invalid_sizing(self):
        cfg = BacktestConfig(
            symbols=["AAPL"],
            strategy=StrategyConfig(type="momentum"),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
            position_sizing="magic",
        )
        with pytest.raises(ValueError):
            cfg.validate()


# -----------------------------------------------------------------------
# T94 — run_backtest_from_config wires correct OrderManager
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT94:
    def test_all_three_sizing_methods(self, data_dir, has_aapl_data):
        from run_backtest import run_backtest_from_config

        base = dict(
            symbols=["AAPL"],
            strategy=StrategyConfig(type="moving_average_crossover",
                                     parameters={"short_window": 5, "long_window": 20}),
            start_date=date(2022, 1, 1),
            end_date=date(2024, 1, 1),
        )

        for sizing in ["fixed", "percentage", "risk_based"]:
            cfg = BacktestConfig(**base, position_sizing=sizing)
            result = run_backtest_from_config(cfg, data_dir=data_dir)
            assert isinstance(result, MetricsResult), (
                f"position_sizing='{sizing}' did not return MetricsResult"
            )


# -----------------------------------------------------------------------
# T95 — Existing run_backtest() still works unchanged
# -----------------------------------------------------------------------
@pytest.mark.integration
class TestT95:
    def test_backwards_compatible(self, data_dir):
        from run_backtest import run_backtest
        result = run_backtest(
            symbols=["AAPL"], data_dir=data_dir, initial_cash=100_000,
        )
        assert isinstance(result, MetricsResult)
