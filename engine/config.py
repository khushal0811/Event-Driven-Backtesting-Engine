"""
config.py — BacktestConfig dataclass for the backtesting engine.

The single structured input the engine accepts. All parameters
validated at construction time — the engine never receives invalid config.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Dict


@dataclass
class StrategyConfig:
    """Configuration for a strategy instance."""
    type:        str              # must match STRATEGY_REGISTRY key
    parameters:  Dict = field(default_factory=dict)
    python_code: Optional[str] = None   # for user-defined Python strategies


@dataclass
class BacktestConfig:
    """
    Complete configuration for a backtest run.

    This is the primary contract between the FastAPI backend and the engine.
    All parameters are validated via validate() before execution.
    """
    # Universe
    symbols:          List[str]

    # Strategy
    strategy:         StrategyConfig

    # Dates
    start_date:       date
    end_date:         date
    interval:         str = "1d"

    # Capital
    initial_capital:  float = 100_000.0

    # Position sizing
    #
    # CONTRACT (read before connecting the FastAPI layer):
    #   position_sizing : selects the order manager (see below).
    #   position_size   : meaning depends on position_sizing:
    #       "fixed"      → number of shares/units (e.g. 100)
    #       "percentage" → percentage of portfolio equity, in the range (0, 100]
    #                      (e.g. 10.0 means 10%).  run_backtest_from_config()
    #                      divides by 100 before passing to PercentageOrderManager.
    #                      DO NOT pass a fraction (0.10) — pass the percentage (10.0).
    #   risk_per_trade  : fraction of equity to risk per trade, in (0, 0.20].
    #                     Note: this is a fraction (not a percentage), consistent
    #                     with the Kelly / ATR risk literature.
    position_sizing:  str   = "risk_based"   # "fixed" | "percentage" | "risk_based"
    position_size:    float = 100.0          # shares (fixed) OR percentage 0–100 (percentage)
    risk_per_trade:   float = 0.02           # fraction for risk_based, e.g. 0.02 = 2 %
    stop_fraction:    float = 0.02           # ATR fallback stop distance (fraction)

    # Optional
    benchmark_symbol:     Optional[str]   = "SPY"
    include_dividends:    bool            = True

    # Transaction costs — applied per fill
    commission_model:  str   = "flat"   # "flat" | "per_share" | "percentage"
    commission_value:  float = 0.0      # $0 flat | $0.005/share | 0.001 = 0.1%
    slippage_bps:      float = 0.0      # basis points slippage on fill price

    def validate(self) -> None:
        """Raise ValueError if config is invalid."""
        if not self.symbols:
            raise ValueError("symbols list cannot be empty.")
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not (0 < self.risk_per_trade <= 0.20):
            raise ValueError("risk_per_trade must be in (0, 0.20].")
        valid_sizing = {"fixed", "percentage", "risk_based"}
        if self.position_sizing not in valid_sizing:
            raise ValueError(f"position_sizing must be one of {valid_sizing}.")
        if self.position_sizing == "percentage":
            if not (0 < self.position_size <= 100):
                raise ValueError(
                    "position_size must be in (0, 100] when position_sizing='percentage'. "
                    f"Got {self.position_size}. Pass a percentage (e.g. 10.0 for 10%), "
                    "not a fraction."
                )
        valid_commission_models = {"flat", "per_share", "percentage"}
        if self.commission_model not in valid_commission_models:
            raise ValueError(f"commission_model must be one of {valid_commission_models}")
        if self.commission_value < 0:
            raise ValueError("commission_value must be >= 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
