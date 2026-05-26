# Event-Driven Backtesting Engine

**A high-performance, deterministic, event-driven backtesting engine designed for quant researchers. It simulates multi-symbol portfolio execution over historical market data with lookahead-free pricing, realistic transaction cost settings, and precise metrics calculations.**

Part of the three-component Strategy Research Platform:
- [Market Data Pipeline](file:///Users/khushalarora/Documents/Career/Trading-System-Workspace/market-data-pipeline/Backtester-Oriented-Market-Data-Pipeline) — Data ingestion, normalization, event streaming
- **Event-Driven Backtesting Engine** — Strategy simulation core (this component)
- [Strategy Research Terminal](file:///Users/khushalarora/Documents/Career/Trading-System-Workspace/market-data-pipeline/strategy-research-terminal) — REST/WS API layer and Next.js frontend dashboard

---

## Engine Architecture

The engine uses a strict event-driven loop where components interact exclusively by passing events through a shared First-In-First-Out (FIFO) queue. This design ensures that each historical bar is processed in isolation, eliminating vector shortcuts and lookahead bias.

```
Market & Corporate Event Stream
            │
            ▼
       DataHandler
            │
            ▼
       Event Queue  <───────────────────────────┐
            │                                   │
      [MarketEvent] ──> Strategy ──> [SignalEvent]
            │                              │
            │                              ▼
            │                         OrderManager
            │                              │
            │                              ▼
            │                         [OrderEvent]
            │                              │
            │                              ▼
            │                       ExecutionEngine
            │                              │
            │                              ▼
      [DividendEvent] ──> Portfolio <── [FillEvent]
            │
            ▼
     Performance Metrics
```

---

## Key Features

### ⚡ $O(1)$ Incremental Strategy Calculations
* Key technical indicators (such as the Moving Average Convergence Divergence `MACD` and `Trend Following` moving averages) have been refactored from legacy $O(n^2)$ array-based recalculations to **incremental $O(1)$ state updates**.
* The strategy object caches mathematical properties (such as previous-bar Exponential Moving Averages) and updates them bar-by-bar, providing a massive speedup on large datasets without altering mathematical correctness.

### 🛡️ Lookahead-Free Execution
* In standard simulation, orders might be filled at the same bar's close price, introducing potential lookahead bias if a strategy triggers a signal at the close.
* The engine supports **Next-Bar Pricing** (`next_bar_pricing=True` inside `SimulatedExecutionEngine`). When active, order events generated at the close of bar $t$ are queued and executed at the open price of the subsequent bar $t+1$.

### 💸 Realistic Transaction Cost Profiles
* **Commission Profiles**: Supports multiple commission calculation rules:
  * `flat`: A constant charge per trade fill (e.g. $1.00).
  * `per_share`: A flat fee charged per share traded (e.g. $0.005/share).
  * `percentage`: A percentage of the total trade value (e.g. 0.001 for 0.1% transaction cost).
* **Slippage Models**: Simulates transaction slippage by applying a basis point (bps) penalty to the fill price (e.g., executing a BUY at $Price \times (1 + \frac{slippage\_bps}{10000})$).

### 📝 Complete Round-Trip Trade Tracking
The Portfolio tracks capital allocations, active positions, dividend distributions, and registers completed round-trip trades. Completed trades are returned as a list of dictionaries with the following schema, suitable for JSON storage in relational databases:
* `symbol` (str) — Symbol traded.
* `qty` (float) — Number of units executed.
* `entry_price` (float) — Weighted entry cost.
* `exit_price` (float) — Weighted exit cost.
* `entry_time` (datetime) — UTC timezone-aware entry timestamp.
* `exit_time` (datetime) — UTC timezone-aware exit timestamp.
* `pnl` (float) — Absolute dollar profit or loss.
* `return` (float) — Rate of return for the trade.
* `direction` (str) — Trade direction: `"long"` or `"short"`.

---

## Setup & Installation

### 1. Structure Check
Ensure that the pipeline, engine, and terminal repositories are cloned side-by-side in the same parent directory:
```
parent_workspace/
├── Backtester-Oriented-Market-Data-Pipeline/
├── Event-Driven-Backtesting-Engine/   <-- This repo
└── strategy-research-terminal/
```

### 2. Configure Virtual Environment
Create and activate a virtual environment:
```bash
cd Event-Driven-Backtesting-Engine
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies & Editable Submodule
Install dependencies and install the data pipeline package in editable mode:
```bash
pip install -r requirements.txt
pip install -e ../Backtester-Oriented-Market-Data-Pipeline
```
Installing the data pipeline as an editable package (`-e`) allows the engine to import the `market_data` module without copying source files.

### 4. Run the Test Suite
Run the test suite using the virtual environment's pytest:
```bash
.venv/bin/pytest tests/ -v
```
Verify that all **126 tests** pass successfully.

---

## Usage Guide

### Command Line Interface

Run a quick test simulation from the terminal:
```bash
# Run with default settings (moving average crossover on AAPL)
python run_backtest.py

# Backtest multiple symbols with a custom strategy configuration
python run_backtest.py \
  --symbols AAPL MSFT \
  --strategy rsi \
  --cash 250000 \
  --qty 150
```

### Python API Integration

Import the runner inside your backend or custom notebooks:
```python
from datetime import date
from engine.config import BacktestConfig, StrategyConfig
from run_backtest import run_backtest_from_config

# Configure strategy parameters
strategy_cfg = StrategyConfig(
    type="macd",
    parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}
)

# Set up complete simulation parameters
config = BacktestConfig(
    symbols=["AAPL", "MSFT"],
    strategy=strategy_cfg,
    start_date=date(2022, 1, 1),
    end_date=date(2025, 1, 1),
    initial_capital=100000.0,
    position_sizing="risk_based",  # "fixed" | "percentage" | "risk_based"
    position_size=100.0,
    risk_per_trade=0.02,           # Kelly/ATR percentage size
    commission_model="percentage", # "flat" | "per_share" | "percentage"
    commission_value=0.001,        # 0.1% fee
    slippage_bps=5.0               # 5 basis points slippage
)

# Run the simulation
result = run_backtest_from_config(config)

# Output summary metrics
print(f"Final Equity    : ${result.final_value:,.2f}")
print(f"Total Return    : {result.total_return * 100:.2f}%")
print(f"Sharpe Ratio    : {result.sharpe_ratio:.4f}")
print(f"Max Drawdown    : {result.max_drawdown * 100:.2f}%")
print(f"Total Trades    : {result.total_trades}")
print(f"Dividend Income : ${result.total_dividend_income:,.2f}")
```

---

## Performance Metrics Suite

The engine computes a comprehensive set of statistics on every snapshot, including:
* **Sharpe Ratio**: Annualized risk-adjusted return (calculated using daily returns vs. a risk-free rate of 0%, scaled by $\sqrt{252}$).
* **Max Drawdown**: Largest peak-to-trough drop in net asset value.
* **CAGR**: Compound Annual Growth Rate over the simulation period.
* **Alpha vs. Benchmark**: Excess return relative to the benchmark index (defaults to `SPY`).
* **Win Rate**: Percentage of round-trip trades that closed with positive PnL.
* **Volatility**: Annualized standard deviation of daily portfolio returns.

---

## License

MIT — see [LICENSE](LICENSE).
