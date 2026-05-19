"""
Shared fixtures and configuration for the engine test suite.
"""

import sys
import os
import pytest  # type: ignore[import-not-found]
from datetime import datetime, date

# ---------------------------------------------------------------------------
# Path setup — ensure engine package and pipeline are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "Backtester-Oriented-Market-Data-Pipeline")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

# Default data directory — pipeline's data folder.
# load_from_parquet expects files named <SYMBOL>.parquet (no interval suffix).
DATA_DIR = os.path.join(PIPELINE_ROOT, "data")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires real Parquet data on disk")
    config.addinivalue_line("markers", "slow: long-running stress tests")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def now():
    return datetime(2024, 1, 15, 10, 0, 0)


@pytest.fixture
def data_dir():
    """Path to the pipeline's data directory with real Parquet files."""
    return DATA_DIR


@pytest.fixture
def has_aapl_data(data_dir):
    """Skip test if AAPL.parquet is not available in the data directory."""
    aapl_path = os.path.join(data_dir, "AAPL.parquet")
    if not os.path.exists(aapl_path):
        pytest.skip(
            f"AAPL.parquet not found at {aapl_path} — "
            "run: python scripts/fetch_data.py --symbols AAPL,MSFT "
            "--start 2020-01-01 --end 2024-01-01 --interval 1d --dividends"
        )
    return True


@pytest.fixture
def has_msft_data(data_dir):
    """Skip test if MSFT.parquet is not available in the data directory."""
    msft_path = os.path.join(data_dir, "MSFT.parquet")
    if not os.path.exists(msft_path):
        pytest.skip(
            f"MSFT.parquet not found at {msft_path} — "
            "run: python scripts/fetch_data.py --symbols AAPL,MSFT "
            "--start 2020-01-01 --end 2024-01-01 --interval 1d --dividends"
        )
    return True

