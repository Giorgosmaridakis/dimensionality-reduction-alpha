import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder

# Phase 2 Test Configuration Matrix
# ================================
TEST_PARAMS = [
    ("ETFs", "2006-2022"),
    ("ETFs", "2023-2025"),
    ("S&P500", "2020-2024"),
    ("S&P500", "2025-End")
]

@pytest.fixture(params=TEST_PARAMS, scope="session")
def panel_context(request):
    """
    Builds the panel for each (universe, period) pair only ONCE
    and shares it across all Phase 2 tests.
    """
    universe, period = request.param
    dl = DataLoader(universe=universe, time_period=period)
    
    if not dl.universe_path.exists():
        pytest.skip(f"Data not found for {universe} {period}")

    builder_raw = DatasetBuilder(dropna=False)
    builder_clean = DatasetBuilder(dropna=True)
    
    df_raw = builder_raw.prepare_stacked_data(dl)
    df_clean = builder_clean.prepare_stacked_data(dl)
    
    return {
        "universe": universe,
        "period": period,
        "dl": dl,
        "df_raw": df_raw,
        "df_clean": df_clean
    }
