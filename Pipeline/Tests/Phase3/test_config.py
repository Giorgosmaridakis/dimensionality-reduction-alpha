import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder

# Phase 3 Test Configuration Matrix
# ================================
TEST_PARAMS = [
    ("ETFs", "2006-2022"),
    ("ETFs", "2023-2025"),
    ("S&P500", "2020-2024"),
    ("S&P500", "2025-End")
]

@pytest.fixture(params=TEST_PARAMS, scope="session")
def merged_context(request):
    """
    Builds the final signal-date panel for Phase 3 tests.

    The current target convention is:
        X_t -> adjusted-close return from t+1 to t+2

    This matches alpha availability after close_t and the backtester's one-row
    weight shift.
    Caches the result per (universe, period) for the entire session.
    """
    universe, period = request.param
    dl = DataLoader(universe=universe, time_period=period)
    
    if not dl.universe_path.exists():
        pytest.skip(f"Data not found for {universe} {period}")

    builder = DatasetBuilder(dropna=True)
    
    # 1. Step 2 logic: Panel stacking
    panel_df = builder.prepare_stacked_data(dl)
    if panel_df.empty:
        pytest.skip(f"No alpha data found for {universe} {period}")
        
    # 2. Step 3 logic: target merging from adjusted close t+1 -> t+2.
    final_df = builder.merge_features_and_targets(panel_df, dl)
    
    return {
        "universe": universe,
        "period": period,
        "dl": dl,
        "builder": builder,
        "panel_df": panel_df,
        "final_df": final_df
    }
