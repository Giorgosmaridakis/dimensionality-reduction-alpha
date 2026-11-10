"""
Phase 2: Dimensional Accuracy
Requirement: Assert total rows match unique dates multiplied by tickers.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase2.test_config import panel_context

def test_dimensional_accuracy(panel_context):
    """
    Asserts that the total number of rows in the final panel matches 
    the intersection of dates and tickers across all alpha files.
    Check with dropna=False to verify the perfect grid logic.
    """
    universe = panel_context["universe"]
    period = panel_context["period"]
    df_raw = panel_context["df_raw"]

    if df_raw.empty:
        pytest.skip(f"No alpha data found for {universe} {period}")

    # Analyze source files for expected dimensions
    unique_dates = df_raw['Date'].nunique()
    unique_tickers = df_raw['ticker'].nunique()
    expected_rows = unique_dates * unique_tickers
    
    print(f"\n[OK] {universe} {period}: Grid consistency check: {unique_dates} dates * {unique_tickers} tickers = {expected_rows} rows.")
    assert len(df_raw) == expected_rows, f"Panel dimensions {len(df_raw)} do not match grid logic {expected_rows}!"
