import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data_loader import DataLoader
from Tests.Phase1.test_config import TEST_PARAMS

"""
Phase 1: Mathematical Return Formula
Requirement: Verify the (Close_t - Close_t-1) / Close_t-1 identity using real data.
"""

@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_real_data_roc_identity(universe, period):
    """
    Loads real Close.csv and Returns.csv and ensures the formula holds.
    """
    dl = DataLoader(universe=universe, time_period=period)
    
    if not dl.universe_path.exists():
        pytest.skip("Data not found.")

    # 1. Look for Adjusted_Close.csv only
    adj_close_matches = list(dl.universe_path.rglob('Adjusted_Close.csv'))
    return_matches = list(dl.universe_path.rglob('Returns.csv'))
    
    if not adj_close_matches or not return_matches:
        pytest.skip(f"Could not find both Adjusted_Close.csv and Returns.csv for {universe} {period}")

    price_file = adj_close_matches[0]

    # Use the discovered price file
    df_close = pd.read_csv(price_file, index_col=0)
    df_returns = pd.read_csv(return_matches[0], index_col=0)
    
    # 2. Pick a random Asset (Column) to check
    # We use the first asset found in both
    common_assets = list(set(df_close.columns) & set(df_returns.columns))
    if not common_assets:
        pytest.fail("No common assets found between Close and Returns!")
        
    asset = common_assets[0]
    
    # 3. Calculate expected ROC using Close data: (P_t - P_{t-1}) / P_{t-1}
    expected = df_close[asset].pct_change().dropna()
    actual = df_returns[asset].loc[expected.index] # Align dates
    
    # 4. Assertion with tolerance
    # NOTE: This may still fail on Dividend/Split days if Close.csv is not 'Adjusted'
    pd.testing.assert_series_equal(
        expected, 
        actual, 
        check_names=False,
        atol=1e-5,
        obj=f"RoC Verification for {asset} ({universe} {period})"
    )
    
    print(f"\n[OK] Math identity confirmed for {asset} in {universe} - {period}")
