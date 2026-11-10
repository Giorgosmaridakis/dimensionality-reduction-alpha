import pytest
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data_loader import DataLoader
from Tests.Phase1.test_config import TEST_PARAMS

"""
Phase 1: Real Price Discovery
Requirement: Every dataset period must contain a 'Returns.csv' or 'Close.csv'.
"""

@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_price_file_existence(universe, period):
    """
    Checks the disk to ensure the mandatory price files exist.
    """
    dl = DataLoader(universe=universe, time_period=period)
    
    if not dl.universe_path.exists():
        pytest.skip(f"Path {dl.universe_path} not found.")

    found_returns = dl.get_returns()
    
    # We expect at least one returns or close file in the price-relevant directory
    assert len(found_returns) > 0, (
        f"Price Discovery Failure for {universe} ({period}).\n"
        f"No 'Returns.csv' found in {dl.universe_path / period}"
    )
    
    print(f"\n[OK] Price data found for {universe} - {period}")
