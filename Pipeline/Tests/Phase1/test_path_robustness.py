import pytest
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data_loader import DataLoader
from Tests.Phase1.test_config import TEST_PARAMS

"""
Phase 1: Path Robustness
Requirement: Ensure DataLoader resolves paths correctly for all configurations.
"""

@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_path_resolution_and_type(universe, period):
    """
    Verifies that for every universe/period, the DataLoader creates valid Path objects.
    """
    
    dl = DataLoader(universe=universe, time_period=period)
    
    # Check that paths are non-empty and correctly typed
    assert isinstance(dl.base_path, Path)
    assert isinstance(dl.universe_path, Path)
    assert str(universe) in str(dl.universe_path)
    
    # Verify that the path is not just a root but contains the specific period
    assert str(period) in str(dl.universe_path / period)
    
    print(f"\n[OK] Path Robustness verified for {universe} - {period}")
