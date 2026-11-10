import pytest
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data_loader import DataLoader
from Tests.Phase1.test_config import TEST_PARAMS

"""
Phase 1: Universe Isolation
Requirement: Ensure different universe configurations do not overlap.
"""

@pytest.mark.parametrize("universe, period", TEST_PARAMS)
def test_logical_isolation(universe, period):
    """
    Checks that a DataLoader for one universe doesn't have 
    access to the strings or paths of others.
    """
    dl = DataLoader(universe=universe, time_period=period)
    
    # We get all other universes mentioned in the config
    other_universes = {p[0] for p in TEST_PARAMS if p[0] != universe}
    
    for other in other_universes:
        assert str(other) not in str(dl.universe_path), (
            f"Leakage detected: {universe} path contains reference to {other}!"
        )
        
    print(f"\n[OK] Isolation verified for {universe} - {period}")
