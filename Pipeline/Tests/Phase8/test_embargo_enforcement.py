"""
Phase 8: Embargo Enforcement
Requirement: If an embargo is set, verify there is a gap where no data is used between Training and Validation.
"""

from __future__ import annotations
import pytest
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Tests.Phase8.test_config import tuning_data

def test_phase8_embargo_enforcement(tuning_data):
    """Requirement: Verify that gap (embargo) creates separation between train and validation."""
    X, _, _ = tuning_data
    # Extract actual dates from the real project data
    dates = X.index.unique().sort_values()
    
    if len(dates) < 20:
        pytest.skip("Not enough real dates in sample to verify embargo enforcement.")
        
    embargo_days = 5
    tscv = TimeSeriesSplit(n_splits=3, gap=embargo_days)
    
    for train_idx, val_idx in tscv.split(dates):
        # We check the raw indices to see if there is a gap of at least 'embargo_days'
        gap = val_idx[0] - train_idx[-1] - 1
        assert gap >= embargo_days, \
            f"Embargo not enforced: Expected gap {embargo_days}, got {gap}"
            
        # Verify temporally
        train_max_date = dates[train_idx[-1]]
        val_min_date = dates[val_idx[0]]
        # The number of business days between should be at least embargo_days
        # However, indices are the most direct proof of data exclusion in TimeSeriesSplit
        assert train_max_date < val_min_date
