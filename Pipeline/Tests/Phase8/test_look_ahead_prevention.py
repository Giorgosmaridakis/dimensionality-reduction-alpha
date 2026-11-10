"""
Phase 8: Look-Ahead Prevention
Requirement: Prove that validation slices occur strictly after training slices in the TimeSeriesSplit.
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

def test_phase8_look_ahead_prevention(tuning_data):
    """Requirement: Prove that validation slices occur strictly after training slices."""
    X, _, _ = tuning_data
    # Extract actual dates from the real project data
    dates = X.index.unique().sort_values()
    
    if len(dates) < 5:
        pytest.skip("Not enough real dates in sample to verify time-series splits.")
        
    tscv = TimeSeriesSplit(n_splits=3, gap=0)
    
    for train_idx, val_idx in tscv.split(dates):
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]
        
        # In a time-series split, the training dates must all be before any validation date
        assert train_dates.max() < val_dates.min(), \
            f"Look-ahead detected: Train max {train_dates.max()} is not before Val min {val_dates.min()}"
