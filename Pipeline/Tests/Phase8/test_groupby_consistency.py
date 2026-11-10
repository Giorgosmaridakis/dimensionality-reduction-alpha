"""
Phase 8: Group-By Consistency
Requirement: Assert that the IC is calculated cross-sectionally (per date) before being averaged, rather than across the entire pool at once.
"""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.tuning import HyperparameterTuner
from Tests.Phase8.test_config import tuning_data

def test_phase8_groupby_consistency_cross_sectional(tuning_data):
    """
    Requirement: Assert that the IC is calculated cross-sectionally (per date) before being averaged.
    """
    X, y, qids = tuning_data
    tuner = HyperparameterTuner(model_type='lightgbm', task_type='regression', loss_function='regression')
    
    # We take a small slice of data with 2 dates
    all_dates = X.index.unique().sort_values()
    if len(all_dates) < 2:
        pytest.skip("Not enough dates for cross-sectional test")
        
    date1, date2 = all_dates[0], all_dates[1]
    mask = X.index.isin([date1, date2])
    
    X_sub = X[mask].copy()
    y_sub = y[mask].copy()
    qids_sub = qids[mask].copy()
    
    # We patch the Model to return specific predictions per date
    with patch('modeling.tuning.Model') as MockModel:
        model_inst = MockModel.return_value
        
        # We define a side effect for predict that returns different values based on the input length
        # But even better, we know evaluate_combo calls predict once for the whole validation fold
        # and then slices it using the qids.
        
        # Let's say date1 has 3 tickers and date2 has 3 tickers
        n1 = len(X_sub.loc[[date1]])
        n2 = len(X_sub.loc[[date2]])
        
        # Define mock predictions
        preds = np.random.randn(len(X_sub))
        model_inst.predict.return_value = preds
        
        # We need to manually calculate the expected IC
        # evaluate_combo uses np.corrcoef for regression
        
        def calc_ic(p, a):
            return np.corrcoef(p, a)[0, 1]
            
        ic1 = calc_ic(preds[:n1], y_sub.iloc[:n1].values)
        ic2 = calc_ic(preds[n1:n1+n2], y_sub.iloc[n1:n1+n2].values)
        expected_avg_ic = (ic1 + ic2) / 2
        
        # Mock tscv to return our specific Indices
        # Instead of mocking tscv, we can just call evaluate_combo directly with our data
        # but evaluate_combo iterates over tscv.split(unique_dates).
        
        mock_tscv = MagicMock()
        mock_tscv.split.return_value = [(np.array([], dtype=int), np.array([0, 1], dtype=int))] # Indices of unique_dates
        
        # Call evaluate_combo
        grid_params = {'learning_rate': 0.1}
        
        # We need to inject our tscv into the call or use the tuner's logic
        # Actually, evaluate_combo takes tscv and unique_dates as arguments now!
        
        actual_avg_ic = tuner.evaluate_combo(
            grid_params, X_sub, y_sub, qids_sub, 
            tscv=mock_tscv, unique_dates=np.array([date1, date2]),
            tuning_metric='ic', base_params={}
        )
        
        assert np.isclose(actual_avg_ic, expected_avg_ic), \
            f"Cross-sectional IC failed. Expected {expected_avg_ic}, got {actual_avg_ic}"
