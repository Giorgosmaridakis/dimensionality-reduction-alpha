"""
Phase 8: Rank vs Linear Logic
Requirement: Verify that for 'ranking' tasks, a Spearman (rank-based) correlation is used, and for 'regression', a Pearson correlation is used.
"""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
import scipy.stats as stats
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.tuning import HyperparameterTuner
from Tests.Phase8.test_config import tuning_data

def test_phase8_rank_vs_linear_logic_distinctness(tuning_data):
    """
    Requirement: Verify that for 'ranking' tasks, a Spearman (rank-based) correlation is used, 
    and for 'regression', a Pearson correlation is used.
    """
    X, y, qids = tuning_data
    
    # We'll use a small slice for fast but real calculation
    all_dates = X.index.unique().sort_values()
    if len(all_dates) < 1:
        pytest.skip("Not enough dates for correlation test")
        
    date = all_dates[0]
    X_sub = X.loc[[date]].copy()
    y_sub = y.loc[[date]].copy()
    qids_sub = qids.loc[[date]].copy()
    
    if len(X_sub) < 5:
         pytest.skip("Not enough tickers on this date for a meaningful correlation comparison.")

    # We patch the Model to return specific predictions
    with patch('modeling.tuning.Model') as MockModel:
        model_inst = MockModel.return_value
        
        # Consistent predictions to compare Pearson vs Spearman
        preds = np.random.randn(len(X_sub))
        model_inst.predict.return_value = preds
        
        # --- TEST 1: REGRESSION (Pearson) ---
        tuner_reg = HyperparameterTuner(model_type='lightgbm', task_type='regression', loss_function='regression')
        
        mock_tscv = MagicMock()
        mock_tscv.split.return_value = [(np.array([], dtype=int), np.array([0], dtype=int))] # Index 0 of unique_dates
        
        actual_ic_reg = tuner_reg.evaluate_combo(
            {'learning_rate': 0.1}, X_sub, y_sub, qids_sub, 
            tscv=mock_tscv, unique_dates=np.array([date]),
            tuning_metric='ic', base_params={}
        )
        
        # Manual Pearson
        expected_pearson = np.corrcoef(preds, y_sub.values)[0, 1]
        assert np.isclose(actual_ic_reg, expected_pearson, equal_nan=True), f"Regression IC should be Pearson. Got {actual_ic_reg}, expected {expected_pearson}"
        
        # --- TEST 2: RANKING (Spearman) ---
        tuner_rank = HyperparameterTuner(model_type='lightgbm', task_type='ranking', loss_function='lambdarank')
        
        actual_ic_rank = tuner_rank.evaluate_combo(
            {'learning_rate': 0.1}, X_sub, y_sub, qids_sub, 
            tscv=mock_tscv, unique_dates=np.array([date]),
            tuning_metric='ic', base_params={}
        )
        
        # Manual Spearman
        expected_spearman, _ = stats.spearmanr(preds, y_sub.values)
        assert np.isclose(actual_ic_rank, expected_spearman, equal_nan=True), f"Ranking IC should be Spearman. Got {actual_ic_rank}, expected {expected_spearman}"
        
        # Verify they are distinct (sanity check for real data)
        assert not np.isclose(actual_ic_reg, actual_ic_rank), "Pearson and Spearman results should typically differ on real data."
