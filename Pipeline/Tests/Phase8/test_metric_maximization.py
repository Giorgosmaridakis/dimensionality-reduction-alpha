"""
Phase 8: Metric Maximization
Requirement: Feed a known parameter grid where one setting is clearly superior; ensure the tuner selects that setting correctly.
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

def test_phase8_metric_maximization(tuning_data):
    """Requirement: Verify the tuner selects the mathematically superior setting."""
    X, y, qids = tuning_data
    tuner = HyperparameterTuner(model_type='lightgbm', task_type='regression', loss_function='regression')
    
    # Grid with two settings
    param_grid = {'learning_rate': [0.1, 0.01]}
    
    # We patch the evaluate_combo method to simulate specific scores
    with patch.object(HyperparameterTuner, 'evaluate_combo') as mock_eval:
        def mock_eval_side_effect(grid_params, **kwargs):
            # We want the tuner to get a better score for learning_rate 0.1
            score = 0.5 if grid_params.get('learning_rate') == 0.1 else 0.2
            
            # Simulate population of results list
            if 'tuning_results' in kwargs and kwargs['tuning_results'] is not None:
                res = grid_params.copy()
                res['Validation_Score'] = score
                kwargs['tuning_results'].append(res)
            return score
            
        mock_eval.side_effect = mock_eval_side_effect
        
        best_params, results_df = tuner.tune(
            X, y, qids, param_grid, 
            use_optuna=False
        )
        
        # Verify that 0.1 was chosen
        assert best_params['learning_rate'] == 0.1
        assert results_df.iloc[0]['Validation_Score'] == 0.5
        assert len(results_df) == 2
