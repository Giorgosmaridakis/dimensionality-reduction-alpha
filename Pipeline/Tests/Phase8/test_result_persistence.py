"""
Phase 8: Result Persistence
Requirement: Confirm that the tuning metrics for all combinations are saved to a CSV for post-hoc diagnostic review.
"""
from __future__ import annotations
import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.tuning import HyperparameterTuner
from Tests.Phase8.test_config import tuning_data

def test_phase8_result_persistence(tuning_data, tmp_path):
    """Requirement: Confirm that the tuning metrics for all combinations are saved to a CSV."""
    X, y, qids = tuning_data
    tuner = HyperparameterTuner(model_type='lightgbm', task_type='regression', loss_function='regression')
    
    param_grid = {'learning_rate': [0.1]}
    export_csv = tmp_path / "tuning_results_log.csv"
    
    # Mock evaluate_combo to avoid real training
    with patch.object(HyperparameterTuner, 'evaluate_combo') as mock_eval:
        def mock_eval_side_effect(grid_params, **kwargs):
            if 'tuning_results' in kwargs and kwargs['tuning_results'] is not None:
                res = grid_params.copy()
                res['Validation_Score'] = 0.5
                kwargs['tuning_results'].append(res)
            return 0.5
            
        mock_eval.side_effect = mock_eval_side_effect
        
        tuner.tune(
            X, y, qids, param_grid, 
            results_export_path=str(export_csv)
        )
        
        # Verify persistence
        assert export_csv.exists(), "The results CSV was not created."
        
        df_saved = pd.read_csv(export_csv)
        assert 'learning_rate' in df_saved.columns
        assert 'Validation_Score' in df_saved.columns
        assert df_saved.iloc[0]['Validation_Score'] == 0.5
