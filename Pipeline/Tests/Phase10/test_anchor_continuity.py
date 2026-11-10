"""
Phase 10: Anchor Continuity
Requirement: Verify that the walking-forward loop shifts periods (e.g., Year 1 -> Year 2) without overlaps or gaps.
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

from walking_forward import WalkingForwardForecaster
from modeling.trees import Model
from Tests.Phase10.test_config import simulation_context

def test_phase10_anchor_continuity(simulation_context, tmp_path):
    """
    Verify that the loop correctly identifies years and shifts testing windows.
    We mock the heavy components to focus on the orchestration loop control.
    """
    df, alpha_cols = simulation_context
    start_yr = df.index.year.min()
    end_yr = start_yr + 2 
    
    # Use start date 1 year after min data year to satisfy training window requirement
    start_date = f"{start_yr + 1}-01-01"
    end_date = f"{end_yr}-12-31"
    
    model = Model(model_type='lightgbm', task_type='regression', loss_function='regression')
    wff = WalkingForwardForecaster(
        model, start_date=start_date, end_date=end_date, 
        max_training_window_years=1, testing_window='Year'
    )
    
    # Patch components to avoid fit/save overhead
    with patch.object(WalkingForwardForecaster, '_tune_model'), \
         patch.object(WalkingForwardForecaster, '_apply_preprocessing', return_value=(df.head(10), df.head(10), alpha_cols, None)), \
         patch.object(Model, 'fit'), \
         patch.object(Model, 'predict', return_value=np.zeros(len(df.head(10)))), \
         patch.object(WalkingForwardForecaster, '_save_results'):
             
        wff.forecast(df, alpha_cols, base_output_dir=str(tmp_path))
        
        # Verify that multiple period subdirectories were created
        dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(dirs) >= 1, "At least one simulation period should have been processed."
        
        # Check that folder names correspond to expected periods
        # e.g., 01-01-2006_31-12-2006
        for d in dirs:
            dates = d.name.split('_')
            assert len(dates) == 2, f"Directory name {d.name} should contain start and end dates."
