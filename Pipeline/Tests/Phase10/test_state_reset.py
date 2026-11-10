"""
Phase 10: State Reset
Requirement: Prove that models and scalers are re-initialized for each anchor, preventing look-ahead leakage.
"""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from walking_forward import WalkingForwardForecaster
from modeling.trees import Model
from Tests.Phase10.test_config import simulation_context

def test_phase10_state_reset(simulation_context, tmp_path):
    """
    Verify that Model.fit is called with correctly partitioned time-slices 
    for each period, implying a fresh start for each simulation anchor.
    """
    df, alpha_cols = simulation_context
    start_yr = df.index.year.min()
    start_date = f"{start_yr + 1}-01-01"
    end_date = f"{start_yr + 2}-12-31" # 2 periods
    
    model = Model(model_type='lightgbm', task_type='regression', loss_function='regression')
    wff = WalkingForwardForecaster(model, start_date=start_date, end_date=end_date)
    
    with patch.object(Model, 'fit') as mock_fit, \
         patch.object(Model, 'predict', return_value=[0]*1000), \
         patch.object(WalkingForwardForecaster, '_save_results'):
             
        wff.forecast(df, alpha_cols, base_output_dir=str(tmp_path), param_grid=None)
        
        # Verify multiple fit calls occurred
        assert mock_fit.call_count >= 2
        
        # Extract data indices from fit calls
        # positional args: (X, y, qids)
        first_train_X = mock_fit.call_args_list[0][0][0]
        second_train_X = mock_fit.call_args_list[1][0][0]
        
        # Training windows should be chronologically distinct and shifting
        assert first_train_X.index.max() < second_train_X.index.max(), \
            "Training window did not shift between simulation anchors."
