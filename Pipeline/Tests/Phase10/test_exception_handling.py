"""
Phase 10: Exception Handling
Requirement: Ensure the loop skips periods with missing data gracefully without crashing the full run.
"""

from __future__ import annotations
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from walking_forward import WalkingForwardForecaster
from modeling.trees import Model
from Tests.Phase10.test_config import simulation_context

def test_phase10_graceful_skipping(simulation_context, tmp_path, capsys):
    """
    Verify that if a period splits into empty data, the loop logs the skip 
    and continues to the next period.
    """
    df, alpha_cols = simulation_context
    start_yr = df.index.year.min()
    start_date = f"{start_yr + 1}-01-01"
    
    model = Model(model_type='lightgbm', task_type='regression', loss_function='regression')
    # Run for 1 period
    wff = WalkingForwardForecaster(model, start_date=start_date, end_date=start_date)
    
    # Force empty train/test split during the loop
    with patch('pandas.DataFrame.loc', return_value=pd.DataFrame()):
        wff.forecast(df, alpha_cols, base_output_dir=str(tmp_path))
        
    captured = capsys.readouterr()
    assert "Skipping" in captured.out, "Missing data skip message not found in stdout."
    assert "Complete" in captured.out, "Forecaster did not finish gracefully after skip."
