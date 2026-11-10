"""
Phase 10: FileSystem Structure
Requirement: Verify that results (predictions, IC metrics, tuning logs) are saved into correctly nested directories.
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from walking_forward import WalkingForwardForecaster
from modeling.trees import Model
from Tests.Phase10.test_config import simulation_context

def test_phase10_filesystem_nesting(simulation_context, tmp_path):
    """
    Verify that results are saved into folders named by the period date range 
    and contain all expected diagnostic files.
    """
    df, alpha_cols = simulation_context
    start_yr = df.index.year.min()
    # Find the first valid date in the target year to avoid holidays/weekends
    start_date = df.loc[f"{start_yr + 1}-01"].index[0].strftime('%Y-%m-%d')
    
    model = Model(model_type='lightgbm', task_type='regression', loss_function='regression')
    # Run for just one period to verify structure
    wff = WalkingForwardForecaster(model, start_date=start_date, end_date=start_date)
    
    # We allow _save_results to run but mock the model behavior for speed
    with patch.object(Model, 'fit'), \
         patch.object(Model, 'predict', return_value=pd.Series([0]*len(df.loc[start_date:start_date]), index=df.loc[start_date:start_date].index)):
             
        wff.forecast(df, alpha_cols, base_output_dir=str(tmp_path), param_grid={'learning_rate': [0.1]})
        
        # Verify period directory exists
        period_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(period_dirs) >= 1
        
        period_dir = period_dirs[0]
        p_name = period_dir.name # e.g. 01-01-2006_01-01-2006
        
        # Check for core files
        assert (period_dir / f"predictions_{p_name}.csv").exists(), "Predictions CSV missing."
        assert (period_dir / f"prediction_metrics_{p_name}.csv").exists(), "Prediction metrics CSV missing."
        assert (period_dir / f"daily_oos_ic_{p_name}.csv").exists(), "Daily OOS IC CSV missing."
        assert (tmp_path / "daily_oos_ic_series.csv").exists(), "Aggregated daily OOS IC series missing."
        assert (period_dir / f"tuning_metrics_{p_name}.csv").exists(), "Tuning logs missing."
        assert (period_dir / f"ic_raw_features_{p_name}.csv").exists(), "Raw feature IC logs missing."
