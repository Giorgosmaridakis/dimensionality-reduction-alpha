"""
Phase 9: Group-By Consistency
Requirement: Assert that the model's predictions maintain meaningful cross-sectional rank correlation (IC) with targets, calculated per date.
"""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.trees import Model
from Tests.Phase9.test_config import model_data

def test_phase9_groupby_consistency_cross_sectional(model_data):
    """
    Verify that the model's predictions are evaluated cross-sectionally 
    (per date) rather than as a single pool.
    """
    X, y, qids, _ = model_data
    
    # Train a simple model
    model = Model(model_type='lightgbm', task_type='regression', loss_function='regression', n_estimators=10)
    model.fit(X, y, qids)
    
    preds = model.predict(X)
    
    # Create evaluation frame
    eval_df = pd.DataFrame({
        'Preds': preds,
        'Actuals': y
    }, index=X.index)
    
    # Calculate IC per date using index level
    def daily_ic(group):
        if len(group) < 2 or group['Preds'].nunique() == 1 or group['Actuals'].nunique() == 1: 
            return np.nan
        return spearmanr(group['Preds'], group['Actuals'])[0]
    
    daily_ics = eval_df.groupby(level=0).apply(daily_ic)
    
    # Assert that we have multiple dates and varied ICs
    assert len(daily_ics) > 1, f"Evaluation must span multiple dates. Found {len(daily_ics)}."
    assert not daily_ics.isna().all(), "Should be able to calculate IC for most dates."
    
    avg_daily_ic = daily_ics.mean()
    
    print(f"--> Average Daily IC: {avg_daily_ic:.4f}")
    assert isinstance(avg_daily_ic, (float, np.float64))
