"""
Phase 9: Prediction Scale
Requirement: For regression, ensure output range is consistent with returns; for ranking, ensure relative ordering.
"""

from __future__ import annotations
import pytest
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.trees import Model
from Tests.Phase9.test_config import model_data

def test_phase9_regression_scale(model_data):
    """Verify regression output is numeric and in a float range."""
    X, y, qids, _ = model_data
    model = Model(model_type='lightgbm', task_type='regression', loss_function='regression', n_estimators=5)
    
    model.fit(X, y, qids)
    preds = model.predict(X)
    
    assert isinstance(preds, np.ndarray)
    assert preds.dtype.kind in 'fc' # float or complex (usually float)
    assert len(preds) == len(X)
    # Basic check: financial returns aren't usually in the millions
    assert np.abs(preds).mean() < 10.0 

def test_phase9_ranking_ordering(model_data):
    """Verify ranking output provides relative values."""
    X, y, qids, _ = model_data
    # Ranking targets must be varied integers (0-9 scores)
    y_int = (y.groupby(level=0).rank(pct=True) * 9.999).astype(int)
    
    model = Model(model_type='lightgbm', task_type='ranking', loss_function='lambdarank', n_estimators=10)
    
    model.fit(X, y_int, qids)
    preds = model.predict(X)
    
    # In ranking, we care about unique values and spread
    assert len(np.unique(preds)) > 1, "Ranking model should produce a variety of scores."
    assert len(preds) == len(X)
