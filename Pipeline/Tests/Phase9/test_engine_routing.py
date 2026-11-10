"""
Phase 9: Engine Routing
Requirement: If the user selects 'catboost' or 'lightgbm', verify that the internal engine initialized is correctly routed.
"""

from __future__ import annotations
import pytest
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.trees import Model

def test_phase9_engine_routing_catboost():
    """Verify routing to CatBoost."""
    model = Model(model_type='catboost', task_type='regression', loss_function='RMSE')
    assert model.model_type == 'catboost'
    # We check if the internal params are set correctly
    assert model.params['loss_function'] == 'RMSE'

def test_phase9_engine_routing_lightgbm():
    """Verify routing to LightGBM."""
    model = Model(model_type='lightgbm', task_type='regression', loss_function='rmse')
    assert model.model_type == 'lightgbm'
    assert model.params['objective'] == 'rmse'
