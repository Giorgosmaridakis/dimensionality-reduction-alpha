"""
Phase 9: GPU Handshake
Requirement: If GPU is enabled, verify that the correct flags are passed to native engines.
"""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.trees import Model

def test_phase9_gpu_handshake_catboost():
    """Verify CatBoost GPU flag."""
    model = Model(model_type='catboost', task_type='regression', loss_function='RMSE', use_gpu=True)
    assert model.params['task_type'] == 'GPU'

def test_phase9_gpu_handshake_lightgbm():
    """Verify LightGBM GPU flag."""
    model = Model(model_type='lightgbm', task_type='regression', loss_function='rmse', use_gpu=True)
    assert model.params['device'] == 'gpu'
