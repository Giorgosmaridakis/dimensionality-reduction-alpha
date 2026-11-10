"""
Phase 9: Hyperparameter Application
Requirement: Assert that the 'Best Params' found in Phase 8 are actually being used during the final fit() call.
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
from Tests.Phase9.test_config import model_data

def test_phase9_hyperparameter_propagation(model_data):
    """
    Verify that custom hyperparameters (like depth or iterations) are 
    actually passed to the underlying engine during initialization.
    """
    X, y, qids, _ = model_data
    custom_params = {'max_depth': 7, 'n_estimators': 50, 'learning_rate': 0.05}
    
    # We test with LightGBM as an example
    model = Model(
        model_type='lightgbm', 
        task_type='regression', 
        loss_function='regression', 
        **custom_params
    )
    
    # Assert they are stored in self.params
    assert model.params['max_depth'] == 7
    assert model.params['n_estimators'] == 50
    assert model.params['learning_rate'] == 0.05
    
    # In 'fit', tree_params should filter out decomposition prefixes and keep the rest
    with patch('modeling.trees.lgb.train') as mock_train:
        # We also need to patch _prepare_data to return a dummy Dataset
        with patch.object(Model, '_prepare_data', return_value=MagicMock()):
            model.fit(X, y, qids, verbose=False)
            
            # The first argument to lgb.train's 'params' should contain our custom ones
            # Note: Model.fit pops n_estimators and uses it as num_boost_round
            args, kwargs = mock_train.call_args
            actual_params = kwargs['params']
            assert actual_params['max_depth'] == 7
            assert actual_params['learning_rate'] == 0.05
            assert kwargs['num_boost_round'] == 50
