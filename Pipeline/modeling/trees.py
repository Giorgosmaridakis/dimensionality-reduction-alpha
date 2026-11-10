"""
Unified tree-model wrapper used by the walk-forward forecaster.

The rest of the pipeline can call Model.fit/predict without caring whether the
underlying implementation is LightGBM or CatBoost.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# We use optional imports to avoid crashing
try:
    from catboost import CatBoostRegressor, Pool, CatBoostRanker
except ImportError:
    CatBoostRegressor, Pool, CatBoostRanker = None, None, None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class Model:
    """
    A unified wrapper for tree-based ranking and regression models.
    Provides a consistent API for interacting with CatBoost and LightGBM,
    handling the specific data format requirements and parameter mappings for each.
    """
    
    def __init__(self, model_type: str, task_type: str, loss_function: str, use_gpu: bool = False, **kwargs):
        """
        Args:
            model_type (str): 'CatBoost' or 'LightGBM'.
            task_type (str): 'regression' or 'ranking'.
            loss_function (str): The loss function to use.
            use_gpu (bool): Whether to train the model on GPU.
            **kwargs: Default hyperparameters for the underlying model.
        """
        self.model_type = model_type.lower()
        self.task_type = task_type.lower()
        self.use_gpu = use_gpu
        
        if self.task_type not in ['regression', 'ranking']:
            raise ValueError("task_type must be either 'regression' or 'ranking'.")
                             
        self.loss_function = loss_function
        self.params = kwargs
        
        # Map internal parameter names to library-specific ones
        if self.model_type == 'catboost':
            self.params['loss_function'] = loss_function
        elif self.model_type == 'lightgbm':
            self.params['objective'] = loss_function
            
        # Handle GPU device configuration
        if self.use_gpu:
            if self.model_type == 'catboost':
                self.params['task_type'] = 'GPU'
            elif self.model_type == 'lightgbm':
                self.params['device'] = 'gpu'
                
        self.model = None
        self.best_params_ = None
        
        # Internal list of categorical feature names used for model preparation
        self.cat_features = []
        
        # Verify that requested library is available
        if self.model_type == 'catboost':
            if CatBoostRegressor is None:
                raise ImportError("CatBoost is not installed.")
        elif self.model_type == 'lightgbm':
            if lgb is None:
                raise ImportError("LightGBM is not installed.")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series, qids: Optional[pd.Series] = None):
        """
        Converts pandas DataFrames into model-specific data containers (Pool, Dataset).
        
        Args:
            X: Input features.
            y: Target values (continuous for regression, labels/relevance for ranking).
            qids: Query IDs for ranking tasks, used to group items.
        """

        if self.model_type == 'catboost':
            # CatBoost requires categorical feature indices rather than names
            cat_indices = [X.columns.get_loc(c) for c in self.cat_features if c in X.columns]
            # Convert group IDs to strings for robust ranking grouping
            groups = qids.astype(str).values if qids is not None else None
            return Pool(data=X, label=y, group_id=groups, cat_features=cat_indices)
            
        elif self.model_type == 'lightgbm':
            # For LightGBM ranking, we provide group sizes (counts per query)
            if qids is not None:
                group_sizes = X.groupby(qids.values, sort=False).size().values
            else:
                group_sizes = None
            # free_raw_data=False prevents deleting the internal pandas objects
            return lgb.Dataset(data=X, label=y, group=group_sizes, categorical_feature=self.cat_features, free_raw_data=False)

    def fit(self, X: pd.DataFrame, y: pd.Series, qids: Optional[pd.Series] = None, eval_set=None, cat_features: List[str] = None, verbose=False):
        """
        Trains the model with early stopping support and automated parameter filtering.
        
        Args:
            X: Training features.
            y: Training targets.
            qids: Training query IDs (for ranking).
            eval_set: Optional tuple of (X_val, y_val, qids_val) for validation.
            cat_features: List of column names to treat as categorical.
            verbose: Whether to print training progress.
        """
        self.cat_features = cat_features or []
        train_data = self._prepare_data(X, y, qids)
        
        if eval_set is not None:
            X_val, y_val, qids_val = eval_set
            val_data = self._prepare_data(X_val, y_val, qids_val)
        else:
            val_data = None
            
        # Filter out design-time parameters (e.g., autoencoder configs) 
        # that shouldn't be passed to the underlying tree model constructor.
        tree_params = {k: v for k, v in self.params.items() if not k.startswith(('pca__', 'ae__', 'fa__', 'kpca__', 'factorvae__'))}
            
        if self.model_type == 'catboost':
            if self.task_type == 'ranking':
                self.model = CatBoostRanker(**tree_params, allow_writing_files=False)
            else:
                self.model = CatBoostRegressor(**tree_params, allow_writing_files=False)
                
            eval_sets = [val_data] if val_data is not None else None
            
            # Enable early stopping for CatBoost
            early_stop = 50 if eval_sets else None
            self.model.fit(train_data, eval_set=eval_sets, early_stopping_rounds=early_stop, verbose=verbose)
            
        elif self.model_type == 'lightgbm':
            if 'n_estimators' not in tree_params:
                raise KeyError("Parameter 'n_estimators' must be provided in the model configuration.")
            tree_params.setdefault('random_seed', 42)
                
            num_boost_round = tree_params.pop('n_estimators')
            tree_params['seed'] = tree_params.pop('random_seed')
            tree_params['verbose'] = -1  # Silence noisy warnings
            
            # LightGBM requires num_leaves to be explicitly set if max_depth is provided
            if 'max_depth' in tree_params and tree_params['max_depth'] > 0 and 'num_leaves' not in tree_params:
                tree_params['num_leaves'] = min(2 ** tree_params['max_depth'] - 1, 128)
                
            valid_sets = [val_data] if val_data is not None else []
            callbacks = [lgb.log_evaluation(period=0)] if not verbose else []
            if valid_sets:
                # Standardized to 50 rounds to match CatBoost
                callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            self.model = lgb.train(
                params=tree_params,
                train_set=train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks if callbacks else None
            )

    def predict(self, X: pd.DataFrame, qids: Optional[pd.Series] = None) -> np.ndarray:
        """
        Generates predictions for the input features.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        if self.model_type == 'catboost':
            return self.model.predict(X)
            
        elif self.model_type == 'lightgbm':
            return self.model.predict(X)
