"""Tree model hyperparameter tuning bridge."""

try:
    from ..modeling.tuning import HyperparameterTuner
except (ImportError, ValueError):  # Supports running Pipeline/run_pipeline.py as a script.
    from modeling.tuning import HyperparameterTuner


class TuningMixin:
    def _tune_model(self, X_train, y_train, ae_train_target, train_qids, train_tickers, close_prices, universe, param_grid, candidate_features, 
                    use_standardization, scaling_method, use_pca, use_fa, use_kpca, use_autoencoder,
                    use_factor_vae, tuning_metric, use_optuna, optuna_trials, period_dir, period_str, selection_pct):
        print("--> Tuning Hyperparameters...")
        # HyperparameterTuner is model-agnostic: it only knows that a
        # "preprocessor" object has fit_transform_* and transform_* methods.
        # The walk-forward forecaster owns the real scaler, decomposer, and
        # neural extractor instances, so this small bridge adapts those concrete
        # objects to the interface expected by the tuner.
        #
        # During tuning, the tuner creates inner time-series validation folds.
        # For each fold it calls fit_transform_* on the inner train slice and
        # transform_* on the inner validation slice. That keeps normalization,
        # PCA/FA, and neural feature extraction fit only on past fold data.
        class TuningBridge:
             def __init__(self, scaler, decomposer, neural, scaling_method='standard'):
                  self.scaler, self.decomposer, self.neural = scaler, decomposer, neural
                  self.scaling_method = scaling_method
             def fit_transform_standardize(self, X): 
                  # Fit scaling parameters on the inner train fold.
                  # If cross_sectional is selected, normalization is done
                  # separately within each date; otherwise train-fold means/stds
                  # are stored and reused by transform_standardize.
                  if self.scaling_method == 'cross_sectional': return self.scaler.fit_transform_cross_sectional(X)
                  return self.scaler.fit_transform_standardize(X)
             def transform_standardize(self, X): 
                  # Apply the same scaling choice to the validation fold.
                  if self.scaling_method == 'cross_sectional': return self.scaler.transform_cross_sectional(X)
                  return self.scaler.transform_standardize(X)
             # PCA/FA are fit on the inner train fold, then applied to the
             # validation fold using the fitted decomposition object.
             def fit_transform_pca(self, X, cols, **kwargs): return self.decomposer.fit_transform_pca(X, cols, **kwargs)
             def transform_pca(self, X, **kwargs): return self.decomposer.transform_pca(X, **kwargs)
             def fit_transform_fa(self, X, cols, **kwargs): return self.decomposer.fit_transform_fa(X, cols, **kwargs)
             def transform_fa(self, X, **kwargs): return self.decomposer.transform_fa(X, **kwargs)
             # Neural feature extractors may need ae_train_target, dates, and
             # tickers. Those are passed through **kwargs by HyperparameterTuner.
             def fit_transform_autoencoder(self, X, cols, **kwargs): return self.neural.fit_transform_autoencoder(X, cols, **kwargs)
             def transform_autoencoder(self, X, **kwargs): return self.neural.transform_autoencoder(X, **kwargs)
             def fit_transform_factor_vae(self, X, cols, **kwargs): return self.neural.fit_transform_factor_vae(X, cols, **kwargs)
             def transform_factor_vae(self, X, **kwargs): return self.neural.transform_factor_vae(X, **kwargs)
             # The tuner needs these names after a transform so it knows which
             # columns to pass to the temporary tree model for validation.
             @property
             def pca_col_names(self): return self.decomposer.pca_col_names
             @property
             def fa_col_names(self): return self.decomposer.fa_col_names
             @property
             def ae_col_names(self): return self.neural.ae_col_names
             @property
             def ae_history(self): return getattr(self.neural, 'ae_history', None)
             @property
             def factor_vae_col_names(self): return self.neural.factor_vae_col_names
             @property
             def factor_vae_history(self): return getattr(self.neural, 'factor_vae_history', None)

        # Bridge the forecaster-owned preprocessing objects into the tuner.
        bridge = TuningBridge(self.scaler, self.decomposer, self.neural_extractor, scaling_method)
        # Build a temporary tuner with the same tree family/task/loss as the
        # final model wrapper. It will repeatedly train temporary fold models.
        tuner = HyperparameterTuner(self.model_wrapper.model_type, self.model_wrapper.task_type, 
                                    self.model_wrapper.loss_function, use_gpu=self.model_wrapper.use_gpu)
        
        # tune(...) searches param_grid either by full grid search or Optuna.
        # It evaluates each candidate on inner validation folds inside X_train,
        # exports the tuning table, and returns the best parameter dictionary.
        best_params, tuning_results = tuner.tune(
            X_train, y_train, train_qids, param_grid, 
            embargo_days=self.embargo_days, preprocessor=bridge, 
            use_standardization=use_standardization, use_pca=use_pca, use_fa=use_fa, 
            use_autoencoder=use_autoencoder, use_factor_vae=use_factor_vae,
            candidate_features=candidate_features, tuning_metric=tuning_metric, 
            use_optuna=use_optuna, optuna_trials=optuna_trials, base_params=self.model_wrapper.params,
            results_export_path=period_dir / f"tuning_metrics_{period_str}.csv",
            training_history_export_path=period_dir / f"deep_learning_training_history_{period_str}.csv",
            ae_target=ae_train_target,
            tickers=train_tickers, close_prices=close_prices,
            universe=universe, selection_pct=selection_pct
        )
        # Merge selected params into the real model wrapper so final fitting and
        # final preprocessing use the same choices selected during tuning.
        self.model_wrapper.params.update(best_params)
        # Keep an explicit copy of the winning params for downstream code that
        # reads best_params_ directly, especially final preprocessing.
        self.model_wrapper.best_params_ = best_params
