"""
Walk-forward experiment engine orchestration.

The helper logic lives in sibling modules under ``walking_forward`` so the
main forecast loop stays readable.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from ..modeling.trees import Model
    from ..features.scaling import FeatureScaler
    from ..features.decomposition import FeatureDecomposer
    from ..features.neural_extractors import NeuralFeatureExtractor
except (ImportError, ValueError):  # Supports running Pipeline/run_pipeline.py as a script.
    from modeling.trees import Model
    from features.scaling import FeatureScaler
    from features.decomposition import FeatureDecomposer
    from features.neural_extractors import NeuralFeatureExtractor

from .config import ConfigMixin
from .data import DataMixin
from .neural import NeuralMixin
from .outputs import OutputMixin
from .preprocessing import PreprocessingMixin
from .tuning import TuningMixin

class WalkingForwardForecaster(ConfigMixin, NeuralMixin, DataMixin, TuningMixin, PreprocessingMixin, OutputMixin):
    """
    Manages the Nested Walking-Forward validation flow.
    """
    
    def __init__(self, model_wrapper: Model, start_date: str, end_date: str, 
                 max_training_window_years: int = 5, target_col: str = 'Target', 
                 testing_window: str = 'Year', embargo_days: int = 1):
        # The unified model wrapper (LightGBM or CatBoost) used for training/prediction
        self.model_wrapper = model_wrapper
        
        # --- Internal toolkits used for feature engineering in every walk-forward fold ---
        self.scaler = FeatureScaler()             # Handles standardization (z-scoring)
        self.decomposer = FeatureDecomposer()     # Handles linear dimensionality reduction (PCA, FA)
        self.neural_extractor = NeuralFeatureExtractor() # Handles GKX and FactorVAE feature extraction
        
        # --- Experiment Configuration ---
        self.start_date = start_date              # The beginning of the first out-of-sample test period
        self.end_date = end_date                  # The final date of the entire experiment
        self.max_training_window = max_training_window_years # Lookback period for training (e.g., use 5 years of history)
        self.target_col = target_col              # The name of the column we are trying to predict (e.g., 'Target_1w')
        self.testing_window = testing_window      # Duration of each walk-forward test step (e.g., 'Year' or 'Month')
        self.embargo_days = embargo_days          # Days to exclude between train/test to prevent data leakage (overlap)

    def _prepare_forecast_inputs(self, full_df, candidate_features, as_percentages):
        full_df = self._prepare_dataframe(full_df, as_percentages)
        candidate_features = [
            col for col in candidate_features
            if col not in {self.target_col, 'Target_prc'}
        ]

        current_test_start = pd.to_datetime(self.start_date)
        min_data_dt = full_df.index.min()
        available_training_years = current_test_start.year - min_data_dt.year
        if available_training_years < 1:
            raise ValueError("Not enough data for walking forward! Need at least 1 year of history before start_date.")

        self.rolling_training_window = min(self.max_training_window, available_training_years)
        return full_df, candidate_features, current_test_start, pd.to_datetime(self.end_date), self._get_test_offset()

    def _build_period_context(self, full_df, candidate_features, output_path, current_test_start, current_test_end):
        period_str = f"{current_test_start.strftime('%d-%m-%Y')}_{current_test_end.strftime('%d-%m-%Y')}"
        period_dir = output_path / period_str
        period_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n====================== Validating Period {period_str} ======================")

        train_start = current_test_start - pd.DateOffset(years=self.rolling_training_window)
        train_end = current_test_start - pd.Timedelta(days=self.embargo_days)
        target_subset = [self.target_col, 'Target_prc']
        train_data = full_df.loc[train_start:train_end].dropna(subset=target_subset)
        test_data = full_df.loc[current_test_start:current_test_end].dropna(subset=target_subset)
        if train_data.empty or test_data.empty:
            print(f"Skipping {period_str} due to empty data.")
            return None

        return {
            'period_str': period_str,
            'period_dir': period_dir,
            'train_data': train_data,
            'test_data': test_data,
            'train_qids': train_data.index.to_series(),
            'test_qids': test_data.index.to_series(),
            'train_tickers': train_data.get('ticker', None),
            'test_tickers': test_data.get('ticker', None),
            'X_train': train_data[candidate_features].copy(),
            'X_test': test_data[candidate_features].copy(),
            'y_train': train_data[self.target_col].copy(),
            'y_test': test_data[self.target_col].copy(),
            'ae_train_target': train_data['Target_prc'].copy(),
        }

    def _load_or_tune_period_params(
        self,
        ctx,
        param_grid,
        candidate_features,
        use_standardization,
        scaling_method,
        use_pca,
        use_fa,
        use_autoencoder,
        use_factor_vae,
        tuning_metric,
        use_optuna,
        optuna_trials,
        close_prices,
        universe,
        selection_pct,
        prediction_source,
        neural_selection_metric,
        reuse_neural_artifacts_from,
    ):
        neural_grid, tree_grid = self._split_neural_tree_grid(
            param_grid,
            use_autoencoder=use_autoencoder,
            use_factor_vae=use_factor_vae,
        )
        reused_neural_params = None
        if prediction_source != 'tree' and reuse_neural_artifacts_from:
            reused_neural_params = self._load_reused_neural_params(
                reuse_neural_artifacts_from,
                ctx['period_str'],
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
            )
            self.model_wrapper.best_params_ = reused_neural_params
            self._save_best_neural_params(
                reused_neural_params,
                ctx['period_dir'],
                ctx['period_str'],
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
                neural_selection_metric=f"reused_from:{reuse_neural_artifacts_from}",
            )
            self._save_neural_reuse_manifest(
                reused_neural_params,
                reuse_neural_artifacts_from,
                ctx['period_dir'],
                ctx['period_str'],
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
            )

        if not param_grid:
            return tree_grid

        if prediction_source == 'tree':
            self._tune_model(
                ctx['X_train'], ctx['y_train'], ctx['ae_train_target'],
                ctx['train_qids'], ctx['train_tickers'], close_prices, universe,
                param_grid, candidate_features, use_standardization, scaling_method,
                use_pca, use_fa, False, use_autoencoder, use_factor_vae,
                tuning_metric, use_optuna, optuna_trials,
                ctx['period_dir'], ctx['period_str'], selection_pct,
            )
            return tree_grid

        if reused_neural_params is not None:
            print(f"--> Reusing neural params for {ctx['period_str']} from: {reuse_neural_artifacts_from}")
        elif neural_grid:
            best_neural_params, _ = self._tune_neural_outputs(
                X_train=ctx['X_train'],
                ae_train_target=ctx['ae_train_target'],
                train_qids=ctx['train_qids'],
                train_tickers=ctx['train_tickers'],
                param_grid=neural_grid,
                candidate_features=candidate_features,
                use_standardization=use_standardization,
                scaling_method=scaling_method,
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
                use_optuna=use_optuna,
                optuna_trials=optuna_trials,
                neural_selection_metric=neural_selection_metric,
                close_prices=close_prices,
                universe=universe,
                selection_pct=selection_pct,
                period_dir=ctx['period_dir'],
                period_str=ctx['period_str'],
            )
            self.model_wrapper.best_params_ = best_neural_params
            self._save_best_neural_params(
                best_neural_params,
                ctx['period_dir'],
                ctx['period_str'],
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
                neural_selection_metric=neural_selection_metric,
            )
        else:
            self.model_wrapper.best_params_ = {}

        return tree_grid

    def _fit_predict_period(self, X_train, X_test, y_train, train_qids, test_qids, final_features,
                            prediction_source, use_autoencoder, use_factor_vae):
        print("--> Fitting and Predicting...")
        if prediction_source == 'neural_expected_return':
            expected_col = self._neural_expected_return_col(
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
            )
            return X_test[expected_col].to_numpy()

        self.model_wrapper.fit(X_train[final_features], y_train, qids=train_qids)
        return self.model_wrapper.predict(X_test[final_features], qids=test_qids)

    @staticmethod
    def _print_oos_summary(fold_ics):
        print("\n" + "=" * 60)
        print(" WALK-FORWARD OOS PERFORMANCE SUMMARY")
        print("=" * 60)
        for fold_name, ic_val, sharpe_val in fold_ics:
            print(f"  Fold [{fold_name}]:  OOS IC = {ic_val:.4f} | OOS Sharpe = {sharpe_val:.4f}")
        if fold_ics:
            avg_ic = sum(ic for _, ic, _ in fold_ics) / len(fold_ics)
            avg_sharpe = sum(s for _, _, s in fold_ics) / len(fold_ics)
            print(f"  {'-' * 52}")
            print(f"  Average OOS IC across {len(fold_ics)} fold(s): {avg_ic:.4f}")
            print(f"  Average OOS Sharpe across {len(fold_ics)} fold(s): {avg_sharpe:.4f}")
        print("=" * 60)
        print("\n Nested Walking Forward Forecasting Complete!")
        
    def forecast(self, full_df: pd.DataFrame, candidate_features: List[str], base_output_dir: str = 'Nested_Results', 
                 use_pca: bool = False, use_fa: bool = False, 
                 use_autoencoder: bool = False, use_factor_vae: bool = False,
                 pca_components: int = 10, ae_hidden_dims: List[int] = None, ae_learning_rate: float = 1e-3,
                 ae_weight_decay: float = 1e-5, ae_batch_size: int = 64, ae_max_epochs: int = 200, ae_patience: int = 10,
                 ae_l1_lambda: float = 0.0,
                 factor_vae_seq_len: int = 20, factor_vae_hidden_size: int = 32, factor_vae_num_factors: int = 8,
                 factor_vae_num_portfolios: int = 32, factor_vae_learning_rate: float = 1e-4,
                 factor_vae_weight_decay: float = 1e-5, factor_vae_batch_size: int = 256,
                 factor_vae_max_epochs: int = 200, factor_vae_patience: int = 10,
                 keep_original_features: bool = False, use_standardization: bool = False, scaling_method: str = 'standard',
                 param_grid: dict = None, as_percentages: bool = True, tuning_metric: str = 'ic', 
                 use_optuna: bool = False, optuna_trials: int = 10, close_prices: pd.DataFrame = None,
                 universe: str = 'ETFs', selection_pct: float = None,
                 prediction_source: str = 'tree', neural_selection_metric: str = 'validation_rank_ic',
                 reuse_neural_artifacts_from: str = None):
        """
        Execute the walk-forward loop for one run_pipeline.py profile.

        prediction_source controls whether the final signal comes from a tree,
        a neural expected-return column, or a tree trained on frozen neural
        pricing outputs only.
        """
        prediction_source = self._validate_prediction_source(
            prediction_source,
            use_autoencoder=use_autoencoder,
            use_factor_vae=use_factor_vae,
        )
        neural_selection_metric = self._validate_neural_selection_metric(neural_selection_metric)

        full_df, candidate_features, current_test_start, end_dt, test_offset = self._prepare_forecast_inputs(
            full_df,
            candidate_features,
            as_percentages,
        )

        # Step 5: Directory Setup
        output_path = Path(base_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # This keeps track of the PCA/Decomposition model so we can reuse it (Sign-Flipping consistency)
        old_decomposition_model = None
        
        # Collect IC values from each fold for the final summary
        fold_ics = []

        # --- THE MAIN WALK-FORWARD LOOP ---
        while current_test_start <= end_dt:
            current_test_end = min(current_test_start + test_offset - pd.Timedelta(days=1), end_dt)
            ctx = self._build_period_context(
                full_df,
                candidate_features,
                output_path,
                current_test_start,
                current_test_end,
            )
            if ctx is None:
                current_test_start += test_offset
                continue

            tree_grid = self._load_or_tune_period_params(
                ctx, param_grid, candidate_features, use_standardization,
                scaling_method, use_pca, use_fa, use_autoencoder,
                use_factor_vae, tuning_metric, use_optuna, optuna_trials,
                close_prices, universe, selection_pct, prediction_source,
                neural_selection_metric, reuse_neural_artifacts_from,
            )

            # E. FINAL PREPROCESSING: Apply Scaling/Decomposition learned on TRAIN data to TEST data
            effective_keep_original_features = keep_original_features if prediction_source == 'tree' else False
            X_train, X_test, final_features, old_decomposition_model = self._apply_preprocessing(
                ctx['X_train'], ctx['X_test'], ctx['y_train'], ctx['ae_train_target'],
                candidate_features, ctx['period_dir'], ctx['period_str'],
                ctx['train_qids'], ctx['test_qids'], ctx['train_tickers'], ctx['test_tickers'],
                use_standardization, scaling_method, use_pca, use_fa, use_autoencoder, use_factor_vae,
                pca_components, ae_hidden_dims, ae_learning_rate, ae_weight_decay,
                ae_batch_size, ae_max_epochs, ae_patience, ae_l1_lambda,
                factor_vae_seq_len,
                factor_vae_hidden_size, factor_vae_num_factors, factor_vae_num_portfolios,
                factor_vae_learning_rate, factor_vae_weight_decay, factor_vae_batch_size,
                factor_vae_max_epochs, factor_vae_patience,
                effective_keep_original_features, old_decomposition_model
            )
            if prediction_source != 'tree':
                # In neural modes, throw away raw alpha columns for the final
                # model surface. Direct mode ranks expected_return; two-stage
                # mode trains the tree on these pricing-equation outputs only.
                final_features = self._neural_pricing_features(
                    X_train,
                    use_autoencoder=use_autoencoder,
                    use_factor_vae=use_factor_vae,
                )
                self._save_neural_output_artifacts(
                    train_data=ctx['train_data'],
                    test_data=ctx['test_data'],
                    X_train=X_train,
                    X_test=X_test,
                    output_cols=final_features,
                    period_dir=ctx['period_dir'],
                    period_str=ctx['period_str'],
                    use_autoencoder=use_autoencoder,
                    use_factor_vae=use_factor_vae,
                    close_prices=close_prices,
                    universe=universe,
                    selection_pct=selection_pct,
                    neural_selection_metric=neural_selection_metric,
                )
                if prediction_source == 'neural_then_tree' and tree_grid:
                    print("--> Tuning Tree on frozen neural pricing outputs only...")
                    self._tune_model(
                        X_train,
                        ctx['y_train'],
                        ctx['ae_train_target'],
                        ctx['train_qids'],
                        ctx['train_tickers'],
                        close_prices,
                        universe,
                        tree_grid,
                        final_features,
                        False,
                        scaling_method,
                        False,
                        False,
                        False,
                        False,
                        False,
                        tuning_metric,
                        use_optuna,
                        optuna_trials,
                        ctx['period_dir'],
                        ctx['period_str'],
                        selection_pct,
                    )
            
            # F. TARGET SCALING: Convert to percentages for regression or discrete ints for ranking
            y_train = ctx['y_train']
            y_test = ctx['y_test']
            if as_percentages and self.model_wrapper.task_type != 'ranking':
                y_train = y_train * 100
                y_test = y_test * 100
            
            if self.model_wrapper.task_type == 'ranking':
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)

            preds = self._fit_predict_period(
                X_train,
                X_test,
                y_train,
                ctx['train_qids'],
                ctx['test_qids'],
                final_features,
                prediction_source,
                use_autoencoder,
                use_factor_vae,
            )
            
            # H. SAVE RESULTS: Gather signals for later backtesting
            save_result = self._save_results(
                ctx['test_data'],
                preds,
                y_test,
                ctx['period_dir'],
                ctx['period_str'],
                close_prices=close_prices,
                universe=universe,
                tuning_metric=tuning_metric,
                selection_pct=selection_pct,
            )
            if isinstance(save_result, (tuple, list)) and len(save_result) == 2:
                fold_ic, fold_sharpe = save_result
            else:
                fold_ic, fold_sharpe = np.nan, np.nan
            fold_ics.append((ctx['period_str'], fold_ic, fold_sharpe))
            print(f"   Fold OOS IC [{ctx['period_str']}]: {fold_ic:.4f} | Fold OOS Sharpe: {fold_sharpe:.4f}")
            
            # I. ADVANCE TIMELINE: Slide the window forward (e.g. by 1 year)
            current_test_start += test_offset
            
        self._print_oos_summary(fold_ics)

        self._aggregate_daily_oos_ic(output_path)
        
        return fold_ics
