"""Neural expected-return hyperparameter tuning."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from ..features.scaling import FeatureScaler
    from ..features.neural_extractors import NeuralFeatureExtractor
except (ImportError, ValueError):  # Supports running Pipeline/run_pipeline.py as a script.
    from features.scaling import FeatureScaler
    from features.neural_extractors import NeuralFeatureExtractor


class NeuralTuningMixin:
    @staticmethod
    def _cap_close_prices_to_training_window(close_prices, train_qids):
        """Restrict validation-Sharpe prices to the current outer train window."""
        if close_prices is None:
            return None

        train_end = pd.to_datetime(np.asarray(train_qids)).max()
        capped = close_prices.copy()

        if 'Date' in capped.columns:
            date_values = pd.to_datetime(capped['Date'])
            return capped.loc[date_values <= train_end].copy()

        if not pd.api.types.is_datetime64_any_dtype(capped.index):
            capped.index = pd.to_datetime(capped.index)
        return capped.loc[capped.index <= train_end].copy()

    def _fit_transform_neural_for_tuning(
        self,
        X_train_fold,
        X_val_fold,
        ae_target_train_fold,
        qids_train_fold,
        qids_val_fold,
        tickers_train_fold,
        tickers_val_fold,
        grid_params,
        candidate_features,
        use_standardization,
        scaling_method,
        use_autoencoder,
        use_factor_vae,
    ):
        scaler = FeatureScaler()
        neural = NeuralFeatureExtractor()
        X_train_fold = X_train_fold.copy()
        X_val_fold = X_val_fold.copy()

        if use_standardization:
            if scaling_method == 'cross_sectional':
                X_train_fold = scaler.fit_transform_cross_sectional(X_train_fold, feature_cols=candidate_features)
                X_val_fold = scaler.transform_cross_sectional(X_val_fold, feature_cols=candidate_features)
            else:
                X_train_fold = scaler.fit_transform_standardize(X_train_fold, feature_cols=candidate_features)
                X_val_fold = scaler.transform_standardize(X_val_fold, feature_cols=candidate_features)

        if use_autoencoder:
            ae_params = {key.replace('ae__', ''): value for key, value in grid_params.items() if key.startswith('ae__')}
            progress_bar = None
            try:
                from tqdm import tqdm
                fold_label = (
                    f"Neural GKX-CA {pd.to_datetime(qids_val_fold).min().strftime('%Y-%m-%d')} "
                    f"to {pd.to_datetime(qids_val_fold).max().strftime('%Y-%m-%d')}"
                )
                progress_bar = tqdm(total=1, desc=fold_label, leave=False)
            except Exception:
                progress_bar = None
            try:
                X_train_fold = neural.fit_transform_autoencoder(
                    X_train_fold,
                    candidate_features,
                    target=ae_target_train_fold,
                    dates=qids_train_fold,
                    tickers=tickers_train_fold,
                    keep_original_features=False,
                    progress_bar=progress_bar,
                    **ae_params,
                )
            finally:
                if progress_bar is not None:
                    progress_bar.close()
            X_val_fold = neural.transform_autoencoder(
                X_val_fold,
                keep_original_features=False,
            )
        elif use_factor_vae:
            factor_params = {
                key.replace('factorvae__', ''): value
                for key, value in grid_params.items()
                if key.startswith('factorvae__')
            }
            progress_bar = None
            try:
                from tqdm import tqdm
                fold_label = (
                    f"Neural FactorVAE {pd.to_datetime(qids_val_fold).min().strftime('%Y-%m-%d')} "
                    f"to {pd.to_datetime(qids_val_fold).max().strftime('%Y-%m-%d')}"
                )
                progress_bar = tqdm(total=1, desc=fold_label, leave=False)
            except Exception:
                progress_bar = None
            try:
                X_train_fold = neural.fit_transform_factor_vae(
                    X_train_fold,
                    candidate_features,
                    target=ae_target_train_fold,
                    dates=qids_train_fold,
                    tickers=tickers_train_fold,
                    keep_original_features=False,
                    progress_bar=progress_bar,
                    **factor_params,
                )
            finally:
                if progress_bar is not None:
                    progress_bar.close()
            X_val_fold = neural.transform_factor_vae(
                X_val_fold,
                dates=qids_val_fold,
                tickers=tickers_val_fold,
                keep_original_features=False,
            )
        else:
            raise ValueError("Neural tuning requires use_autoencoder or use_factor_vae.")

        return X_train_fold, X_val_fold

    def _tune_neural_outputs(
        self,
        X_train,
        ae_train_target,
        train_qids,
        train_tickers,
        param_grid,
        candidate_features,
        use_standardization,
        scaling_method,
        use_autoencoder,
        use_factor_vae,
        use_optuna,
        optuna_trials,
        neural_selection_metric,
        close_prices,
        universe,
        selection_pct,
        period_dir,
        period_str,
    ):
        print("--> Tuning Neural Expected-Return Outputs...")
        combinations = self._iter_param_combinations(param_grid)
        if use_optuna and optuna_trials:
            rng = np.random.default_rng(42)
            if len(combinations) > optuna_trials:
                selected = rng.choice(len(combinations), size=optuna_trials, replace=False)
                combinations = [combinations[i] for i in selected]

        unique_dates = train_qids.drop_duplicates().sort_values()
        n_splits = min(3, max(2, len(unique_dates) - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=self.embargo_days)
        results = []
        best_score = float('-inf')
        best_params = {}
        validation_close_prices = self._cap_close_prices_to_training_window(
            close_prices,
            train_qids,
        )

        expected_col = self._neural_expected_return_col(
            use_autoencoder=use_autoencoder,
            use_factor_vae=use_factor_vae,
        )

        for combo_index, grid_params in enumerate(combinations, start=1):
            fold_scores = []
            print(f"   Neural Trial {combo_index}/{len(combinations)}: {grid_params}")
            if neural_selection_metric == 'in_sample_rank_ic':
                try:
                    X_train_neural, _ = self._fit_transform_neural_for_tuning(
                        X_train_fold=X_train,
                        X_val_fold=X_train,
                        ae_target_train_fold=ae_train_target,
                        qids_train_fold=train_qids,
                        qids_val_fold=train_qids,
                        tickers_train_fold=train_tickers,
                        tickers_val_fold=train_tickers,
                        grid_params=grid_params,
                        candidate_features=candidate_features,
                        use_standardization=use_standardization,
                        scaling_method=scaling_method,
                        use_autoencoder=use_autoencoder,
                        use_factor_vae=use_factor_vae,
                    )
                    if neural_selection_metric == 'validation_sharpe':
                        validation_score = self._score_neural_validation_fold(
                            X_val_neural=X_train_neural,
                            expected_col=expected_col,
                            ae_target_val_fold=ae_train_target,
                            qids_val_fold=train_qids,
                            tickers_val_fold=train_tickers,
                            neural_selection_metric=neural_selection_metric,
                            close_prices=validation_close_prices,
                            universe=universe,
                            selection_pct=selection_pct,
                            period_dir=period_dir,
                            period_str=period_str,
                            fold_number=0,
                        )
                    else:
                        validation_score = self._daily_rank_ic(
                            X_train_neural[expected_col],
                            ae_train_target,
                            train_qids,
                        )
                    fold_scores.append(validation_score)
                except Exception as exc:
                    print(f"      Neural in-sample scoring error: {exc}")
                    validation_score = float('-inf')
                    fold_scores.append(validation_score)
            else:
                for fold_number, (train_idx, val_idx) in enumerate(tscv.split(unique_dates), start=1):
                    fold_train_dates = set(unique_dates.iloc[train_idx])
                    fold_val_dates = set(unique_dates.iloc[val_idx])
                    train_mask = train_qids.isin(fold_train_dates)
                    val_mask = train_qids.isin(fold_val_dates)

                    X_train_fold = X_train.loc[train_mask].copy()
                    X_val_fold = X_train.loc[val_mask].copy()
                    ae_target_train_fold = ae_train_target.loc[train_mask]
                    ae_target_val_fold = ae_train_target.loc[val_mask]
                    qids_train_fold = train_qids.loc[train_mask]
                    qids_val_fold = train_qids.loc[val_mask]
                    tickers_train_fold = train_tickers.loc[train_mask] if train_tickers is not None else None
                    tickers_val_fold = train_tickers.loc[val_mask] if train_tickers is not None else None

                    try:
                        _, X_val_neural = self._fit_transform_neural_for_tuning(
                            X_train_fold=X_train_fold,
                            X_val_fold=X_val_fold,
                            ae_target_train_fold=ae_target_train_fold,
                            qids_train_fold=qids_train_fold,
                            qids_val_fold=qids_val_fold,
                            tickers_train_fold=tickers_train_fold,
                            tickers_val_fold=tickers_val_fold,
                            grid_params=grid_params,
                            candidate_features=candidate_features,
                            use_standardization=use_standardization,
                            scaling_method=scaling_method,
                            use_autoencoder=use_autoencoder,
                            use_factor_vae=use_factor_vae,
                        )
                        fold_score = self._score_neural_validation_fold(
                            X_val_neural=X_val_neural,
                            expected_col=expected_col,
                            ae_target_val_fold=ae_target_val_fold,
                            qids_val_fold=qids_val_fold,
                            tickers_val_fold=tickers_val_fold,
                            neural_selection_metric=neural_selection_metric,
                            close_prices=validation_close_prices,
                            universe=universe,
                            selection_pct=selection_pct,
                            period_dir=period_dir,
                            period_str=period_str,
                            fold_number=fold_number,
                        )
                    except Exception as exc:
                        print(f"      Neural fold {fold_number} error: {exc}")
                        fold_score = float('-inf')
                    fold_scores.append(fold_score)

                finite_scores = [score for score in fold_scores if score != float('-inf')]
                is_complete = len(fold_scores) == n_splits and len(finite_scores) == n_splits
                validation_score = float(np.mean(finite_scores)) if is_complete else float('-inf')

            row = grid_params.copy()
            expected_folds = 1 if neural_selection_metric == 'in_sample_rank_ic' else n_splits
            successful_folds = len([score for score in fold_scores if score != float('-inf')])
            row['Neural_Selection_Metric'] = neural_selection_metric
            row['Validation_Score'] = validation_score
            row['Complete_CV'] = len(fold_scores) == expected_folds and successful_folds == expected_folds
            row['Successful_Folds'] = successful_folds
            for fold_idx, score in enumerate(fold_scores, start=1):
                row[f'Fold_{fold_idx}_Score'] = score
            results.append(row)
            print(f"   => Neural Score: {validation_score:.5f}")

            if validation_score > best_score:
                best_score = validation_score
                best_params = grid_params.copy()

        results_df = pd.DataFrame(results)
        out_path = period_dir / f"neural_tuning_metrics_{period_str}.csv"
        self._safe_to_csv(results_df, out_path, index=False)
        model_name = 'factorvae' if use_factor_vae else 'gkx_ae' if use_autoencoder else 'neural'
        artifact_dir = period_dir / f"best_{model_name}"
        self._safe_to_csv(
            results_df,
            artifact_dir / f"{model_name}_tuning_metrics_{period_str}.csv",
            index=False,
        )
        print(f"   Best neural params: {best_params} | {neural_selection_metric}: {best_score:.5f}")
        return best_params, results_df
