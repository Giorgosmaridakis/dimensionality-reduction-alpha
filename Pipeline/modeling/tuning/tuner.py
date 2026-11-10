"""Main hyperparameter tuner orchestration."""

import time
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from ..trees import Model
except (ImportError, ValueError):  # Supports running Pipeline/run_pipeline.py as a script.
    from modeling.trees import Model

from .preprocessing import apply_fold_preprocessing, split_grid_params, tree_grid_params
from .records import append_training_history, append_tuning_result, export_training_history
from .scoring import score_validation_fold
from .search import (
    TrialPruned,
    build_param_combinations,
    run_grid_search,
    run_optuna_search,
    should_use_optuna,
)


class HyperparameterTuner:
    def __init__(self, model_type: str, task_type: str, loss_function: str, use_gpu: bool = False):
        # These describe the temporary tree models trained during tuning.
        # The final production model is owned by walking_forward/forecaster.py.
        self.model_type = model_type
        self.task_type = task_type
        self.loss_function = loss_function
        self.use_gpu = use_gpu

    def tune(self, X: pd.DataFrame, y: pd.Series, qids: pd.Series, param_grid: Dict[str, List[Any]],
             cat_features: List[str] = None, embargo_days: int = 0, preprocessor=None,
             use_standardization: bool = False, use_pca: bool = False, use_fa: bool = False,
             use_autoencoder: bool = False, use_factor_vae: bool = False,
             candidate_features: List[str] = None, tuning_metric: str = 'ic',
             use_optuna: bool = False, optuna_trials: int = 20, base_params: Dict[str, Any] = None,
             results_export_path: Optional[str] = None, training_history_export_path: Optional[str] = None,
             ae_target=None, tickers=None, close_prices=None,
             universe: str = 'ETFs', selection_pct: float = None):
        """
        Unified tuning loop.

        The outer walk-forward split is handled in walking_forward/forecaster.py;
        this method only sees the current training window and splits it again
        into three validation folds.
        """
        base_params = base_params or {}
        # Neural preprocessors need the raw return target aligned to X. Some
        # callers may pass a NumPy array/list, so normalize it to a Series here.
        if ae_target is not None and not hasattr(ae_target, 'iloc'):
            ae_target = pd.Series(ae_target, index=X.index)

        # Tune inside the current walk-forward training window only. The outer
        # test period is never visible here.
        unique_dates = X.index.unique().sort_values()
        tscv = TimeSeriesSplit(n_splits=3, gap=embargo_days)

        print(f"   Total unique dates in training set: {len(unique_dates)}")
        print(f"   Tuning metric: {tuning_metric.upper()}")

        tuning_results = []
        training_history_records = []
        # Expand the grid once. Grid search will evaluate all of these; Optuna
        # samples from the same discrete choices and avoids duplicate samples.
        combinations = build_param_combinations(param_grid)

        # Keep evaluate_combo stateless-ish by passing all fold context through
        # one dictionary. This is also what Optuna receives per trial.
        eval_context = {
            'X': X, 'y': y, 'qids': qids,
            'tscv': tscv, 'unique_dates': unique_dates,
            'cat_features': cat_features,
            'preprocessor': preprocessor,
            'use_standardization': use_standardization,
            'use_pca': use_pca, 'use_fa': use_fa,
            'use_autoencoder': use_autoencoder, 'use_factor_vae': use_factor_vae,
            'candidate_features': candidate_features,
            'tuning_metric': tuning_metric,
            'base_params': base_params,
            'tuning_results': tuning_results,
            'training_history_records': training_history_records,
            'training_history_export_path': training_history_export_path,
            'ae_target': ae_target,
            'tickers': tickers,
            'close_prices': close_prices,
            'universe': universe,
            'selection_pct': selection_pct,
            'progress_factory': None,
        }

        if should_use_optuna(use_optuna):
            # Optuna is useful for large grids. It still samples categorical
            # values from the supplied param_grid, so the search space remains
            # exactly the experimenter's explicit choices.
            best_params, _ = run_optuna_search(
                param_grid=param_grid,
                combinations=combinations,
                evaluate_combo=self.evaluate_combo,
                eval_context=eval_context,
                optuna_trials=optuna_trials,
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
            )
        else:
            # Deterministic exhaustive search over every parameter combination.
            best_params, _ = run_grid_search(
                combinations=combinations,
                evaluate_combo=self.evaluate_combo,
                eval_context=eval_context,
            )

        results_df = pd.DataFrame(tuning_results)
        if results_export_path:
            # One row per tried/pruned parameter combination, including fold
            # scores and whether all CV folds completed.
            results_df.to_csv(results_export_path, index=False)
            print(f"   [DONE] Tuning results saved to {results_export_path}")

        training_history_df = pd.DataFrame(training_history_records)
        if training_history_export_path and not training_history_df.empty:
            # Neural models can append per-epoch training losses/metrics. This
            # is separate from tuning_results because it is much more granular.
            training_history_df.to_csv(training_history_export_path, index=False)
            print(f"   [DONE] Deep learning training history saved to {training_history_export_path}")

        return best_params, results_df

    def evaluate_combo(self, grid_params, X, y, qids, tscv, unique_dates, combo_index=0,
                       cat_features=None, preprocessor=None, use_standardization=False,
                       use_pca=False, use_fa=False, use_autoencoder=False,
                       use_factor_vae=False, candidate_features=None, tuning_metric='ic',
                       base_params=None, tuning_results=None, training_history_records=None,
                       training_history_export_path=None, ae_target=None, tickers=None, close_prices=None,
                       universe='ETFs', selection_pct=None, progress_factory=None,
                       trial=None, enable_pruning=False):
        """Evaluates a single hyperparameter combination."""
        base_params = base_params or {}
        print(f"-> Tuning [{combo_index}]: {grid_params}")
        start_time = time.time()
        fold_scores = []

        grid_parts = split_grid_params(grid_params)
        trial_number = trial.number if trial is not None else combo_index

        for fold_number, (train_date_idx, val_date_idx) in enumerate(tscv.split(unique_dates), start=1):
            # TimeSeriesSplit gives date positions. Convert them to row
            # positions because each date can contain many tickers/assets.
            train_idx, val_idx = self._fold_row_indices(X, unique_dates, train_date_idx, val_date_idx)

            X_train_fold = X.iloc[train_idx].copy()
            X_val_fold = X.iloc[val_idx].copy()
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            # AE/FactorVAE price raw returns. If no separate raw target is
            # provided, fall back to y for old/regression-only callers.
            ae_target_train_fold = ae_target.iloc[train_idx] if ae_target is not None else y_train_fold

            # Ranking libraries expect integer relevance labels. Regression
            # keeps the continuous target as-is.
            y_train_fold_fit = y_train_fold.astype(int) if self.task_type == 'ranking' else y_train_fold
            y_val_fold_fit = y_val_fold.astype(int) if self.task_type == 'ranking' else y_val_fold

            # qids are date/group labels for ranking models and daily scoring.
            qids_train_fold = qids.iloc[train_idx] if qids is not None else None
            qids_val_fold = qids.iloc[val_idx] if qids is not None else None
            # Tickers are only needed by neural sequence models and Sharpe
            # tuning/backtesting.
            tickers_train_fold = tickers.iloc[train_idx] if tickers is not None else None
            tickers_val_fold = tickers.iloc[val_idx] if tickers is not None else None

            fold_features = candidate_features.copy() if candidate_features else list(X_train_fold.columns)

            # Fit preprocessing only on this inner train fold and transform the
            # validation fold. This prevents validation-period leakage.
            X_train_fold, X_val_fold, fold_features = apply_fold_preprocessing(
                preprocessor=preprocessor,
                X_train_fold=X_train_fold,
                X_val_fold=X_val_fold,
                fold_features=fold_features,
                grid_parts=grid_parts,
                use_standardization=use_standardization,
                use_pca=use_pca,
                use_fa=use_fa,
                use_autoencoder=use_autoencoder,
                use_factor_vae=use_factor_vae,
                ae_target_train_fold=ae_target_train_fold,
                qids_train_fold=qids_train_fold,
                qids_val_fold=qids_val_fold,
                tickers_train_fold=tickers_train_fold,
                tickers_val_fold=tickers_val_fold,
                progress_factory=progress_factory,
                combo_index=combo_index,
                fold_number=fold_number,
                grid_params=grid_params,
                trial_number=trial_number,
                training_history_records=training_history_records,
                training_history_export_path=training_history_export_path,
            )

            temp_model = self._build_model_for_combo(grid_params, base_params)

            try:
                # Each parameter combination trains a fresh temporary model.
                # The real model_wrapper is updated only after tuning finishes.
                temp_model.fit(
                    X=X_train_fold[fold_features],
                    y=y_train_fold_fit,
                    qids=qids_train_fold,
                    eval_set=(X_val_fold[fold_features], y_val_fold_fit, qids_val_fold),
                    cat_features=cat_features,
                    verbose=False,
                )

                # Score this validation fold using the requested tuning metric:
                # daily IC/rank-IC, validation Sharpe, or model eval metric.
                fold_score = score_validation_fold(
                    temp_model=temp_model,
                    X_val_fold=X_val_fold,
                    y_val_fold=y_val_fold,
                    fold_features=fold_features,
                    qids_val_fold=qids_val_fold,
                    tickers_val_fold=tickers_val_fold,
                    val_idx=val_idx,
                    tuning_metric=tuning_metric,
                    task_type=self.task_type,
                    model_type=self.model_type,
                    close_prices=close_prices,
                    tickers=tickers,
                    universe=universe,
                    selection_pct=selection_pct,
                )

                fold_scores.append(fold_score)
                if enable_pruning and trial is not None:
                    # For neural searches, report partial CV progress to Optuna.
                    # After at least two folds, weak trials can be pruned early.
                    partial_score = np.mean(fold_scores)
                    trial.report(partial_score, step=fold_number)
                    if fold_number >= 2 and trial.should_prune():
                        print(f"   -> Pruned after fold {fold_number}: partial score={partial_score:.5f}")
                        append_tuning_result(
                            tuning_results,
                            grid_params,
                            fold_scores,
                            pruned=True,
                            trial_number=trial_number,
                            pruning_enabled=enable_pruning,
                        )
                        raise TrialPruned()
            except TrialPruned:
                raise
            except Exception as exc:
                print(f"Fold Error: {exc}")
                fold_scores.append(float('-inf'))

        avg_score = np.mean(fold_scores) if fold_scores else float('-inf')
        fold_scores_str = ", ".join([f"{score:.4f}" if score != float('-inf') else "Error" for score in fold_scores])
        print(f"   => Time: {time.time() - start_time:.1f}s | Score: {avg_score:.5f} (Folds: {fold_scores_str})")

        append_tuning_result(
            tuning_results,
            grid_params,
            fold_scores,
            pruned=False,
            trial_number=trial_number,
            pruning_enabled=enable_pruning,
        )

        return avg_score

    def _build_model_for_combo(self, grid_params, base_params):
        # Strip preprocessing params from the tree model kwargs. Parameters like
        # pca__n_components or factorvae__seq_len are consumed before fitting.
        temp_params = {**base_params, **tree_grid_params(grid_params)}
        # Avoid passing duplicate objective/loss keys into Model, which already
        # receives loss_function as an explicit constructor argument.
        temp_params.pop('loss_function', None)
        temp_params.pop('objective', None)
        if self.use_gpu and self.model_type.lower() == 'catboost':
            temp_params.pop('task_type', None)

        # Tests and legacy scripts patch modeling.tuning.Model. Looking it up
        # from the package keeps that compatibility after splitting tuning.py
        # into a package.
        tuning_package = sys.modules.get(__package__)
        model_cls = getattr(tuning_package, 'Model', Model)
        return model_cls(
            self.model_type,
            self.task_type,
            self.loss_function,
            use_gpu=self.use_gpu,
            **temp_params,
        )

    @staticmethod
    def _fold_row_indices(X, unique_dates, train_date_idx, val_date_idx):
        # Map fold date selections back to row numbers. The data is panel data:
        # one date usually has many rows, one per ticker.
        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]
        train_idx = np.where(X.index.isin(train_dates))[0]
        val_idx = np.where(X.index.isin(val_dates))[0]
        return train_idx, val_idx

    # Compatibility aliases for tests or external scripts that called the old
    # private static helpers directly.
    _append_tuning_result = staticmethod(append_tuning_result)
    _append_training_history = staticmethod(append_training_history)
    _export_training_history = staticmethod(export_training_history)
