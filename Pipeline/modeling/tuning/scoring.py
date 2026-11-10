"""Scoring helpers for tuning folds."""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def score_validation_fold(
    temp_model,
    X_val_fold,
    y_val_fold,
    fold_features,
    qids_val_fold,
    tickers_val_fold,
    val_idx,
    tuning_metric,
    task_type,
    model_type,
    close_prices=None,
    tickers=None,
    universe='ETFs',
    selection_pct=None,
):
    metric = tuning_metric.lower()
    if metric in ['ic', 'sharpe', 'sharp']:
        # Metrics in this branch need model predictions, not just the library's
        # built-in validation loss.
        val_preds = temp_model.predict(X_val_fold[fold_features])
        val_dates_arr = qids_val_fold.values if qids_val_fold is not None else np.zeros(len(y_val_fold))

        if metric in ['sharpe', 'sharp'] and close_prices is not None and tickers is not None:
            # Sharpe tuning runs the validation predictions through the same
            # long/short backtester convention as the final OOS evaluation.
            return score_sharpe(
                val_preds,
                val_dates_arr,
                tickers.iloc[val_idx],
                task_type=task_type,
                close_prices=close_prices,
                universe=universe,
                selection_pct=selection_pct,
            )

        # Default validation metric is daily cross-sectional IC. Regression uses
        # Pearson correlation; ranking uses Spearman/rank correlation.
        fold_score = score_daily_ic(
            val_preds,
            y_val_fold.values,
            val_dates_arr,
            task_type=task_type,
        )
        return fold_score

    # Fallback: rely on the tree library's own eval metric where available.
    return score_model_eval_metric(temp_model, model_type, task_type)


def score_daily_ic(val_preds, y_val_values, val_dates_arr, task_type):
    # Calculate IC separately per date, then average across dates. This matches
    # the cross-sectional nature of asset ranking.
    daily_ics = []
    for date in np.unique(val_dates_arr):
        mask = val_dates_arr == date
        if mask.sum() > 2:
            # Skip tiny cross-sections and zero-variance cases because
            # correlation would be undefined or meaningless.
            preds_day = val_preds[mask]
            actuals_day = y_val_values[mask]
            if np.std(preds_day) > 1e-10 and np.std(actuals_day) > 1e-10:
                if task_type == 'regression':
                    ic = np.corrcoef(preds_day, actuals_day)[0, 1]
                else:
                    ic, _ = spearmanr(preds_day, actuals_day)
                if not np.isnan(ic):
                    daily_ics.append(ic)

    return np.mean(daily_ics) if daily_ics else 0.0


def score_sharpe(val_preds, val_dates_arr, tickers_val_fold, task_type, close_prices, universe, selection_pct):
    # Import lazily so IC-only tuning does not pay the Backtester import cost.
    _ensure_pipeline_on_path()
    from Backtester.backtester import Backtester

    if task_type == 'ranking':
        # The backtester expects ranks for ranking tasks. Higher predicted score
        # receives better/lower rank number.
        pred_df = pd.DataFrame({
            'Date': val_dates_arr,
            'Ticker': tickers_val_fold.values,
            'Predicted_Score': val_preds,
        })
        pred_df['Predicted_Rank'] = (
            pred_df.groupby('Date')['Predicted_Score']
            .rank(ascending=False, method='min')
            .astype(int)
        )
    else:
        pred_df = pd.DataFrame({
            'Date': val_dates_arr,
            'Ticker': tickers_val_fold.values,
            'Prediction': val_preds,
        })

    try:
        # export=False keeps tuning trials from writing temporary portfolio
        # reports for every validation fold.
        bt = Backtester(output_folder='temp', universe=universe)
        fold_score = bt.run_backtest(
            data=close_prices,
            predictions=pred_df,
            strategy='long_short',
            export=False,
            selection_pct=selection_pct,
        )
        return 0.0 if pd.isna(fold_score) else fold_score
    except Exception as exc:
        print(f"Backtester Tuning Error: {exc}")
        return 0.0


def score_model_eval_metric(temp_model, model_type, task_type):
    # Different tree libraries expose their validation metric through different
    # attributes. Normalize them into "higher is better" for the tuner.
    raw_fold_score = 0
    model_key = model_type.lower()

    if model_key == 'catboost':
        # CatBoost stores a dictionary of eval-set metric histories.
        evals = temp_model.model.get_evals_result()
        valid_key = next((k for k in evals.keys() if 'valid' in k or 'test' in k), None)
        if valid_key:
            metric_key = next((k for k in evals[valid_key].keys()), None)
            if task_type == 'regression':
                raw_fold_score = np.min(evals[valid_key][metric_key])
            else:
                raw_fold_score = np.max(evals[valid_key][metric_key])
    elif model_key == 'lightgbm':
        # LightGBM exposes the best validation scores directly.
        if hasattr(temp_model.model, 'best_score') and temp_model.model.best_score:
            valid_key = list(temp_model.model.best_score.keys())[0]
            metric_key = list(temp_model.model.best_score[valid_key].keys())[0]
            raw_fold_score = temp_model.model.best_score[valid_key][metric_key]

    # Regression losses are usually minimized, so negate them to keep the
    # tuner's convention: larger score is better.
    return -raw_fold_score if task_type == 'regression' else raw_fold_score


def _ensure_pipeline_on_path():
    # Supports running this module both as part of the package and from scripts
    # that execute inside Pipeline/ directly.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if pipeline_dir not in sys.path:
        sys.path.append(pipeline_dir)
