"""Result and training-history bookkeeping for hyperparameter tuning."""

from pathlib import Path

import numpy as np
import pandas as pd


def append_tuning_result(tuning_results, grid_params, fold_scores, pruned, trial_number, pruning_enabled):
    if tuning_results is None:
        return

    # Ignore failed folds when reporting partial score, but require all three
    # folds to succeed before treating the candidate as a valid final score.
    finite_scores = [score for score in fold_scores if score != float('-inf')]
    partial_score = np.mean(finite_scores) if finite_scores else float('-inf')
    is_complete = len(fold_scores) == 3 and len(finite_scores) == 3
    validation_score = partial_score if is_complete and not pruned else float('-inf')

    # Preserve the grid params on every result row so the CSV is self-contained.
    result_row = grid_params.copy()
    result_row['Trial'] = trial_number
    result_row['Validation_Score'] = validation_score
    result_row['Partial_Validation_Score'] = partial_score
    result_row['Pruned'] = bool(pruned)
    result_row['Pruning_Enabled'] = bool(pruning_enabled)
    result_row['Complete_CV'] = bool(is_complete)
    result_row['Completed_Folds'] = len(fold_scores)
    result_row['Successful_Folds'] = len(finite_scores)

    for i, score in enumerate(fold_scores, start=1):
        # Keep per-fold scores for debugging unstable parameter combinations.
        result_row[f'Fold_{i}_Score'] = score

    tuning_results.append(result_row)


def append_training_history(training_history_records, history, grid_params, trial_number, fold_number, method):
    if training_history_records is None or not history:
        return

    # Neural history is stored as dict-of-lists, one list per tracked metric.
    # Convert it to row records with Trial/Fold/Method attached.
    max_len = max((len(values) for values in history.values() if hasattr(values, '__len__')), default=0)
    for epoch_idx in range(max_len):
        row = {
            'Trial': trial_number,
            'Fold': fold_number,
            'Method': method,
        }

        for key, values in history.items():
            if hasattr(values, '__len__') and epoch_idx < len(values):
                # Avoid colliding with the output row's own meaning of epoch.
                history_key = 'training_epoch' if key == 'epoch' else key
                row[history_key] = values[epoch_idx]

        for key, value in grid_params.items():
            # Prefix grid params so they are easy to distinguish from training
            # metrics like loss, val_loss, or IC.
            row[f'grid__{key}'] = value
        training_history_records.append(row)


def export_training_history(training_history_records, training_history_export_path):
    if not training_history_export_path or not training_history_records:
        return

    # Write incrementally during long neural tuning runs so progress is not lost
    # if a later trial fails.
    Path(training_history_export_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_history_records).to_csv(training_history_export_path, index=False)
