"""Search strategies for hyperparameter tuning."""

import itertools

import numpy as np

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None

    class TrialPruned(Exception):
        pass

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def build_param_combinations(param_grid):
    # Convert a sklearn-style grid into explicit dictionaries. Example:
    # {'lr': [0.1, 0.01], 'depth': [3, 5]} -> four candidate dictionaries.
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, value_combo)) for value_combo in itertools.product(*values)]


def should_use_optuna(use_optuna):
    # If Optuna is not installed, silently fall back to full grid search.
    return bool(use_optuna and optuna is not None)


def maybe_add_progress_factory(eval_context, use_autoencoder, use_factor_vae):
    # Deep-learning trials can take long enough that a compact progress bar is
    # useful. Tree-only tuning stays quiet to avoid noisy logs.
    if (use_autoencoder or use_factor_vae) and tqdm is not None:
        def progress_factory(total, desc):
            return tqdm(
                total=total,
                desc=desc,
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )

        eval_context['progress_factory'] = progress_factory


def run_grid_search(combinations, evaluate_combo, eval_context):
    # Exhaustive and deterministic: evaluate every combination once and keep
    # the highest validation score.
    best_score = float('-inf')
    best_params = {}

    for i, grid_params in enumerate(combinations):
        avg_score = evaluate_combo(
            grid_params,
            combo_index=f"{i + 1}/{len(combinations)}",
            **eval_context,
        )
        if avg_score > best_score:
            best_score = avg_score
            best_params = grid_params

    return best_params, best_score


def run_optuna_search(
    param_grid,
    combinations,
    evaluate_combo,
    eval_context,
    optuna_trials,
    use_autoencoder=False,
    use_factor_vae=False,
):
    # Optuna samples from the same discrete param_grid values, but it can stop
    # early after optuna_trials unique candidates instead of exhausting the grid.
    maybe_add_progress_factory(eval_context, use_autoencoder, use_factor_vae)

    enable_pruning = use_autoencoder or use_factor_vae
    # Pruning is most helpful for expensive neural trials. Tree trials are cheap
    # enough that disabling pruning keeps their scores easier to interpret.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=2,
        interval_steps=1,
    ) if enable_pruning else optuna.pruners.NopPruner()

    seen_param_keys = set()
    queued_param_keys = set()
    completed_unique_trials = {'count': 0}
    param_order = list(param_grid.keys())
    total_unique_combinations = len(combinations)

    def canonical_value(value):
        # Normalize NumPy scalar types so parameter dictionaries can be compared
        # reliably inside tuples/sets.
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        return str(value)

    def make_param_key(params):
        # Preserve param_grid order so the same combination always produces the
        # same duplicate-detection key.
        return tuple((key, canonical_value(params[key])) for key in param_order)

    # Optuna may suggest the same categorical combination multiple times. The
    # following state tracks what has already run and queues unused fallbacks.
    combination_keys = [make_param_key(combo) for combo in combinations]
    fallback_order = list(np.random.default_rng(42).permutation(total_unique_combinations))
    fallback_state = {'cursor': 0, 'duplicates': 0, 'enqueued': 0}

    def enqueue_unused_combination():
        # When Optuna proposes a duplicate, enqueue one unused grid combination
        # so the next trial still gives useful information to the study.
        while fallback_state['cursor'] < len(fallback_order):
            combo_idx = fallback_order[fallback_state['cursor']]
            fallback_state['cursor'] += 1
            fallback_key = combination_keys[combo_idx]

            if fallback_key in seen_param_keys or fallback_key in queued_param_keys:
                continue

            fallback_params = combinations[combo_idx]
            study.enqueue_trial(fallback_params)
            queued_param_keys.add(fallback_key)
            fallback_state['enqueued'] += 1
            return fallback_params

        return None

    def objective(trial):
        # Ask Optuna for one categorical choice per hyperparameter.
        grid_params = {}
        for key, values in param_grid.items():
            grid_params[key] = trial.suggest_categorical(key, values)

        param_key = make_param_key(grid_params)
        if param_key in seen_param_keys:
            # Do not waste a full CV run on a combination that has already been
            # evaluated. Prune this duplicate trial and queue a replacement.
            fallback_state['duplicates'] += 1
            fallback_params = enqueue_unused_combination()
            if fallback_params is not None:
                print(
                    f"-> Skipping duplicate Optuna trial {trial.number + 1}: {grid_params}. "
                    f"Queued unused fallback: {fallback_params}"
                )
            else:
                print(f"-> Skipping duplicate Optuna trial {trial.number + 1}: {grid_params}. No unused fallback left.")
            raise TrialPruned()

        queued_param_keys.discard(param_key)
        seen_param_keys.add(param_key)
        # evaluate_combo performs the actual inner CV training/scoring.
        score = evaluate_combo(
            grid_params,
            combo_index=trial.number + 1,
            trial=trial,
            enable_pruning=enable_pruning,
            **eval_context,
        )
        completed_unique_trials['count'] += 1
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    max_attempts = min(total_unique_combinations * 3, max(optuna_trials * 10, optuna_trials))
    attempts = 0
    target_unique_trials = min(optuna_trials, total_unique_combinations)

    # Optimize one trial at a time so duplicate handling can enqueue fallback
    # trials between attempts.
    while (
        completed_unique_trials['count'] < target_unique_trials
        and len(seen_param_keys) < total_unique_combinations
        and attempts < max_attempts
    ):
        study.optimize(
            objective,
            n_trials=1,
            show_progress_bar=False,
        )
        attempts += 1

    print(
        f"   Optuna unique completed trials: {completed_unique_trials['count']}/{target_unique_trials} "
        f"(attempts={attempts}, unique sampled={len(seen_param_keys)}, "
        f"duplicates={fallback_state['duplicates']}, fallbacks queued={fallback_state['enqueued']}, "
        f"total grid={total_unique_combinations})"
    )
    return study.best_params, study.best_value
