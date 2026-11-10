"""Fold-level preprocessing helpers for hyperparameter tuning."""

from .records import append_training_history, export_training_history


DECOMPOSITION_PREFIXES = ('pca__', 'ae__', 'fa__', 'factorvae__')


def split_grid_params(grid_params):
    # Separate the shared param grid by owner. Prefixes are stripped because the
    # underlying transformers expect plain names such as n_components or seq_len.
    return {
        'pca': {k.replace('pca__', ''): v for k, v in grid_params.items() if k.startswith('pca__')},
        'fa': {k.replace('fa__', ''): v for k, v in grid_params.items() if k.startswith('fa__')},
        'ae': {k.replace('ae__', ''): v for k, v in grid_params.items() if k.startswith('ae__')},
        'factor_vae': {
            k.replace('factorvae__', ''): v
            for k, v in grid_params.items()
            if k.startswith('factorvae__')
        },
    }


def tree_grid_params(grid_params):
    # Tree constructors should see only tree-model params. Preprocessing params
    # are consumed by apply_fold_preprocessing before the tree is fit.
    return {
        key: value
        for key, value in grid_params.items()
        if not key.startswith(DECOMPOSITION_PREFIXES)
    }


def apply_fold_preprocessing(
    preprocessor,
    X_train_fold,
    X_val_fold,
    fold_features,
    grid_parts,
    use_standardization=False,
    use_pca=False,
    use_fa=False,
    use_autoencoder=False,
    use_factor_vae=False,
    ae_target_train_fold=None,
    qids_train_fold=None,
    qids_val_fold=None,
    tickers_train_fold=None,
    tickers_val_fold=None,
    progress_factory=None,
    combo_index=0,
    fold_number=0,
    grid_params=None,
    trial_number=None,
    training_history_records=None,
    training_history_export_path=None,
):
    if preprocessor is None:
        # No scaler/decomposer/neural extractor was requested for this run.
        return X_train_fold, X_val_fold, fold_features

    if use_standardization:
        # Fit normalization on the inner train fold and apply it to validation.
        # The bridge decides whether this means global standardization or
        # cross-sectional per-date normalization.
        X_train_fold = preprocessor.fit_transform_standardize(X_train_fold)
        X_val_fold = preprocessor.transform_standardize(X_val_fold)

    if use_pca:
        # PCA is fit on the train fold only. Validation data is projected using
        # that fitted PCA object.
        n_components = _require_n_components(grid_parts['pca'], 'pca')
        X_train_fold = preprocessor.fit_transform_pca(X_train_fold, fold_features, n_components=n_components)
        X_val_fold = preprocessor.transform_pca(X_val_fold)
        fold_features = preprocessor.pca_col_names

    elif use_fa:
        # Factor Analysis follows the same train-fit / validation-transform
        # pattern as PCA.
        n_components = _require_n_components(grid_parts['fa'], 'fa')
        X_train_fold = preprocessor.fit_transform_fa(X_train_fold, fold_features, n_components=n_components)
        X_val_fold = preprocessor.transform_fa(X_val_fold)
        fold_features = preprocessor.fa_col_names

    elif use_autoencoder:
        # The conditional autoencoder can use raw returns, dates, and tickers to
        # learn pricing features. Only the train fold is fit here.
        progress_bar = _make_progress_bar(progress_factory, combo_index, fold_number, 'AE')
        try:
            X_train_fold = preprocessor.fit_transform_autoencoder(
                X_train_fold,
                fold_features,
                target=ae_target_train_fold,
                dates=qids_train_fold,
                tickers=tickers_train_fold,
                progress_bar=progress_bar,
                **grid_parts['ae'],
            )
        finally:
            _close_progress_bar(progress_bar)
        _record_neural_history(
            preprocessor,
            history_attr='ae_history',
            method='autoencoder',
            training_history_records=training_history_records,
            training_history_export_path=training_history_export_path,
            grid_params=grid_params,
            trial_number=trial_number,
            fold_number=fold_number,
        )
        X_val_fold = preprocessor.transform_autoencoder(X_val_fold)
        fold_features = preprocessor.ae_col_names

    elif use_factor_vae:
        # FactorVAE uses return targets and panel identifiers to learn latent
        # factors. The validation fold is transformed after fitting on train.
        progress_bar = _make_progress_bar(progress_factory, combo_index, fold_number, 'FactorVAE')
        try:
            X_train_fold = preprocessor.fit_transform_factor_vae(
                X_train_fold,
                fold_features,
                target=ae_target_train_fold,
                dates=qids_train_fold,
                tickers=tickers_train_fold,
                progress_bar=progress_bar,
                **grid_parts['factor_vae'],
            )
        finally:
            _close_progress_bar(progress_bar)
        _record_neural_history(
            preprocessor,
            history_attr='factor_vae_history',
            method='factor_vae',
            training_history_records=training_history_records,
            training_history_export_path=training_history_export_path,
            grid_params=grid_params,
            trial_number=trial_number,
            fold_number=fold_number,
        )
        X_val_fold = preprocessor.transform_factor_vae(
            X_val_fold,
            dates=qids_val_fold,
            tickers=tickers_val_fold,
        )
        fold_features = preprocessor.factor_vae_col_names

    return X_train_fold, X_val_fold, fold_features


def _make_progress_bar(progress_factory, combo_index, fold_number, method):
    # The factory is injected by search.py only for neural tuning runs.
    if progress_factory is None:
        return None
    return progress_factory(1, f"Trial {combo_index} Fold {fold_number} {method}")


def _require_n_components(params, prefix):
    if 'n_components' not in params:
        raise ValueError(
            f"use_{prefix}=True requires '{prefix}__n_components' in param_grid. "
            "Do not rely on an implicit PCA/FA default during tuning."
        )
    return params['n_components']


def _close_progress_bar(progress_bar):
    # Always close progress bars, including when neural fitting raises.
    if progress_bar is not None:
        progress_bar.close()


def _record_neural_history(
    preprocessor,
    history_attr,
    method,
    training_history_records,
    training_history_export_path,
    grid_params,
    trial_number,
    fold_number,
):
    # Neural extractors expose training history dictionaries. Record them with
    # trial/fold/grid metadata so failed or unstable trials can be diagnosed.
    append_training_history(
        training_history_records,
        history=getattr(preprocessor, history_attr, None),
        grid_params=grid_params,
        trial_number=trial_number,
        fold_number=fold_number,
        method=method,
    )
    export_training_history(training_history_records, training_history_export_path)
