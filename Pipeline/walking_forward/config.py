"""Configuration and parameter-grid helpers for walk-forward runs."""

import itertools


class ConfigMixin:
    @staticmethod
    def _validate_prediction_source(prediction_source, use_autoencoder=False, use_factor_vae=False):
        valid_sources = {'tree', 'neural_expected_return', 'neural_then_tree'}
        if prediction_source not in valid_sources:
            raise ValueError(f"Invalid prediction_source '{prediction_source}'. Supported values: {sorted(valid_sources)}")

        if prediction_source != 'tree':
            selected_neural_models = int(bool(use_autoencoder)) + int(bool(use_factor_vae))
            if selected_neural_models != 1:
                raise ValueError(
                    "prediction_source='neural_expected_return' or 'neural_then_tree' "
                    "requires exactly one of use_autoencoder=True or use_factor_vae=True."
                )
        return prediction_source

    @staticmethod
    def _validate_neural_selection_metric(neural_selection_metric):
        aliases = {
            'rank_ic': 'validation_rank_ic',
            'validation_rank_ic': 'validation_rank_ic',
            'in_sample_rank_ic': 'in_sample_rank_ic',
            'sharpe': 'validation_sharpe',
            'sharp': 'validation_sharpe',
            'validation_sharpe': 'validation_sharpe',
            'validation_sharp': 'validation_sharpe',
        }
        metric_key = str(neural_selection_metric).lower()
        if metric_key not in aliases:
            raise ValueError(
                f"Invalid neural_selection_metric '{neural_selection_metric}'. "
                f"Supported values: {sorted(aliases)}"
            )
        return aliases[metric_key]

    @staticmethod
    def _split_neural_tree_grid(param_grid, use_autoencoder=False, use_factor_vae=False):
        if not param_grid:
            return {}, {}

        if use_autoencoder:
            neural_prefixes = ('ae__',)
        elif use_factor_vae:
            neural_prefixes = ('factorvae__',)
        else:
            neural_prefixes = tuple()

        neural_grid = {
            key: value
            for key, value in param_grid.items()
            if key.startswith(neural_prefixes)
        }
        tree_grid = {
            key: value
            for key, value in param_grid.items()
            if not key.startswith(('pca__', 'fa__', 'ae__', 'factorvae__'))
        }
        return neural_grid, tree_grid

    @staticmethod
    def _neural_expected_return_col(use_autoencoder=False, use_factor_vae=False):
        if use_autoencoder:
            return 'gkx_expected_return'
        if use_factor_vae:
            return 'factorvae_expected_return'
        raise ValueError("No neural expected-return source selected.")

    @staticmethod
    def _neural_pricing_features(df, use_autoencoder=False, use_factor_vae=False):
        if use_autoencoder:
            feature_cols = (
                ['gkx_expected_return'] +
                [col for col in df.columns if col.startswith('ae_component_')] +
                [col for col in df.columns if col.startswith('gkx_factor_forecast_')]
            )
        elif use_factor_vae:
            feature_cols = (
                ['factorvae_expected_return'] +
                [col for col in df.columns if col.startswith('factorvae_beta_')] +
                [col for col in df.columns if col in ['factorvae_alpha_mu', 'factorvae_alpha_sigma']]
            )
        else:
            feature_cols = []

        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Neural pricing output columns missing: {missing}")
        if not feature_cols:
            raise ValueError("No neural pricing output columns found.")
        return feature_cols

    @staticmethod
    def _iter_param_combinations(param_grid):
        if not param_grid:
            return [{}]
        keys, values = zip(*param_grid.items())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
