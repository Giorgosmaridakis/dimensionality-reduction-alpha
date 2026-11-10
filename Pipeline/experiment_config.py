"""Experiment grids and naming helpers for run_pipeline.py."""

import math
import os


def _env_flag(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


USE_CUDA = _env_flag("PIPELINE_USE_CUDA", default=True)


DEFAULT_EXPERIMENTS = [
    {
        "model": "lightgbm",
        "task": "ranking",
        "universe": "S&P500",
        "start_date": "2022-01-01",
        "end_date": "2025-12-30",
        "profiles": [
            "FactorVAE-Then-LightGBM-ReuseBest-Standard",
            "GKX-AE-Then-LightGBM-ReuseBest-Standard",
        ],
    },
    {
        "model": "catboost",
        "task": "ranking",
        "universe": "S&P500",
        "start_date": "2022-01-01",
        "end_date": "2025-12-30",
        "profiles": [
            "FactorVAE-Then-CatBoost-ReuseBest-Standard",
            "GKX-AE-Then-CatBoost-ReuseBest-Standard",
        ],
    },
]


MODEL_DISPLAY_NAMES = {
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
}


GKX_AE_REUSE_ARTIFACTS = {
    "ETFs": {
        "standard": "Results_Pipeline_no_tree_GKX-AE-ETFs_standard",
        "cross_sectional": (
            "Results_Pipeline_gkx_ae_expected_return_no_tree_"
            "GKX-AE-ExpectedReturn-CrossSectional_cross_sectional_sharp_top5pct"
        ),
    },
    "S&P500": {
        "standard": "Results_Pipeline_gkxno_tree_GKX-Expected_S_P500_Return_standard",
        "cross_sectional": (
            "Results_Pipeline_gkx_ae_expected_return_no_tree_"
            "GKX-AE-ExpectedReturn_cross_sectional_sharp_top5pct_SP500"
        ),
    },
}


FACTORVAE_REUSE_ARTIFACTS = {
    "ETFs": {
        "standard": "Results_Pipeline_no_tree_FactorVAE-ETFs_standard",
        "cross_sectional": "Results_Pipeline_no_tree_FactorVAE-ExpectedReturn_cross_sectional_sharp_top5pct_etfs",
    },
    "S&P500": {
        "standard": "Results_Pipeline_no_tree_FactorVAE-ExpectedReturn_standard_S_P500",
        "cross_sectional": (
            "Results_Pipeline_factorvae_expected_return_no_tree_"
            "FactorVAE-ExpectedReturn_cross_sectional_sharp_top5pct_SP500"
        ),
    },
}


def grid_size(param_grid):
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def sqrt_grid_trials(param_grid, cap=50):
    return min(cap, max(1, math.ceil(math.sqrt(grid_size(param_grid))) + 10))


def loss_for(model_choice, task):
    if task == "ranking":
        if model_choice == "catboost":
            return "YetiRank"
        return "lambdarank"

    if model_choice == "catboost":
        return "RMSE"
    return "regression"


def valid_component_counts(num_features):
    counts = [5, 8, 10, 12, 15, 20]
    return [count for count in counts if count < num_features] or [max(1, num_features - 1)]


def build_base_tree_grid(model_choice):
    if model_choice == "catboost":
        return {
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6],
            "l2_leaf_reg": [3, 7, 10],
            "iterations": [300, 500],
            "colsample_bylevel": [1.0],
        }
    if model_choice == "lightgbm":
        return {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "reg_lambda": [3.0, 5.0, 10.0],
            "n_estimators": [300, 500],
            "min_child_samples": [25, 50, 100],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    raise ValueError(f"Unsupported model_type: {model_choice}")


def build_param_grids(model_choice, valid_components):
    base_tree_grid = build_base_tree_grid(model_choice)

    grids = {
        "baseline": base_tree_grid.copy(),
        "pca": {**base_tree_grid, "pca__n_components": valid_components},
        "fa": {**base_tree_grid, "fa__n_components": valid_components},
    }

    grids["ae"] = {
        **base_tree_grid,
        "ae__latent_dim": valid_components,
        "ae__learning_rate": [5e-4, 1e-4, 5e-5, 1e-5],
        "ae__weight_decay": [1e-6, 1e-5, 1e-4],
        "ae__l1_lambda": [0.0, 1e-5, 1e-4],
        "ae__max_epochs": [200],
        "ae__patience": [20],
    }

    grids["factor_vae"] = {
        **base_tree_grid,
        "factorvae__seq_len": [10,20],
        "factorvae__hidden_size": [32,64,128],
        "factorvae__num_factors": valid_components,
        "factorvae__num_portfolios": [8,16,32],
        "factorvae__learning_rate": [5e-4, 1e-4, 5e-5, 1e-5],
        "factorvae__weight_decay": [1e-6, 1e-5, 1e-4],
        "factorvae__max_epochs": [200],
        "factorvae__patience": [25],
    }
    return grids


def _display_model_name(model_choice):
    return MODEL_DISPLAY_NAMES.get(model_choice, model_choice.upper())


def _filter_profiles(configs, profile_names):
    if not profile_names:
        return configs

    # Baseline, PCA-Only and FA-Only exist once per scaling method, so keep every match.
    return [
        config
        for name in profile_names
        for config in configs
        if config["name"] == name
    ]


def _universe_output_label(universe):
    if not universe:
        return ""
    return str(universe).replace("&", "").replace(" ", "")


def _reuse_artifacts_for(artifact_map, universe, scaling_method):
    universe_artifacts = artifact_map.get(universe)
    if isinstance(universe_artifacts, dict):
        return universe_artifacts.get(scaling_method)
    return universe_artifacts


def _neural_then_tree_config(
    name,
    grids,
    universe,
    scaling_method,
    reuse_neural_artifacts_from,
    use_autoencoder=False,
    use_factor_vae=False,
):
    return {
        "name": name,
        "use_pca": False,
        "use_fa": False,
        "use_ae": use_autoencoder,
        "use_factor_vae": use_factor_vae,
        "keep_orig": False,
        "grid": grids["baseline"],
        "universe": universe,
        "scaling_method": scaling_method,
        "use_optuna": True,
        "optuna_trials": 40,
        "selection_pct": 0.05,
        "prediction_source": "neural_then_tree",
        "neural_selection_metric": "sharp",
        "reuse_neural_artifacts_from": reuse_neural_artifacts_from,
    }


def _neural_expected_return_config(name, neural_grid, universe, scaling_method, use_autoencoder=False, use_factor_vae=False):
    """Standalone neural model: stocks are ranked by its expected return, no tree."""
    return {
        "name": name,
        "use_pca": False,
        "use_fa": False,
        "use_ae": use_autoencoder,
        "use_factor_vae": use_factor_vae,
        "keep_orig": False,
        "grid": neural_grid,
        "universe": universe,
        "scaling_method": scaling_method,
        "use_optuna": True,
        "optuna_trials": 40,
        "selection_pct": 0.05,
        "prediction_source": "neural_expected_return",
        "neural_selection_metric": "sharp",
    }


def standalone_neural_configs(grids, universe="S&P500", profile_names=None):
    """GKX-AE and FactorVAE run on their own. Their saved settings are reused by the neural_then_tree profiles."""
    gkx_grid = {k: v for k, v in grids["ae"].items() if k.startswith("ae__")}
    factorvae_grid = {k: v for k, v in grids["factor_vae"].items() if k.startswith("factorvae__")}
    configs = []
    for scaling_method, suffix in [("standard", "Standard"), ("cross_sectional", "CrossSectional")]:
        configs.append(_neural_expected_return_config(
            f"GKX-AE-ExpectedReturn-{suffix}", gkx_grid, universe, scaling_method, use_autoencoder=True))
        configs.append(_neural_expected_return_config(
            f"FactorVAE-ExpectedReturn-{suffix}", factorvae_grid, universe, scaling_method, use_factor_vae=True))
    return _filter_profiles(configs, profile_names)


def active_feature_reduction_configs(
    grids,
    model_choice="catboost",
    universe="S&P500",
    profile_names=None,
):
    tree_configs = []
    for scaling_method in ["standard", "cross_sectional"]:
        tree_configs.extend(
            [
                {
                    "name": "Baseline",
                    "use_pca": False,
                    "use_fa": False,
                    "use_ae": False,
                    "use_factor_vae": False,
                    "keep_orig": True,
                    "grid": grids["baseline"],
                    "universe": universe,
                    "scaling_method": scaling_method,
                    "use_optuna": True,
                    "optuna_trials": sqrt_grid_trials(grids["baseline"]),
                    "selection_pct": 0.05,
                },
                {
                    "name": "PCA-Only",
                    "use_pca": True,
                    "use_fa": False,
                    "use_ae": False,
                    "use_factor_vae": False,
                    "keep_orig": False,
                    "grid": grids["pca"],
                    "universe": universe,
                    "scaling_method": scaling_method,
                    "use_optuna": True,
                    "optuna_trials": sqrt_grid_trials(grids["pca"]),
                    "selection_pct": 0.05,
                },
                {
                    "name": "FA-Only",
                    "use_pca": False,
                    "use_fa": True,
                    "use_ae": False,
                    "use_factor_vae": False,
                    "keep_orig": False,
                    "grid": grids["fa"],
                    "universe": universe,
                    "scaling_method": scaling_method,
                    "use_optuna": True,
                    "optuna_trials": sqrt_grid_trials(grids["fa"]),
                    "selection_pct": 0.05,
                },
            ]
        )

    model_display = _display_model_name(model_choice)
    configs = [*tree_configs]

    for neural_scaling_method, profile_suffix in [
        ("standard", "Standard"),
        ("cross_sectional", "CrossSectional"),
    ]:
        gkx_reuse_path = _reuse_artifacts_for(
            GKX_AE_REUSE_ARTIFACTS,
            universe,
            neural_scaling_method,
        )
        if gkx_reuse_path:
            configs.append(
                _neural_then_tree_config(
                    name=f"GKX-AE-Then-{model_display}-ReuseBest-{profile_suffix}",
                    grids=grids,
                    universe=universe,
                    scaling_method=neural_scaling_method,
                    reuse_neural_artifacts_from=gkx_reuse_path,
                    use_autoencoder=True,
                )
            )

        factorvae_reuse_path = _reuse_artifacts_for(
            FACTORVAE_REUSE_ARTIFACTS,
            universe,
            neural_scaling_method,
        )
        if factorvae_reuse_path:
            configs.append(
                _neural_then_tree_config(
                    name=f"FactorVAE-Then-{model_display}-ReuseBest-{profile_suffix}",
                    grids=grids,
                    universe=universe,
                    scaling_method=neural_scaling_method,
                    reuse_neural_artifacts_from=factorvae_reuse_path,
                    use_factor_vae=True,
                )
            )

    if profile_names:
        # Standalone neural profiles run only when named, because they do not use the tree model.
        configs.extend(standalone_neural_configs(grids, universe=universe))
    return _filter_profiles(configs, profile_names)


def experiment_labels(model_choice, task, config):
    prediction_source = config.get("prediction_source", "tree")
    if prediction_source == "neural_expected_return":
        if config.get("use_factor_vae", False):
            return "factorvae_expected_return", "no_tree", "FactorVAE ExpectedReturn", "NO TREE"
        if config.get("use_ae", False):
            return "gkx_ae_expected_return", "no_tree", "GKX-AE ExpectedReturn", "NO TREE"
        return "neural_expected_return", "no_tree", "Neural ExpectedReturn", "NO TREE"

    if prediction_source == "neural_then_tree":
        if config.get("use_factor_vae", False):
            return f"factorvae_then_{model_choice}", task, f"FactorVAE + {model_choice.upper()}", task.upper()
        if config.get("use_ae", False):
            return f"gkx_ae_then_{model_choice}", task, f"GKX-AE + {model_choice.upper()}", task.upper()

    return model_choice, task, model_choice.upper(), task.upper()


def output_names(model_choice, task, config):
    model_name, task_name, display_model, display_task = experiment_labels(model_choice, task, config)
    tuning_metric = config.get("tuning_metric", "sharp").lower()
    selection_pct = config.get("selection_pct", 0.05)
    selection_label = f"top{int(selection_pct * 100)}pct"
    suffix = f"{config['name']}_{config.get('scaling_method', 'standard')}_{tuning_metric}_{selection_label}"
    universe_label = _universe_output_label(config.get("universe"))
    if universe_label:
        suffix = f"{suffix}_{universe_label}"
    return {
        "output_dir": f"Results_Pipeline_{model_name}_{task_name}_{suffix}",
        "portfolio_dir": f"portfolio_metrics_{model_name}_{task_name}_{suffix}",
        "display_model": display_model,
        "display_task": display_task,
        "tuning_metric": tuning_metric,
        "selection_pct": selection_pct,
    }
