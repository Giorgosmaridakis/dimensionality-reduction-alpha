from __future__ import annotations

from experiment_config import (
    DEFAULT_EXPERIMENTS,
    active_feature_reduction_configs,
    build_param_grids,
    loss_for,
    output_names,
    valid_component_counts,
)


def test_default_sp500_experiments_run_neural_standard_reuse_profiles_2022_to_2025():
    assert DEFAULT_EXPERIMENTS == [
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


def test_active_experiment_configs_can_select_lightgbm_sp500_standard_reuse_profiles():
    exp = DEFAULT_EXPERIMENTS[0]
    valid_components = valid_component_counts(num_features=25)
    grids = build_param_grids(exp["model"], valid_components)
    configs = active_feature_reduction_configs(
        grids,
        model_choice=exp["model"],
        universe=exp["universe"],
        profile_names=exp["profiles"],
    )

    assert len(configs) == 2
    configs_by_name = {config["name"]: config for config in configs}

    factorvae_config = configs_by_name["FactorVAE-Then-LightGBM-ReuseBest-Standard"]
    assert factorvae_config["universe"] == "S&P500"
    assert factorvae_config["use_ae"] is False
    assert factorvae_config["use_factor_vae"] is True
    assert factorvae_config["prediction_source"] == "neural_then_tree"
    assert factorvae_config["scaling_method"] == "standard"
    assert factorvae_config["neural_selection_metric"] == "sharp"
    assert factorvae_config["reuse_neural_artifacts_from"].endswith(
        "Results_Pipeline_no_tree_FactorVAE-ExpectedReturn_standard_S_P500"
    )
    assert all(not key.startswith("factorvae__") for key in factorvae_config["grid"])
    assert "n_estimators" in factorvae_config["grid"]
    assert "min_child_samples" in factorvae_config["grid"]

    gkx_config = configs_by_name["GKX-AE-Then-LightGBM-ReuseBest-Standard"]
    assert gkx_config["universe"] == "S&P500"
    assert gkx_config["use_ae"] is True
    assert gkx_config["use_factor_vae"] is False
    assert gkx_config["prediction_source"] == "neural_then_tree"
    assert gkx_config["scaling_method"] == "standard"
    assert gkx_config["neural_selection_metric"] == "sharp"
    assert gkx_config["reuse_neural_artifacts_from"].endswith(
        "Results_Pipeline_gkxno_tree_GKX-Expected_S_P500_Return_standard"
    )
    assert all(not key.startswith("ae__") for key in gkx_config["grid"])
    assert "n_estimators" in gkx_config["grid"]
    assert "min_child_samples" in gkx_config["grid"]


def test_active_experiment_configs_can_select_catboost_sp500_standard_reuse_profiles():
    exp = DEFAULT_EXPERIMENTS[1]
    valid_components = valid_component_counts(num_features=25)
    grids = build_param_grids(exp["model"], valid_components)
    configs = active_feature_reduction_configs(
        grids,
        model_choice=exp["model"],
        universe=exp["universe"],
        profile_names=exp["profiles"],
    )

    assert len(configs) == 2
    configs_by_name = {config["name"]: config for config in configs}

    factorvae_config = configs_by_name["FactorVAE-Then-CatBoost-ReuseBest-Standard"]
    assert factorvae_config["universe"] == "S&P500"
    assert factorvae_config["use_ae"] is False
    assert factorvae_config["use_factor_vae"] is True
    assert factorvae_config["prediction_source"] == "neural_then_tree"
    assert factorvae_config["scaling_method"] == "standard"
    assert factorvae_config["neural_selection_metric"] == "sharp"
    assert factorvae_config["reuse_neural_artifacts_from"].endswith(
        "Results_Pipeline_no_tree_FactorVAE-ExpectedReturn_standard_S_P500"
    )
    assert all(not key.startswith("factorvae__") for key in factorvae_config["grid"])
    assert "iterations" in factorvae_config["grid"]
    assert "depth" in factorvae_config["grid"]

    gkx_config = configs_by_name["GKX-AE-Then-CatBoost-ReuseBest-Standard"]
    assert gkx_config["universe"] == "S&P500"
    assert gkx_config["use_ae"] is True
    assert gkx_config["use_factor_vae"] is False
    assert gkx_config["prediction_source"] == "neural_then_tree"
    assert gkx_config["scaling_method"] == "standard"
    assert gkx_config["neural_selection_metric"] == "sharp"
    assert gkx_config["reuse_neural_artifacts_from"].endswith(
        "Results_Pipeline_gkxno_tree_GKX-Expected_S_P500_Return_standard"
    )
    assert all(not key.startswith("ae__") for key in gkx_config["grid"])
    assert "iterations" in gkx_config["grid"]
    assert "depth" in gkx_config["grid"]


def test_active_experiment_configs_include_tree_profiles_for_both_scalers_when_unfiltered():
    valid_components = valid_component_counts(num_features=25)
    grids = build_param_grids("lightgbm", valid_components)
    configs = active_feature_reduction_configs(
        grids,
        model_choice="lightgbm",
        universe="S&P500",
    )

    tree_configs = [
        config
        for config in configs
        if config.get("prediction_source", "tree") == "tree"
    ]
    observed = {(config["name"], config["scaling_method"]) for config in tree_configs}

    assert observed == {
        ("Baseline", "standard"),
        ("PCA-Only", "standard"),
        ("FA-Only", "standard"),
        ("Baseline", "cross_sectional"),
        ("PCA-Only", "cross_sectional"),
        ("FA-Only", "cross_sectional"),
    }

    configs_by_profile = {
        (config["name"], config["scaling_method"]): config
        for config in tree_configs
    }
    baseline = configs_by_profile[("Baseline", "standard")]
    pca = configs_by_profile[("PCA-Only", "standard")]
    fa = configs_by_profile[("FA-Only", "standard")]

    assert baseline["keep_orig"] is True
    assert baseline["use_pca"] is False
    assert baseline["use_fa"] is False
    assert baseline["grid"] == grids["baseline"]

    assert pca["keep_orig"] is False
    assert pca["use_pca"] is True
    assert pca["use_fa"] is False
    assert pca["grid"] == grids["pca"]

    assert fa["keep_orig"] is False
    assert fa["use_pca"] is False
    assert fa["use_fa"] is True
    assert fa["grid"] == grids["fa"]


def test_factorvae_standard_and_cross_sectional_sp500_profiles_are_available():
    grids = build_param_grids("lightgbm", valid_component_counts(num_features=25))
    configs = active_feature_reduction_configs(
        grids,
        model_choice="lightgbm",
        universe="S&P500",
    )
    configs_by_name = {config["name"]: config for config in configs}

    assert any("FactorVAE" in profile for exp in DEFAULT_EXPERIMENTS for profile in exp["profiles"])
    factorvae_standard = configs_by_name["FactorVAE-Then-LightGBM-ReuseBest-Standard"]
    assert factorvae_standard["scaling_method"] == "standard"
    assert factorvae_standard["reuse_neural_artifacts_from"].endswith(
        "Results_Pipeline_no_tree_FactorVAE-ExpectedReturn_standard_S_P500"
    )
    factorvae_config = configs_by_name["FactorVAE-Then-LightGBM-ReuseBest-CrossSectional"]
    assert factorvae_config["reuse_neural_artifacts_from"].endswith(
        "Results_Pipeline_factorvae_expected_return_no_tree_FactorVAE-ExpectedReturn_cross_sectional_sharp_top5pct_SP500"
    )


def test_output_names_add_gkx_then_lightgbm_convention_for_sp500():
    grids = build_param_grids("lightgbm", valid_component_counts(num_features=25))
    config = next(
        config
        for config in active_feature_reduction_configs(
            grids,
            model_choice="lightgbm",
            universe="S&P500",
        )
        if config["name"] == "GKX-AE-Then-LightGBM-ReuseBest-Standard"
    )
    names = output_names("lightgbm", "ranking", config)

    assert names["display_model"] == "GKX-AE + LIGHTGBM"
    assert names["display_task"] == "RANKING"
    assert names["tuning_metric"] == "sharp"
    assert names["output_dir"].startswith("Results_Pipeline_gkx_ae_then_lightgbm_ranking_")
    assert "GKX-AE-Then-LightGBM-ReuseBest-Standard" in names["output_dir"]
    assert names["output_dir"].endswith("_sharp_top5pct_SP500")
    assert names["portfolio_dir"].endswith("_sharp_top5pct_SP500")
    assert "_standard_" in names["output_dir"]


def test_output_names_add_gkx_then_catboost_convention_for_sp500():
    grids = build_param_grids("catboost", valid_component_counts(num_features=25))
    config = next(
        config
        for config in active_feature_reduction_configs(
            grids,
            model_choice="catboost",
            universe="S&P500",
        )
        if config["name"] == "GKX-AE-Then-CatBoost-ReuseBest-Standard"
    )
    names = output_names("catboost", "ranking", config)

    assert names["display_model"] == "GKX-AE + CATBOOST"
    assert names["display_task"] == "RANKING"
    assert names["tuning_metric"] == "sharp"
    assert names["output_dir"].startswith("Results_Pipeline_gkx_ae_then_catboost_ranking_")
    assert "GKX-AE-Then-CatBoost-ReuseBest-Standard" in names["output_dir"]
    assert names["output_dir"].endswith("_sharp_top5pct_SP500")
    assert names["portfolio_dir"].endswith("_sharp_top5pct_SP500")
    assert "_standard_" in names["output_dir"]


def test_tree_loss_mapping_remains_stable():
    assert loss_for("lightgbm", "ranking") == "lambdarank"
    assert loss_for("catboost", "ranking") == "YetiRank"
    assert loss_for("lightgbm", "regression") == "regression"


def test_named_tree_profile_selects_both_scaling_methods():
    grids = build_param_grids("lightgbm", valid_component_counts(80))
    configs = active_feature_reduction_configs(grids, model_choice="lightgbm", universe="ETFs", profile_names=["Baseline"])

    assert [c["scaling_method"] for c in configs] == ["standard", "cross_sectional"]


def test_standalone_neural_profiles_run_only_when_named():
    grids = build_param_grids("lightgbm", valid_component_counts(80))
    unfiltered = active_feature_reduction_configs(grids, model_choice="lightgbm", universe="ETFs")
    assert not any(c.get("prediction_source") == "neural_expected_return" for c in unfiltered)

    names = [
        "GKX-AE-ExpectedReturn-Standard",
        "GKX-AE-ExpectedReturn-CrossSectional",
        "FactorVAE-ExpectedReturn-Standard",
        "FactorVAE-ExpectedReturn-CrossSectional",
    ]
    configs = active_feature_reduction_configs(grids, model_choice="lightgbm", universe="ETFs", profile_names=names)

    assert [c["name"] for c in configs] == names
    assert all(c["prediction_source"] == "neural_expected_return" for c in configs)
    assert all(c["optuna_trials"] == 40 for c in configs)
    assert all(k.startswith("ae__") for k in configs[0]["grid"])
    assert all(k.startswith("factorvae__") for k in configs[2]["grid"])
    assert output_names("lightgbm", "ranking", configs[2])["output_dir"].startswith(
        "Results_Pipeline_factorvae_expected_return_no_tree_"
    )
