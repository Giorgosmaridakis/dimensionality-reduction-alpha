import datetime
import glob
import os
import sys

import pandas as pd

# Add the parent directory to sys.path so local imports work from script runs.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_builder import DatasetBuilder
from data.data_loader import DataLoader
from experiment_config import (
    DEFAULT_EXPERIMENTS,
    USE_CUDA,
    active_feature_reduction_configs,
    build_param_grids,
    loss_for,
    output_names,
    valid_component_counts,
)
from modeling.trees import Model
from walking_forward import WalkingForwardForecaster


def _build_dataset(data_loader):
    print("\n========================================")
    print("2. PREPROCESSING & STACKING DATA")
    print("========================================")

    dataset_builder = DatasetBuilder(date_col="Date", ticker_name="ticker")
    print("-> Stacking Alpha Features (This may take a minute)...")
    stacked_alphas = dataset_builder.prepare_stacked_data(data_loader)
    stacked_alphas.to_csv("stacked_alphas.csv", index=False)

    print("-> Merging Features with Target Returns...")
    full_dataset = dataset_builder.merge_features_and_targets(
        stacked_alphas,
        data_loader,
        target_col="Target",
    )
    print("\nDataset Shape:", full_dataset.shape)
    full_dataset.to_csv("full_dataset.csv", index=False)
    return full_dataset


def _candidate_features(full_dataset):
    return [
        col
        for col in full_dataset.columns
        if col not in ["Date", "ticker", "Target", "Target_prc"]
    ]


def _build_model(model_choice, task):
    return Model(
        model_type=model_choice,
        task_type=task,
        loss_function=loss_for(model_choice, task),
        use_gpu=USE_CUDA,
        random_seed=42,
    )


def _run_backtest(output_dir, portfolio_dir, close_df, universe, selection_pct):
    print("\n-> Backtesting Configuration...")
    if close_df is None:
        return

    from Backtester.backtester import Backtester

    pred_files = glob.glob(os.path.join(output_dir, "*", "predictions_*.csv"))
    if not pred_files:
        return

    predictions_df = pd.concat(
        [pd.read_csv(path) for path in sorted(pred_files)],
        ignore_index=True,
    )
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"], dayfirst=True)
    predictions_df = predictions_df.sort_values(["Date", "Ticker"]).drop_duplicates(
        subset=["Date", "Ticker"],
        keep="last",
    )

    bt = Backtester(output_folder=portfolio_dir, universe=universe)
    bt.run_backtest(
        data=close_df,
        predictions=predictions_df,
        strategy="long_short",
        selection_pct=selection_pct,
    )


def _log_experiment(display_model, display_task, config_name, universe, tuning_metric, selection_pct, output_dir):
    log_file = "experiment_history_log.csv"
    log_exists = os.path.isfile(log_file)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a") as f:
        if not log_exists:
            f.write("Completion_Time,Model,Task,Profile,Universe,Tuning_Metric,Selection_Pct,Output_Directory\n")
        f.write(
            f"{timestamp},{display_model},{display_task},{config_name},"
            f"{universe},{tuning_metric},{selection_pct},{output_dir}\n"
        )

    print(f"Experiment logged to {log_file} at {timestamp}\n")


def _resolve_reuse_path(reuse_path):
    if not reuse_path or os.path.isabs(reuse_path) or os.path.exists(reuse_path):
        return reuse_path

    script_dir = os.path.abspath(os.path.dirname(__file__) or os.getcwd())
    script_relative_path = os.path.join(script_dir, reuse_path)
    if os.path.exists(script_relative_path):
        return os.path.abspath(script_relative_path)

    def asciiish_name(value):
        translation = {
            ord("\u0395"): "E",
            ord("\u03A4"): "T",
            ord("\u03B5"): "e",
            ord("\u03C4"): "t",
        }
        return str(value).translate(translation).casefold()

    target_name = asciiish_name(os.path.basename(reuse_path))
    for entry in os.scandir(script_dir):
        if entry.is_dir() and asciiish_name(entry.name) == target_name:
            return os.path.abspath(entry.path)

    return reuse_path


def _run_experiment_config(
    forecaster,
    full_dataset,
    candidate_features,
    close_df,
    universe,
    model_choice,
    task,
    config,
    current_index,
    total_experiments,
):
    names = output_names(model_choice, task, config)
    output_dir = names["output_dir"]
    portfolio_dir = names["portfolio_dir"]
    selection_pct = names["selection_pct"]
    tuning_metric = names["tuning_metric"]
    reuse_neural_artifacts_from = _resolve_reuse_path(config.get("reuse_neural_artifacts_from"))

    print("\n========================================")
    print(f"RUNNING EXPERIMENT {current_index} / {total_experiments}")
    print(f"MODEL: {names['display_model']} | TASK: {names['display_task']} | PROFILE: {config['name']}")
    print(f"Directory: {output_dir}")
    print("========================================")

    forecaster.forecast(
        full_df=full_dataset,
        candidate_features=candidate_features,
        base_output_dir=output_dir,
        use_pca=config.get("use_pca", False),
        use_fa=config.get("use_fa", False),
        use_autoencoder=config.get("use_ae", False),
        use_factor_vae=config.get("use_factor_vae", False),
        keep_original_features=config.get("keep_orig", False),
        use_standardization=config.get("use_standardization", True),
        scaling_method=config.get("scaling_method", "standard"),
        param_grid=config["grid"],
        as_percentages=True,
        tuning_metric=tuning_metric,
        use_optuna=config.get("use_optuna", True),
        optuna_trials=config.get("optuna_trials", 40),
        close_prices=close_df,
        universe=universe,
        selection_pct=selection_pct,
        prediction_source=config.get("prediction_source", "tree"),
        neural_selection_metric=config.get("neural_selection_metric", "validation_rank_ic"),
        reuse_neural_artifacts_from=reuse_neural_artifacts_from,
    )

    _run_backtest(output_dir, portfolio_dir, close_df, universe, selection_pct)
    _log_experiment(
        names["display_model"],
        names["display_task"],
        config["name"],
        universe,
        tuning_metric,
        selection_pct,
        output_dir,
    )


def run_end_to_end_pipeline():
    """Main thesis experiment entry point."""
    print(f"CUDA/GPU acceleration requested: {USE_CUDA}")
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed; neural CUDA availability not checked.")

    for exp in DEFAULT_EXPERIMENTS:
        model_choice = exp["model"]
        task = exp["task"]
        universe = exp.get("universe", "S&P500")
        start_date = exp.get("start_date", "2022-01-01")
        end_date = exp.get("end_date", "2025-12-30")
        max_training_window_years = exp.get("max_training_window_years", 4)

        print("========================================")
        print(f"1. INITIALIZING DATA LOADER | UNIVERSE: {universe}")
        print("========================================")
        data_loader = DataLoader(universe=universe)

        print("-> Discovering Alpha Feature Files...")
        alpha_files = data_loader.get_alpha_files()
        print(f"   Found {len(alpha_files)} unique alpha features.")

        print("-> Loading Universe Adjusted Close Prices for Tuning and Backtesting...")
        close_df = data_loader.get_adjusted_close()
        print("-> Target returns will be built from tradable next adjusted-close-to-close prices.")

        full_dataset = _build_dataset(data_loader)

        print("\n========================================")
        print("3. MODEL SELECTION & WALKING FORWARD")
        print("========================================")

        candidate_features = _candidate_features(full_dataset)
        valid_components = valid_component_counts(len(candidate_features))
        print(
            f"-> Discovered {len(candidate_features)} candidate features. "
            f"Component counts to test: {valid_components}"
        )

        print("\n" + "=" * 60)
        print(f" STARTING EXPERIMENT SEQUENCE: {model_choice.upper()} | {task.upper()} | {universe}")
        print("=" * 60)

        ml_model = _build_model(model_choice, task)
        grids = build_param_grids(model_choice, valid_components)
        configs = active_feature_reduction_configs(
            grids,
            model_choice=model_choice,
            universe=universe,
            profile_names=exp.get("profiles"),
        )
        forecaster = WalkingForwardForecaster(
            model_wrapper=ml_model,
            start_date=start_date,
            end_date=end_date,
            max_training_window_years=max_training_window_years,
            target_col="Target",
        )

        for idx, config in enumerate(configs, start=1):
            _run_experiment_config(
                forecaster=forecaster,
                full_dataset=full_dataset,
                candidate_features=candidate_features,
                close_df=close_df,
                universe=universe,
                model_choice=model_choice,
                task=task,
                config=config,
                current_index=idx,
                total_experiments=len(configs),
            )


if __name__ == "__main__":
    run_end_to_end_pipeline()
