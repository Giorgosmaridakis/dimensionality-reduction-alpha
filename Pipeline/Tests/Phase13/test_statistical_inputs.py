"""
Phase 13: Statistical Test Inputs
Requirement: Verify that the pipeline exports daily OOS IC and daily portfolio
return series suitable for statistical significance tests.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_statistical_inputs import build_statistical_inputs, load_portfolio_returns
from walking_forward import WalkingForwardForecaster


def _workspace_tmp(name: str) -> Path:
    """Use a repo-local test dir because pytest tmp_path is locked in some Windows sandboxes."""
    path = ROOT / "temp" / "phase13_tests" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_phase13_daily_oos_ic_export_and_aggregation():
    tmp_path = _workspace_tmp("daily_ic")
    model = SimpleNamespace(model_type="lightgbm", task_type="ranking")
    forecaster = WalkingForwardForecaster(
        model_wrapper=model,
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    dates = pd.to_datetime([
        "2025-01-01", "2025-01-01", "2025-01-01",
        "2025-01-02", "2025-01-02", "2025-01-02",
    ])
    test_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "A", "B", "C"],
            "Actual_Return": [0.01, 0.02, 0.03, -0.03, -0.02, -0.01],
        },
        index=dates,
    )
    y_test = pd.Series([0, 1, 2, 0, 1, 2], index=dates)
    preds = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    period_str = "01-01-2025_02-01-2025"
    period_dir = tmp_path / period_str
    period_dir.mkdir()

    forecaster._save_results(
        test_data=test_data,
        preds=preds,
        y_test=y_test,
        period_dir=period_dir,
        period_str=period_str,
        tuning_metric="ic",
    )
    forecaster._aggregate_daily_oos_ic(tmp_path)

    period_ic_path = period_dir / f"daily_oos_ic_{period_str}.csv"
    aggregate_ic_path = tmp_path / "daily_oos_ic_series.csv"

    assert period_ic_path.exists()
    assert aggregate_ic_path.exists()

    daily_ic = pd.read_csv(period_ic_path)
    expected_columns = {
        "Date",
        "Daily_IC",
        "N_Assets",
        "Model",
        "Task",
        "Tuning_Metric",
        "Period",
    }
    assert expected_columns.issubset(daily_ic.columns)
    assert len(daily_ic) == 2
    assert daily_ic["N_Assets"].tolist() == [3, 3]
    assert np.isclose(daily_ic.loc[0, "Daily_IC"], 1.0)
    assert np.isclose(daily_ic.loc[1, "Daily_IC"], 1.0)

    aggregate_ic = pd.read_csv(aggregate_ic_path)
    assert len(aggregate_ic) == len(daily_ic)


def test_phase13_backtester_standardized_return_series():
    tmp_path = _workspace_tmp("return_series")
    time_series = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "Total PnL(%)": [0.0, 1.5],
            "Cumulative PnL(%)": [0.0, 1.5],
            "Drawdown(%)": [0.0, -0.2],
        }
    )
    portfolio_dir = tmp_path / "portfolio_metrics_lightgbm_ranking_Baseline_standard"
    portfolio_dir.mkdir()
    time_series.to_csv(portfolio_dir / "long_short_time_series.csv", index=False)

    return_series = load_portfolio_returns(portfolio_dir)

    expected_columns = [
        "Date",
        "Daily_Return",
        "Daily_Return_Pct",
        "Cumulative_Return_Pct",
        "Drawdown_Pct",
        "Strategy",
    ]
    assert expected_columns == list(return_series.columns[:len(expected_columns)])
    assert np.isclose(return_series.loc[1, "Daily_Return"], 0.015)
    assert np.isclose(return_series.loc[1, "Daily_Return_Pct"], 1.5)
    assert return_series.loc[1, "Strategy"] == "long_short"


def test_phase13_master_statistical_input_builder():
    tmp_path = _workspace_tmp("master_builder")
    results_dir = tmp_path / "Results_Pipeline_lightgbm_ranking_Baseline_standard"
    portfolio_dir = tmp_path / "portfolio_metrics_lightgbm_ranking_Baseline_standard"
    output_dir = tmp_path / "Statistical_Results"
    results_dir.mkdir()
    portfolio_dir.mkdir()

    pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02"],
            "Daily_IC": [0.1, 0.2],
            "N_Assets": [3, 3],
            "Model": ["lightgbm", "lightgbm"],
            "Task": ["ranking", "ranking"],
            "Tuning_Metric": ["ic", "ic"],
            "Period": ["p1", "p1"],
        }
    ).to_csv(results_dir / "daily_oos_ic_series.csv", index=False)

    pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02"],
            "Daily_Return": [0.0, 0.015],
            "Daily_Return_Pct": [0.0, 1.5],
            "Cumulative_Return_Pct": [0.0, 1.5],
            "Drawdown_Pct": [0.0, -0.2],
            "Strategy": ["long_short", "long_short"],
        }
    ).to_csv(portfolio_dir / "daily_portfolio_return_series.csv", index=False)

    build_statistical_inputs(root_dir=tmp_path, output_dir=output_dir)

    master_ic = pd.read_csv(output_dir / "daily_oos_ic_series.csv")
    master_returns = pd.read_csv(output_dir / "daily_portfolio_return_series.csv")

    assert len(master_ic) == 2
    assert len(master_returns) == 2
    assert set(["Model", "Task", "Method", "Scaling", "Configuration"]).issubset(master_ic.columns)
    assert set(["Model", "Task", "Method", "Scaling", "Configuration"]).issubset(master_returns.columns)
    assert master_ic.loc[0, "Method"] == "Baseline"
    assert master_returns.loc[0, "Scaling"] == "standard"
