"""
Phase 11: Annualized metrics
Requirement: long_short_Annualized_oos_metric.csv must hold the true annual
return and a 365-day volatility, and must leave the original metrics file alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Backtester.build_annualized_metrics import (
    OUTPUT_NAME,
    annualized_return_and_volatility,
    build_annualized_metrics,
)


def _time_series():
    # 2020-01-01 to 2024-01-01 is 1461 days, exactly 4 years at 365.25 days.
    # Growing 16x over 4 years doubles the money each year: +100% per year.
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2024-01-01"]),
            "Total PnL(%)": [0.0, 1.0, -1.0, 2.0],
            "Cumulative PnL(%)": [0.0, 100.0, 300.0, 1500.0],
        }
    )


def test_annual_return_uses_calendar_years_not_total_return():
    annual_return, _ = annualized_return_and_volatility(_time_series())
    assert annual_return == pytest.approx(100.0)


def test_annual_return_handles_a_loss():
    ts = _time_series()
    # Losing 75% over 4 years is a factor of 0.25, or 0.25 ** (1 / 4) per year.
    ts["Cumulative PnL(%)"] = [0.0, -30.0, -50.0, -75.0]
    annual_return, _ = annualized_return_and_volatility(ts)
    assert annual_return == pytest.approx((0.25 ** 0.25 - 1) * 100)


def test_volatility_uses_365_days_and_skips_the_first_row():
    _, annual_volatility = annualized_return_and_volatility(_time_series())
    expected = np.std([1.0, -1.0, 2.0], ddof=1) * np.sqrt(365)
    assert annual_volatility == pytest.approx(expected)


def test_new_file_corrects_two_columns_and_leaves_the_original_alone(tmp_path):
    _time_series().to_csv(tmp_path / "long_short_time_series.csv", index=False)
    original = tmp_path / "long_short_final_oos_metrics.csv"
    pd.DataFrame(
        [{
            "IC": 0.01,
            "Annualized Return": 1500.0,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": -0.5,
        }]
    ).to_csv(original, index=False)
    original_bytes = original.read_bytes()

    output_path = build_annualized_metrics(tmp_path)

    assert output_path == tmp_path / OUTPUT_NAME
    assert original.read_bytes() == original_bytes
    result = pd.read_csv(output_path).iloc[0]
    assert list(result.index) == ["IC", "Annualized Return", "Annualized Volatility", "Sharpe Ratio"]
    assert result["Annualized Return"] == pytest.approx(100.0)
    assert result["Annualized Volatility"] > 0
    assert result["IC"] == pytest.approx(0.01)
    assert result["Sharpe Ratio"] == pytest.approx(-0.5)
