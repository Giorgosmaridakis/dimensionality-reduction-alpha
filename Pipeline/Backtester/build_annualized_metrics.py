"""
Write long_short_Annualized_oos_metric.csv next to each saved backtest.

The new file copies long_short_final_oos_metrics.csv and corrects two columns,
using the saved daily time series. Nothing is re-simulated, and the original
metrics file is not changed.

    Annualized Return     = true annual rate over calendar time, in percent
    Annualized Volatility = daily std * sqrt(365), in percent, on the same
                            365-day basis as the reported Sharpe Ratio

Usage, from Pipeline:
    python Backtester/build_annualized_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

METRICS_NAME = "long_short_final_oos_metrics.csv"
TIME_SERIES_NAME = "long_short_time_series.csv"
OUTPUT_NAME = "long_short_Annualized_oos_metric.csv"


def annualized_return_and_volatility(time_series):
    """Return (annual return %, annual volatility %) from a daily time series."""
    dates = pd.to_datetime(time_series["Date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    total_return = time_series["Cumulative PnL(%)"].iloc[-1] / 100
    annual_return = ((1 + total_return) ** (1 / years) - 1) * 100

    # The first row has no prior day, so it is not a real daily return.
    daily_returns = time_series["Total PnL(%)"].iloc[1:]
    annual_volatility = daily_returns.std() * np.sqrt(365)
    return annual_return, annual_volatility


def build_annualized_metrics(portfolio_dir):
    """Write the corrected metrics file for one backtest folder and return its path."""
    portfolio_dir = Path(portfolio_dir)
    metrics = pd.read_csv(portfolio_dir / METRICS_NAME)
    time_series = pd.read_csv(portfolio_dir / TIME_SERIES_NAME)

    annual_return, annual_volatility = annualized_return_and_volatility(time_series)
    metrics["Annualized Return"] = annual_return
    metrics["Annualized Volatility"] = annual_volatility

    output_path = portfolio_dir / OUTPUT_NAME
    metrics.to_csv(output_path, index=False)
    return output_path


def main():
    pipeline_dir = Path(__file__).resolve().parents[1]
    written, skipped = 0, []
    for portfolio_dir in sorted(pipeline_dir.glob("portfolio_metrics_*")):
        if (portfolio_dir / METRICS_NAME).exists() and (portfolio_dir / TIME_SERIES_NAME).exists():
            build_annualized_metrics(portfolio_dir)
            written += 1
        else:
            skipped.append(portfolio_dir.name)

    print(f"Wrote {OUTPUT_NAME} in {written} folder(s).")
    for name in skipped:
        print(f"Skipped (missing input files): {name}")


if __name__ == "__main__":
    main()
