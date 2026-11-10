"""
run_backtester.py
-----------------
Applies the new Backtester to every Results_Pipeline_* folder found in the
Pipeline directory.

For each experiment configuration it:
  1. Globs all predictions_*.csv files from the year sub-folders.
  2. Runs the backtester with three strategies: long, short, long_short.
  3. Saves outputs to a matching portfolio_metrics_* folder.

Usage (from the Pipeline directory):
    python run_backtester.py

Optional CLI overrides:
    python run_backtester.py --strategies long long_short
    python run_backtester.py --results-dir /custom/path
    python run_backtester.py --close-csv /path/to/Close.csv
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from typing import Optional

import pandas as pd

# ── resolve project root so the script works from any CWD ──────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add the pipeline root to sys.path so local package imports work from any CWD.
sys.path.insert(0, str(PROJECT_ROOT))

from Backtester.backtester import Backtester                     # noqa: E402
from data.data_loader import DataLoader                          # noqa: E402


# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_RESULTS_DIR = SCRIPT_DIR                      # scan inside Pipeline/
DEFAULT_STRATEGIES  = [ "long_short"]
DEFAULT_INIT_CASH   = 10_000_000


def parse_args():
    p = argparse.ArgumentParser(description="Run Backtester over all Results_Pipeline_* folders.")
    p.add_argument(
        "--close-csv",
        default=None,
        help="Optional path to a wide close-price CSV. If omitted, DataLoader loads all adjusted-close periods.",
    )
    p.add_argument(
        "--universe",
        default="ETFs",
        choices=["ETFs", "S&P500"],
        help="Universe used for close-price loading and backtester cost assumptions. Default: ETFs.",
    )
    p.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to search for Results_Pipeline_* folders."
             f"  Default: {DEFAULT_RESULTS_DIR}",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        choices=["long_short"],
        help="Strategies to run for each experiment.  Default:  long_short",
    )
    p.add_argument(
        "--init-cash",
        type=float,
        default=DEFAULT_INIT_CASH,
        help=f"Starting portfolio cash.  Default: {DEFAULT_INIT_CASH}",
    )
    return p.parse_args()


def load_close_prices(close_csv_path: str = None, universe: str = "ETFs"):
    """Load the wide-format Close price file."""
    if close_csv_path is None:
        print(f"\n[INFO] Loading adjusted close prices through DataLoader for universe={universe}")
        loader = DataLoader(universe=universe)
        close_df = loader.get_adjusted_close()
        print(f"       Shape: {close_df.shape}  |  "
              f"Date range: {close_df.index.min().date()} -> {close_df.index.max().date()}")
        return close_df

    path = Path(close_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Close.csv not found at: {path}\n"
            "Please pass --close-csv with the correct path."
        )
    print(f"\n[INFO] Loading close prices from: {path}")
    close_df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    print(f"       Shape: {close_df.shape}  |  "
          f"Date range: {close_df.index.min().date()} → {close_df.index.max().date()}")
    return close_df


def find_results_folders(results_dir: str) -> list:
    """Return all Results_Pipeline_* sub-directories, sorted."""
    pattern = os.path.join(results_dir, "Results_Pipeline_*")
    folders = sorted([f for f in glob.glob(pattern) if os.path.isdir(f)])
    return folders


def load_predictions(results_folder: str) -> Optional[pd.DataFrame]:
    """
    Glob all predictions_*.csv files inside year sub-folders of a
    Results_Pipeline_* directory and concatenate them.
    """
    pred_files = glob.glob(
        os.path.join(results_folder, "*", "predictions_*.csv")
    )
    if not pred_files:
        print(f"  [WARN] No predictions_*.csv files found in: {results_folder}")
        return None

    print(f"  [INFO] Found {len(pred_files)} prediction file(s):")
    for fp in sorted(pred_files):
        print(f"         {os.path.relpath(fp, results_folder)}")

    dfs = [pd.read_csv(f) for f in sorted(pred_files)]
    predictions_df = pd.concat(dfs, ignore_index=True)

    # Normalise the Date column (dayfirst because files use DD-MM-YYYY)
    predictions_df["Date"] = pd.to_datetime(
        predictions_df["Date"], dayfirst=True
    )
    print(f"  [INFO] Total predictions: {len(predictions_df):,} rows  |  "
          f"Dates: {predictions_df['Date'].min().date()} → "
          f"{predictions_df['Date'].max().date()}")
    return predictions_df


def portfolio_folder_for(results_folder: str) -> str:
    """
    Derive the output portfolio folder name from the Results_Pipeline_* name.
    Results_Pipeline_lightgbm_ranking_PCA-Only_standard_sharp_top5pct_ETFs
      → portfolio_metrics_lightgbm_ranking_PCA-Only_standard_sharp_top5pct_ETFs
    """
    base_name = os.path.basename(results_folder)               # Results_Pipeline_…
    portfolio_name = base_name.replace("Results_Pipeline_", "portfolio_metrics_", 1)
    return os.path.join(os.path.dirname(results_folder), portfolio_name)


def run_all(args):
    # 1. Load prices once
    close_df = load_close_prices(args.close_csv, args.universe)

    # 2. Find all experiment result folders
    results_folders = find_results_folders(args.results_dir)
    if not results_folders:
        print(f"\n[WARN] No Results_Pipeline_* folders found in: {args.results_dir}")
        return

    print(f"\n[INFO] Discovered {len(results_folders)} Results_Pipeline_* folder(s):")
    for f in results_folders:
        print(f"       {os.path.basename(f)}")

    # 3. Loop over each experiment
    for results_folder in results_folders:
        exp_name = os.path.basename(results_folder)
        portfolio_folder = portfolio_folder_for(results_folder)

        print(f"\n{'='*70}")
        print(f"  EXPERIMENT : {exp_name}")
        print(f"  OUTPUT     : {os.path.basename(portfolio_folder)}")
        print(f"{'='*70}")

        # 3a. Load predictions
        predictions_df = load_predictions(results_folder)
        if predictions_df is None:
            print("  [SKIP] No predictions – skipping this folder.")
            continue

        # 3b. Run each strategy
        for strategy in args.strategies:
            print(f"\n  --- Strategy: {strategy.upper()} ---")
            strategy_output = os.path.join(portfolio_folder, strategy)
            bt = Backtester(output_folder=strategy_output, init_cash=args.init_cash, universe=args.universe)
            try:
                bt.run_backtest(
                    data=close_df,
                    predictions=predictions_df,
                    strategy=strategy,
                )
                print(f"  [OK] Results saved to: {strategy_output}")
            except Exception as exc:
                print(f"  [ERROR] Strategy '{strategy}' failed for '{exp_name}': {exc}")

    print(f"\n{'='*70}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
