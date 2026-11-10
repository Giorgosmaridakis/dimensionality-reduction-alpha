import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


RESULTS_PREFIX = "Results_Pipeline_"
PORTFOLIO_PREFIX = "portfolio_metrics_"


def parse_configuration(folder_name, prefix):
    raw = folder_name.replace(prefix, "", 1)
    parts = raw.split("_")

    model = parts[0] if len(parts) > 0 else ""
    task = parts[1] if len(parts) > 1 else ""
    scaling = parts[-1] if len(parts) > 3 else ""
    method = "_".join(parts[2:-1]) if len(parts) > 3 else "_".join(parts[2:])

    return {
        "Model": model,
        "Task": task,
        "Method": method,
        "Scaling": scaling,
        "Configuration": raw,
    }


def load_config_daily_ic(results_dir):
    labels = parse_configuration(results_dir.name, RESULTS_PREFIX)

    config_series = results_dir / "daily_oos_ic_series.csv"
    if config_series.exists():
        df = pd.read_csv(config_series)
    else:
        period_files = sorted(results_dir.glob("*/daily_oos_ic_*.csv"))
        if not period_files:
            return None
        df = pd.concat((pd.read_csv(path) for path in period_files), ignore_index=True)

    for key, value in labels.items():
        df[key] = value

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


def load_portfolio_returns(portfolio_dir):
    labels = parse_configuration(portfolio_dir.name, PORTFOLIO_PREFIX)

    standardized_path = portfolio_dir / "daily_portfolio_return_series.csv"
    if standardized_path.exists():
        df = pd.read_csv(standardized_path)
    else:
        fallback_path = portfolio_dir / "long_short_time_series.csv"
        if not fallback_path.exists():
            return None
        time_series_df = pd.read_csv(fallback_path)
        df = pd.DataFrame({
            "Date": pd.to_datetime(time_series_df["Date"]).dt.normalize(),
            "Daily_Return": time_series_df["Total PnL(%)"] / 100.0,
            "Daily_Return_Pct": time_series_df["Total PnL(%)"],
            "Cumulative_Return_Pct": time_series_df["Cumulative PnL(%)"],
            "Drawdown_Pct": time_series_df["Drawdown(%)"],
            "Strategy": "long_short",
        })

    for key, value in labels.items():
        df[key] = value

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


def build_statistical_inputs(root_dir, output_dir):
    if pd is None:
        raise SystemExit("pandas is required to build statistical input files. Install pandas or use the pipeline environment.")

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ic_frames = []
    for results_dir in sorted(root_dir.glob(f"{RESULTS_PREFIX}*")):
        if results_dir.is_dir():
            df = load_config_daily_ic(results_dir)
            if df is not None and not df.empty:
                ic_frames.append(df)

    if ic_frames:
        daily_ic = pd.concat(ic_frames, ignore_index=True)
        daily_ic = daily_ic.sort_values(["Configuration", "Date"]).reset_index(drop=True)
        daily_ic.to_csv(output_dir / "daily_oos_ic_series.csv", index=False)
        print(f"[DONE] Saved daily OOS IC master file: {output_dir / 'daily_oos_ic_series.csv'}")
    else:
        print("[WARN] No daily OOS IC files found.")

    return_frames = []
    for portfolio_dir in sorted(root_dir.glob(f"{PORTFOLIO_PREFIX}*")):
        if portfolio_dir.is_dir():
            df = load_portfolio_returns(portfolio_dir)
            if df is not None and not df.empty:
                return_frames.append(df)

    if return_frames:
        daily_returns = pd.concat(return_frames, ignore_index=True)
        daily_returns = daily_returns.sort_values(["Configuration", "Date"]).reset_index(drop=True)
        daily_returns.to_csv(output_dir / "daily_portfolio_return_series.csv", index=False)
        print(f"[DONE] Saved daily portfolio return master file: {output_dir / 'daily_portfolio_return_series.csv'}")
    else:
        print("[WARN] No daily portfolio return files found.")


def main():
    parser = argparse.ArgumentParser(
        description="Build master daily IC and daily portfolio return datasets for statistical tests."
    )
    parser.add_argument("--root-dir", default=".", help="Pipeline root containing Results_Pipeline_* folders.")
    parser.add_argument("--output-dir", default="Statistical_Results", help="Output folder for master CSV files.")
    args = parser.parse_args()

    build_statistical_inputs(args.root_dir, args.output_dir)


if __name__ == "__main__":
    main()
