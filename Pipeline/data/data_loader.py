"""
Data loading utilities used at the very start of run_pipeline.py.

The loader intentionally keeps file discovery separate from feature stacking:
DataLoader finds alpha/price files, while DatasetBuilder decides how to align
those files into the model panel.
"""

import os
from pathlib import Path
import pandas as pd
# --- CONFIGURATION VARIABLES ---
# Define the supported universes and their corresponding time-period subdirectories.
# This makes it easy to change folder names globally in the future without modifying code logic.
UNIVERSE_ETFS = 'ETFs'
UNIVERSE_SP500 = 'S&P500'

UNIVERSE_TIME_PERIODS = {
    UNIVERSE_ETFS: ['2006-2022', '2023-2025'],
    UNIVERSE_SP500: ['2020-2024', '2025-End']
}

class DataLoader:
    def __init__(self, universe=UNIVERSE_ETFS, time_period=None, base_path=None):
        """
        Initializes the DataLoader for the given universe and selected time period(s).
        
        Args:
            universe (str): The target universe (e.g., 'ETFs' or 'Yahoo_Dailies').
            time_period (str): Optional specific time period folder. If omitted, all
                configured periods for the universe are loaded.
            base_path (str): The root path of the project.
        """
        if universe not in UNIVERSE_TIME_PERIODS:
            raise ValueError(f"Universe must be one of {list(UNIVERSE_TIME_PERIODS.keys())}")
            
        if time_period is not None and time_period not in UNIVERSE_TIME_PERIODS[universe]:
            raise ValueError(f"Time period '{time_period}' is not valid for universe '{universe}'. Available periods are: {UNIVERSE_TIME_PERIODS[universe]}")
            
        if base_path is None:
            base_path = Path(__file__).resolve().parent.parent.parent
            
        self.universe = universe
        self.time_period = time_period or 'all'
        self.time_periods = [time_period] if time_period is not None else UNIVERSE_TIME_PERIODS[universe]
        self.base_path = Path(base_path)
        self.universe_path = self.base_path / self.universe

    def _period_paths(self):
        return [self.universe_path / period for period in self.time_periods]

    def _load_wide_price_file(self, filename):
        """Load one wide price file name across all configured periods."""
        frames = []
        for period_path in self._period_paths():
            file_path = period_path / filename
            if file_path.exists():
                frames.append(pd.read_csv(file_path, parse_dates=['Date'], index_col='Date'))

        if not frames:
            raise FileNotFoundError(
                f"CRITICAL ERROR: No {filename} found for {self.universe} "
                f"in periods {self.time_periods}."
            )

        df = pd.concat(frames).sort_index()
        # Later period folders can overlap at boundaries; keep the most recent row.
        df = df[~df.index.duplicated(keep='last')]
        return df
        
    def get_alpha_files(self):
        """
        Scans the selected time-period subdirectories within the universe directory 
        to find all 'zuras' folders and gathers 'alpha*.csv' files.
        """
        if not self.universe_path.exists():
            raise FileNotFoundError(f"Universe path {self.universe_path} does not exist.")
            
        alpha_files = {}

        for period_path in self._period_paths():
            if not period_path.exists():
                print(f"Warning: Time period path {period_path} does not exist.")
                continue

            zuras_dirs = sorted(period_path.rglob('zuras'))

            for zuras_dir in zuras_dirs:
                for file_path in sorted(zuras_dir.glob('alpha*.csv')):
                    feature_name = file_path.stem  # e.g., 'alpha001'
                    alpha_files.setdefault(feature_name, []).append(str(file_path))
                
        return alpha_files

    def get_returns(self):
        """
        Gathers the Returns.csv files from the selected time-period subdirectories.
        """
        return_files = []

        for period_path in self._period_paths():
            if period_path.exists():
                return_matches = sorted(period_path.rglob('Returns.csv'))
                return_files.extend([str(p) for p in return_matches])
            
        return sorted(return_files)

    def get_adjusted_close(self):
        """
        Loads and combines Adjusted_Close.csv files from the selected periods.
        """
        return self._load_wide_price_file("Adjusted_Close.csv")
        
   
