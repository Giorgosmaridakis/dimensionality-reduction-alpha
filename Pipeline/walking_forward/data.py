"""Data preparation and walk-forward date-window helpers."""

import pandas as pd


class DataMixin:
    def _prepare_dataframe(self, df: pd.DataFrame, as_percentages: bool):
        """
        Ensures data integrity: handles indexing, sorting, and target formatting.
        """
        # 1. Indexing: Ensure the dataframe is indexed by Date and chronologically sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                 df = df.set_index('Date').sort_index()
            else:
                 raise ValueError("Dataframe must have a DatetimeIndex or a 'Date' column.")
        
        df = df.copy()

        # Preserve the realized return before any ranking/binning or display scaling.
        # Neural pricing models must price returns, not rank labels.
        df['Target_prc'] = df[self.target_col]
        
        # 2. Return Scaling: Create a standardized 'Actual_Return' column for evaluation
        if as_percentages:
            df['Actual_Return'] = df[self.target_col] * 100
        else:
            df['Actual_Return'] = df[self.target_col]
            
        # 3. Ranking Transformation: If task is 'ranking', convert continuous returns into discrete bins (0-9)
        if self.model_wrapper.task_type == 'ranking':
            # Group by Date (level 0), calculate percentile rank (0.0 to 1.0), 
            # scale to 0-9.999, and truncate to integers (0, 1, ..., 9).
            df[self.target_col] = (df.groupby(level=0)[self.target_col].rank(pct=True) * 9.999).astype(int)
            
        return df

    def _get_test_offset(self):
        """
        Translates a human-readable testing window string into a pandas DateOffset.
        This determines the "stride" or how far we walk forward in each iteration.
        """
        test_offset_map = {
            'Month': pd.DateOffset(months=1),       # Walk forward month-by-month
            'Quarter': pd.DateOffset(months=3),     # Walk forward quarter-by-quarter
            '6Month': pd.DateOffset(months=6),      # Walk forward every six months
            'Year': pd.DateOffset(years=1)          # Walk forward year-by-year (Default)
        }
        # Ensure the provided testing window is supported
        if self.testing_window not in test_offset_map:
            raise KeyError(f"Invalid testing_window: '{self.testing_window}'. "
                           f"Supported values are: {list(test_offset_map.keys())}")
                           
        return test_offset_map[self.testing_window]
