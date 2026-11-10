"""
Builds the modeling panel consumed by run_pipeline.py.

The key contract is: one row per Date/ticker, alpha columns as features, and a
Target column that matches the backtest execution convention.
"""

import pandas as pd
import numpy as np

class DatasetBuilder:
    def __init__(self, date_col='Date', ticker_name='ticker', index_date_name='Date', dropna=True):
        """
        Initializes the DatasetBuilder.
        """
        self.date_col = date_col
        self.ticker_name = ticker_name
        self.index_date_name = index_date_name
        self.dropna = dropna

    def prepare_stacked_data(self, data_loader):
        """
        Reads and stacks alpha feature files into a single panel dataframe.
        """
        feature_files = data_loader.get_alpha_files()
        stacked_features = []
        all_start_dates = []

        for feature_name, file_paths in feature_files.items():
            if isinstance(file_paths, str):
                file_paths = [file_paths]
                
            dfs = []
            for fp in file_paths:
                temp_df = pd.read_csv(fp)
                dfs.append(temp_df)
            
            df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

            found_date_col = next((c for c in df.columns if c.lower() == self.date_col.lower()), None)

            if found_date_col:
                df[found_date_col] = pd.to_datetime(df[found_date_col])
                df = df.set_index(found_date_col)
            else:
                if isinstance(df.index, pd.RangeIndex):
                    raise ValueError(f"Could not find date column '{self.date_col}' in {feature_name}.")
                df.index = pd.to_datetime(df.index)

            df.index.name = self.index_date_name
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            all_start_dates.append(df.index.min())
            
            # Alpha files arrive wide (Date x ticker). The model pipeline works
            # long-form, so each feature is melted to Date/ticker/value first.
            df_long = df.reset_index().melt(
                id_vars=[self.index_date_name], 
                var_name=self.ticker_name, 
                value_name=feature_name
            )
            df_long[self.index_date_name] = pd.to_datetime(df_long[self.index_date_name])
            df_long = df_long.drop_duplicates(
                subset=[self.index_date_name, self.ticker_name],
                keep='last'
            )
            stacked_features.append(df_long)

        if not stacked_features:
            return pd.DataFrame()

        # Start where every alpha has history. This avoids silently forward
        # filling missing early feature values.
        global_start_date = max(all_start_dates)
        panel_df = stacked_features[0]
        for df_next in stacked_features[1:]:
            panel_df = panel_df.merge(
                df_next, 
                on=[self.index_date_name, self.ticker_name], 
                how='inner'
            )

        panel_df = panel_df[panel_df[self.index_date_name] >= global_start_date].copy()
        
        if self.dropna:
            panel_df = panel_df.dropna().reset_index(drop=True)

        panel_df = panel_df.sort_values(
            [self.index_date_name, self.ticker_name]
        ).drop_duplicates(
            subset=[self.index_date_name, self.ticker_name],
            keep='last'
        ).reset_index(drop=True)
            
        return panel_df

    def merge_features_and_targets(self, panel_df, data_loader, target_col='Target'):
        """
        Builds tradable close-to-close targets and merges them with features.

        Alpha rows dated t use the full OHLCV bar through close_t, so they are
        available after close_t. The backtester shifts weights one row, so the
        first execution price is adjusted close_{t+1}. Therefore the supervised
        target for signal date t is the holding return after that execution:

            Target_t = Adjusted_Close_{t+2} / Adjusted_Close_{t+1} - 1
        """
        adjusted_close = data_loader.get_adjusted_close()

        adjusted_close = adjusted_close.sort_index()

        # Backtester execution is next adjusted close because size_df is shifted
        # by one row. Shift close-to-close returns by -2 to align the signal date
        # t with the return earned after entry at close_{t+1}.
        close_to_close_returns = adjusted_close.pct_change()
        tradable_next_close_returns = close_to_close_returns.shift(-2)
        tradable_next_close_returns.index.name = self.index_date_name

        returns_long = tradable_next_close_returns.reset_index().melt(
            id_vars=[self.index_date_name],
            var_name=self.ticker_name,
            value_name=target_col
        )
        returns_long[self.index_date_name] = pd.to_datetime(returns_long[self.index_date_name])
        returns_long = returns_long.drop_duplicates(
            subset=[self.index_date_name, self.ticker_name],
            keep='last'
        )
        returns_long = returns_long.dropna(subset=[target_col])
        
        final_df = panel_df.merge(
            returns_long, 
            on=[self.index_date_name, self.ticker_name], 
            how='inner'
        )
        return final_df
