"""
Portfolio simulation for saved pipeline predictions.

The convention used throughout the current project is close-only execution:
signals dated t become target weights on the next available adjusted close via
the one-row size_df shift below.
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
import os


class Backtester:
    def __init__(self, output_folder='s&p500_performance', init_cash=10_000_000, universe=None, fees=None, slippage=None):
        self.output_folder = output_folder
        self.init_cash = init_cash

        # Auto-detect universe if not provided
        if universe is None:
            if 'etf' in output_folder.lower():
                universe = 'ETFs'
            else:
                universe = 'S&P500'

        self.universe = universe
        
        # Apply slippage and fees based on universe
        if universe == 'ETFs':
            self.slippage = slippage if slippage is not None else 0.0002
            self.fees = fees if fees is not None else 0.0001
        else:
            # Default to S&P500 parameters
            self.slippage = slippage if slippage is not None else 0.0005
            self.fees = fees if fees is not None else 0.0001

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    @staticmethod
    def _to_wide_price_frame(data: pd.DataFrame, value_col: str = 'Close') -> pd.DataFrame:
        """Accept either long Date/Ticker/Close data or already-wide price data."""
        if 'Ticker' in data.columns and value_col in data.columns:
            data_copy = data.copy()
            if 'Date' in data_copy.columns and not pd.api.types.is_datetime64_any_dtype(data_copy['Date']):
                data_copy['Date'] = pd.to_datetime(data_copy['Date'])
            if 'Date' not in data_copy.columns:
                data_copy = data_copy.reset_index()
                data_copy['Date'] = pd.to_datetime(data_copy['Date'])
            price_frame = data_copy.pivot(index='Date', columns='Ticker', values=value_col)
        else:
            price_frame = data.copy()
            if not pd.api.types.is_datetime64_any_dtype(price_frame.index):
                price_frame.index = pd.to_datetime(price_frame.index)

        return price_frame.sort_index()

    # ------------------------------------------------------------------
    # Helper: build the daily time-series DataFrame that matches the
    # Excel template (Simulator_Time_series.xlsx)
    # ------------------------------------------------------------------
    def _build_time_series(self, pf, close_price, size_df, strategy):
        """
        Returns a DataFrame with one row per trading day containing:
            Date, Total PnL, Total PnL(%), Cumulative PnL(%),
            Drawdown(%), No Longs, No Shorts, Total Positions,
            Longs-Shorts Imbalance, Long $ Value, Short $ Value,
            Gross Exposure, Net Exposure, Turnover
        """
        # ---- Portfolio value / cash series ----
        portfolio_value = pf.value()          # total portfolio value per day
        cash             = pf.cash()          # uninvested cash per day

        # ---- Daily P&L ----
        daily_pnl        = portfolio_value.diff()
        daily_pnl.iloc[0] = portfolio_value.iloc[0] - self.init_cash

        daily_pnl_pct    = daily_pnl / portfolio_value.shift(1).replace(0, np.nan) * 100
        daily_pnl_pct.iloc[0] = 0.0

        # ---- Cumulative P&L % ----
        cum_pnl_pct = (portfolio_value / self.init_cash - 1) * 100

        # ---- Drawdown % ----
        rolling_max  = portfolio_value.cummax()
        drawdown_pct = (portfolio_value - rolling_max) / rolling_max * 100

        # ---- Position values (long / short) per asset per day ----
        # asset_value > 0  => long position, < 0 => short position
        asset_value = pf.asset_value(group_by=False)   # shape: (days, assets)

        long_value_per_asset  = asset_value.clip(lower=0)
        short_value_per_asset = asset_value.clip(upper=0).abs()

        long_dollar  = long_value_per_asset.sum(axis=1)
        short_dollar = short_value_per_asset.sum(axis=1)

        # ---- Position counts ----
        no_longs  = (asset_value > 0).sum(axis=1)
        no_shorts = (asset_value < 0).sum(axis=1)
        total_pos = no_longs + no_shorts
        ls_imbal  = no_longs - no_shorts

        # ---- Exposure ----
        gross_exposure = (long_dollar + short_dollar) / portfolio_value * 100
        net_exposure   = (long_dollar - short_dollar) / portfolio_value * 100

        # ---- Turnover ----
        # Daily turnover = sum of absolute change in position values / portfolio value
        pos_change = asset_value.diff().abs().sum(axis=1)
        pos_change.iloc[0] = asset_value.iloc[0].abs().sum()   # first day: full deployment
        turnover = pos_change / portfolio_value * 100

        ts = pd.DataFrame({
            'Date':                    portfolio_value.index,
            'Total PnL':               daily_pnl.values,
            'Total PnL(%)':            daily_pnl_pct.values,
            'Cumulative PnL(%)':       cum_pnl_pct.values,
            'Drawdown(%)':             drawdown_pct.values,
            'No Longs':                no_longs.values,
            'No Shorts':               no_shorts.values,
            'Total Positions':         total_pos.values,
            'Longs-Shorts Imbalance':  ls_imbal.values,
            'Long $ Value':            long_dollar.values,
            'Short $ Value':           short_dollar.values,
            'Gross Exposure':          gross_exposure.values,
            'Net Exposure':            net_exposure.values,
            'Turnover':                turnover.values,
        })

        ts['Date'] = pd.to_datetime(ts['Date']).dt.normalize()
        ts = ts.reset_index(drop=True)
        return ts

    # ------------------------------------------------------------------
    # Main backtest runner
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame = None,
        strategy: str = 'long',
        export: bool = True,
        selection_pct: float = None,
    ):
        """
        Runs a backtest.
        If predictions is provided, it ranks and trades the selected top/bottom
        percentage of assets. By default this is 10% for S&P500 and 20% for ETFs.
        according to the strategy ('long', 'short', 'long_short').
        Otherwise it falls back to equal-weighted buy-and-hold on all assets.
        If export=False, it skips saving files and returns the Sharpe Ratio.
        Orders execute on the adjusted close series after the one-bar weight shift.
        """
        # ---- Handle Data Formatting ----
        close_price = self._to_wide_price_frame(data, value_col='Close')

        if export and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # ---- Regression Z-Score Pre-computation ----
        rolling_mean = None
        rolling_std  = None
        is_regression = False

        if predictions is not None:
            if 'Date' in predictions.columns and not pd.api.types.is_datetime64_any_dtype(predictions['Date']):
                predictions['Date'] = pd.to_datetime(predictions['Date'], dayfirst=True)
            predictions = predictions.sort_values(['Date', 'Ticker']).drop_duplicates(
                subset=['Date', 'Ticker'],
                keep='last'
            )

            if 'Predicted_Rank' not in predictions.columns and 'Prediction' in predictions.columns:
                is_regression = True
                print("-> Regression predictions detected. Computing 10-day trailing return Z-Scores...")
                rets = close_price.pct_change()
                rolling_mean = rets.shift(1).rolling(window=10, min_periods=1).mean()
                rolling_std  = rets.shift(1).rolling(window=10, min_periods=1).std()

        # Truncate to prediction period
        if predictions is not None:
            min_date = predictions['Date'].min()
            max_date = predictions['Date'].max()
            available_dates = close_price.index[close_price.index >= min_date]
            future_dates = available_dates[available_dates > max_date]
            slice_end = future_dates[0] if len(future_dates) > 0 else max_date

            close_price = close_price.loc[min_date:slice_end].copy()
            if is_regression:
                rolling_mean = rolling_mean.loc[min_date:slice_end].copy()
                rolling_std  = rolling_std.loc[min_date:slice_end].copy()
            print(
                f"-> Sliced Backtest Period: {min_date.date()} to {slice_end.date()} "
                f"(signals through {max_date.date()})"
            )

        num_assets = len(close_price.columns)
        if num_assets == 0:
            print("No assets found to backtest.")
            return None

        # ---- Signal Generation ----
        entries       = pd.DataFrame(False, index=close_price.index, columns=close_price.columns)
        exits         = pd.DataFrame(False, index=close_price.index, columns=close_price.columns)
        short_entries = pd.DataFrame(False, index=close_price.index, columns=close_price.columns)
        short_exits   = pd.DataFrame(False, index=close_price.index, columns=close_price.columns)
        size_df       = pd.DataFrame(0.0,   index=close_price.index, columns=close_price.columns)

        if predictions is not None:
            print(f"-> Applying '{strategy.upper()}' Strategy based on predictions...")

            if is_regression:
                raw_pred_wide = predictions.pivot(index='Date', columns='Ticker', values='Prediction')
                raw_pred_wide = raw_pred_wide.reindex(index=close_price.index, columns=close_price.columns)
                pred_wide = raw_pred_wide.rank(axis=1, ascending=False, method='min')
            else:
                pred_wide = predictions.pivot(index='Date', columns='Ticker', values='Predicted_Rank')
                pred_wide = pred_wide.reindex(index=close_price.index, columns=close_price.columns)

            for date, row in pred_wide.iterrows():
                valid_ranks = row.dropna()
                n = len(valid_ranks)
                if n == 0:
                    continue

                percentage = selection_pct if selection_pct is not None else (0.20 if self.universe == 'ETFs' else 0.10)
                top_k    = max(1, int(np.ceil(n * percentage)))
                bottom_k = max(1, int(np.ceil(n * percentage)))

                long_tickers  = valid_ranks.nsmallest(top_k).index.tolist()
                short_tickers = valid_ranks.nlargest(bottom_k).index.tolist()

                exits.loc[date]       = True
                short_exits.loc[date] = True

                if strategy in ['long', 'long_short']:
                    entries.loc[date, long_tickers]  = True
                    exits.loc[date, long_tickers]    = False
                    alloc = 0.5 if strategy == 'long_short' else 1.0
                    size_df.loc[date, long_tickers]  = alloc / top_k

                if strategy in ['short', 'long_short']:
                    short_entries.loc[date, short_tickers] = True
                    short_exits.loc[date, short_tickers]   = False
                    alloc = 0.5 if strategy == 'long_short' else 1.0
                    size_df.loc[date, short_tickers] = -alloc / bottom_k

            # Alpha_t is computed after close_t. With close-only execution, the
            # first tradable bar is the next adjusted close, so shift weights.
            size_df = size_df.shift(1).fillna(0.0)

        else:
            print("-> No predictions provided. Using default Buy-and-Hold Equal-Weight Strategy...")
            size_per_asset = 1 / num_assets
            size_df[:] = size_per_asset
            for ticker in close_price.columns:
                first_valid_date = close_price[ticker].first_valid_index()
                if first_valid_date is not None:
                    entries.loc[first_valid_date, ticker] = True

        # ---- Run VectorBT Backtest ----
        pf = vbt.Portfolio.from_orders(
            close_price,
            price=close_price,
            size=size_df,
            size_type='targetpercent',
            freq='D',
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            group_by=True,
            cash_sharing=True,
            call_seq='auto',
            log=True
        )

        # ---- Standard outputs ----
        if export:
            vbt_position_path = os.path.join(self.output_folder, 'vbt_generated_positions.csv')
            size_df.to_csv(vbt_position_path) # Exporting your actual strategy weights
    
            stats_df   = pf.stats().to_frame(name='Portfolio_Value')
            stats_path = os.path.join(self.output_folder, f'{strategy}_portfolio_performance_results.csv')
            stats_df.to_csv(stats_path)
    
            plot_path = os.path.join(self.output_folder, f'{strategy}_portfolio_cumulative_returns.html')
            fig = pf.value().vbt.plot()
            fig.write_html(plot_path)
    
            print("\n--- Portfolio Summary Statistics ---")
            print(pf.stats())
    
            # ---- Daily Time-Series Output (matches Excel template) ----
            time_series_df = self._build_time_series(pf, close_price, size_df, strategy)
            ts_path = os.path.join(self.output_folder, f'{strategy}_time_series.csv')
            time_series_df.to_csv(ts_path, index=False)
            print(f"--> Saved daily time series to: {ts_path}")
        else:
            return pf.sharpe_ratio()

        # ---- IC / OOS Metrics ----
        mean_ic = np.nan
        ic_ir   = np.nan

        if predictions is not None:
            import scipy.stats as stats_lib
            ic_list = []
            for date, group in predictions.groupby('Date'):
                if len(group) > 1:
                    pred_col = 'Predicted_Score' if 'Predicted_Score' in group.columns else 'Prediction'
                    act_col = (
                        'Actual_Return' if 'Actual_Return' in group.columns
                        else 'Actual_Score' if 'Actual_Score' in group.columns
                        else 'Actual'
                    )
                    if pred_col in group.columns and act_col in group.columns:
                        ic, _ = stats_lib.spearmanr(group[pred_col], group[act_col])
                        if not np.isnan(ic):
                            ic_list.append(ic)

            if ic_list:
                mean_ic = np.mean(ic_list)
                ic_ir   = mean_ic / np.std(ic_list) if np.std(ic_list) > 0 else np.nan
                print(f"--> IC: {mean_ic:.4f} | ICIR: {ic_ir:.4f}")

        vbt_stats = dict(pf.stats())
        metrics_dict = {
            'IC':                             mean_ic,
            'ICIR':                           ic_ir,
            'Annualized Return':              vbt_stats.get('Ann. Return [%]',    vbt_stats.get('Total Return [%]', np.nan)),
            'Annualized Volatility':          vbt_stats.get('Ann. Volatility [%]', np.nan),
            'Sharpe Ratio':                   vbt_stats.get('Sharpe Ratio',       np.nan),
            'Sortino Ratio':                  vbt_stats.get('Sortino Ratio',      np.nan),
            'Max Drawdown [%]':               vbt_stats.get('Max Drawdown [%]',   np.nan),
            'Calmar Ratio':                   vbt_stats.get('Calmar Ratio',       np.nan),
            'Total Trades (Proxy Turnover)':  vbt_stats.get('Total Trades',       np.nan),
        }

        metrics_df   = pd.DataFrame([metrics_dict])
        metrics_path = os.path.join(self.output_folder, f'{strategy}_final_oos_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"--> Saved OOS metrics to: {metrics_path}")
        print(f"Plots and results saved to: {self.output_folder}")

        return pf
