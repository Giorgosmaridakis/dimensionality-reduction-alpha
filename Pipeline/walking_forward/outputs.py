"""Prediction, metric, and aggregate output helpers."""

import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path


class OutputMixin:
    def _save_results(
        self,
        test_data,
        preds,
        y_test,
        period_dir,
        period_str,
        close_prices=None,
        universe=None,
        tuning_metric='ic',
        selection_pct=None,
    ):
        if self.model_wrapper.task_type == 'ranking':
            output_df = pd.DataFrame({
                'Date': test_data.index,
                'Ticker': test_data.get('ticker', 'Unknown'),
                'Predicted_Score': preds,
                'Actual_Score': y_test,
                'Actual_Return': test_data['Actual_Return']
            }).reset_index(drop=True)

            output_df['Predicted_Rank'] = output_df.groupby('Date')['Predicted_Score'].rank(ascending=False, method='min').astype(int)
            output_df['Actual_Rank'] = output_df.groupby('Date')['Actual_Score'].rank(ascending=False, method='min').astype(int)
        else:
            output_df = pd.DataFrame({
                'Date': test_data.index,
                'Ticker': test_data.get('ticker', 'Unknown'),
                'Prediction': preds,
                'Actual': y_test
            }).reset_index(drop=True)
        
        # Sort and Save
        sort_col = ['Date', 'Predicted_Rank'] if 'Predicted_Rank' in output_df.columns else ['Date', 'Prediction']
        output_df = output_df.sort_values(sort_col, ascending=[True, True if 'Rank' in sort_col[1] else False])
        output_df['Date'] = output_df['Date'].dt.strftime('%d-%m-%Y')
        self._safe_to_csv(output_df, period_dir / f"predictions_{period_str}.csv", index=False)
        
        # IC Metrics
        print("--> Evaluating Final Prediction IC...")
        pred_col = 'Predicted_Score' if 'Predicted_Score' in output_df.columns else 'Prediction'
        if self.model_wrapper.task_type == 'ranking':
            act_col = 'Actual_Return' if 'Actual_Return' in output_df.columns else 'Actual_Score'
        else:
            act_col = 'Actual'
        
        output_df['Date'] = pd.to_datetime(output_df['Date'], dayfirst=True)
        daily_ic_rows = []
        for date, group in output_df.groupby('Date'):
            valid_group = group[[pred_col, act_col]].dropna()
            n_assets = len(valid_group)
            daily_ic = np.nan
            if n_assets > 1 and valid_group[pred_col].nunique() > 1 and valid_group[act_col].nunique() > 1:
                if self.model_wrapper.task_type == 'ranking':
                    daily_ic = stats.spearmanr(valid_group[pred_col], valid_group[act_col])[0]
                else:
                    daily_ic = stats.pearsonr(valid_group[pred_col], valid_group[act_col])[0]

            daily_ic_rows.append({
                'Date': date,
                'Daily_IC': daily_ic,
                'N_Assets': n_assets,
                'Model': self.model_wrapper.model_type,
                'Task': self.model_wrapper.task_type,
                'Tuning_Metric': tuning_metric,
                'Period': period_str,
            })

        daily_ic_df = pd.DataFrame(daily_ic_rows)
        self._safe_to_csv(daily_ic_df, period_dir / f"daily_oos_ic_{period_str}.csv", index=False)
        daily_ics = daily_ic_df['Daily_IC'].dropna()
        fold_ic_mean = float(daily_ics.mean())
        fold_sharpe = 0.0
        if close_prices is not None:
            try:
                try:
                    from ..Backtester.backtester import Backtester
                except (ImportError, ValueError):
                    from Backtester.backtester import Backtester
                bt = Backtester(output_folder=period_dir, universe=universe)
                fold_sharpe = bt.run_backtest(
                    data=close_prices,
                    predictions=output_df,
                    strategy='long_short',
                    export=False,
                    selection_pct=selection_pct,
                )
                if pd.isna(fold_sharpe): fold_sharpe = 0.0
            except Exception as e:
                print(f"   Warning: Could not compute OOS Sharpe for {period_str}: {e}")
                
        # Export evaluation metrics in a clear long format.
        metrics_df = pd.DataFrame([
            {'TUNING_METRIC': tuning_metric.upper(), 'EVALUATION_METRIC': 'OOS_IC', 'VALUE': fold_ic_mean},
            {'TUNING_METRIC': tuning_metric.upper(), 'EVALUATION_METRIC': 'OOS_SHARPE', 'VALUE': fold_sharpe},
        ])
        self._safe_to_csv(metrics_df, period_dir / f"prediction_metrics_{period_str}.csv", index=False)
                
        # Return the fold IC and Sharpe so the main loop can collect it
        return fold_ic_mean, fold_sharpe

    def _aggregate_daily_oos_ic(self, output_path: Path):
        daily_ic_files = sorted(output_path.glob("*/daily_oos_ic_*.csv"))
        if not daily_ic_files:
            return

        daily_ic_df = pd.concat((pd.read_csv(path) for path in daily_ic_files), ignore_index=True)
        daily_ic_df['Date'] = pd.to_datetime(daily_ic_df['Date']).dt.normalize()
        daily_ic_df = daily_ic_df.sort_values(['Date', 'Period']).reset_index(drop=True)
        out_path = output_path / "daily_oos_ic_series.csv"
        self._safe_to_csv(daily_ic_df, out_path, index=False)
        print(f"   [DONE] Daily OOS IC series saved to {out_path}")
