"""Shared neural scoring and CSV helpers."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats


class NeuralUtilsMixin:
    @staticmethod
    def _daily_rank_ic(signal, actual, dates):
        score_df = pd.DataFrame({
            'Date': pd.to_datetime(np.asarray(dates)),
            'Signal': np.asarray(signal, dtype=float),
            'Actual': np.asarray(actual, dtype=float),
        }).dropna()

        daily_ics = []
        for _, group in score_df.groupby('Date'):
            if len(group) > 2 and group['Signal'].nunique() > 1 and group['Actual'].nunique() > 1:
                ic = stats.spearmanr(group['Signal'], group['Actual'])[0]
                if not np.isnan(ic):
                    daily_ics.append(ic)
        return float(np.mean(daily_ics)) if daily_ics else 0.0

    @staticmethod
    def _io_path(path):
        """Return a filesystem path string that can handle long paths on Windows."""
        path = Path(path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        path_str = str(path)
        if os.name != "nt":
            return path_str
        if path_str.startswith("\\\\?\\"):
            return path_str
        if path_str.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path_str[2:]
        return "\\\\?\\" + path_str

    @staticmethod
    def _safe_to_csv(df, path, **kwargs):
        """Write CSVs defensively on Windows, where long paths can misreport as missing parents."""
        path = Path(path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        os.makedirs(NeuralUtilsMixin._io_path(path.parent), exist_ok=True)
        output_path = NeuralUtilsMixin._io_path(path)
        try:
            df.to_csv(output_path, **kwargs)
        except FileNotFoundError:
            os.makedirs(NeuralUtilsMixin._io_path(path.parent), exist_ok=True)
            df.to_csv(output_path, **kwargs)

    @staticmethod
    def _build_neural_predictions_frame(dates, tickers, predictions, actual_returns=None):
        if tickers is None:
            tickers = pd.Series(['Unknown'] * len(predictions))

        output_df = pd.DataFrame({
            'Date': pd.to_datetime(np.asarray(dates)),
            'Ticker': np.asarray(tickers),
            'Prediction': np.asarray(predictions, dtype=float),
        })
        if actual_returns is not None:
            output_df['Actual_Return'] = np.asarray(actual_returns, dtype=float)

        output_df = output_df.dropna(subset=['Date', 'Ticker', 'Prediction'])
        output_df['Predicted_Rank'] = output_df.groupby('Date')['Prediction'].rank(
            ascending=False,
            method='min',
        ).astype(int)
        if 'Actual_Return' in output_df.columns:
            output_df['Actual_Rank'] = output_df.groupby('Date')['Actual_Return'].rank(
                ascending=False,
                method='min',
            ).astype(int)
        return output_df.sort_values(['Date', 'Predicted_Rank']).reset_index(drop=True)

    def _score_neural_validation_fold(
        self,
        X_val_neural,
        expected_col,
        ae_target_val_fold,
        qids_val_fold,
        tickers_val_fold,
        neural_selection_metric,
        close_prices,
        universe,
        selection_pct,
        period_dir,
        period_str,
        fold_number,
    ):
        """Score one neural validation fold by Rank IC or by validation backtest Sharpe."""
        if neural_selection_metric != 'validation_sharpe':
            return self._daily_rank_ic(
                X_val_neural[expected_col],
                ae_target_val_fold,
                qids_val_fold,
            )

        if close_prices is None or tickers_val_fold is None:
            print("      Neural Sharpe scoring needs close prices and tickers; falling back to Rank IC.")
            return self._daily_rank_ic(
                X_val_neural[expected_col],
                ae_target_val_fold,
                qids_val_fold,
            )

        predictions = self._build_neural_predictions_frame(
            dates=qids_val_fold,
            tickers=tickers_val_fold,
            predictions=X_val_neural[expected_col],
            actual_returns=ae_target_val_fold,
        )
        if predictions.empty or predictions['Ticker'].nunique() <= 1:
            return float('-inf')

        try:
            try:
                from ..Backtester.backtester import Backtester
            except (ImportError, ValueError):
                from Backtester.backtester import Backtester
            temp_dir = period_dir / "_neural_validation_sharpe" / f"{period_str}_fold_{fold_number}"
            bt = Backtester(output_folder=temp_dir, universe=universe)
            fold_sharpe = bt.run_backtest(
                data=close_prices,
                predictions=predictions,
                strategy='long_short',
                export=False,
                selection_pct=selection_pct,
            )
            return 0.0 if pd.isna(fold_sharpe) else float(fold_sharpe)
        except Exception as exc:
            print(f"      Neural validation Sharpe scoring error: {exc}")
            return float('-inf')
