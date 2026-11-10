"""Saving, loading, and exporting neural run artifacts."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    from ..evaluation.metrics import calculate_ic_and_pvalue
except (ImportError, ValueError):  # Supports running Pipeline/run_pipeline.py as a script.
    from evaluation.metrics import calculate_ic_and_pvalue


class NeuralArtifactsMixin:
    @staticmethod
    def _neural_artifact_dir(period_dir, model_name):
        # The period folder already identifies the year; keeping this nested
        # folder short avoids Windows MAX_PATH failures for long experiment names.
        return Path(period_dir) / f"best_{model_name}"

    @staticmethod
    def _legacy_neural_artifact_dir(period_dir, model_name, period_str):
        return Path(period_dir) / f"best_{model_name}_{period_str}"

    def _save_best_neural_params(
        self,
        best_params,
        period_dir,
        period_str,
        use_autoencoder=False,
        use_factor_vae=False,
        neural_selection_metric='validation_rank_ic',
    ):
        model_name = 'factorvae' if use_factor_vae else 'gkx_ae' if use_autoencoder else 'neural'
        period_dir = Path(period_dir)
        if not period_dir.is_absolute():
            period_dir = (Path.cwd() / period_dir).resolve()
        artifact_dir = self._neural_artifact_dir(period_dir, model_name)
        os.makedirs(self._io_path(artifact_dir), exist_ok=True)

        payload = {
            'period': period_str,
            'model': model_name,
            'neural_selection_metric': neural_selection_metric,
            'best_params': best_params,
        }
        with open(self._io_path(artifact_dir / "best_params.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

        rows = [{'Parameter': key, 'Value': value} for key, value in best_params.items()]
        self._safe_to_csv(
            pd.DataFrame(rows),
            artifact_dir / "best_params.csv",
            index=False,
        )

    def _load_reused_neural_params(
        self,
        reuse_root,
        period_str,
        use_autoencoder=False,
        use_factor_vae=False,
    ):
        """Load best neural hyperparameters saved by an earlier expected-return run."""
        model_name = 'factorvae' if use_factor_vae else 'gkx_ae' if use_autoencoder else 'neural'
        reuse_root = Path(reuse_root)
        if not reuse_root.is_absolute():
            reuse_root = (Path.cwd() / reuse_root).resolve()

        period_dir = reuse_root / period_str
        artifact_dirs = [
            self._legacy_neural_artifact_dir(period_dir, model_name, period_str),
            self._neural_artifact_dir(period_dir, model_name),
        ]
        candidate_paths = [
            artifact_dir / candidate
            for artifact_dir in artifact_dirs
            for candidate in [
                "best_params.json",
                f"best_{model_name}_params_{period_str}.json",
            ]
        ]
        params_path = next((path for path in candidate_paths if os.path.exists(self._io_path(path))), None)
        if params_path is None:
            tried = "\n".join(str(path) for path in candidate_paths)
            raise FileNotFoundError(
                f"Could not find reusable {model_name} params for period {period_str}. Tried:\n{tried}"
            )

        with open(self._io_path(params_path), "r") as f:
            payload = json.load(f)

        best_params = payload.get("best_params", payload)
        if not isinstance(best_params, dict) or not best_params:
            raise ValueError(f"Reusable params file is empty or invalid: {params_path}")

        expected_prefix = 'factorvae__' if use_factor_vae else 'ae__' if use_autoencoder else ''
        if expected_prefix:
            best_params = {
                key: value
                for key, value in best_params.items()
                if key.startswith(expected_prefix)
            }
        if not best_params:
            raise ValueError(f"No {expected_prefix} params found in reusable params file: {params_path}")

        print(f"   [REUSE] Loaded {model_name} params from {params_path}")
        return best_params

    def _save_neural_reuse_manifest(
        self,
        reused_params,
        reuse_root,
        period_dir,
        period_str,
        use_autoencoder=False,
        use_factor_vae=False,
    ):
        model_name = 'factorvae' if use_factor_vae else 'gkx_ae' if use_autoencoder else 'neural'
        period_dir = Path(period_dir)
        if not period_dir.is_absolute():
            period_dir = (Path.cwd() / period_dir).resolve()
        artifact_dir = self._neural_artifact_dir(period_dir, model_name)
        os.makedirs(self._io_path(artifact_dir), exist_ok=True)
        payload = {
            'period': period_str,
            'model': model_name,
            'reuse_neural_artifacts_from': str(reuse_root),
            'reused_best_params': reused_params,
        }
        with open(self._io_path(artifact_dir / "reuse_manifest.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _save_neural_output_artifacts(
        self,
        train_data,
        test_data,
        X_train,
        X_test,
        output_cols,
        period_dir,
        period_str,
        use_autoencoder=False,
        use_factor_vae=False,
        close_prices=None,
        universe='ETFs',
        selection_pct=None,
        neural_selection_metric='validation_rank_ic',
    ):
        model_name = 'factorvae' if use_factor_vae else 'gkx_ae' if use_autoencoder else 'neural'
        artifact_dir = self._neural_artifact_dir(period_dir, model_name)
        os.makedirs(self._io_path(artifact_dir), exist_ok=True)

        expected_col = self._neural_expected_return_col(
            use_autoencoder=use_autoencoder,
            use_factor_vae=use_factor_vae,
        )

        for split_name, source_data, neural_frame in [
            ('train', train_data, X_train),
            ('test', test_data, X_test),
        ]:
            output_df = pd.DataFrame({
                'Date': source_data.index,
                'Ticker': source_data.get('ticker', 'Unknown'),
                'Prediction': neural_frame[expected_col].to_numpy(),
                'Actual_Return': source_data['Actual_Return'].to_numpy(),
            }).reset_index(drop=True)

            for col in output_cols:
                output_df[col] = neural_frame[col].to_numpy()

            output_df['Predicted_Rank'] = output_df.groupby('Date')['Prediction'].rank(
                ascending=False,
                method='min',
            ).astype(int)
            output_df['Actual_Rank'] = output_df.groupby('Date')['Actual_Return'].rank(
                ascending=False,
                method='min',
            ).astype(int)
            output_df = output_df.sort_values(['Date', 'Predicted_Rank'])
            output_df['Date'] = pd.to_datetime(output_df['Date']).dt.strftime('%d-%m-%Y')

            self._safe_to_csv(
                output_df,
                artifact_dir / f"{split_name}_predictions.csv",
                index=False,
            )
            if split_name == 'train':
                train_predictions_df = output_df.copy()
            else:
                test_predictions_df = output_df.copy()

        train_ic_df = calculate_ic_and_pvalue(
            X_train[output_cols],
            train_data['Actual_Return'],
        )
        train_ic_df['Period'] = period_str
        train_ic_df['Model'] = model_name
        self._safe_to_csv(
            train_ic_df,
            artifact_dir / "training_ic.csv",
            index=False,
        )

        test_predictions_for_metrics = test_predictions_df.copy()
        test_predictions_for_metrics['Date'] = pd.to_datetime(
            test_predictions_for_metrics['Date'],
            dayfirst=True,
        )
        daily_ic_rows = []
        for date, group in test_predictions_for_metrics.groupby('Date'):
            valid_group = group[['Prediction', 'Actual_Return']].dropna()
            daily_ic = np.nan
            if len(valid_group) > 1 and valid_group['Prediction'].nunique() > 1 and valid_group['Actual_Return'].nunique() > 1:
                daily_ic = stats.spearmanr(valid_group['Prediction'], valid_group['Actual_Return'])[0]
            daily_ic_rows.append({
                'Date': date,
                'Daily_IC': daily_ic,
                'N_Assets': len(valid_group),
                'Model': model_name,
                'Task': 'neural_expected_return',
                'Tuning_Metric': neural_selection_metric,
                'Period': period_str,
            })
        neural_daily_ic_df = pd.DataFrame(daily_ic_rows)
        self._safe_to_csv(
            neural_daily_ic_df,
            artifact_dir / "daily_oos_ic.csv",
            index=False,
        )

        neural_fold_ic = float(neural_daily_ic_df['Daily_IC'].dropna().mean())
        neural_fold_sharpe = 0.0
        if close_prices is not None:
            try:
                try:
                    from ..Backtester.backtester import Backtester
                except (ImportError, ValueError):
                    from Backtester.backtester import Backtester
                backtest_dir = artifact_dir / f"{model_name}_expected_return_backtest_{period_str}"
                bt = Backtester(output_folder=backtest_dir, universe=universe)
                pf = bt.run_backtest(
                    data=close_prices,
                    predictions=test_predictions_for_metrics.copy(),
                    strategy='long_short',
                    export=True,
                    selection_pct=selection_pct,
                )
                neural_fold_sharpe = pf.sharpe_ratio() if pf is not None else np.nan
                if pd.isna(neural_fold_sharpe):
                    neural_fold_sharpe = 0.0
            except Exception as exc:
                print(f"   Warning: could not export {model_name} expected-return backtest for {period_str}: {exc}")

        neural_metrics_df = pd.DataFrame([
            {'TUNING_METRIC': neural_selection_metric.upper(), 'EVALUATION_METRIC': 'OOS_IC', 'VALUE': neural_fold_ic},
            {'TUNING_METRIC': neural_selection_metric.upper(), 'EVALUATION_METRIC': 'OOS_SHARPE', 'VALUE': neural_fold_sharpe},
        ])
        self._safe_to_csv(
            neural_metrics_df,
            artifact_dir / "prediction_metrics.csv",
            index=False,
        )

        model = None
        if use_factor_vae:
            model = getattr(self.neural_extractor, 'factor_vae_model', None)
        elif use_autoencoder:
            model = getattr(self.neural_extractor, 'autoencoder', None)

        if model is not None:
            try:
                import torch
                state_path = artifact_dir / "state_dict.pt"
                os.makedirs(self._io_path(state_path.parent), exist_ok=True)
                torch.save(
                    model.state_dict(),
                    self._io_path(state_path),
                )
            except Exception as exc:
                print(f"   Warning: could not save {model_name} state dict for {period_str}: {exc}")

        history = None
        if use_factor_vae:
            history = getattr(self.neural_extractor, 'factor_vae_history', None)
        elif use_autoencoder:
            history = getattr(self.neural_extractor, 'ae_history', None)
        if history:
            self._safe_to_csv(
                pd.DataFrame(history),
                artifact_dir / "training_history.csv",
                index=False,
            )
