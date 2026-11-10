from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from walking_forward.neural_tuning import NeuralTuningMixin
from walking_forward.neural_utils import NeuralUtilsMixin


class _DummyNeuralTuner(NeuralTuningMixin, NeuralUtilsMixin):
    embargo_days = 1

    @staticmethod
    def _iter_param_combinations(param_grid):
        keys, values = zip(*param_grid.items())
        return [dict(zip(keys, combo)) for combo in zip(*values)]

    @staticmethod
    def _neural_expected_return_col(use_autoencoder=False, use_factor_vae=False):
        return "expected_return"

    def _fit_transform_neural_for_tuning(
        self,
        X_train_fold,
        X_val_fold,
        ae_target_train_fold,
        qids_train_fold,
        qids_val_fold,
        tickers_train_fold,
        tickers_val_fold,
        grid_params,
        candidate_features,
        use_standardization,
        scaling_method,
        use_autoencoder,
        use_factor_vae,
    ):
        self._active_trial = grid_params["trial"]
        X_val_neural = X_val_fold.copy()
        X_val_neural["expected_return"] = np.arange(len(X_val_neural), dtype=float)
        return X_train_fold.copy(), X_val_neural

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
        if close_prices is not None:
            self.max_close_price_seen = max(
                getattr(self, "max_close_price_seen", pd.Timestamp.min),
                pd.to_datetime(close_prices.index).max(),
            )
        if self._active_trial == "incomplete" and fold_number == 1:
            return float("-inf")
        if self._active_trial == "incomplete":
            return 0.99
        return 0.50


def test_validation_sharpe_prices_are_capped_to_train_window():
    close_prices = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0, 99.0],
            "BBB": [20.0, 21.0, 22.0, 88.0],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    )
    train_qids = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))

    capped = NeuralTuningMixin._cap_close_prices_to_training_window(close_prices, train_qids)

    assert capped.index.max() == pd.Timestamp("2020-01-03")
    assert close_prices.index.max() == pd.Timestamp("2020-01-04")


def test_incomplete_neural_cv_candidate_cannot_win(tmp_path):
    dates = pd.date_range("2020-01-01", periods=8, freq="D").repeat(2)
    X_train = pd.DataFrame({"feature_1": np.arange(len(dates), dtype=float)}, index=dates)
    ae_train_target = pd.Series(np.linspace(-0.1, 0.1, len(dates)), index=dates)
    train_qids = pd.Series(dates, index=dates)
    train_tickers = pd.Series(["AAA", "BBB"] * 8, index=dates)

    tuner = _DummyNeuralTuner()
    close_prices = pd.DataFrame(
        {"AAA": np.arange(9, dtype=float), "BBB": np.arange(9, dtype=float)},
        index=pd.date_range("2020-01-01", periods=9, freq="D"),
    )
    best_params, results_df = tuner._tune_neural_outputs(
        X_train=X_train,
        ae_train_target=ae_train_target,
        train_qids=train_qids,
        train_tickers=train_tickers,
        param_grid={"trial": ["incomplete", "complete"]},
        candidate_features=["feature_1"],
        use_standardization=False,
        scaling_method="standard",
        use_autoencoder=False,
        use_factor_vae=True,
        use_optuna=False,
        optuna_trials=0,
        neural_selection_metric="validation_rank_ic",
        close_prices=close_prices,
        universe="ETFs",
        selection_pct=0.5,
        period_dir=tmp_path,
        period_str="01-01-2020_31-12-2020",
    )

    assert best_params == {"trial": "complete"}
    incomplete_row = results_df.loc[results_df["trial"] == "incomplete"].iloc[0]
    assert incomplete_row["Validation_Score"] == float("-inf")
    assert incomplete_row["Successful_Folds"] < 3
    assert tuner.max_close_price_seen == pd.Timestamp("2020-01-08")
