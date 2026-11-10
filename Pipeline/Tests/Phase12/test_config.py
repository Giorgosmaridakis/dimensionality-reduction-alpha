"""
Phase 12: Data Leakage & Inflated Results Detection
Shared fixtures used across all leakage tests.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(n_dates: int = 60, n_tickers: int = 20, n_features: int = 5,
                seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic panel DataFrame with a DatetimeIndex + ticker column,
    random alpha features, and a forward-return 'Target' column.

    The Target is deliberately *not* related to the features so that any
    inflated IC we detect must come from leakage, not genuine signal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for date in dates:
        for ticker in tickers:
            row = {"Date": date, "ticker": ticker}
            for f in range(n_features):
                row[f"alpha{f+1:03d}"] = rng.standard_normal()
            # Target: pure noise (no genuine signal)
            row["Target"] = rng.standard_normal()
            rows.append(row)

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


@pytest.fixture(scope="module")
def panel_df():
    """Full synthetic panel with 60 dates × 20 tickers."""
    return _make_panel()


@pytest.fixture(scope="module")
def split_data(panel_df):
    """
    Returns (train_df, test_df, feature_cols) with a strict temporal split:
    first 40 dates → train, last 20 dates → test.
    """
    dates = panel_df.index.unique().sort_values()
    train_dates = dates[:40]
    test_dates = dates[40:]

    train_df = panel_df.loc[train_dates]
    test_df = panel_df.loc[test_dates]
    feature_cols = [c for c in panel_df.columns if c.startswith("alpha")]
    return train_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# Fixture for migrated test_pipeline.py tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dummy_df():
    """
    Recreates the synthetic dataset from the original test_pipeline.py.
    - feature_1, feature_2: random signals
    - Target: next-day feature_1 + noise  (honest, slight predictability)
    - feature_3_cheat: identical to Target  (perfect future-peek feature)
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="B")
    n = len(dates)

    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    f3 = rng.standard_normal(n)
    target = np.roll(f1, -1) + rng.standard_normal(n) * 1.0  # Increased noise
    
    df = pd.DataFrame(
        {"feature_1": f1, "feature_2": f2, "feature_4": f3, "Target": target,
         "feature_3_cheat": target},
        index=dates,
    )
    return df.iloc[:-1].copy()


@pytest.fixture()
def clean_test_dirs(tmp_path):
    """
    Yields a tmp_path and guarantees cleanup of any named result directories
    that the pipeline writes to disk during the legacy tests.
    """
    yield tmp_path
    for d in [
        "Test_Results_1", "Test_Results_Honest", "Test_Results_Cheat",
        "Test_Results_Det_1", "Test_Results_Det_2",
    ]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
