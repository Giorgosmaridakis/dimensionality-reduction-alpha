"""
Phase 12 – Test 1: Temporal (Row-Level) Leakage
================================================
Verify that the train / test split contains zero date overlap.
Any shared date means the model had access to future observations
while being evaluated, which directly inflates all performance metrics.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_temporal_leakage.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest_plugins = ["Tests.Phase12.test_config"]



class TestTemporalLeakage:
    """
    Verify that the train / test split contains zero date overlap.
    Any shared date means the model had access to future observations
    while being evaluated, which directly inflates all performance metrics.
    """

    def test_no_date_overlap_in_split(self, split_data):
        """Train and test sets must cover disjoint date ranges."""
        train_df, test_df, _ = split_data

        train_dates = set(train_df.index.unique())
        test_dates = set(test_df.index.unique())
        overlap = train_dates & test_dates

        assert len(overlap) == 0, (
            f"LEAKAGE DETECTED – {len(overlap)} date(s) appear in both train and test: "
            f"{sorted(overlap)[:5]} ..."
        )

    def test_train_ends_before_test_begins(self, split_data):
        """The last training date must strictly precede the first test date."""
        train_df, test_df, _ = split_data

        train_max = train_df.index.max()
        test_min = test_df.index.min()

        assert train_max < test_min, (
            f"LEAKAGE DETECTED – train ends on {train_max} but test starts on {test_min}. "
            "Future data was available during training."
        )

    def test_no_index_row_duplication_across_split(self, split_data):
        """
        Concatenating train + test must not produce duplicated (Date, ticker) pairs.
        Duplicates indicate the same observation was placed in both folds.
        """
        train_df, test_df, _ = split_data

        combined = pd.concat([train_df, test_df])
        combined["_key"] = combined.index.astype(str) + "_" + combined["ticker"].astype(str)
        duplicated = combined["_key"].duplicated(keep=False)

        assert not duplicated.any(), (
            f"LEAKAGE DETECTED – {duplicated.sum()} duplicate (Date, Ticker) rows across train/test."
        )
