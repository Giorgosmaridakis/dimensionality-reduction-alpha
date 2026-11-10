"""
Phase 12 – Test 3: PCA / Decomposition Leakage
================================================
PCA components must be fitted only on training data.
If PCA is fitted on the full dataset (train + test), the principal directions
encode test-set variance and inflate OOS reconstruction fidelity.

Run with:
    cd Pipeline
    pytest Tests/Phase12/test_decomposition_leakage.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.decomposition import FeatureDecomposer

pytest_plugins = ["Tests.Phase12.test_config"]


class TestDecompositionLeakage:
    """
    PCA components must be fitted only on training data.
    If PCA is fitted on the full dataset (train + test), the principal
    directions encode test-set variance and inflate OOS reconstruction fidelity.
    """

    def test_pca_components_differ_from_global_fit(self, split_data):
        """
        A PCA fitted on train-only must produce different components than one
        fitted on the full dataset.  If they are identical, test data leaked
        into the PCA fitting step.
        """
        train_df, test_df, feature_cols = split_data
        combined = pd.concat([train_df, test_df])

        decomposer_train = FeatureDecomposer()
        decomposer_train.fit_transform_pca(train_df, feature_cols=feature_cols, n_components=2)

        decomposer_global = FeatureDecomposer()
        decomposer_global.fit_transform_pca(combined, feature_cols=feature_cols, n_components=2)

        components_match = np.allclose(
            np.abs(decomposer_train.pca.components_),
            np.abs(decomposer_global.pca.components_),
            atol=1e-6,
        )
        assert not components_match, (
            "LEAKAGE DETECTED – PCA components fitted on train-only are identical to "
            "those fitted on the full dataset. This suggests PCA is being fitted on "
            "train+test data simultaneously."
        )

    def test_pca_transform_uses_train_fitted_model(self, split_data):
        """
        After fitting PCA on train, transforming the test set must use the
        train-fitted projection matrix (not re-fitted on test).
        Concretely: transform(test) with train-PCA ≠ fit_transform(test).
        """
        train_df, test_df, feature_cols = split_data

        decomposer = FeatureDecomposer()
        decomposer.fit_transform_pca(train_df, feature_cols=feature_cols, n_components=2)

        # Correct pipeline: apply train PCA to test
        test_projected_correct = decomposer.transform_pca(test_df.copy())

        # Simulated leakage: re-fit PCA on the test set
        decomposer_leak = FeatureDecomposer()
        test_projected_leak = decomposer_leak.fit_transform_pca(
            test_df, feature_cols=feature_cols, n_components=2
        )

        pca_cols = decomposer.pca_col_names
        correct_vals = test_projected_correct[pca_cols].values
        leak_vals = test_projected_leak[pca_cols].values

        assert not np.allclose(np.abs(correct_vals), np.abs(leak_vals), atol=1e-6), (
            "LEAKAGE DETECTED – Applying train-fitted PCA to the test set gives the same "
            "result as re-fitting PCA on the test set. This is statistically near-impossible "
            "on random data and indicates a shared fitting path."
        )
