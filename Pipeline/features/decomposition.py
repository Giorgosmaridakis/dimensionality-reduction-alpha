"""
Linear feature compression utilities.

The forecaster fits these on the fold training window and applies the learned
mapping to the test window. Optional sign alignment keeps PCA/FA component
orientation more stable across walk-forward periods.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis

class FeatureDecomposer:
    def __init__(self):
        """
        Initializes the FeatureDecomposer for dimensionality reduction.
        """
        self.pca = None
        self.fa = None

    def fit_transform_pca(self, df, feature_cols, n_components=10, align_signs_with=None, keep_original_features=False):
        """
        Fits PCA on the specified features.
        """
        self.pca = PCA(n_components=n_components)
        self.pca_feature_cols = feature_cols
        self.keep_original_features = keep_original_features
        
        df_pca = df.copy()
        
        pca_features = self.pca.fit_transform(df_pca[feature_cols].fillna(0))
        
        pca_reconstructed = self.pca.inverse_transform(pca_features)
        self.pca_reconstruction_mse = np.mean((df_pca[feature_cols].fillna(0).values - pca_reconstructed) ** 2)
        
        previous_pca = None
        if align_signs_with is not None:
            previous_pca = getattr(align_signs_with, 'pca', align_signs_with)

        if previous_pca is not None and hasattr(previous_pca, 'components_'):
            for i in range(min(self.pca.components_.shape[0], previous_pca.components_.shape[0])):
                dot_product = np.dot(self.pca.components_[i], previous_pca.components_[i])
                if dot_product < 0:
                    self.pca.components_[i] = -self.pca.components_[i]
                    pca_features[:, i] = -pca_features[:, i]
        
        n_comp_out = pca_features.shape[1]
        self.pca_col_names = [f'pca_component_{i+1}' for i in range(n_comp_out)]
        
        if not self.keep_original_features:
            df_pca = df_pca.drop(columns=feature_cols)
        
        for i, col_name in enumerate(self.pca_col_names):
            df_pca[col_name] = pca_features[:, i]
            
        return df_pca

    def transform_pca(self, df, keep_original_features=None):
        """
        Transforms features using the previously fitted PCA model.
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_transform_pca first.")
            
        df_pca = df.copy()
        missing_cols = [col for col in self.pca_feature_cols if col not in df_pca.columns]
        if missing_cols:
            raise KeyError(f"Missing required features for PCA transform: {missing_cols}")
        
        pca_features = self.pca.transform(df_pca[self.pca_feature_cols].fillna(0))
        
        keep_orig = getattr(self, 'keep_original_features', False) if keep_original_features is None else keep_original_features
        if not keep_orig:
            df_pca = df_pca.drop(columns=self.pca_feature_cols)
        
        for i, col_name in enumerate(self.pca_col_names):
            df_pca[col_name] = pca_features[:, i]
            
        return df_pca

    def fit_transform_fa(self, df, feature_cols, n_components=10, align_signs_with=None, keep_original_features=False):
        """
        Fits FactorAnalysis on the specified features.
        """
        self.fa = FactorAnalysis(n_components=n_components)
        self.fa_feature_cols = feature_cols
        self.keep_original_features_fa = keep_original_features
        
        df_fa = df.copy()
        fa_features = self.fa.fit_transform(df_fa[feature_cols].fillna(0))
        
        fa_reconstructed = np.dot(fa_features, self.fa.components_) + self.fa.mean_
        self.fa_reconstruction_mse = np.mean((df_fa[feature_cols].fillna(0).values - fa_reconstructed) ** 2)
        
        previous_fa = None
        if align_signs_with is not None:
            previous_fa = getattr(align_signs_with, 'fa', align_signs_with)

        if previous_fa is not None and hasattr(previous_fa, 'components_'):
            for i in range(min(self.fa.components_.shape[0], previous_fa.components_.shape[0])):
                dot_product = np.dot(self.fa.components_[i], previous_fa.components_[i])
                if dot_product < 0:
                    self.fa.components_[i] = -self.fa.components_[i]
                    fa_features[:, i] = -fa_features[:, i]
        
        n_comp_out = fa_features.shape[1]
        self.fa_col_names = [f'fa_component_{i+1}' for i in range(n_comp_out)]
        
        if not self.keep_original_features_fa:
            df_fa = df_fa.drop(columns=feature_cols)
        
        for i, col_name in enumerate(self.fa_col_names):
            df_fa[col_name] = fa_features[:, i]
            
        return df_fa

    def transform_fa(self, df, keep_original_features=None):
        """
        Transforms features using the previously fitted FA model.
        """
        if self.fa is None:
            raise ValueError("FA not fitted. Call fit_transform_fa first.")
            
        df_fa = df.copy()
        missing_cols = [col for col in self.fa_feature_cols if col not in df_fa.columns]
        if missing_cols:
            raise KeyError(f"Missing required features for FA transform: {missing_cols}")
        
        fa_features = self.fa.transform(df_fa[self.fa_feature_cols].fillna(0))
        
        keep_orig = getattr(self, 'keep_original_features_fa', False) if keep_original_features is None else keep_original_features
        if not keep_orig:
            df_fa = df_fa.drop(columns=self.fa_feature_cols)
        
        for i, col_name in enumerate(self.fa_col_names):
            df_fa[col_name] = fa_features[:, i]
            
        return df_fa


