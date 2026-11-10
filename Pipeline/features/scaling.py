"""
Scaling helpers used inside each walk-forward fold.

Fit methods are called only on the training slice; transform methods are used
on validation/test slices to avoid leaking future distribution information.
"""

import pandas as pd

class FeatureScaler:
    def __init__(self):
        """
        Initializes the FeatureScaler.
        """
        self.scalers = {}

    def fit_transform_standardize(self, df, feature_cols=None):
        """
        Fits StandardScalers for the specified features and standardizes them.
        If feature_cols is None, it automatically finds columns starting with 'alpha'.
        """
        df_standardized = df.copy()
        
        # Auto-discovery if columns not provided
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith('alpha')]
            
        for col in feature_cols:
            if col in df_standardized.columns:
                mean = df_standardized[col].mean()
                std = df_standardized[col].std()
                
                if std == 0 or pd.isna(std):
                    std = 1.0
                    
                self.scalers[col] = {'mean': mean, 'std': std}
                df_standardized[col] = (df_standardized[col] - mean) / std
                
        return df_standardized

    def transform_standardize(self, df, feature_cols=None):
        """
        Transforms features using previously fitted StandardScalers.
        If feature_cols is None, it targets all columns currently in self.scalers.
        """
        if not self.scalers:
            pass # Allows no-op transforms when scaling was not fitted.
            
        df_standardized = df.copy()
        
        # Use learned scalers if columns not provided
        if feature_cols is None:
            feature_cols = list(self.scalers.keys())
        
        for col in feature_cols:
            if col in df_standardized.columns and col in self.scalers:
                mean = self.scalers[col]['mean']
                std = self.scalers[col]['std']
                df_standardized[col] = (df_standardized[col] - mean) / std
                
        return df_standardized

    def _apply_cross_sectional(self, df, feature_cols=None):
        """
        Internal method to apply cross-sectional z-score normalization per date.
        """
        df_normalized = df.copy()
        
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith('alpha')]
            
        # Determine how to group by date
        if 'Date' in df_normalized.columns:
            grouper = 'Date'
        elif df_normalized.index.name == 'Date':
            grouper = df_normalized.index
        else:
            raise ValueError("Cross-sectional normalization requires a 'Date' column or index.")
            
        # Group by date and standardize each feature cross-sectionally
        def standardize(x):
            std = x.std()
            # If standard deviation is 0 or NaN (e.g. only one asset), keep it 0 centered
            if pd.isna(std) or std == 0:
                return x - x.mean()
            return (x - x.mean()) / std
            
        for col in feature_cols:
            if col in df_normalized.columns:
                df_normalized[col] = df_normalized.groupby(grouper)[col].transform(standardize)
                
        return df_normalized

    def fit_transform_cross_sectional(self, df, feature_cols=None):
        """
        Applies cross-sectional normalization. There is no historical state to fit,
        so this just calls the internal cross-sectional logic.
        """
        return self._apply_cross_sectional(df, feature_cols)

    def transform_cross_sectional(self, df, feature_cols=None):
        """
        Applies cross-sectional normalization to out-of-sample data.
        Since cross-sectional scaling is stateless, this is identical to fit_transform.
        """
        return self._apply_cross_sectional(df, feature_cols)
