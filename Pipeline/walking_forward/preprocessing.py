"""Final fold preprocessing for walk-forward experiments."""

import pandas as pd

try:
    from ..evaluation.metrics import calculate_ic_and_pvalue
except (ImportError, ValueError):  # Supports running Pipeline/run_pipeline.py as a script.
    from evaluation.metrics import calculate_ic_and_pvalue


class PreprocessingMixin:
    def _apply_preprocessing(self, X_train, X_test, y_train, ae_train_target, candidate_features, period_dir, period_str,
                             train_qids, test_qids, train_tickers, test_tickers,
                             use_standardization, scaling_method, use_pca, use_fa, use_autoencoder, use_factor_vae,
                             pca_components, ae_hidden_dims, ae_learning_rate, ae_weight_decay,
                             ae_batch_size, ae_max_epochs, ae_patience, ae_l1_lambda,
                             factor_vae_seq_len,
                             factor_vae_hidden_size, factor_vae_num_factors, factor_vae_num_portfolios,
                             factor_vae_learning_rate, factor_vae_weight_decay, factor_vae_batch_size,
                             factor_vae_max_epochs, factor_vae_patience,
                             keep_original_features, old_decomposition_model):
        period_dir.mkdir(parents=True, exist_ok=True)
        print("--> Applying Final Preprocessing...")
        if use_standardization:
            if scaling_method == 'cross_sectional':
                X_train = self.scaler.fit_transform_cross_sectional(X_train, feature_cols=candidate_features)
                X_test = self.scaler.transform_cross_sectional(X_test, feature_cols=candidate_features)
            else:
                X_train = self.scaler.fit_transform_standardize(X_train, feature_cols=candidate_features)
                X_test = self.scaler.transform_standardize(X_test, feature_cols=candidate_features)

        raw_ic_df = calculate_ic_and_pvalue(X_train[candidate_features], y_train, task_type=self.model_wrapper.task_type)
        self._safe_to_csv(raw_ic_df, period_dir / f"ic_raw_features_{period_str}.csv", index=False)
        
        final_features = candidate_features
        
        # Decompositions
        if use_pca:
            k = getattr(self.model_wrapper, 'best_params_', {}).get('pca__n_components', pca_components)
            X_train = self.decomposer.fit_transform_pca(X_train, candidate_features, n_components=k, 
                                                        align_signs_with=old_decomposition_model, 
                                                        keep_original_features=keep_original_features)
            X_test = self.decomposer.transform_pca(X_test, keep_original_features=keep_original_features)
            old_decomposition_model = self.decomposer.pca
            final_features = self.decomposer.pca_col_names
            if keep_original_features: final_features = candidate_features + final_features
        elif use_fa:
            k = getattr(self.model_wrapper, 'best_params_', {}).get('fa__n_components', pca_components)
            X_train = self.decomposer.fit_transform_fa(X_train, candidate_features, n_components=k, 
                                                       align_signs_with=old_decomposition_model, 
                                                       keep_original_features=keep_original_features)
            X_test = self.decomposer.transform_fa(X_test, keep_original_features=keep_original_features)
            old_decomposition_model = self.decomposer.fa
            final_features = self.decomposer.fa_col_names
            if keep_original_features: final_features = candidate_features + final_features
        elif use_autoencoder:
            k = getattr(self.model_wrapper, 'best_params_', {}).get('ae__latent_dim', pca_components)
            h = getattr(self.model_wrapper, 'best_params_', {}).get('ae__hidden_dims', ae_hidden_dims)
            lr = getattr(self.model_wrapper, 'best_params_', {}).get('ae__learning_rate', ae_learning_rate)
            wd = getattr(self.model_wrapper, 'best_params_', {}).get('ae__weight_decay', ae_weight_decay)
            batch_size = getattr(self.model_wrapper, 'best_params_', {}).get('ae__batch_size', ae_batch_size)
            max_epochs = getattr(self.model_wrapper, 'best_params_', {}).get('ae__max_epochs', ae_max_epochs)
            patience = getattr(self.model_wrapper, 'best_params_', {}).get('ae__patience', ae_patience)
            l1_lambda = getattr(self.model_wrapper, 'best_params_', {}).get('ae__l1_lambda', ae_l1_lambda)
            progress_bar = None
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=1, desc=f"Final GKX-CA {period_str}", leave=False)
            except ImportError:
                progress_bar = None
            try:
                X_train = self.neural_extractor.fit_transform_autoencoder(X_train, candidate_features, latent_dim=k, hidden_dims=h,
                                                                        target=ae_train_target,
                                                                        dates=train_qids,
                                                                        tickers=train_tickers,
                                                                        l1_lambda=l1_lambda,
                                                                        learning_rate=lr, weight_decay=wd,
                                                                        batch_size=batch_size,
                                                                        max_epochs=max_epochs, patience=patience,
                                                                        keep_original_features=keep_original_features,
                                                                        progress_bar=progress_bar)
            finally:
                if progress_bar is not None:
                    progress_bar.close()
            X_test = self.neural_extractor.transform_autoencoder(X_test, keep_original_features=keep_original_features)
            final_features = self.neural_extractor.ae_col_names
            if keep_original_features: final_features = candidate_features + final_features
        elif use_factor_vae:
            seq_len = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__seq_len', factor_vae_seq_len)
            hidden_size = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__hidden_size', factor_vae_hidden_size)
            num_factors = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__num_factors', factor_vae_num_factors)
            num_portfolios = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__num_portfolios', factor_vae_num_portfolios)
            lr = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__learning_rate', factor_vae_learning_rate)
            wd = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__weight_decay', factor_vae_weight_decay)
            batch_size = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__batch_size', factor_vae_batch_size)
            max_epochs = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__max_epochs', factor_vae_max_epochs)
            patience = getattr(self.model_wrapper, 'best_params_', {}).get('factorvae__patience', factor_vae_patience)
            progress_bar = None
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=1, desc=f"Final FactorVAE {period_str}", leave=False)
            except Exception:
                progress_bar = None
            try:
                X_train = self.neural_extractor.fit_transform_factor_vae(
                    X_train,
                    candidate_features,
                    target=ae_train_target,
                    dates=train_qids,
                    tickers=train_tickers,
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    num_factors=num_factors,
                    num_portfolios=num_portfolios,
                    learning_rate=lr,
                    weight_decay=wd,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    keep_original_features=keep_original_features,
                    progress_bar=progress_bar,
                )
            finally:
                if progress_bar is not None:
                    progress_bar.close()
            X_test = self.neural_extractor.transform_factor_vae(
                X_test,
                dates=test_qids,
                tickers=test_tickers,
                keep_original_features=keep_original_features,
            )
            final_features = self.neural_extractor.factor_vae_col_names
            if keep_original_features: final_features = candidate_features + final_features

        # --- REPORTING: Calculate IC and P-Value for the final (decomposed) features ---
        if use_pca or use_fa or use_autoencoder or use_factor_vae:
            print(f"--> Calculating IC for decomposed features ({len(final_features)} components)...")
            decomposed_ic_df = calculate_ic_and_pvalue(X_train[final_features], y_train, task_type=self.model_wrapper.task_type)
            prefix = "pca" if use_pca else ("fa" if use_fa else ("ae" if use_autoencoder else "factorvae"))
            self._safe_to_csv(decomposed_ic_df, period_dir / f"ic_{prefix}_features_{period_str}.csv", index=False)

        return X_train, X_test, final_features, old_decomposition_model
