"""
Optional statistical comparison tools for experiment results.

These functions are not part of the main training loop; they are meant for
post-run analysis once IC or return series have been saved.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Union, Tuple, Callable

class StatisticalTests:
    """
    Advanced Statistical Tests for Quant Model Evaluation.
    Includes Newey-West adjustments, Diebold-Mariano tests, 
    Sharpe Ratio comparisons, and Multiple Testing corrections.
    """

    @staticmethod
    def newey_west_se(series: np.ndarray, max_lag: int = 6) -> float:
        """
        Newey-West heteroskedasticity and autocorrelation consistent (HAC) standard error.
        Corrects for serial correlation in time series metrics.
        """
        T = len(series)
        mean = np.mean(series)
        demeaned = series - mean

        # Variance (lag 0)
        gamma_0 = np.dot(demeaned, demeaned) / T

        # Add weighted autocovariances
        nw_var = gamma_0
        for lag in range(1, max_lag + 1):
            gamma_lag = np.dot(demeaned[lag:], demeaned[:-lag]) / T
            # Bartlett kernel weight
            weight = 1 - lag / (max_lag + 1)
            nw_var += 2 * weight * gamma_lag

        return np.sqrt(nw_var / T)

    @staticmethod
    def newey_west_se_statsmodels(series: np.ndarray, max_lag: int = 6) -> float:
        """
        Calculates Newey-West SE using statsmodels OLS.
        """
        T = len(series)
        ones = np.ones(T)
        model = sm.OLS(series, ones)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
        return results.bse[0]

    @classmethod
    def ic_statistics(cls, ic_series: np.ndarray, max_lag: int = 6) -> Dict:
        """
        Computes Mean IC, ICIR, T-Stat (NW-adjusted), P-Value, and 95% Confidence Interval.
        """
        T = len(ic_series)
        mean_ic = np.mean(ic_series)
        std_ic = np.std(ic_series, ddof=1)

        # Newey-West standard error — corrects for serial correlation
        se_nw = cls.newey_west_se(ic_series, max_lag=max_lag)

        # t-statistic (ICIR adjusted for NW SE)
        t_stat = mean_ic / se_nw
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=T-1))

        # 95% confidence interval
        ci_lower = mean_ic - 1.96 * se_nw
        ci_upper = mean_ic + 1.96 * se_nw

        return {
            'mean_ic': mean_ic,
            'icir': mean_ic / std_ic,  # Standard ICIR (Mean/Std)
            't_stat': t_stat,
            'p_value': p_value,
            'ci_95': (ci_lower, ci_upper)
        }

    @classmethod
    def diebold_mariano_test(cls, errors_model1: np.ndarray, errors_model2: np.ndarray, max_lag: int = 6) -> Tuple[float, float]:
        """
        Tests H0: models have equal predictive accuracy.
        Positive DM = model2 is better (smaller errors).
        Negative DM = model1 is better.
        """
        # Loss differential (using squared error loss)
        d = errors_model1**2 - errors_model2**2

        # Mean loss differential
        mean_d = np.mean(d)

        # Newey-West SE of the loss differential
        se_d = cls.newey_west_se(d, max_lag=max_lag)

        # DM statistic
        dm_stat = mean_d / se_d

        # p-value (two-sided)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return dm_stat, p_value

    @staticmethod
    def sharpe_ratio_test(returns1: np.ndarray, returns2: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Tests whether two Sharpe ratios are significantly different.
        Jobson-Korkie test with Memmel correction.
        """
        n = len(returns1)
        mu1, mu2 = np.mean(returns1), np.mean(returns2)
        s1, s2 = np.std(returns1, ddof=1), np.std(returns2, ddof=1)
        cov_12 = np.cov(returns1, returns2)[0,1]

        sr1 = mu1 / s1
        sr2 = mu2 / s2

        # Jobson-Korkie statistic
        theta = (1/n) * (
            2*s1**2*s2**2
            - 2*s1*s2*cov_12
            + 0.5*mu1**2*s2**2
            + 0.5*mu2**2*s1**2
            - (mu1*mu2*cov_12**2)/(s1*s2)
        )

        jk_stat = (sr1 - sr2) / np.sqrt(theta)
        p_value = 2 * (1 - stats.norm.cdf(abs(jk_stat)))

        return sr1, sr2, jk_stat, p_value

    @staticmethod
    def block_bootstrap_indices(T: int, block_size: int = 6) -> np.ndarray:
        """
        Resamples blocks of consecutive observations to preserve time series structure.
        """
        n_blocks = int(np.ceil(T / block_size))
        block_starts = np.random.randint(0, T - block_size + 1, size=n_blocks)
        indices = []
        for start in block_starts:
            indices.extend(range(start, start + block_size))
        return np.array(indices[:T])

    @classmethod
    def bootstrap_metric_difference(cls, metric_fn: Callable, data1: np.ndarray, data2: np.ndarray,
                                     n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple:
        """
        Bootstrap test for difference in any metric between two models.
        """
        T = len(data1)
        observed_diff = metric_fn(data1) - metric_fn(data2)

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            indices = cls.block_bootstrap_indices(T, block_size=6)
            boot_data1 = data1[indices]
            boot_data2 = data2[indices]
            bootstrap_diffs.append(metric_fn(boot_data1) - metric_fn(boot_data2))

        bootstrap_diffs = np.array(bootstrap_diffs)
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_diffs, 100*alpha/2)
        ci_upper = np.percentile(bootstrap_diffs, 100*(1-alpha/2))

        p_value = np.mean(bootstrap_diffs <= 0) if observed_diff > 0 else np.mean(bootstrap_diffs >= 0)
        p_value = 2 * min(p_value, 1-p_value)

        return observed_diff, (ci_lower, ci_upper), p_value

    @staticmethod
    def correct_multiple_tests(p_values: List[float], method: str = 'fdr_bh', alpha: float = 0.05) -> Tuple:
        """
        Correct p-values for multiple testing (e.g., FDR, Bonferroni).
        """
        reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
        return reject, p_corrected
