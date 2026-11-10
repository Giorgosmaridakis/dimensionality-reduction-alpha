# Financial Machine Learning Pipeline: Modular Architecture Manual

**Dimensionality Reduction as Alpha? Comparing Linear and Deep Feature Extractors for Tree-Based Cross-Sectional Return Prediction**

Code for my Bachelor's thesis, Department of Computer Science, University of Crete (2026). Advisor: Dr George P. Kafentzis. Supervisor: Dr George Tzagkarakis.

The thesis tests whether compressing a library of 80 formulaic alpha signals before prediction improves a daily long/short strategy, compared with giving all the signals directly to a gradient-boosted ranking model.

**Thesis:** [full text (PDF)](docs/_BSc__Dimensionality_Reduction_as_Alpha__Comparing_Linear_and_Deep_Feature_Extractors_for_Tree_Based_Cross_Sectional_Return_Prediction.pdf) · **Defence slides:** [presentation (PDF)](docs/presentation.pdf)

> **Note on the thesis:** the market regression in Section 4.3 (Tables 4.4–4.5, Figure 4.1) aligned each day's portfolio return with the market return two days later. With same-day returns the books are dollar-neutral but carry a market beta of about 0.2, so they are not market-neutral. The other results are unaffected.

**Contents:** [Project overview](#project-overview) · [Repository layout](#repository-layout) · [1. Architecture](#1-modular-architecture-overview) · [2. 13-phase lifecycle](#2-the-13-phase-lifecycle-breakdown) · [3. Refactoring decisions](#3-summary-of-major-refactoring-decisions) · [4. Data](#4-data) · [5. Results](#5-results) · [6. Setup, running and tests](#6-setup-running-and-tests) · [7. Known limitations](#7-known-limitations) · [8. Reproducibility](#8-reproducibility) · [9. Hyperparameter search spaces](#9-hyperparameter-search-spaces) · [10. Output files](#10-output-files) · [11. Further documentation](#11-further-documentation) · [12. References](#12-references) · [13. Citation](#13-citation) · [14. License](#14-license)

## Project overview

### Research questions

1. Does compressing the alpha library through dimensionality reduction improve the out-of-sample performance of a tree-based predictor, compared with the raw signals?
2. Do deep extractors (the GKX autoencoder and FactorVAE) produce better features than linear methods (PCA and Factor Analysis)?
3. How does the choice of feature normalization, time-series (standard) or cross-sectional, interact with the reduction method and the breadth of the universe?

### What it compares

| | |
|---|---|
| **Feature extractors** | None (Baseline), PCA, Factor Analysis, GKX conditional autoencoder (Gu, Kelly & Xiu, 2021), FactorVAE (Duan et al., 2022) |
| **Ranking models** | LightGBM (LambdaRank), CatBoost (YetiRank) |
| **Neural models** | Used directly to rank assets, or as inputs to a tree model |
| **Normalization** | Standard and cross-sectional |
| **Universes** | 60 ETFs and 483 S&P 500 stocks |

This gives 48 configurations, 24 per universe:

| Universe | Out-of-sample years | Folds | Training window |
|---|---|---:|---|
| ETFs | 2019–2025 | 7 | 4 years |
| S&P 500 | 2022–2025 | 4 | 2 years, because the data starts in 2020 |

Each year, the models are retuned on validation Sharpe, using three time-ordered folds of the training window only, then refitted and used to predict that year. Tree models are tuned with Optuna, 16 to 40 trials per year depending on the profile. Standalone neural models test 40 settings drawn at random from their grid; see [Tuning_Architecture_README.md](Pipeline/Tuning_Architecture_README.md#4-trial-budget).

The strategy goes long the top 5% and short the bottom 5% of assets each day, with 50% gross exposure per side. The backtest includes fees of 1 bp, plus slippage of 2 bp for ETFs and 5 bp for S&P 500 stocks.

### Repository layout

```text
Bachelor_Thesis/
├── README.md                     this file
├── LICENSE
├── requirements.txt
├── docs/
│   ├── _BSc__Dimensionality_Reduction_as_Alpha_...pdf   the thesis
│   └── presentation.pdf          the defence slides
├── results/
│   └── results_summary.csv       Sharpe, Rank IC and risk metrics for all 48 configurations
└── Pipeline/
    ├── run_pipeline.py           entry point: runs the experiments in DEFAULT_EXPERIMENTS
    ├── experiment_config.py      experiments, search grids, profiles and output folder names
    ├── build_statistical_inputs.py
    ├── README.md                 code structure, prediction sources and output folders
    ├── data/                     data loading, panel and target construction, ticker lists
    ├── features/                 scaling, PCA, FA, GKX and FactorVAE extractors
    ├── modeling/                 tree wrappers, Optuna tuning, GKX and FactorVAE models
    ├── walking_forward/          yearly train, tune and predict loop
    ├── evaluation/               IC metrics and statistical tests
    ├── Backtester/               vectorbt long/short backtester and comparison scripts
    └── Tests/                    191 tests in 13 phases
```

---

This document provides a microscopic breakdown of the refactored, modular pipeline. It is a technical reference for the thesis code and maps each stage of the system to its tests.

---

## 1. Modular Architecture Overview

The pipeline has been decoupled into four primary layers, each in its own subdirectory of [`Pipeline/`](Pipeline/).

```text
┌─────────────────────────────────────────────────────────────────┐
│                        run_pipeline.py                          │
│              (Orchestrator & Experiment Loop)                    │
└────────┬───────────────┬──────────────────┬─────────────────────┘
         │               │                  │
    ┌────▼────┐   ┌──────▼──────┐   ┌───────▼──────────┐
    │  Data   │   │ Preprocessing │   │   Evaluation     │
    │  Layer  │   │   Layer     │   │     Layer        │
    │ (loader,│   │  (scaling,  │   │ (walk-forward,   │
    │ builder)│   │  decompos.) │   │  metric calc)    │
    └─────────┘   └─────────────┘   └───────┬──────────┘
                                            │
                                    ┌───────▼──────────┐
                                    │     Modeling     │
                                    │      Layer       │
                                    │ (trees engine,   │
                                    │  tuning/grid)    │
                                    └───────┬──────────┘
                                            │
                                    ┌───────▼──────────┐
                                    │    Backtesting   │
                                    │      Layer       │
                                    │ (vectorbt sim,   │
                                    │  strategy logic) │
                                    └──────────────────┘
```

---

## 2. The 13-Phase Lifecycle Breakdown

The following stages describe the exact data flow from raw CSV to final performance metrics. Each phase has a matching folder under `Pipeline/Tests/`. Run the commands below from inside `Pipeline/`.

### Phase 1: Data Discovery & Lifecycle
*   **Location**: `data/data_loader.py` -> `DataLoader`
*   **Logical Goal**: Isolate the filesystem from the logic and ensure data integrity.
*   **Transformation**: Finds `alpha*.csv` within `zuras/` and locates `Returns.csv` / `Adjusted_Close.csv`.



#### Verification Checklist (English & Pytest)
*   **File Discovery**: Prove that the `DataLoader` finds the expected number of alpha signals.
    *   *Test File*: [test_file_discovery.py](Pipeline/Tests/Phase1/test_file_discovery.py)
*   **Path Robustness**: Verify OS-agnostic path handling (Windows vs Linux).
    *   *Test File*: [test_path_robustness.py](Pipeline/Tests/Phase1/test_path_robustness.py)
*   **Universe Isolation**: Confirm selection of 'ETFs' strictly ignores 'S&P500' data.
    *   *Test File*: [test_universe_isolation.py](Pipeline/Tests/Phase1/test_universe_isolation.py)
*   **Price Discovery**: Ensure critical benchmark files exist in every subdirectory.
    *   *Test File*: [test_price_discovery.py](Pipeline/Tests/Phase1/test_price_discovery.py)
*   **Return Formula Quality**: Mathematically prove that `Returns.csv` follows the Rate of Change formula. Note: This test prioritizes **Adjusted_Close.csv** to account for dividends.
    *   *Test File*: [test_return_formula.py](Pipeline/Tests/Phase1/test_return_formula.py)

#### How to Run
```powershell
pytest Tests/Phase1/ -v
```

### Phase 2: Panel Construction (Melt & Stack)
*   **Location**: `data/dataset_builder.py` -> `DatasetBuilder.prepare_stacked_data`
*   **Logical Goal**: Transform "Wide" data (Time x Asset) to "Long" data (Time x Asset x Value).
*   **Transformation**: `pd.melt()` on multiple alpha files, then joining them on `Date` and `Ticker`.

#### Verification Checklist (English & Pytest)
*   **Dimensional Accuracy**: Assert total rows match unique dates multiplied by tickers.
    *   *Test File*: [test_dimensional_accuracy.py](Pipeline/Tests/Phase2/test_dimensional_accuracy.py)
*   **Data Integrity**: Pick random samples and verify against source CSVs.
    *   *Test File*: [test_data_integrity.py](Pipeline/Tests/Phase2/test_data_integrity.py)
*   **Column Mapping**: Ensure every alpha file corresponds to exactly one column.
    *   *Test File*: [test_column_mapping.py](Pipeline/Tests/Phase2/test_column_mapping.py)
*   **NaN Persistence**: Verify `dropna` logic works as intended.
    *   *Test File*: [test_nan_persistence.py](Pipeline/Tests/Phase2/test_nan_persistence.py)

#### How to Run
```powershell
pytest Tests/Phase2/ -v
```

### Phase 3: Target Engineering (The Predictive Window)
*   **Location**: `data/dataset_builder.py` -> `DatasetBuilder.merge_features_and_targets`
*   **Logical Goal**: Create the signal-to-tradable-return relationship: features at $t$ predict the adjusted-close holding return from $t+1$ to $t+2$.
*   **Transformation**: Builds close-to-close returns from `Adjusted_Close.csv` and shifts them by `-2` rows so `Target_t = AdjustedClose_{t+2} / AdjustedClose_{t+1} - 1`.


#### Verification Checklist (English)
*   **Temporal Causality**: This is the most critical test. Prove that for every row at time $T$, the `Target` column contains the adjusted-close return from $T+1$ to $T+2$. 
*   **Zero Leakage**: Verify that no part of the calculation of the 'Target' uses any information that would not have been available at the end of day $T$.
*   **Ticker Alignment**: Ensure that the merge didn't accidentally shift tickers, causing features for one company to be aligned with another company's adjusted-close target return.
*   **Boundary Conditions**: Check what happens at the very end of the dataset--ensure dates without a full $T+1$ entry and $T+2$ exit window are handled correctly.

### Phase 4: Statistical Normalization (Standardization)
*   **Location**: `features/scaling.py` -> `FeatureScaler`
*   **Logical Goal**: Ensure GBDT initialization and PCA aren't biased by feature magnitude.
*   **Transformation**: $X_{norm} = (X - \mu) / \sigma$.
*   **Test Blueprint**:
    ```python
    from features.scaling import FeatureScaler
    import numpy as np

    def test_phase4_scaling():
        scaler = FeatureScaler()
        df = pd.DataFrame({'f': [10.0, 20.0, 30.0]})
        out = scaler.fit_transform_standardize(df, ['f'])
        assert np.isclose(out['f'].mean(), 0.0)
        assert np.isclose(out['f'].std(), 1.0)
    ```

#### Verification Checklist (English)
*   **Zero-Mean Verification**: Confirm that the mathematical mean of each scaled feature is within a tiny tolerance ($10^{-9}$) of zero.
*   **Unit-Variance Verification**: Confirm that the standard deviation is exactly $1.0$.
*   **Out-of-Sample Integrity**: Fit the scaler on a 'Train' period and apply it to a 'Test' period. Verify that the 'Test' data is scaled using the mean and std of the 'Train' data, not its own.
*   **Zero-Std Edge Case**: If a feature has zero variance (all values identical), ensure the scaler returns 0.0 rather than `NaN` or infinity.

### Phase 5: Linear Latent Extraction (PCA/FA)
*   **Location**: `features/decomposition.py` -> `FeatureDecomposer`
*   **Logical Goal**: Reduce dimensionality while maintaining variance.
*   **Transformation**: Projects high-dimensional alphas into orthogonal components. Handles sign-alignment to prevent "Component Flipping" across rolling windows.
*   **Test Blueprint**:
    ```python
    from features.decomposition import FeatureDecomposer
    def test_phase5_pca_shape():
        decom = FeatureDecomposer()
        df = pd.DataFrame(np.random.randn(100, 10), columns=[f'f{i}' for i in range(10)])
        out = decom.fit_transform_pca(df, list(df.columns), n_components=3)
        assert len([c for c in out.columns if 'pca' in c]) == 3
    ```

#### Verification Checklist (English)
*   **Component Accuracy**: Ensure the resulting DataFrame contains exactly $K$ principal components, where $K$ is the `n_components` parameter.
    *   *Test File*: [test_component_accuracy.py](Pipeline/Tests/Phase5/test_component_accuracy.py)
*   **Structural Integrity**: Ensure the transformed DataFrame is non-empty, contains a valid `Date` column, has no duplicate `(Date, ticker)` pairs, and the max date does not exceed the period's declared end year.
    *   *Test File*: [test_structural_integrity.py](Pipeline/Tests/Phase5/test_structural_integrity.py)
*   **Orthogonality**: (For PCA) Prove that the resulting components have zero correlation with each other.
    *   *Test File*: [test_orthogonality.py](Pipeline/Tests/Phase5/test_orthogonality.py)
*   **Sign-Deterministic Alignment**: If you repeat the transformation on slightly noisier data, ensure the "direction" of the components remains the same (no random flipping).
*   **Reconstruction Path**: Verify that the decomposer can transform new test data using the projection matrix learned during the training phase.

### Phase 6: Non-Linear Neural Feature Extraction (GKX/FactorVAE)
*   **Location**: `features/neural_extractors.py` -> `NeuralFeatureExtractor`
*   **Logical Goal**: Capture manifold relationships that linear PCA misses.
*   **Transformation**: PyTorch training loop -> Bottleneck latent vector extraction.


#### Verification Checklist (English)
*   **Architecture Validity**: Verify that the latent layer (the bottleneck) has exactly the dimensions specified in the `latent_dim` config.
*   **Standardization Assumption**: Confirm that the input data was normalized before entering the network, as Neural Networks are sensitive to scale.
*   **Training Convergence**: Verify that the Loss Function (usually MSE) is tracked and reaches a stable floor after several epochs.
*   **Handling NaNs**: Ensure the extractor fills missing values with the feature's mean or zero to prevent the entire network output from becoming `NaN`.

### Phase 7: Feature Quality Diagnostic (IC)
*   **Location**: `evaluation/metrics.py` -> `calculate_ic_and_pvalue`
*   **Logical Goal**: Rank signals before they ever reach the ML model.
*   **Transformation**: Cross-sectional Spearman group-by or Pearson correlation.

#### Verification Checklist (English)
*   **Perfect Correlation Test**: Create a feature that is a 100% copy of the Target; the function must return an IC of 1.0.
*   **Rank vs Linear Logic**: Verify that for 'ranking' tasks, a Spearman (rank-based) correlation is used, and for 'regression', a Pearson correlation is used.
*   **Group-By Consistency**: Assert that the IC is calculated cross-sectionally (per date) before being averaged, rather than across the entire pool at once.
*   **P-Value Accuracy**: Confirm that statistically significant relationships yield p-values below the 0.05 threshold.

### Phase 8: Optimization Engine (Hyperparameter Tuning)
*   **Location**: `modeling/tuning/` -> `HyperparameterTuner`
*   **Logical Goal**: Find the "Global Optimal" configuration for the current market regime.
*   **Transformation**: Iterates through parameter grids (or Optuna trials) using `TimeSeriesSplit` + Embargo.
#### Verification Checklist (English)
*   **Look-Ahead Prevention**: Prove that the validation slices occur strictly after the training slices in the TimeSeriesSplit.
*   **Embargo Enforcement**: Verify there is an `embargo_days` gap where no data is used between Training and Validation. The thesis runs use `embargo_days=1`.
*   **Metric Maximization**: Feed a known parameter grid where one setting is clearly superior; ensure the tuner selects that setting correctly.
*   **Result Persistence**: Confirm that the tuning metrics for all combinations are saved to a CSV for post-hoc diagnostic review.

### Phase 9: Model Training & Inference (GBDT)
*   **Location**: `modeling/trees.py` -> `Model`
*   **Logical Goal**: Unified interface for CatBoost and LightGBM.
*   **Transformation**: Fits tree ensembles on training data; outputs scores/ranks for test data.


#### Verification Checklist (English)
*   **Engine Routing**: If the user selects 'catboost', verify that the internal engine initialized is indeed `catboost.CatBoostRegressor` or `Classifier`.
*   **Hyperparameter Application**: Assert that the 'Best Params' found in Phase 8 are actually being used during the final `fit()` call.
*   **Prediction Scale**: For regression, ensure the output range is consistent with historical returns; for ranking, ensure the output provides a relative ordering of assets.
*   **GPU Handshake**: If GPU is enabled, verify that the `device='cuda'` or equivalent flag is passed to the native model.
*   **Group-By Consistency**: Assert that the model's predictions maintain meaningful cross-sectional rank correlation (IC) with targets, calculated per date.

### Phase 10: Walking-Forward Orchestration
*   **Location**: `walking_forward/forecaster.py` -> `WalkingForwardForecaster`
*   **Logical Goal**: Simulate real-time model degradation and retraining.
*   **Transformation**: Re-runs Phases 4–9 for every anchor period (e.g., annually).

#### Verification Checklist (English)
*   **Anchor Continuity**: Confirm that the loop moves from 2019 to 2020 to 2021 without skipping any periods or overlapping the test windows.
*   **State Reset**: Ensure that for each new anchor period, the scalers and models are completely re-initialized (no memory of previous years' data).
*   **FileSystem Structure**: Verify that the code correctly creates a nested directory structure (e.g., `Result/Year/Metric.csv`) for every config.
*   **Exception Handling**: If one year's data is missing, ensure the loop can skip gracefully or log the error rather than crashing the full 5-year run.

### Phase 11: Portfolio Simulation
*   **Location**: `Backtester/backtester.py` -> `Backtester`
*   **Logical Goal**: Convert ML ranks into realistic PnL.
*   **Transformation**: VectorBT simulation with long and short books chosen by `selection_pct` (5% per side in the thesis experiments), and Z-score normalization for regression predictions.

#### Verification Checklist (English)
*   **Top/Bottom Tier Logic**: Verify that only the top `selection_pct` of signals result in 'Long' positions and the bottom `selection_pct` result in 'Short' positions.
*   **Z-Score Normalization**: For regression models, prove that the 10-day rolling Z-score properly maps signal strength to mean-reverting or trend-following behaviors.
*   **VectorBT Alignment**: Confirm that the index of the signals perfectly matches the index of the price data to avoid price-shift errors.
*   **Performance Metrics**: Manually calculate the Sharpe Ratio for a small toy PnL stream and check that the Backtester's output matches it.

### Phase 12: Data Leakage & Reliability Testing
*   **Location**: `Tests/Phase12/`
*   **Logical Goal**: Proactive detection of inflated results and causality violations.
*   **Suite Overview**:
    *   **Temporal Integrity**: Ensures zero date overlap between train/test splits.
    *   **Preprocessing Isolation**: Verifies that Scalers and PCA/FA components are fit *strictly* on training data.
    *   **Target-Peek Detection**: Checks for features derived from future returns or "cheating" variables.
    *   **Walk-Forward Reliability**: Validates inner-CV temporal ordering, embargo enforcement, and fold independence.
    *   **Determinism**: Checks byte-identical predictions across two small fixed-seed runs. This does not guarantee that full GPU experiments repeat exactly; see [Reproducibility](#8-reproducibility).
    *   **Permutation IC Testing**: Confirms IC collapses to zero when predictions are shuffled, exposing hidden look-ahead bias.

#### Verification Checklist (English)
*   **Zero Leakage**: Assert that `train.max_date < test.min_date` and no row indices are duplicated across the split.
*   **Isolate Scalers**: Demonstrate that fitted scaler means differ significantly between walk-forward folds.
*   **Causality Check**: Ensure no raw alpha features have an $|IC| > 0.5$ with the *future* target return.
*   **Determinism**: Run a small test case twice with the same seed and confirm the resulting prediction files are identical.
*   **Sensitivity**: Introduce a "cheat feature" (Target + Noise) and verify the test suite flags it by detecting a near-zero MSE and extreme IC.

### Phase 13: Statistical Inputs
*   **Location**: `build_statistical_inputs.py`
*   **Logical Goal**: Collect the daily OOS IC and daily portfolio return series of every run into one place for statistical tests.
*   **Test File**: [test_statistical_inputs.py](Pipeline/Tests/Phase13/test_statistical_inputs.py)


---

## 3. Summary of Major Refactoring Decisions

1.  **Decoupled Tuning**: The `tune_hyperparameters` function was moved out of the `Model` class into `modeling/tuning/`, so the tuning algorithm can be tested without a complex model instance.
2.  **Removal of SignalFilter**: The experimental KNN/NaiveBayes filters were removed to simplify the pipeline focus on GBDT models.
3.  **Standardized Preprocessing API**: Each extractor (`PCA`, `GKX`, `FactorVAE`, `Scaling`) now follows a strict `fit_transform` and `transform` pattern to prevent Data Leakage.

---

## 4. Data

All price data comes from Yahoo Finance through the [`yfinance`](https://github.com/ranaroussi/yfinance) Python library.

| Universe | Tickers | Price history |
|---|---|---|
| ETFs | [60 ETFs](Pipeline/data/tickers/etf_tickers.csv) covering equity sectors, regions, fixed income and commodities | 2006–2025 |
| S&P 500 | [483 constituents](Pipeline/data/tickers/sp500_tickers.csv) with a complete price and alpha history | 2020–2025 |

The alpha signals are derived from the *101 Formulaic Alphas* (Kakushadze, 2016) and computed following Jansen (2020), *Machine Learning for Algorithmic Trading*. They use price and volume data only. The data has 82 alpha files, but `alpha068` and `alpha086` are constant in both universes and carry no information, so 80 signals are informative. The pipeline still loads all 82 columns.

The price and alpha files are not in this repository, and neither is the code that downloads prices or computes the alphas. To run the experiments, place the files next to `Pipeline/` as described in [Pipeline/data/README.md](Pipeline/data/README.md). Full per-fold outputs, about 9 GB, are also kept outside the repository.

---

## 5. Results

All 48 configurations are in [results/results_summary.csv](results/results_summary.csv). Sharpe ratios use a 365-day year; divide by 1.2035 to compare with the 252-day convention (see [Backtester/README.md](Pipeline/Backtester/README.md#things-to-watch)).

**ETFs, 2019–2025**

| Extractor | Model | Standard: Rank IC | Standard: Sharpe | Cross-sectional: Rank IC | Cross-sectional: Sharpe |
|---|---|---:|---:|---:|---:|
| Baseline | LightGBM | +0.0034 | −0.648 | +0.0029 | −1.333 |
| Baseline | CatBoost | +0.0128 | −0.734 | +0.0158 | −1.369 |
| PCA | LightGBM | +0.0014 | −1.451 | −0.0023 | −1.801 |
| PCA | CatBoost | +0.0031 | −1.740 | +0.0153 | −1.077 |
| FA | LightGBM | −0.0027 | −2.142 | −0.0024 | −2.630 |
| FA | CatBoost | +0.0057 | −1.912 | +0.0097 | −1.322 |
| GKX-AE | LightGBM | +0.0112 | −1.981 | −0.0003 | −1.661 |
| GKX-AE | CatBoost | +0.0040 | −1.901 | +0.0073 | −1.765 |
| GKX-AE | none (standalone) | +0.0087 | −0.995 | +0.0035 | −1.854 |
| FactorVAE | LightGBM | +0.0049 | −0.537 | −0.0008 | −0.641 |
| FactorVAE | CatBoost | +0.0119 | −0.906 | +0.0099 | −1.420 |
| FactorVAE | none (standalone) | +0.0117 | **−0.248** | +0.0057 | −0.944 |

**S&P 500, 2022–2025**

| Extractor | Model | Standard: Rank IC | Standard: Sharpe | Cross-sectional: Rank IC | Cross-sectional: Sharpe |
|---|---|---:|---:|---:|---:|
| Baseline | LightGBM | +0.0061 | −1.375 | +0.0044 | −2.120 |
| Baseline | CatBoost | +0.0018 | −5.071 | +0.0038 | −4.605 |
| PCA | LightGBM | +0.0049 | −4.331 | +0.0048 | −3.873 |
| PCA | CatBoost | +0.0021 | −3.845 | +0.0034 | −3.717 |
| FA | LightGBM | +0.0013 | −4.460 | +0.0034 | −4.524 |
| FA | CatBoost | −0.0010 | −4.866 | +0.0015 | −3.920 |
| GKX-AE | LightGBM | +0.0082 | −3.994 | +0.0017 | −5.121 |
| GKX-AE | CatBoost | +0.0022 | −5.386 | −0.0003 | −6.510 |
| GKX-AE | none (standalone) | −0.0018 | −4.225 | −0.0030 | −5.227 |
| FactorVAE | LightGBM | +0.0032 | −1.296 | +0.0066 | **−0.742**\* |
| FactorVAE | CatBoost | −0.0007 | −2.456 | −0.0001 | −2.485 |
| FactorVAE | none (standalone) | −0.0007 | −1.833 | −0.0017 | −1.049 |

\* This run does not fully reproduce. Its 2022 and 2023 prediction files were saved about two hours after its portfolio, by a later run. Rebuilding the portfolio from the saved predictions gives a Sharpe of −0.133 instead of −0.742. Treat it as unconfirmed until the run is repeated.

#### Main findings

- **No configuration is profitable after costs.** Every one of the 48 has a negative out-of-sample Sharpe ratio, so the comparison is about relative performance.
- **FactorVAE is the least unprofitable feature extractor in both universes.** The best ETF result is standalone FactorVAE with standard normalization (−0.248), and the best tree-based result in each universe uses FactorVAE with LightGBM.
- **Deep does not mean better.** The GKX autoencoder is among the weakest configurations and has the worst Sharpe in the study (−6.510, S&P 500, CatBoost, cross-sectional).
- **Rank IC is a poor guide to portfolio value.** On the ETFs, CatBoost gives every extractor a positive Rank IC, yet LightGBM has the better Sharpe for most extractors.
- **Normalization matters, and in different directions.** Cross-sectional normalization raises Rank IC in 8 of the 12 linear-method comparisons, but it often lowers the Sharpe on the 60-ETF universe, where the daily cross-section is small.

---

## 6. Setup, running and tests

### Setup

Python 3.12 is recommended.

```bash
git clone https://github.com/Giorgosmaridakis/Bachelor_Thesis.git
cd Bachelor_Thesis
python -m venv .venv
.venv\Scripts\activate          # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Running

```bash
cd Pipeline
python run_pipeline.py
```

- Choose what runs by editing `DEFAULT_EXPERIMENTS` in [experiment_config.py](Pipeline/experiment_config.py). As committed, it runs four S&P 500 neural + tree profiles.
- Those profiles need the saved standalone neural runs inside `Pipeline/`, so they cannot run from a fresh clone.
- The standalone neural runs (no tree) run only when named in `profiles`: `GKX-AE-ExpectedReturn-Standard`, `GKX-AE-ExpectedReturn-CrossSectional`, `FactorVAE-ExpectedReturn-Standard` and `FactorVAE-ExpectedReturn-CrossSectional`. The tree settings of that experiment entry are ignored for them.
- A GPU is used by default. Set `PIPELINE_USE_CUDA=0` to turn it off.

See [Pipeline/README.md](Pipeline/README.md) for the code structure, output files and the other documentation.

### Tests

```bash
cd Pipeline
pytest
```

There are 191 tests across 13 phases. With the data in place, all 191 pass. Without it, 67 pass and the 124 that need data are skipped, including every test in phases 2, 3, 5, 7 and 8.

---

## 7. Known limitations

- **Survivorship bias:** the S&P 500 universe is one fixed list of 483 tickers, so companies that left the index early are missing. See [data/README.md](Pipeline/data/README.md#known-limitations).
- **Dollar-neutral, not market-neutral:** the long and short books are equal in size, but some configurations still have significant market exposure.
- **Sharpe comparison test:** `sharpe_ratio_test` in [evaluation/statistical_tests.py](Pipeline/evaluation/statistical_tests.py) uses an incorrect variance formula and overstates significance. It was not used for the results in the thesis. Do not use it before correcting it, for example with the Memmel (2003) formula.
- **Training windows:** S&P 500 models effectively use a 2-year window, because the data starts in 2020 and the window is set once for the whole run. ETF models use 4 years. The saved standalone S&P 500 GKX runs are an exception: some years used a different window, for example 2021–2024 for the 2025 fold.
- **Trial budgets:** the LightGBM and ETF CatBoost runs used 40 trials per year, while the current code computes the budget with a square-root rule, so rerunning them gives a different number of trials.

---

## 8. Reproducibility

- **From saved files:** 47 of the 48 portfolios rebuild from their saved predictions to the exact Sharpe and IC. The exception is the run marked \* in the results.
- **By retraining:** results do not always match exactly. In checks on one GPU machine, CatBoost on saved neural outputs matched exactly, LightGBM refits were usually close but not exact, and FactorVAE standard-normalization folds on the S&P 500 differed noticeably, even between two repeats. The causes are not fully determined.

---

## 9. Hyperparameter search spaces

The grids are defined in [experiment_config.py](Pipeline/experiment_config.py). The four reduction methods share the same latent sizes, so they are compared at matched representation sizes.

| Component | Hyperparameter | Values |
|---|---|---|
| LightGBM | `learning_rate` | 0.01, 0.05, 0.1 |
| | `max_depth` | 3, 5, 7 |
| | `n_estimators` | 300, 500 |
| | `min_child_samples` | 25, 50, 100 |
| | `reg_lambda` | 3, 5, 10 |
| | `subsample` | 0.8, 1.0 |
| | `colsample_bytree` | 0.8, 1.0 |
| CatBoost | `learning_rate` | 0.01, 0.05, 0.1 |
| | `depth` | 4, 6 |
| | `iterations` | 300, 500 |
| | `l2_leaf_reg` | 3, 7, 10 |
| | `colsample_bylevel` | 1.0 |
| PCA / FA | `n_components` | 5, 8, 10, 12, 15, 20 |
| GKX-AE | `latent_dim` | 5, 8, 10, 12, 15, 20 |
| | `learning_rate` | 1e-5, 5e-5, 1e-4, 5e-4 |
| | `weight_decay` | 1e-6, 1e-5, 1e-4 |
| | `l1_lambda` | 0, 1e-5, 1e-4 |
| | max epochs / early-stop patience | 200 / 20 |
| FactorVAE | `seq_len` | 10, 20 |
| | `hidden_size` | 32, 64, 128 |
| | `num_factors` | 5, 8, 10, 12, 15, 20 |
| | `num_portfolios` | 8, 16, 32 |
| | `learning_rate` | 1e-5, 5e-5, 1e-4, 5e-4 |
| | `weight_decay` | 1e-6, 1e-5, 1e-4 |
| | max epochs / early-stop patience | 200 / 25 |

Grid sizes: LightGBM 648, CatBoost 36, GKX-AE 216 and FactorVAE 1,296 neural settings. The number of trials actually run per profile is in [Tuning_Architecture_README.md](Pipeline/Tuning_Architecture_README.md#4-trial-budget).

---

## 10. Output files

Each experiment writes two folders inside `Pipeline/`:

| Folder | Contents |
|---|---|
| `Results_Pipeline_<experiment>/` | One subfolder per test year with `predictions_<period>.csv`, daily IC, fold metrics, tuning results and, for neural runs, the selected parameters and model weights |
| `portfolio_metrics_<experiment>/` | `long_short_final_oos_metrics.csv`, the daily `long_short_time_series.csv`, vectorbt statistics, the executed weights `vbt_generated_positions.csv` and an interactive chart |

Running `python Backtester/build_annualized_metrics.py` adds `long_short_Annualized_oos_metric.csv` with calendar-year annualized return and volatility. The full file list is in [Pipeline/README.md](Pipeline/README.md#output-folders).

---

## 11. Further documentation

| Document | Covers |
|---|---|
| [Thesis (PDF)](docs/_BSc__Dimensionality_Reduction_as_Alpha__Comparing_Linear_and_Deep_Feature_Extractors_for_Tree_Based_Cross_Sectional_Return_Prediction.pdf) | The full Bachelor's thesis |
| [Presentation (PDF)](docs/presentation.pdf) | The defence slides |
| [Pipeline/README.md](Pipeline/README.md) | Code structure, prediction sources and output folders |
| [SIGNAL_TO_TRADES_README.md](Pipeline/SIGNAL_TO_TRADES_README.md) | How a prediction becomes portfolio weights and trades |
| [Tuning_Architecture_README.md](Pipeline/Tuning_Architecture_README.md) | Hyperparameter tuning, trial budgets and neural reuse |
| [data/README.md](Pipeline/data/README.md) | Data source, folder layout, target alignment and survivorship bias |
| [Backtester/README.md](Pipeline/Backtester/README.md) | Backtest logic and the 365-day annualization |
| [features/README.md](Pipeline/features/README.md) | Feature extraction and neural outputs |
| [features/GKX_VALIDATION_README.md](Pipeline/features/GKX_VALIDATION_README.md) | How GKX early stopping is validated |
| [modeling/README.md](Pipeline/modeling/README.md) | Tree models and tuning |
| [walking_forward/README.md](Pipeline/walking_forward/README.md) | The walk-forward modules |
| [evaluation/README.md](Pipeline/evaluation/README.md) | Metrics and statistical tests |

---

## 12. References

- Duan, Y., Wang, L., Zhang, Q., & Li, J. (2022). FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns. *AAAI*, 36(4), 4468–4476.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Gu, S., Kelly, B., & Xiu, D. (2021). Autoencoder Asset Pricing Models. *Journal of Econometrics*, 222(1), 429–450.
- Jansen, S. (2020). *Machine Learning for Algorithmic Trading* (2nd ed.). Packt. Code: [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading).
- Kakushadze, Z. (2016). 101 Formulaic Alphas. *Wilmott*, 2016(84), 72–81. arXiv:1601.00991.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Prokhorenkova, L. et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS*.

---

## 13. Citation

```bibtex
@thesis{maridakis2026dimensionality,
  author = {Maridakis, Georgios},
  title  = {Dimensionality Reduction as Alpha? Comparing Linear and Deep Feature Extractors for Tree-Based Cross-Sectional Return Prediction},
  type   = {Bachelor's thesis},
  school = {Department of Computer Science, University of Crete},
  year   = {2026}
}
```

---

## 14. License

[MIT](LICENSE)
