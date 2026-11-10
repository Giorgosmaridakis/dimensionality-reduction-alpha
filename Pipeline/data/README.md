# Data

This folder loads raw alpha/price files and builds the panel used by `run_pipeline.py`.

## Where It Enters From `run_pipeline.py`

`run_pipeline.py` starts here:

1. `DataLoader(universe='ETFs')`
2. `dl.get_alpha_files()`
3. `dl.get_adjusted_close()`
4. `DatasetBuilder.prepare_stacked_data(dl)`
5. `DatasetBuilder.merge_features_and_targets(...)`

The output is `full_dataset`, which then flows into the walk-forward evaluator.

## Main Files

- `data_loader.py` knows where the universe folders live and loads alpha files plus adjusted-close prices.
- `dataset_builder.py` converts wide alpha CSVs into a long panel and joins future returns as `Target`.

## Data Shape

The stacked feature panel uses one row per:

```text
Date, ticker
```

with columns like:

```text
alpha001, alpha002, ..., Target
```

## Target Alignment

The project now uses adjusted-close execution only. Because the backtester shifts target weights forward one row, the target for signal date `t` is:

```text
Target_t = Adjusted_Close_{t+2} / Adjusted_Close_{t+1} - 1
```

Meaning:

- alpha features dated `t` are assumed known after `close_t`,
- the trade is entered at `adjusted_close_{t+1}`,
- the realized holding return is from `adjusted_close_{t+1}` to `adjusted_close_{t+2}`.

## Source

All prices come from Yahoo Finance through the `yfinance` Python library. The ticker lists are in `tickers/`:

- `tickers/etf_tickers.csv`: the 60 ETFs.
- `tickers/sp500_tickers.csv`: the 483 S&P 500 constituents.

The alpha signals are derived from the *101 Formulaic Alphas* (Kakushadze, 2016) and computed following Jansen (2020). There are 82 alpha files, but `alpha068` and `alpha086` are constant in both universes, so 80 carry information.

## Expected Folder Layout

The data is not in this repository. `DataLoader` looks for it next to `Pipeline/`:

```text
Bachelor_Thesis/
├── Pipeline/
├── ETFs/
│   ├── 2006-2022/
│   │   ├── Adjusted_Close.csv
│   │   └── zuras/alpha001.csv, alpha002.csv, ...
│   └── 2023-2025/  (same contents)
└── S&P500/
    ├── 2020-2024/  (same contents)
    └── 2025-End/   (same contents)
```

- `Adjusted_Close.csv` is wide: a `Date` column plus one column per ticker.
- Each `alpha*.csv` file is a wide table of one alpha signal per date and ticker.
- The alpha files must already exist. This repository does not contain code that computes them.

## Things To Watch

- `prepare_stacked_data(...)` inner-joins alpha features, so a row survives only if every selected alpha exists.
- `merge_features_and_targets(...)` drops rows without a future tradable target.
- `get_adjusted_close()` is the only price loader used by the current backtest path.

## Known Limitations

**Survivorship bias in the S&P 500 universe.** The S&P 500 data uses one fixed list of 483 tickers for the whole sample, not the index membership at each date. Every ticker has prices for at least 99% of the sample, and none stops trading early. Companies that left the index or were delisted during the sample (for example SIVB, FRC and TWTR) are missing, and companies that joined later can appear before they joined. Backtest results therefore describe this fixed set of surviving companies and may not match what could have been traded at the time. They should not be read as the performance of a strategy on the historical S&P 500.
