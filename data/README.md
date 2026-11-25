# Data files

The cleaned data files used by the project are included in this folder.

- `QQQ_clean_1min.csv` and `SPY_clean_1min.csv` are used in the first data exploration notebook.
- `QQQ_clean_5min.parquet` and `SPY_clean_5min.parquet` are used by the modelling and backtest notebooks and scripts.

The data is IEX based and does not include consolidated volume from all US exchanges. Scheduled bars with missing data are left missing and are not filled in.
