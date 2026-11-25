# Data files

Place the data files in this folder.

- `QQQ_clean_1min.csv` and `SPY_clean_1min.csv` are used in the first data exploration notebook.
- `QQQ_clean_5min.parquet` and `SPY_clean_5min.parquet` are used by the modelling and backtest notebooks and scripts.

The files currently cover 7 March 2022 to 16 October 2026. The data is IEX based and does not include consolidated volume from all US exchanges. Scheduled bars with missing data are left missing and are not filled in.

The data files are kept locally and excluded from Git. Someone who clones the project will need to obtain the data separately. Keep the original files unchanged and check the provider's licence before sharing them.
