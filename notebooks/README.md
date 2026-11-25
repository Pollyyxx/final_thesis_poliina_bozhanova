# Notebooks

## Current notebooks

- `../01_data_availability_and_exploration.ipynb`: data availability, integrity checks, trading session alignment, and exploratory analysis.
- `02_feature_engineering.ipynb`: scheduled session data preparation, next bar open to close targets, leakage aware predictors, and a future data invariance check.
- `03_baseline_logistic_regression.ipynb`: majority and direction persistence baselines plus Logistic Regression, evaluated with expanding chronological windows.
- `04_xgboost_lstm.ipynb`: fixed settings comparison of two simple baselines, Logistic Regression, XGBoost and LSTM; it reads saved results and isolates model fitting in a separate process when results need to be generated.
- `05_long_or_cash_backtest.ipynb`: saved out of sample long or cash performance using next bar open entries and open to close returns, with per side cost sensitivity.
- `06_indicator_ablation.ipynb`: paired comparison of the base predictors with MACD, RSI, Bollinger bandwidth, and session VWAP distance, including matched cost sensitivity results.
- `07_robustness_checks.ipynb`: fold by fold predictive and strategy stability review using the existing out of sample predictions.
- `08_results_review.ipynb`: joint review of model metrics, baseline deltas, cost sensitive strategy results, indicator changes, and fold stability using saved outputs.

## Notebook figures

- Notebook 01: session completeness by missing bar count, return distribution, and intraday pattern.
- Notebook 02: balance of next bar direction labels for each ETF.
- Notebook 03: fold by fold baseline and Logistic Regression accuracy.
- Notebook 04: model score means and fold to fold spread.
- Notebook 05: cost sensitivity equity curves.
- Notebook 06: matched indicator changes in accuracy and ROC AUC.
- Notebook 07: fold by fold strategy returns.
- Notebook 08: entries per session and signal accuracy compared with trading returns at execution.

## Planned notebooks

- None currently. The available local history ends on 16 October 2026; the existing folds are used for walk forward evaluation and are not described as an untouched post selection test.
