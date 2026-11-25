# Python Code

Reusable Python code for the thesis project.

## Current modules

- `feature_engineering.py`: validation of source OHLCV bars, regular session selection, next bar open to close target construction, and leakage aware five minute predictors.
- `model_evaluation.py`: chronological expanding date splits and out of sample comparison of majority and direction persistence baselines with Logistic Regression.
- `advanced_models.py`: shared timestamp evaluation of two simple baselines, Logistic Regression, XGBoost and LSTM models with training only scaling.
- `run_advanced_models.py`: reproducible five fold model run; checks fold coverage and prediction integrity, then saves metrics and predictions under `results/`.
- `backtesting.py`: one bar long or cash trades with explicit signal, entry and exit times and per side cost scenarios.
- `run_backtest.py`: validates saved predictions and runs the backtest without importing the model training libraries.
- `run_indicator_ablation.py`: compares base predictors with a fixed set of technical indicators on identical rows and folds, checks future data invariance, and saves matched prediction and cost sensitivity outputs.
- `run_robustness_checks.py`: validates fold coverage and saves predictive metrics and flat start/flat end strategy results by fold, plus fold to fold return ranges.

## Planned components

- No additional code components are planned.
