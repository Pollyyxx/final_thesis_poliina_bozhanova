# Intraday Equity Forecasting

## Project overview

This project studies whether machine learning models can predict short term price direction for SPY and QQQ. SPY tracks the S&P 500, and QQQ tracks the Nasdaq-100. The project also tests a simple long or cash trading strategy and estimates how trading costs affect its results.

## Research question

Can the models predict whether the next five minute bar will close above or below its open for SPY and QQQ? How does a strategy based on those predictions perform after estimated trading costs and slippage?

## Data

The analysis uses five minute open, high, low, close, and volume data for SPY and QQQ. The cleaned CSV and Parquet files are included in the `data/` folder. See `data/README.md` for details.

## Models

The project compares these models:

- Majority class baseline
- Direction persistence baseline
- Logistic Regression
- XGBoost
- Long Short Term Memory (LSTM)

## Method

The notebooks clean and explore the data, create features, train and evaluate the models, and test the trading strategy. Model evaluation uses expanding window walk forward folds to keep the time order of the observations.

A forecast is made after a five minute bar has finished. The backtest assumes a trade at the next bar's open and exit at that bar's close. It tests a long or cash strategy and includes estimated transaction costs and slippage.

## Evaluation

Prediction results are measured with directional accuracy, F1-score, and ROC AUC. The trading strategy is reviewed using cumulative net return, maximum drawdown, Sharpe ratio, number of trades, and turnover.

## Project files

- `01_data_availability_and_exploration.ipynb`: initial data checks and exploration
- `notebooks/`: feature engineering, model training, backtesting, and results review
- `python_code/`: Python scripts used for modelling and analysis
- `data/`: cleaned input data files
- `results/`: saved model predictions and analysis results
- `docs/`: project notes and methodology

## Running the analysis

Use Python 3.13 and install the packages listed in `requirements.txt`. The required five minute Parquet files are in `data/`.

From the project folder, run these scripts in order:

1. `python python_code/run_advanced_models.py`
2. `python python_code/run_backtest.py`
3. `python python_code/run_indicator_ablation.py`
4. `python python_code/run_robustness_checks.py`

## Limitations

The trading costs are estimates. The source data does not include quotes or actual trade fills, so the backtest cannot show how a live strategy would perform. The walk forward folds have already been used to review the models, so they should not be treated as a completely untouched final test set.
