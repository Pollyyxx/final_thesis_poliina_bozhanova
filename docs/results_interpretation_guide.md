# Results interpretation guide

This guide summarizes the main results and where to find them.

## Main questions

- How much better are the models than the majority baseline?
- Do results stay similar across the five chronological folds?
- Do the predictive results lead to positive returns after the per trade costs?
- How do turnover and drawdown affect the business decision?
- What do the data source, bar level execution assumption, illustrative costs, and reviewed folds limit?

## Where to look

- results/advanced_model_fold_metrics.csv: predictive performance by fold.
- results/long_or_cash_backtest_summary.csv: full period returns and trading activity.
- results/robustness_backtest_stability.csv: fold level return ranges.
- results/indicator_ablation_deltas.csv: matched comparison of base and indicator features.
- notebooks/08_results_review.ipynb: combined tables and figures.

Each positive forecast is one trade from the next bar open to its close. Costs are charged per side at both entry and exit. The cost rates are illustrative, not measured broker costs. The five evaluation folds have already been reviewed, so they are not an untouched final test.
