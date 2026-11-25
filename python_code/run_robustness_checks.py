"""Summarise model and strategy stability across the saved chronological folds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting import backtest_long_or_cash


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_fold_assignments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read saved fold definitions and assign each prediction to one test fold."""
    folds = pd.read_csv(RESULTS / "advanced_model_fold_metrics.csv")
    predictions = pd.read_csv(RESULTS / "advanced_model_predictions.csv")
    predictions["datetime"] = pd.to_datetime(predictions["datetime"], errors="raise")

    # The fold file repeats the same test window for each model, so keep one
    # row per ETF and fold when checking the saved test-row counts.
    windows = folds[["ETF", "fold", "test_rows", "test_start", "test_end"]].drop_duplicates()
    if windows.duplicated(["ETF", "fold"]).any():
        raise ValueError("Each ETF/fold must have one unique saved test window")

    # Match each prediction date to one saved test window; ambiguous dates fail.
    assignments = []
    for ticker, ticker_rows in predictions.groupby("ETF", sort=True):
        ticker_windows = windows.loc[windows["ETF"].eq(ticker)].copy()
        if ticker_windows.empty:
            raise ValueError(f"No saved fold definitions for {ticker}")
        ticker_windows["test_start"] = pd.to_datetime(ticker_windows["test_start"]).dt.date
        ticker_windows["test_end"] = pd.to_datetime(ticker_windows["test_end"]).dt.date

        for row in ticker_rows.itertuples(index=False):
            test_date = row.datetime.date()
            matching = ticker_windows.loc[
                ticker_windows["test_start"].le(test_date)
                & ticker_windows["test_end"].ge(test_date)
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"{ticker} prediction on {test_date} maps to {len(matching)} folds"
                )
            assignments.append((ticker, row.datetime, int(matching.iloc[0]["fold"])))

    assigned = pd.DataFrame(assignments, columns=["ETF", "datetime", "fold"])
    data = predictions.merge(assigned, on=["ETF", "datetime"], validate="one_to_one")

    # Row counts must match the original fold definitions before backtesting.
    actual_counts = data.groupby(["ETF", "fold"]).size().rename("assigned_rows")
    expected_counts = windows.set_index(["ETF", "fold"])["test_rows"].astype(int)
    counts = actual_counts.to_frame().join(expected_counts.rename("expected_rows"), how="outer")
    if counts.isna().any().any() or not counts["assigned_rows"].eq(counts["expected_rows"]).all():
        raise AssertionError("Assigned prediction counts do not match the saved fold sizes")
    return folds, data


def main() -> None:
    folds, predictions = load_fold_assignments()

    # Keep the saved model metrics as the fold-by-fold predictive stability table.
    folds.sort_values(["ETF", "fold", "model"]).to_csv(
        RESULTS / "robustness_fold_metrics.csv", index=False
    )

    summaries = []
    daily_parts = []
    for (ticker, fold), fold_predictions in predictions.groupby(["ETF", "fold"], sort=True):
        # Each fold is evaluated as a separate portfolio. This intentionally
        # starts and ends flat, so fold returns are not compounded across folds.
        summary, daily = backtest_long_or_cash(fold_predictions)
        summary.insert(1, "fold", int(fold))
        daily.insert(1, "fold", int(fold))
        summaries.append(summary)
        daily_parts.append(daily)

    by_fold = pd.concat(summaries, ignore_index=True)
    by_fold.to_csv(RESULTS / "robustness_backtest_by_fold.csv", index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(
        RESULTS / "robustness_backtest_daily_by_fold.csv", index=False
    )

    # Summarize the fold range; do not compound these separate fold portfolios.
    grouped = by_fold.groupby(["ETF", "model", "cost_bps_per_side"], sort=True)
    stability = grouped.agg(
        folds=("fold", "nunique"),
        positive_return_folds=("net_cumulative_return", lambda values: int((values > 0).sum())),
        mean_net_cumulative_return=("net_cumulative_return", "mean"),
        median_net_cumulative_return=("net_cumulative_return", "median"),
        minimum_net_cumulative_return=("net_cumulative_return", "min"),
        maximum_net_cumulative_return=("net_cumulative_return", "max"),
        mean_maximum_drawdown=("maximum_drawdown", "mean"),
        minimum_annualized_daily_sharpe=("annualized_daily_sharpe", "min"),
        maximum_annualized_daily_sharpe=("annualized_daily_sharpe", "max"),
    ).reset_index()
    stability.to_csv(RESULTS / "robustness_backtest_stability.csv", index=False)

    print("Fold robustness outputs saved:")
    for filename in (
        "robustness_fold_metrics.csv",
        "robustness_backtest_by_fold.csv",
        "robustness_backtest_daily_by_fold.csv",
        "robustness_backtest_stability.csv",
    ):
        print(f"- results/{filename}")
    print(f"Assigned {len(predictions):,} predictions across {len(by_fold):,} fold/model/cost rows.")


if __name__ == "__main__":
    main()
