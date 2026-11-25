"""Run a matched base-versus-technical-indicator model ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# Import advanced_models before feature data work so XGBoost loads before Torch.
from advanced_models import evaluate_xgb_lstm
from backtesting import backtest_long_or_cash
from feature_engineering import (
    BASE_FEATURE_COLUMNS,
    INDICATOR_FEATURE_COLUMNS,
    add_technical_indicators,
    build_model_frame,
    select_scheduled_bars,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TICKERS = ("QQQ", "SPY")
MODEL_NAMES = (
    "Majority baseline", "Direction persistence baseline",
    "Logistic regression", "XGBoost", "LSTM",
)
METRICS = ("accuracy", "balanced_accuracy", "f1_up", "roc_auc")


def assert_future_invariance(bars: pd.DataFrame) -> None:
    """Changing future bars must not change indicators already observed."""
    day = bars.loc[bars["datetime"].dt.date == bars["datetime"].dt.date.iloc[-1]].copy()
    if len(day) < 40:
        raise AssertionError("Not enough bars for the future-invariance check")
    cut = len(day) // 2
    before = add_technical_indicators(day)
    changed = day.copy()
    future = changed.index[cut:]
    for column in ("Open", "High", "Low", "Close", "Volume"):
        changed[column] = changed[column].astype(float)
    for column in ("Open", "High", "Low", "Close"):
        changed.loc[future, column] = changed.loc[future, column].to_numpy() * 1.03
    # Volume changes too; VWAP at earlier timestamps must remain unchanged.
    changed.loc[future, "Volume"] = changed.loc[future, "Volume"].to_numpy() * 1.7
    after = add_technical_indicators(changed)
    for column in INDICATOR_FEATURE_COLUMNS:
        if not np.allclose(
            before.loc[:cut - 1, column].to_numpy(dtype=float),
            after.loc[:cut - 1, column].to_numpy(dtype=float),
            equal_nan=True,
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError(f"Future data changed earlier values of {column}")


def main() -> None:
    print("Preparing matched indicator ablation; five chronological folds.", flush=True)
    source_frames = {
        ticker: pd.read_parquet(DATA / f"{ticker}_clean_5min.parquet")
        for ticker in TICKERS
    }
    all_dates = [frame["datetime"].min().date() for frame in source_frames.values()]
    all_ends = [frame["datetime"].max().date() for frame in source_frames.values()]
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=min(all_dates), end_date=max(all_ends)
    )

    metric_frames, prediction_frames = [], []
    for ticker, raw in source_frames.items():
        bars = select_scheduled_bars(raw, schedule)
        assert_future_invariance(bars)
        # Build both feature sets from the same session-filtered bars.
        base = build_model_frame(bars)
        enriched = add_technical_indicators(base)

        # Remove warm-up rows from both feature sets so folds and timestamps match.
        common = enriched.dropna(
            subset=[*BASE_FEATURE_COLUMNS, *INDICATOR_FEATURE_COLUMNS, "target", "next_return"]
        ).copy()
        feature_sets = {
            "Base features": BASE_FEATURE_COLUMNS,
            "+ technical indicators": (*BASE_FEATURE_COLUMNS, *INDICATOR_FEATURE_COLUMNS),
        }
        outputs = {}
        for feature_set, columns in feature_sets.items():
            metrics, predictions = evaluate_xgb_lstm(
                common,
                columns,
                n_splits=5,
                initial_train_fraction=0.60,
                sequence_length=12,
                lstm_epochs=8,
                random_state=42,
            )
            metrics.insert(0, "feature_set", feature_set)
            metrics.insert(0, "ETF", ticker)
            predictions.insert(0, "feature_set", feature_set)
            predictions.insert(0, "ETF", ticker)
            metric_frames.append(metrics)
            prediction_frames.append(predictions)
            outputs[feature_set] = predictions
            print(
                f"{ticker} / {feature_set}: {len(predictions):,} OOS rows; "
                f"{metrics['fold'].nunique()} folds",
                flush=True,
            )

        left, right = (outputs[name] for name in feature_sets)
        identity = [
            "datetime", "signal_datetime", "next_datetime", "next_open", "next_close",
            "actual", "next_return",
        ]
        if not left[identity].reset_index(drop=True).equals(
            right[identity].reset_index(drop=True)
        ):
            raise AssertionError(f"{ticker}: base and indicator predictions are not aligned")

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = (
        all_metrics.groupby(["ETF", "feature_set", "model"], sort=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    # Subtract base scores from indicator scores on the same matched sample.
    baseline = summary.loc[summary["feature_set"] == "Base features"].drop(
        columns="feature_set"
    )
    augmented = summary.loc[summary["feature_set"] == "+ technical indicators"].drop(
        columns="feature_set"
    )
    delta = augmented.merge(baseline, on=["ETF", "model"], suffixes=("_indicators", "_base"))
    for metric in METRICS:
        delta[f"{metric}_delta"] = delta[f"{metric}_indicators"] - delta[f"{metric}_base"]
    delta.insert(1, "comparison", "+ technical indicators minus base")

    # Backtest both feature sets on the same shared out-of-sample intervals.
    backtest_input = all_predictions.drop(columns=["feature_set"])
    # Backtest interface expects one ETF per group. Encode feature set in a temporary key.
    backtest_input["ETF"] = (
        all_predictions["ETF"] + " | " + all_predictions["feature_set"]
    )
    bt_summary, bt_daily = backtest_long_or_cash(backtest_input)
    bt_summary[["ETF", "feature_set"]] = bt_summary["ETF"].str.split(" | ", regex=False, expand=True)
    bt_daily[["ETF", "feature_set"]] = bt_daily["ETF"].str.split(" | ", regex=False, expand=True)

    # Basic artifact integrity checks.
    expected = set(MODEL_NAMES)
    if set(all_metrics["model"]) != expected:
        raise AssertionError("Unexpected model set in ablation results")
    if all_predictions.duplicated(["ETF", "feature_set", "datetime"]).any():
        raise AssertionError("Duplicate ETF/feature-set/timestamp predictions")
    if summary.empty or delta.empty or bt_summary.empty:
        raise AssertionError("One or more ablation output tables are empty")

    # Keep the paired predictions and all summary tables for later review.
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(RESULTS / "indicator_ablation_fold_metrics.csv", index=False)
    all_predictions.to_csv(RESULTS / "indicator_ablation_predictions.csv", index=False)
    summary.to_csv(RESULTS / "indicator_ablation_summary.csv", index=False)
    delta.to_csv(RESULTS / "indicator_ablation_deltas.csv", index=False)
    bt_summary.to_csv(RESULTS / "indicator_ablation_backtest_summary.csv", index=False)
    bt_daily.to_csv(RESULTS / "indicator_ablation_backtest_daily_returns.csv", index=False)

    print("\nMean fold metrics:", flush=True)
    print(summary.round(4).to_string(index=False), flush=True)
    print("\nXGBoost backtest sensitivity:", flush=True)
    xgb_bt = bt_summary.loc[bt_summary["model"] == "XGBoost"]
    print(
        xgb_bt[["ETF", "feature_set", "cost_bps_per_side", "net_cumulative_return", "annualized_daily_sharpe"]]
        .round(4).to_string(index=False),
        flush=True,
    )
    print("\nSaved indicator ablation metrics, predictions, and matched backtests.", flush=True)


if __name__ == "__main__":
    main()
