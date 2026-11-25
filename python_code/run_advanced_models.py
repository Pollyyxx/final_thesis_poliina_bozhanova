"""Run the fixed XGBoost/LSTM comparison and save auditable predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from advanced_models import evaluate_xgb_lstm
from feature_engineering import FEATURE_COLUMNS, build_model_frame, select_scheduled_bars
import torch
import xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    print(f"XGBoost {xgboost.__version__}; PyTorch {torch.__version__}", flush=True)
    # Read the local cleaned inputs; these data files are intentionally not in Git.
    sources = {
        ticker: pd.read_parquet(DATA_DIR / f"{ticker}_clean_5min.parquet")
        for ticker in ("QQQ", "SPY")
    }
    start_date = min(frame["datetime"].min().date() for frame in sources.values())
    end_date = max(frame["datetime"].max().date() for frame in sources.values())
    # Use the exchange calendar so early closes and holidays are handled correctly.
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start_date, end_date=end_date)

    # Keep model metrics and predictions together by ETF before writing results.
    metric_frames = []
    prediction_frames = []
    for ticker, source in sources.items():
        model_frame = build_model_frame(select_scheduled_bars(source, schedule))
        metrics, predictions = evaluate_xgb_lstm(
            model_frame,
            FEATURE_COLUMNS,
            n_splits=5,
            initial_train_fraction=0.60,
            sequence_length=12,
            lstm_epochs=8,
            random_state=42,
        )
        metrics.insert(0, "ETF", ticker)
        predictions.insert(0, "ETF", ticker)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        print(
            f"{ticker}: {metrics['fold'].nunique()} folds; "
            f"{len(predictions):,} shared out-of-sample timestamps",
            flush=True,
        )

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)

    required_models = {
        "Majority baseline", "Direction persistence baseline",
        "Logistic regression", "XGBoost", "LSTM",
    }
    for ticker in ("QQQ", "SPY"):
        ticker_metrics = all_metrics.loc[all_metrics["ETF"] == ticker]
        if set(ticker_metrics["model"]) != required_models:
            raise AssertionError(f"{ticker}: the expected model and baseline set was not evaluated")
        if set(ticker_metrics["fold"]) != {1, 2, 3, 4, 5}:
            raise AssertionError(f"{ticker}: expected five chronological folds")
        ticker_predictions = all_predictions.loc[all_predictions["ETF"] == ticker]
        if ticker_predictions["datetime"].duplicated().any():
            raise AssertionError(f"{ticker}: duplicate out-of-sample timestamps")
        if ticker_predictions.isna().any().any():
            raise AssertionError(f"{ticker}: missing values in saved predictions")

    # Save fold-level scores and shared predictions for the review notebooks.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(RESULTS_DIR / "advanced_model_fold_metrics.csv", index=False)
    all_predictions.to_csv(RESULTS_DIR / "advanced_model_predictions.csv", index=False)

    print("\nMean fold metrics:", flush=True)
    summary = (
        all_metrics.groupby(["ETF", "model"])[
            ["accuracy", "balanced_accuracy", "f1_up", "roc_auc"]
        ]
        .mean()
        .round(4)
    )
    print(summary.to_string(), flush=True)
    print("\nSaved fold metrics and shared-timestamp predictions under results/.", flush=True)


if __name__ == "__main__":
    main()
