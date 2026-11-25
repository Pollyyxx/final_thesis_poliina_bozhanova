"""Chronological expanding-window classification evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def expanding_date_splits(dates, initial_train_fraction=0.60, n_splits=5):
    """Return expanding train-date sets and later, non-overlapping test blocks."""
    # Sort unique dates so each whole session stays on one side of a split.
    ordered_dates = sorted(pd.unique(pd.Series(dates).dropna()))
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between 0 and 1")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    initial_count = int(len(ordered_dates) * initial_train_fraction)
    if initial_count < 1 or len(ordered_dates) - initial_count < n_splits:
        raise ValueError("Not enough unique dates for the requested split settings")

    # The first 60% of dates form the initial training window. Split the later
    # dates into test blocks; each fold trains on all dates before its test block.
    remaining = ordered_dates[initial_count:]
    blocks = [block.tolist() for block in np.array_split(remaining, n_splits)]
    splits = []
    for block in blocks:
        if not block:
            continue
        block_start = ordered_dates.index(block[0])
        train_dates = ordered_dates[:block_start]
        if max(train_dates) >= min(block):
            raise AssertionError("Training dates must precede test dates")
        splits.append((train_dates, block))
    return splits


def evaluate_expanding_window(
    frame: pd.DataFrame,
    feature_columns,
    n_splits=5,
    initial_train_fraction=0.60,
):
    """Compare majority, direction-persistence, and logistic-regression predictions out of sample."""
    required = {"datetime", "target", "return_1_bar", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")

    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    data["session_date"] = data["datetime"].dt.date
    # Rows without a complete input or valid next-bar label cannot be scored.
    eligible = data.dropna(subset=[*feature_columns, "target"]).copy()
    eligible["target"] = eligible["target"].astype(int)
    if not set(eligible["target"].unique()).issubset({0, 1}):
        raise ValueError("target must contain only binary labels 0 and 1")

    splits = expanding_date_splits(
        eligible["session_date"].unique(),
        initial_train_fraction=initial_train_fraction,
        n_splits=n_splits,
    )
    metric_rows = []
    prediction_parts = []

    for fold, (train_dates, test_dates) in enumerate(splits, start=1):
        train = eligible.loc[eligible["session_date"].isin(train_dates)]
        test = eligible.loc[eligible["session_date"].isin(test_dates)]
        if train.empty or test.empty:
            raise ValueError(f"Fold {fold} has an empty train or test partition")
        if max(train["session_date"]) >= min(test["session_date"]):
            raise AssertionError(f"Fold {fold} is not chronological")
        if train["target"].nunique() < 2:
            raise ValueError(f"Fold {fold} training labels contain only one class")

        x_train = train[list(feature_columns)]
        y_train = train["target"].astype(int)
        x_test = test[list(feature_columns)]
        y_test = test["target"].astype(int)
        # The majority forecast uses training labels only; persistence instead
        # carries the latest completed direction forward for each test row.
        majority_class = int(y_train.mode().iloc[0])
        baseline_prediction = np.full(len(test), majority_class, dtype=int)
        # This is a small time-series idea: use the last completed direction as
        # the forecast for the next bar. It is different from the training majority.
        persistence_prediction = (test["return_1_bar"].to_numpy() > 0).astype(int)

        # Scaling is fit only on this fold's training rows.
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs"),
        )
        classifier.fit(x_train, y_train)
        logistic_prediction = classifier.predict(x_test).astype(int)
        logistic_probability = classifier.predict_proba(x_test)[:, 1]

        # Save fold boundaries with each score so results can be traced to dates.
        common = {
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": min(train["session_date"]),
            "train_end": max(train["session_date"]),
            "test_start": min(test["session_date"]),
            "test_end": max(test["session_date"]),
        }
        for model_name, prediction in [
            ("Majority baseline", baseline_prediction),
            ("Direction persistence baseline", persistence_prediction),
            ("Logistic regression", logistic_prediction),
        ]:
            metric_rows.append({
                **common,
                "model": model_name,
                "accuracy": accuracy_score(y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "f1_up": f1_score(y_test, prediction, pos_label=1, zero_division=0),
                "roc_auc": (
                    roc_auc_score(y_test, logistic_probability)
                    if model_name == "Logistic regression" and y_test.nunique() == 2
                    else np.nan
                ),
            })

        prediction_parts.append(pd.DataFrame({
            "datetime": test["datetime"].to_numpy(),
            "signal_datetime": test["signal_datetime"].to_numpy(),
            "next_datetime": test["next_datetime"].to_numpy(),
            "session_date": test["session_date"].to_numpy(),
            "actual": y_test.to_numpy(),
            "baseline_prediction": baseline_prediction,
            "persistence_prediction": persistence_prediction,
            "logistic_prediction": logistic_prediction,
            "logistic_probability_up": logistic_probability,
        }))

    return pd.DataFrame(metric_rows), pd.concat(prediction_parts, ignore_index=True)
