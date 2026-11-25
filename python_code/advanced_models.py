"""Out-of-sample XGBoost and LSTM comparisons on shared test rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# On this macOS environment XGBoost and PyTorch load separate OpenMP runtimes.
# Import XGBoost first to avoid a native-library crash when both are used.
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_evaluation import expanding_date_splits


class _SequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16):
        # Keep this one-layer network small so it is easy to compare and inspect.
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values):
        sequence_output, _ = self.lstm(values)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)


def _make_sequences(frame, feature_columns, sequence_length):
    sequences, labels, timestamps = [], [], []
    # Build each window inside one session; a missing bar breaks continuity.
    for _, day in frame.groupby("session_date", sort=True):
        day = day.sort_values("datetime")
        values = day[list(feature_columns)].to_numpy(dtype=np.float32)
        targets = day["target"].to_numpy(dtype=np.int64)
        times = pd.to_datetime(day["datetime"]).to_numpy()
        for end in range(sequence_length - 1, len(day)):
            start = end - sequence_length + 1
            window_times = times[start:end + 1]
            if np.all(np.diff(window_times) == np.timedelta64(5, "m")):
                sequences.append(values[start:end + 1])
                labels.append(targets[end])
                timestamps.append(times[end])
    if not sequences:
        return (
            np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype="datetime64[ns]"),
        )
    return np.stack(sequences), np.asarray(labels), np.asarray(timestamps)


def _score(y_true, prediction, probability):
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "f1_up": f1_score(y_true, prediction, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability) if np.unique(y_true).size == 2 else np.nan,
    }


def _fit_lstm(x_train, y_train, x_test, seed, epochs, batch_size=512):
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("LSTM train and test sequence sets must be non-empty")
    torch.manual_seed(seed)
    # Small CPU batches run more efficiently with one PyTorch worker here.
    torch.set_num_threads(1)
    np.random.seed(seed)
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
    loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = _SequenceClassifier(input_size=x_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # This loss expects raw scores; sigmoid is applied later for probabilities.
    loss_function = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test))
        probability = torch.sigmoid(logits).numpy()
    return (probability >= 0.5).astype(int), probability


def evaluate_xgb_lstm(
    frame,
    feature_columns,
    n_splits=5,
    initial_train_fraction=0.60,
    sequence_length=12,
    lstm_epochs=8,
    random_state=42,
):
    """Evaluate two simple baselines, logistic regression, XGBoost, and LSTM on shared timestamps."""
    feature_columns = list(feature_columns)
    required = {"datetime", "target", "return_1_bar", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")

    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    data["session_date"] = data["datetime"].dt.date
    data = data.dropna(subset=[*feature_columns, "target"]).copy()
    data["target"] = data["target"].astype(int)
    # Reuse the same chronological date blocks used by the simple models.
    splits = expanding_date_splits(
        data["session_date"].unique(),
        initial_train_fraction=initial_train_fraction,
        n_splits=n_splits,
    )
    metrics, predictions = [], []

    for fold, (train_dates, test_dates) in enumerate(splits, start=1):
        train = data.loc[data["session_date"].isin(train_dates)].copy()
        test = data.loc[data["session_date"].isin(test_dates)].copy()
        if max(train["session_date"]) >= min(test["session_date"]):
            raise AssertionError(f"Fold {fold} is not chronological")
        if train["target"].nunique() < 2:
            raise ValueError(f"Fold {fold} training labels contain only one class")

        # Learn scaling from past data only and apply the same transform to test rows.
        scaler = StandardScaler().fit(train[feature_columns])
        x_train_scaled = scaler.transform(train[feature_columns]).astype(np.float32)
        x_test_scaled = scaler.transform(test[feature_columns]).astype(np.float32)
        train_scaled = train[["datetime", "session_date", "target"]].copy()
        test_scaled = test[["datetime", "session_date", "target"]].copy()
        train_scaled[feature_columns] = x_train_scaled
        test_scaled[feature_columns] = x_test_scaled
        x_train_seq, y_train_seq, _ = _make_sequences(
            train_scaled, feature_columns, sequence_length
        )
        x_test_seq, y_test_seq, test_timestamps = _make_sequences(
            test_scaled, feature_columns, sequence_length
        )
        if len(test_timestamps) == 0:
            raise ValueError(f"Fold {fold} has no complete LSTM test sequences")

        # All model metrics use the same test timestamps for a fair comparison.
        # LSTM needs 12 complete bars, so compare every model on those same rows.
        common_timestamps = pd.to_datetime(test_timestamps)
        common_test = (
            test.loc[test["datetime"].isin(common_timestamps)]
            .sort_values("datetime")
            .copy()
        )
        if len(common_test) != len(common_timestamps):
            raise AssertionError("Shared LSTM test timestamps could not be matched to test rows")
        y_test = common_test["target"].astype(int).to_numpy()
        y_train = train["target"].astype(int).to_numpy()
        common_test_scaled = scaler.transform(common_test[feature_columns]).astype(np.float32)
        majority_class = int(train["target"].mode().iloc[0])
        baseline_pred = np.full(len(common_test), majority_class, dtype=int)
        # This benchmark carries the latest completed close-to-close direction forward.
        persistence_pred = (common_test["return_1_bar"].to_numpy() > 0).astype(int)

        # Logistic regression uses the scaled inputs prepared from training data.
        logistic = LogisticRegression(max_iter=1000, solver="lbfgs")
        logistic.fit(x_train_scaled, y_train)
        logistic_pred = logistic.predict(common_test_scaled).astype(int)
        logistic_prob = logistic.predict_proba(common_test_scaled)[:, 1]

        # Trees use the original feature scale; keep settings fixed rather than tuning on test folds.
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=random_state + fold,
        )
        xgb.fit(train[feature_columns], y_train)
        xgb_pred = xgb.predict(common_test[feature_columns]).astype(int)
        xgb_prob = xgb.predict_proba(common_test[feature_columns])[:, 1]

        # The same fixed epoch count is used in each fold for a fair comparison.
        # The recurrent model receives ordered 12-bar windows from the same fold.
        lstm_pred, lstm_prob = _fit_lstm(
            x_train_seq,
            y_train_seq,
            x_test_seq,
            seed=random_state + fold,
            epochs=lstm_epochs,
        )
        if not np.array_equal(y_test_seq, y_test):
            raise AssertionError("LSTM labels do not align with shared test timestamps")

        fold_models = [
            ("Majority baseline", baseline_pred, None),
            ("Direction persistence baseline", persistence_pred, None),
            ("Logistic regression", logistic_pred, logistic_prob),
            ("XGBoost", xgb_pred, xgb_prob),
            ("LSTM", lstm_pred, lstm_prob),
        ]
        # Save each model on the shared test rows; baselines have no probability score.
        for name, prediction, probability in fold_models:
            scores = _score(y_test, prediction, probability if probability is not None else np.full(len(y_test), majority_class, dtype=float))
            if name in {"Majority baseline", "Direction persistence baseline"}:
                scores["roc_auc"] = np.nan
            metrics.append({
                "fold": fold,
                "model": name,
                "train_rows": len(train),
                "test_rows": len(common_test),
                "train_start": min(train["session_date"]),
                "train_end": max(train["session_date"]),
                "test_start": min(test["session_date"]),
                "test_end": max(test["session_date"]),
                **scores,
            })
        predictions.append(pd.DataFrame({
            "datetime": common_test["datetime"],
            "signal_datetime": common_test["signal_datetime"],
            "next_datetime": common_test["next_datetime"],
            "entry_datetime": common_test["next_datetime"],
            "exit_datetime": common_test["next_datetime"] + pd.Timedelta(minutes=5),
            "next_open": common_test["next_open"].astype(float).to_numpy(),
            "next_close": common_test["next_close"].astype(float).to_numpy(),
            "next_return": common_test["next_return"].astype(float).to_numpy(),
            "actual": y_test,
            "baseline_prediction": baseline_pred,
            "persistence_prediction": persistence_pred,
            "logistic_prediction": logistic_pred,
            "xgboost_prediction": xgb_pred,
            "xgboost_probability_up": xgb_prob,
            "lstm_prediction": lstm_pred,
            "lstm_probability_up": lstm_prob,
        }))

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)
