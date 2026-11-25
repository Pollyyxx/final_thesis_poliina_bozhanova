"""Long-or-cash backtesting on chronological out-of-sample predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_PREDICTION_COLUMNS = {
    "Majority baseline": "baseline_prediction",
    "Direction persistence baseline": "persistence_prediction",
    "Logistic regression": "logistic_prediction",
    "XGBoost": "xgboost_prediction",
    "LSTM": "lstm_prediction",
}


def _daily_returns(intervals: pd.DataFrame, return_column: str) -> pd.DataFrame:
    # Compound five-minute returns within each session before summarizing days.
    return (
        intervals.groupby("session_date", sort=True)[return_column]
        .apply(lambda values: (1 + values).prod() - 1)
        .rename(return_column)
        .reset_index()
    )


def _performance_summary(daily: pd.DataFrame) -> dict[str, float]:
    # Risk and return metrics are calculated from the daily net strategy series.
    net = daily["net_return"].to_numpy(dtype=float)
    equity = np.cumprod(1 + net)
    running_peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    standard_deviation = daily["net_return"].std(ddof=1)
    sharpe = (
        daily["net_return"].mean() / standard_deviation * np.sqrt(252)
        if len(daily) > 1 and standard_deviation > 0
        else np.nan
    )
    return {
        "sessions": int(len(daily)),
        "gross_cumulative_return": float((1 + daily["gross_return"]).prod() - 1),
        "net_cumulative_return": float(equity[-1] - 1),
        "maximum_drawdown": float(np.min(equity / running_peak - 1)),
        "annualized_daily_sharpe": float(sharpe),
    }


def backtest_long_or_cash(
    predictions: pd.DataFrame,
    cost_bps_per_side=(0.0, 1.0, 2.5, 5.0, 10.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate long-or-cash positions entered at the next bar's open.

    Each positive forecast is a separate trade: enter at the next bar open and
    exit at that bar close. Per side costs apply to every entry and exit.
    """
    # Check the saved execution prices and timestamps before calculating returns.
    required = {
        "ETF", "datetime", "signal_datetime", "next_datetime",
        "entry_datetime", "exit_datetime", "next_open", "next_close",
        "next_return", "actual",
        *MODEL_PREDICTION_COLUMNS.values(),
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")
    if not cost_bps_per_side or any(cost < 0 for cost in cost_bps_per_side):
        raise ValueError("cost_bps_per_side must contain non-negative values")

    data = predictions.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    for column in ("signal_datetime", "next_datetime", "entry_datetime", "exit_datetime"):
        data[column] = pd.to_datetime(data[column], errors="raise")
    data["next_return"] = pd.to_numeric(data["next_return"], errors="raise")
    data["next_open"] = pd.to_numeric(data["next_open"], errors="raise")
    data["next_close"] = pd.to_numeric(data["next_close"], errors="raise")
    data = data.sort_values(["ETF", "datetime"], kind="stable").reset_index(drop=True)
    if data.duplicated(["ETF", "datetime"]).any():
        raise ValueError("Each ETF must have at most one prediction per timestamp")
    if not np.isfinite(data[["next_open", "next_close", "next_return"]].to_numpy()).all():
        raise ValueError("Target-bar prices and returns must be finite")
    if (data["next_open"] <= 0).any() or (data["next_close"] <= 0).any():
        raise ValueError("Target-bar prices must be positive")
    if (data["next_return"] <= -1).any():
        raise ValueError("next_return must be greater than -100%")
    calculated_return = data["next_close"].div(data["next_open"]) - 1
    if not np.allclose(calculated_return, data["next_return"], rtol=0, atol=1e-12):
        raise AssertionError("next_return must equal next-bar close divided by next-bar open")
    if not np.array_equal((data["next_return"] > 0).astype(int), data["actual"].astype(int)):
        raise AssertionError("Saved direction labels do not match realized next-bar returns")
    if not (data["signal_datetime"] - data["datetime"]).eq(pd.Timedelta(minutes=5)).all():
        raise AssertionError("Signals must be available at the current bar close")
    if not (data["next_datetime"] - data["signal_datetime"]).eq(pd.Timedelta(0)).all():
        raise AssertionError("The next bar must begin when the current bar closes")
    if not data["entry_datetime"].eq(data["next_datetime"]).all():
        raise AssertionError("Trades must enter at the next bar open")
    if not (data["exit_datetime"] - data["entry_datetime"]).eq(pd.Timedelta(minutes=5)).all():
        raise AssertionError("Trades must exit at the close of the five-minute target bar")
    if not (data["datetime"].dt.date == data["next_datetime"].dt.date).all():
        raise AssertionError("Backtest targets must not cross session dates")

    summaries = []
    daily_parts = []
    for ticker, ticker_data in data.groupby("ETF", sort=True):
        ticker_data = ticker_data.sort_values("datetime", kind="stable").copy()
        session_dates = ticker_data["datetime"].dt.date
        for model_name, prediction_column in MODEL_PREDICTION_COLUMNS.items():
            positions = ticker_data[prediction_column].astype(int)
            if not positions.isin([0, 1]).all():
                raise ValueError(f"{ticker}/{model_name} predictions must be binary")

            # Every positive signal opens and closes one separate bar trade.
            entry_turnover = positions
            exit_turnover = positions
            turnover = entry_turnover + exit_turnover
            gross_strategy_return = positions.to_numpy() * ticker_data["next_return"].to_numpy()

            # Costs are an explicit scenario input, not a claim about actual brokerage fees.
            for cost_bps in cost_bps_per_side:
                cost_fraction = float(cost_bps) / 10_000
                entry_cost = entry_turnover.to_numpy() * cost_fraction
                exit_cost = exit_turnover.to_numpy() * cost_fraction
                intervals = pd.DataFrame({
                    "ETF": ticker,
                    "model": model_name,
                    "datetime": ticker_data["datetime"].to_numpy(),
                    "signal_datetime": ticker_data["signal_datetime"].to_numpy(),
                    "next_datetime": ticker_data["next_datetime"].to_numpy(),
                    "entry_datetime": ticker_data["entry_datetime"].to_numpy(),
                    "exit_datetime": ticker_data["exit_datetime"].to_numpy(),
                    "entry_price": ticker_data["next_open"].to_numpy(),
                    "exit_price": ticker_data["next_close"].to_numpy(),
                    "session_date": session_dates.to_numpy(),
                    "position": positions.to_numpy(),
                    "asset_return": ticker_data["next_return"].to_numpy(),
                    "gross_return": gross_strategy_return,
                    "entry_turnover": entry_turnover.to_numpy(),
                    "exit_turnover": exit_turnover.to_numpy(),
                    "one_way_turnover": turnover.to_numpy(),
                    "entry_cost": entry_cost,
                    "exit_cost": exit_cost,
                })
                # Apply one fee at entry and one at exit for each positive signal.
                intervals["net_return"] = (
                    (1 - intervals["entry_cost"])
                    * (1 + intervals["gross_return"])
                    * (1 - intervals["exit_cost"])
                    - 1
                )
                daily = _daily_returns(intervals, "gross_return").merge(
                    _daily_returns(intervals, "net_return"), on="session_date", validate="one_to_one"
                )
                daily.insert(0, "ETF", ticker)
                daily.insert(1, "model", model_name)
                daily.insert(2, "cost_bps_per_side", float(cost_bps))
                daily_parts.append(daily)

                summary = {
                    "ETF": ticker,
                    "model": model_name,
                    "cost_bps_per_side": float(cost_bps),
                    "intervals": int(len(intervals)),
                    "active_intervals": int(positions.sum()),
                    "exposure_fraction": float(positions.mean()),
                    "entries": int(positions.sum()),
                    "exits": int(positions.sum()),
                    "trade_events": int(turnover.sum()),
                    "one_way_turnover": float(turnover.sum()),
                    "entries_per_session": float(
                        positions.sum() / session_dates.nunique()
                    ),
                    "cumulative_cost_fraction_uncompounded": float(
                        intervals["entry_cost"].sum() + intervals["exit_cost"].sum()
                    ),
                    **_performance_summary(daily),
                }
                summaries.append(summary)

    return pd.DataFrame(summaries), pd.concat(daily_parts, ignore_index=True)
