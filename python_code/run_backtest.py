"""Run the long-or-cash cost-sensitivity backtest on saved OOS predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting import backtest_long_or_cash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
COST_SCENARIOS_BPS_PER_SIDE = (0.0, 1.0, 2.5, 5.0, 10.0)


def main() -> None:
    # Reuse the saved out-of-sample forecasts; this script does not refit models.
    prediction_path = RESULTS_DIR / "advanced_model_predictions.csv"
    predictions = pd.read_csv(
        prediction_path,
        parse_dates=["datetime", "signal_datetime", "next_datetime", "entry_datetime", "exit_datetime"],
    )
    # Apply the same trading rule to every model under each per side cost case.
    summaries, daily_returns = backtest_long_or_cash(
        predictions,
        cost_bps_per_side=COST_SCENARIOS_BPS_PER_SIDE,
    )

    # Catch missing model or cost rows before replacing the saved result files.
    expected_models = {
        "Majority baseline", "Direction persistence baseline",
        "Logistic regression", "XGBoost", "LSTM"
    }
    expected_costs = set(COST_SCENARIOS_BPS_PER_SIDE)
    for ticker in ("QQQ", "SPY"):
        subset = summaries.loc[summaries["ETF"] == ticker]
        if set(subset["model"]) != expected_models:
            raise AssertionError(f"{ticker}: missing model strategy results")
        if set(subset["cost_bps_per_side"]) != expected_costs:
            raise AssertionError(f"{ticker}: missing transaction-cost scenarios")
    if summaries.duplicated(["ETF", "model", "cost_bps_per_side"]).any():
        raise AssertionError("Duplicate strategy/scenario summary rows")
    if len(summaries) != 2 * 5 * len(COST_SCENARIOS_BPS_PER_SIDE):
        raise AssertionError("Unexpected number of ETF/model/cost-scenario summaries")
    if not (summaries["entries"] == summaries["exits"]).all():
        raise AssertionError("Every one-bar long trade must have a matching exit")
    zero_cost = summaries.loc[summaries["cost_bps_per_side"] == 0].set_index(["ETF", "model"])
    if not (zero_cost["net_cumulative_return"].sub(zero_cost["gross_cumulative_return"]).abs() < 1e-12).all():
        raise AssertionError("Zero-cost net returns must equal gross returns")
    for _, model_scenarios in summaries.groupby(["ETF", "model"]):
        ordered = model_scenarios.sort_values("cost_bps_per_side")["net_cumulative_return"]
        if not ordered.is_monotonic_decreasing:
            raise AssertionError("Net cumulative return must not increase with higher costs")
    if daily_returns.duplicated(
        ["ETF", "model", "cost_bps_per_side", "session_date"]
    ).any():
        raise AssertionError("Duplicate strategy/scenario daily returns")

    summaries.to_csv(RESULTS_DIR / "long_or_cash_backtest_summary.csv", index=False)
    daily_returns.to_csv(RESULTS_DIR / "long_or_cash_backtest_daily_returns.csv", index=False)
    print("Scenario rates are illustrative all-in costs per side (basis points).")
    print("\nLong-or-cash results:")
    print(
        summaries.loc[summaries["cost_bps_per_side"].isin([0.0, 5.0, 10.0])]
        .sort_values(["ETF", "model", "cost_bps_per_side"])
        [[
            "ETF", "model", "cost_bps_per_side", "net_cumulative_return",
            "maximum_drawdown", "annualized_daily_sharpe", "entries", "exits",
        ]]
        .round(4)
        .to_string(index=False)
    )
    print("\nSaved scenario summary and daily return series under results/.")


if __name__ == "__main__":
    main()
