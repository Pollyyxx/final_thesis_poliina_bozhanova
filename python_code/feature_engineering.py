"""Leakage-aware preparation of five-minute SPY and QQQ model data."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLUMNS = ("datetime", "Open", "High", "Low", "Close", "Volume")
FEATURE_COLUMNS = (
    "return_1_bar",
    "return_3_bars",
    "return_6_bars",
    "realized_volatility_12_bars",
    "bar_range_pct",
    "close_location_in_bar",
    "log_volume",
    "log_volume_change_1_bar",
    "minutes_since_open",
    "time_of_day_sin",
    "time_of_day_cos",
)
BASE_FEATURE_COLUMNS = FEATURE_COLUMNS
INDICATOR_FEATURE_COLUMNS = (
    "macd_line_pct",
    "macd_histogram_pct",
    "rsi_14",
    "bollinger_bandwidth_20",
    "distance_from_session_vwap",
)
BAR_MINUTES = 5
TARGET_COLUMNS = (
    "signal_datetime", "next_datetime", "next_open", "next_close",
    "next_return", "target",
)


def _validated_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, validated bars without changing the input frame."""
    missing = set(BASE_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    if data["datetime"].isna().any():
        raise ValueError("datetime contains missing values")
    if data["datetime"].dt.tz is not None:
        raise ValueError("Expected timezone-naive America/New_York timestamps")

    data = data.sort_values("datetime", kind="stable").reset_index(drop=True)
    if data["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed")
    if data[list(BASE_COLUMNS[1:])].isna().any().any():
        raise ValueError("OHLCV columns contain missing values")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (data["Volume"] < 0).any():
        raise ValueError("Volume must be non-negative")
    if not (data["High"] >= data[["Open", "Close", "Low"]].max(axis=1)).all():
        raise ValueError("High is inconsistent with the OHLC values")
    if not (data["Low"] <= data[["Open", "Close", "High"]].min(axis=1)).all():
        raise ValueError("Low is inconsistent with the OHLC values")
    return data


def select_scheduled_bars(
    df: pd.DataFrame, schedule: pd.DataFrame, interval_minutes: int = 5
) -> pd.DataFrame:
    """Keep observed timestamps on scheduled regular-session bar starts."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    data = _validated_bars(df)
    # Build the expected local bar-start times from the actual exchange schedule.
    expected = set()
    for session_date, row in schedule.iterrows():
        market_open = row["market_open"].tz_convert("America/New_York").tz_localize(None)
        market_close = row["market_close"].tz_convert("America/New_York").tz_localize(None)
        expected.update(pd.date_range(
            start=market_open,
            end=market_close - pd.Timedelta(minutes=interval_minutes),
            freq=f"{interval_minutes}min",
        ))
    # Keep observed bars only; this filter never fills an absent timestamp.
    return data.loc[data["datetime"].isin(expected)].reset_index(drop=True)


def add_next_bar_target(df: pd.DataFrame) -> pd.DataFrame:
    """Predict the next bar's open-to-close direction within the same session.

    Predictors are calculated after the current bar is complete. Using the next
    bar's open as entry avoids assuming we can trade at the just-observed close.
    """
    data = _validated_bars(df)
    next_datetime = data["datetime"].shift(-1)
    next_open = data["Open"].shift(-1)
    next_close = data["Close"].shift(-1)
    # A target is valid only when the next row is the next five-minute bar today.
    same_date = data["datetime"].dt.date == next_datetime.dt.date
    consecutive = next_datetime.sub(data["datetime"]).eq(pd.Timedelta(minutes=5))
    valid = same_date & consecutive

    data["signal_datetime"] = data["datetime"] + pd.Timedelta(minutes=BAR_MINUTES)
    data["next_datetime"] = next_datetime.where(valid)
    data["next_open"] = next_open.where(valid)
    data["next_close"] = next_close.where(valid)
    # This return begins after the signal: enter at the next bar's open, then
    # hold through that bar until its close.
    data["next_return"] = (next_close.div(next_open) - 1).where(valid)
    data["target"] = (data["next_return"] > 0).where(valid).astype("Int64")
    return data


def add_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Add features available at each bar close, using only current/past bars."""
    data = _validated_bars(df)
    data["_session_date"] = data["datetime"].dt.date
    groups = data.groupby("_session_date", sort=False)

    # A lag is kept only when every intervening five-minute bar is present.
    for lag, name in ((1, "return_1_bar"), (3, "return_3_bars"), (6, "return_6_bars")):
        previous_close = groups["Close"].shift(lag)
        previous_datetime = groups["datetime"].shift(lag)
        exact_lookback = data["datetime"].sub(previous_datetime).eq(
            pd.Timedelta(minutes=5 * lag)
        )
        data[name] = (data["Close"].div(previous_close) - 1).where(exact_lookback)

    # A rolling standard deviation gives a simple measure of recent movement.
    data["realized_volatility_12_bars"] = data.groupby(
        "_session_date", sort=False
    )["return_1_bar"].transform(
        lambda values: values.rolling(window=12, min_periods=12).std()
    )
    data["bar_range_pct"] = (data["High"] - data["Low"]).div(data["Close"])
    bar_range = data["High"] - data["Low"]
    data["close_location_in_bar"] = (
        data["Close"].sub(data["Low"]).div(bar_range.where(bar_range.ne(0)))
    )
    # Log scaling reduces the range of raw volume values while preserving order.
    data["log_volume"] = np.log1p(data["Volume"].astype(float))
    previous_log_volume = data.groupby("_session_date", sort=False)["log_volume"].shift(1)
    previous_datetime = data.groupby("_session_date", sort=False)["datetime"].shift(1)
    consecutive = data["datetime"].sub(previous_datetime).eq(pd.Timedelta(minutes=5))
    data["log_volume_change_1_bar"] = (
        data["log_volume"].sub(previous_log_volume).where(consecutive)
    )

    # The signal is available when this bar closes, five minutes after its start.
    signal_time = data["datetime"] + pd.Timedelta(minutes=BAR_MINUTES)
    minutes = signal_time.dt.hour.mul(60).add(signal_time.dt.minute).sub(570)
    # Sine and cosine let the model represent time-of-day as a repeating cycle.
    angle = 2 * np.pi * minutes / 390
    data["minutes_since_open"] = minutes.astype(int)
    data["time_of_day_sin"] = np.sin(angle)
    data["time_of_day_cos"] = np.cos(angle)
    return data.drop(columns="_session_date")


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return observed bars, forward target, and explicit model predictors."""
    return add_predictors(add_next_bar_target(df))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixed-parameter indicators using only current and earlier bars.

    EMA/RSI/Bollinger histories reset after a missing five-minute bar. Session
    VWAP resets at each market open and uses the available source volume.
    """
    data = _validated_bars(df)
    data["_session_date"] = data["datetime"].dt.date
    previous_datetime = data.groupby("_session_date", sort=False)["datetime"].shift(1)
    consecutive = data["datetime"].sub(previous_datetime).eq(pd.Timedelta(minutes=5))
    starts_segment = ~consecutive
    # Restart rolling indicator history after overnight gaps and missing five-minute bars.
    data["_segment_id"] = starts_segment.groupby(data["_session_date"]).cumsum()
    segment_keys = ["_session_date", "_segment_id"]
    segments = data.groupby(segment_keys, sort=False)

    # Standard 12/26/9 MACD settings are kept fixed for this comparison.
    ema_fast = segments["Close"].transform(
        lambda values: values.ewm(span=12, adjust=False, min_periods=12).mean()
    )
    ema_slow = segments["Close"].transform(
        lambda values: values.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.groupby(
        [data["_session_date"], data["_segment_id"]], sort=False
    ).transform(
        lambda values: values.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    # Divide price-based indicators by close so their scale is comparable across ETFs.
    data["macd_line_pct"] = macd_line.div(data["Close"])
    data["macd_histogram_pct"] = macd_line.sub(macd_signal).div(data["Close"])

    # RSI compares smoothed gains with smoothed losses over 14 bars.
    change = segments["Close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    average_gain = gains.groupby(
        [data["_session_date"], data["_segment_id"]], sort=False
    ).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    average_loss = losses.groupby(
        [data["_session_date"], data["_segment_id"]], sort=False
    ).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    relative_strength = average_gain.div(average_loss.where(average_loss.ne(0)))
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask(average_loss.eq(0) & average_gain.eq(0), 50)
    rsi = rsi.mask(average_loss.eq(0) & average_gain.gt(0), 100)
    data["rsi_14"] = rsi

    # Bandwidth measures the recent price spread relative to its rolling average.
    rolling_middle = segments["Close"].transform(
        lambda values: values.rolling(window=20, min_periods=20).mean()
    )
    rolling_std = segments["Close"].transform(
        lambda values: values.rolling(window=20, min_periods=20).std(ddof=0)
    )
    data["bollinger_bandwidth_20"] = rolling_std.mul(4).div(rolling_middle)

    # Session VWAP starts fresh each morning and uses only the volume in this source.
    typical_price = data[["High", "Low", "Close"]].mean(axis=1)
    cumulative_volume = data.groupby("_session_date", sort=False)["Volume"].cumsum()
    cumulative_price_volume = (typical_price * data["Volume"]).groupby(
        data["_session_date"], sort=False
    ).cumsum()
    session_vwap = cumulative_price_volume.div(cumulative_volume.where(cumulative_volume.gt(0)))
    data["distance_from_session_vwap"] = data["Close"].div(session_vwap).sub(1)

    return data.drop(columns=["_session_date", "_segment_id"])
