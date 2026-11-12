"""Features for a day-ahead forecast issued at 00:00 for the 24 hours of that day.

The rule that keeps this honest: every feature for hour h of day D must be known at
00:00 on day D. So the shortest usable lag is 24 h (same hour yesterday), never 1 h.
Observed weather is *not* known a day ahead; it is offered only as an explicitly
labelled oracle variant ("with a perfect weather forecast") to bound its value.
"""
import pandas as pd

from .data import WEATHER

CALENDAR = ["hr", "dow", "month", "holiday", "workingday", "trend_days"]
LAGS = ["lag_24", "lag_48", "lag_168", "lag_336", "same_hour_mean_7d", "yesterday_total", "last_week_total"]


def build(df: pd.DataFrame, with_weather: bool = False) -> pd.DataFrame:
    y = df["cnt"]
    X = pd.DataFrame(index=df.index)
    for c in ["hr", "dow", "month", "holiday", "workingday"]:
        X[c] = df[c]
    X["trend_days"] = (df.index - df.index[0]).days
    for k in (24, 48, 168, 336):
        X[f"lag_{k}"] = y.shift(k)
    X["same_hour_mean_7d"] = sum(y.shift(24 * d) for d in range(1, 8)) / 7
    daily = y.resample("D").sum()
    X["yesterday_total"] = daily.shift(1).reindex(df.index.normalize()).to_numpy()
    X["last_week_total"] = daily.shift(1).rolling(7).sum().reindex(df.index.normalize()).to_numpy()
    if with_weather:
        for c in WEATHER:
            X[c] = df[c]
    return X


RELATIVE = ["hr", "dow", "month", "holiday", "workingday", "rel_lag_24", "rel_lag_48", "rel_lag_168", "rel_lag_336", "rel_same_hour_mean_7d", "rel_yesterday"]


def add_relative(X: pd.DataFrame) -> pd.DataFrame:
    """Lags divided by last week's mean hourly demand: the model learns shape, the level carries growth."""
    X = X.copy()
    X["level"] = X["last_week_total"] / 168
    for c in ["lag_24", "lag_48", "lag_168", "lag_336", "same_hour_mean_7d"]:
        X[f"rel_{c}"] = (X[c] + 1) / (X["level"] + 1)
    X["rel_yesterday"] = (X["yesterday_total"] / 24 + 1) / (X["level"] + 1)
    return X


def feature_names(with_weather: bool, target: str = "level") -> list[str]:
    base = RELATIVE if target == "relative" else CALENDAR + LAGS
    return base + (WEATHER if with_weather else [])
