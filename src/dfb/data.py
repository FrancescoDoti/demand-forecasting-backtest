"""Load UCI Bike Sharing (hourly) onto a complete hourly grid."""
from pathlib import Path

import pandas as pd

WEATHER = ["weathersit", "temp", "atemp", "hum", "windspeed"]


def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dteday"])
    df["ts"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.set_index("ts").sort_index()
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h")
    # 165 hours are absent from the source file: overnight zero-rental hours and a few system outages.
    out = df.reindex(grid)
    out["was_missing"] = out["cnt"].isna()
    out["cnt"] = out["cnt"].fillna(0)
    for c in ["holiday", "workingday"] + WEATHER:
        out[c] = out[c].ffill()
    out["hr"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    return out[["cnt", "hr", "dow", "month", "holiday", "workingday", "was_missing"] + WEATHER]
