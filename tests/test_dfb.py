import numpy as np
import pandas as pd
import pytest

from dfb.conformal import conformal_quantile, intervals
from dfb.features import add_relative, build
from dfb.metrics import coverage, wape


def synthetic(days=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2011-01-01", periods=24 * days, freq="h")
    base = 50 + 40 * np.sin(2 * np.pi * idx.hour / 24) ** 2
    df = pd.DataFrame({"cnt": rng.poisson(base).astype(float)}, index=idx)
    df["hr"], df["dow"], df["month"] = idx.hour, idx.dayofweek, idx.month
    df["holiday"] = df["workingday"] = 0
    for c in ["weathersit", "temp", "atemp", "hum", "windspeed"]:
        df[c] = 0.5
    return df


def test_no_feature_uses_same_day_demand():
    """Change every count on one day: none of that day's features may move (day-ahead contract)."""
    df = synthetic()
    day = "2011-02-10"
    X1 = add_relative(build(df))
    df2 = df.copy()
    df2.loc[day, "cnt"] += 1000
    X2 = add_relative(build(df2))
    pd.testing.assert_frame_equal(X1.loc[day], X2.loc[day])
    assert not X1.loc["2011-02-11"].equals(X2.loc["2011-02-11"])  # ...but the next day's must


def test_conformal_quantile_finite_sample_level():
    scores = np.arange(1, 101, dtype=float)
    assert conformal_quantile(scores, 0.1) == 91.0  # k = ceil(101 * 0.9) = 91 -> 91st smallest
    assert conformal_quantile(scores[:5], 0.1) == float("inf")


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    covs = []
    for _ in range(200):
        y_cal, y_te = 100 + rng.normal(size=500), 100 + rng.normal(size=500)  # positive: intervals clip at 0
        lo, hi = intervals(np.full(500, 100.0), y_cal, np.full(500, 100.0), alpha=0.1)
        covs.append(coverage(y_te, lo, hi))
    assert np.mean(covs) == pytest.approx(0.9, abs=0.01)


def test_scaled_intervals_widen_with_prediction():
    lo, hi = intervals(np.array([1.0, 100.0]), np.array([2.0, 110.0]), np.array([1.0, 400.0]), alpha=0.5, scaled=True)
    assert (hi - lo)[1] > (hi - lo)[0]
    assert (lo >= 0).all()


def test_wape_ignores_zero_actuals_gracefully():
    assert wape(np.array([0.0, 10.0]), np.array([1.0, 9.0])) == pytest.approx(0.2)
