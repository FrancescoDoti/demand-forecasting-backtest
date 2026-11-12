"""Rolling-origin backtest: 12 folds, one per month of 2012, expanding training window.

python -m dfb.backtest --data data/hour.csv
For each fold the model trains on everything before the month except the last 28
days, which are kept for conformal calibration; it then forecasts the month day by day.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .conformal import intervals
from .data import load_hourly
from .features import add_relative, build, feature_names
from .metrics import coverage, mae, rmse, wape

ALPHA = 0.1
CALIB_DAYS = 28
PARAMS = dict(objective="poisson", n_jobs=8, n_estimators=700, learning_rate=0.04, num_leaves=63, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
HOUR_GROUPS = {"night 00-05": range(0, 6), "commute peaks 07-09,16-19": [7, 8, 9, 16, 17, 18, 19], "other daytime": [6, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23]}


class Model:
    """Wraps LightGBM so callers never see the target transform.

    level:    predict counts directly with a Poisson objective.
    relative: predict log((y + 1) / (level + 1)) with L2, where level = last week's mean
              hourly demand, then map back. Trees cannot extrapolate a growing level;
              this way they never have to.
    """

    def __init__(self, target: str, **extra):
        self.target = target
        params = {**PARAMS, **extra}
        if target == "relative" and params["objective"] == "poisson":
            params["objective"] = "regression"
        self.m = lgb.LGBMRegressor(**params)
        self.quantile = params["objective"] == "quantile"

    def _lvl(self, X):
        return np.log1p(X["level"].to_numpy())

    def fit(self, X, y, cols):
        self.cols = cols
        t = np.log1p(y.to_numpy()) - self._lvl(X) if self.target == "relative" else y
        self.m.fit(X[cols], t)
        self.booster_ = self.m.booster_
        return self

    def predict(self, X):
        p = self.m.predict(X[self.cols])
        return np.clip(np.expm1(p + self._lvl(X)), 0, None) if self.target == "relative" else p


def fit_predict(X_tr, y_tr, X_te, cols, target, **extra):
    m = Model(target, **extra).fit(X_tr, y_tr, cols)
    return m, m.predict(X_te)


FOLDS = {"dev": ("2011-07", "2011-12"), "test": ("2012-01", "2012-12")}


def run(path: Path, target: str = "relative", folds: str = "test") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    df = load_hourly(path)
    X_now, X_wx = add_relative(build(df, False)), add_relative(build(df, True))
    y = df["cnt"]
    ok = X_now[feature_names(False, "level") + ["level"]].notna().all(axis=1)
    rows, importances = [], []
    for month in pd.period_range(*FOLDS[folds], freq="M"):
        start, end = month.start_time, month.end_time
        cal_start = start - pd.Timedelta(days=CALIB_DAYS)
        tr = ok & (df.index < cal_start)
        cal = ok & (df.index >= cal_start) & (df.index < start)
        te = (df.index >= start) & (df.index <= end) & ~df["was_missing"]
        out = pd.DataFrame({"y": y[te], "hr": df["hr"][te]}, index=df.index[te])
        out["naive_24"] = X_now.loc[te, "lag_24"]
        out["naive_168"] = X_now.loc[te, "lag_168"]
        out["mean_7d"] = X_now.loc[te, "same_hour_mean_7d"]
        cols = feature_names(False, target)
        m, out["lgbm"] = fit_predict(X_now.loc[tr], y[tr], X_now.loc[te], cols, target)
        cal_pred = m.predict(X_now.loc[cal])
        for scaled, name in ((False, "conf"), (True, "conf_scaled")):
            out[f"{name}_lo"], out[f"{name}_hi"] = intervals(cal_pred, y[cal].to_numpy(), out["lgbm"].to_numpy(), ALPHA, scaled)
        _, out["q_lo"] = fit_predict(X_now.loc[tr], y[tr], X_now.loc[te], cols, target, objective="quantile", alpha=ALPHA / 2)
        _, out["q_hi"] = fit_predict(X_now.loc[tr], y[tr], X_now.loc[te], cols, target, objective="quantile", alpha=1 - ALPHA / 2)
        other = "relative" if target == "level" else "level"
        _, out[f"lgbm_{other}"] = fit_predict(X_now.loc[tr], y[tr], X_now.loc[te], feature_names(False, other), other)
        out["ens_level_relative"] = (out["lgbm"] + out[f"lgbm_{other}"]) / 2
        out["ens_lgbm_naive168"] = (out["lgbm"] + out["naive_168"]) / 2
        _, out["lgbm_oracle_weather"] = fit_predict(X_wx.loc[tr], y[tr], X_wx.loc[te], feature_names(True, target), target)
        out["fold"] = str(month)
        rows.append(out)
        importances.append(pd.Series(m.booster_.feature_importance("gain"), index=cols))
        print(f"{month}: train {int(tr.sum())} h, test {int(te.sum())} h, WAPE lgbm {wape(out.y, out.lgbm):.3f}, naive_168 {wape(out.y, out.naive_168):.3f}", flush=True)
    preds = pd.concat(rows)
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return summarise(preds), preds, imp


def summarise(p: pd.DataFrame) -> dict:
    y = p["y"].to_numpy()
    point = {}
    models = ["naive_24", "naive_168", "mean_7d", "lgbm"] + [c for c in p.columns if c.startswith("lgbm_") and c != "lgbm_oracle_weather"] + ["ens_level_relative", "ens_lgbm_naive168", "lgbm_oracle_weather"]
    for s in models:
        point[s] = {"mae": mae(y, p[s]), "rmse": rmse(y, p[s]), "wape": wape(y, p[s]),
                    "mase_vs_naive_168": mae(y, p[s]) / mae(y, p["naive_168"]),
                    "folds_beating_naive_168": int(sum(wape(g.y, g[s]) < wape(g.y, g.naive_168) for _, g in p.groupby("fold")))}
    per_fold = {f: {s: wape(g.y, g[s]) for s in ["naive_168", "mean_7d", "lgbm", "lgbm_oracle_weather"]} for f, g in p.groupby("fold")}
    iv = {}
    for name, lo, hi in [("split conformal", "conf_lo", "conf_hi"), ("scaled conformal", "conf_scaled_lo", "conf_scaled_hi"), ("LightGBM quantile 5-95%", "q_lo", "q_hi")]:
        iv[name] = {"coverage": coverage(y, p[lo], p[hi]), "mean_width": float((p[hi] - p[lo]).mean()),
                    "by_hour_group": {g: coverage(p.y[p.hr.isin(h)], p[lo][p.hr.isin(h)], p[hi][p.hr.isin(h)]) for g, h in HOUR_GROUPS.items()},
                    "worst_fold_coverage": float(min(coverage(g.y, g[lo], g[hi]) for _, g in p.groupby("fold")))}
    return {"test_hours": int(len(p)), "target_coverage": 1 - ALPHA, "point": point, "per_fold_wape": per_fold, "intervals": iv}


def plots(preds: pd.DataFrame, imp: pd.Series, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wk = preds.loc["2012-09-10":"2012-09-16"]
    if wk.empty:
        wk = preds.iloc[-24 * 7:]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(wk.index, wk.conf_scaled_lo, wk.conf_scaled_hi, color="#93c5fd", alpha=0.6, label="90% scaled conformal interval")
    ax.plot(wk.index, wk.y, c="k", lw=1.2, label="actual")
    ax.plot(wk.index, wk.lgbm, c="#2563eb", lw=1.2, label="LightGBM day-ahead")
    ax.plot(wk.index, wk.naive_168, c="#9ca3af", lw=1, ls="--", label="same hour last week")
    ax.set_ylabel("rentals / hour")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.set_title("Day-ahead forecast, week of 10 Sep 2012 (out of sample)")
    fig.tight_layout()
    fig.savefig(out / "forecast_week.png", dpi=140)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, lo, hi, c in [("split conformal", "conf_lo", "conf_hi", "#9ca3af"), ("scaled conformal", "conf_scaled_lo", "conf_scaled_hi", "#2563eb"), ("quantile LightGBM", "q_lo", "q_hi", "#f59e0b")]:
        cov = preds.groupby("hr").apply(lambda g: coverage(g.y, g[lo], g[hi]))
        ax.plot(cov.index, cov.values, "o-", ms=3, c=c, label=name)
    ax.axhline(0.9, ls="--", c="k", lw=1)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0.5, 1.01)
    ax.legend(fontsize=8)
    ax.set_title("Interval coverage by hour (target 90%)")
    fig.tight_layout()
    fig.savefig(out / "coverage_by_hour.png", dpi=140)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    (imp / imp.sum()).iloc[::-1].plot.barh(ax=ax, color="#2563eb")
    ax.set_title("LightGBM gain importance (mean over folds)")
    fig.tight_layout()
    fig.savefig(out / "importance.png", dpi=140)


def report(s: dict) -> str:
    L = ["# Backtest results", "", f"{s['test_hours']} out-of-sample hours (monthly folds, expanding window; target transform: {s.get('target')}; folds: {s.get('folds')}). Outage hours excluded from scoring.", "",
         "## Point forecasts", "", "| Model | MAE | RMSE | WAPE | MASE vs last-week naive | folds beating naive (of 12) |", "|---|---|---|---|---|---|"]
    for k, v in s["point"].items():
        L.append(f"| {k} | {v['mae']:.1f} | {v['rmse']:.1f} | {v['wape']:.3f} | {v['mase_vs_naive_168']:.3f} | {v['folds_beating_naive_168']} |")
    L += ["", "`lgbm_oracle_weather` uses the weather that actually happened, which is not known a day ahead. It is an upper bound on the value of a perfect weather forecast, not a deployable model.", "",
          "## 90% prediction intervals", "", "| Method | Coverage | Mean width | Night | Commute peaks | Other daytime | Worst month |", "|---|---|---|---|---|---|---|"]
    for k, v in s["intervals"].items():
        g = v["by_hour_group"]
        L.append(f"| {k} | {v['coverage']:.3f} | {v['mean_width']:.0f} | {g['night 00-05']:.3f} | {g['commute peaks 07-09,16-19']:.3f} | {g['other daytime']:.3f} | {v['worst_fold_coverage']:.3f} |")
    L += ["", "## WAPE per fold", "", "| Month | last-week naive | 7-day same-hour mean | LightGBM | LightGBM + oracle weather |", "|---|---|---|---|---|"]
    for f, v in s["per_fold_wape"].items():
        L.append(f"| {f} | {v['naive_168']:.3f} | {v['mean_7d']:.3f} | {v['lgbm']:.3f} | {v['lgbm_oracle_weather']:.3f} |")
    L += ["", "![week](forecast_week.png)", "", "![coverage](coverage_by_hour.png)", "", "![importance](importance.png)", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hour.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--target", choices=["level", "relative"], default="relative")
    ap.add_argument("--folds", choices=["dev", "test"], default="test")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    s, preds, imp = run(Path(a.data), a.target, a.folds)
    s["target"], s["folds"] = a.target, a.folds
    s["feature_importance_share"] = (imp / imp.sum()).round(4).to_dict()
    (out / "results.json").write_text(json.dumps(s, indent=2))
    (out / "report.md").write_text(report(s))
    plots(preds, imp, out)
    print(report(s))


if __name__ == "__main__":
    main()
