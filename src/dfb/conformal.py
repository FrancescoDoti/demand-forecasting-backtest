"""Split conformal prediction intervals, plain and locally scaled.

Plain: [yhat - q, yhat + q], q = the ceil((n+1)(1-a))/n quantile of |residual| on a
calibration set the model never trained on. Distribution-free marginal coverage under
exchangeability, but the same width at 3 a.m. and at rush hour.

Scaled: residuals divided by sqrt(yhat + 1) (count noise grows with the mean), so the
interval widens where demand is high and narrows where it is near zero.
"""
import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The k-th smallest score, k = ceil((n + 1)(1 - alpha)); infinite if k > n (too little calibration data)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def scale(yhat: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(yhat, 0, None) + 1.0)


def intervals(yhat_cal, y_cal, yhat_test, alpha: float = 0.1, scaled: bool = False):
    if scaled:
        q = conformal_quantile(np.abs(y_cal - yhat_cal) / scale(yhat_cal), alpha)
        half = q * scale(yhat_test)
    else:
        q = conformal_quantile(np.abs(y_cal - yhat_cal), alpha)
        half = np.full(len(yhat_test), q)
    return np.clip(yhat_test - half, 0, None), yhat_test + half
