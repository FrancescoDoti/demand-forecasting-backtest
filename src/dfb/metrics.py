import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y, p):
    """Weighted absolute percentage error: sum|e| / sum|y|. Unlike MAPE it survives zero-demand hours."""
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)))


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))
