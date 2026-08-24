"""
Outlier removal helpers for NanoLocz localization tables.

This module ports MATLAB ``Rem_outliers.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import zscore

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def rem_outliers(locs: np.ndarray, threshold: float) -> FloatArray:
    """Remove rows whose x/y z-score exceeds ``threshold``."""
    arr = np.asarray(locs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("locs must be an Nx2 or wider array")

    valid_rows = np.all(~np.isnan(arr[:, 0:2]), axis=1)
    locs_valid = arr[valid_rows]

    if locs_valid.size == 0:
        return np.empty((0, arr.shape[1]), dtype=np.float64)

    z = np.abs(zscore(locs_valid[:, 0:2], axis=0, nan_policy="omit"))
    idx_keep_valid = np.all(z < float(threshold), axis=1)

    keep = np.zeros(arr.shape[0], dtype=bool)
    keep[np.flatnonzero(valid_rows)] = idx_keep_valid

    return np.asarray(arr[keep], dtype=np.float64)


# MATLAB-style alias.
Rem_outliers = rem_outliers

__all__ = ["rem_outliers", "Rem_outliers"]
