"""
Rotational symmetry averaging for NanoLocz-compatible images.

Ports MATLAB ``rotation_sym.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import rotate

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def rotation_sym(d: np.ndarray, fold: int) -> FloatArray:
    """Average an image with its rotated copies."""
    arr = np.asarray(d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("rotation_sym expects a 2-D image")
    if int(fold) <= 0:
        raise ValueError("fold must be a positive integer")

    acc = arr.copy()
    for i in range(2, int(fold) + 1):
        angle = (i - 1) * 360.0 / float(fold)
        acc += rotate(arr, angle=angle, reshape=False, order=3, mode="constant", cval=0.0, prefilter=True)

    return np.asarray(acc / float(fold), dtype=np.float64)


__all__ = ["rotation_sym"]
