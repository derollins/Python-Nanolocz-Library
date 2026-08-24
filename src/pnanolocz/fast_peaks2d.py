"""
Fast 2-D peak detection for NanoLocz-compatible AFM workflows.

This module ports MATLAB ``Fast_peaks2D.m``.  It detects local maxima above an
intensity threshold and optionally filters peaks by a simple prominence measure.

Coordinate convention
---------------------
By default this function returns MATLAB-style 1-based coordinates because
NanoLocz localisation tables are originally MATLAB tables:

    locs[:, 0] = x / column coordinate
    locs[:, 1] = y / row coordinate
    locs[:, 2] = peak height
    locs[:, 3] = prominence

Set ``matlab_indexing=False`` to return 0-based Python coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter, map_coordinates

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _sample_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sample image values along a line using bilinear interpolation."""
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    n = max(n, 2)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return map_coordinates(np.asarray(img, dtype=np.float64), [ys, xs], order=1, mode="nearest")


def fast_peaks2d(
    img: np.ndarray,
    thresh: float,
    kernel_size: int,
    min_prom: float = 0.0,
    *,
    matlab_indexing: bool = True,
) -> FloatArray:
    """Detect local maxima in a 2-D image.

    Parameters
    ----------
    img:
        2-D grayscale image.
    thresh:
        Minimum height threshold.
    kernel_size:
        MATLAB kernel size parameter.  The effective maximum-filter window is
        ``kernel_size + 2``.
    min_prom:
        Optional minimum prominence.  If <= 0, all detected peaks are returned
        with prominence set to zero.
    matlab_indexing:
        If true, output x/y coordinates are 1-based.

    Returns
    -------
    ndarray
        ``N x 4`` table ``[x, y, height, prominence]``.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("fast_peaks2d expects a 2-D image")

    k = int(kernel_size) + 2
    k = max(k, 1)

    # MATLAB: ordfilt2(Img, kernel_size^2, true(kernel_size), 'zeros')
    max_filtered = maximum_filter(arr, size=(k, k), mode="constant", cval=0.0)
    maxima = (max_filtered == arr) & (arr > float(thresh))

    # MATLAB excludes first two and last three rows/columns.
    if maxima.shape[0] >= 5:
        maxima[:2, :] = False
        maxima[-3:, :] = False
    if maxima.shape[1] >= 5:
        maxima[:, :2] = False
        maxima[:, -3:] = False

    rows, cols = np.nonzero(maxima)
    heights = arr[rows, cols]

    if rows.size == 0:
        return np.empty((0, 4), dtype=np.float64)

    # Internal coordinates are 0-based for computation.
    coords0 = np.column_stack([cols.astype(float), rows.astype(float)])

    if float(min_prom) > 0:
        prom = np.zeros(rows.size, dtype=np.float64)
        for j in range(rows.size):
            d = np.sqrt(np.sum((coords0[j, :2] - coords0[:, :2]) ** 2, axis=1))
            order = np.argsort(d)

            # If this is the highest peak, MATLAB assigns its own height.
            if np.all(heights[j] >= heights):
                prom[j] = heights[j]
                continue

            higher = heights[order] > heights[j]
            higher_idx = np.flatnonzero(higher)
            if higher_idx.size == 0:
                prom[j] = heights[j]
                continue

            nearest_higher = order[higher_idx[0]]
            line_vals = _sample_line(
                arr,
                coords0[nearest_higher, 0],
                coords0[nearest_higher, 1],
                coords0[j, 0],
                coords0[j, 1],
            )
            prom[j] = heights[j] - float(np.nanmin(line_vals))

        keep = prom > float(min_prom)
        coords0 = coords0[keep]
        heights = heights[keep]
        prom = prom[keep]
    else:
        prom = np.zeros(rows.size, dtype=np.float64)

    coords = coords0 + 1.0 if matlab_indexing else coords0
    return np.asarray(np.column_stack([coords, heights, prom]), dtype=np.float64)


# MATLAB-compatible alias.
Fast_peaks2D = fast_peaks2d

__all__ = ["fast_peaks2d", "Fast_peaks2D"]
