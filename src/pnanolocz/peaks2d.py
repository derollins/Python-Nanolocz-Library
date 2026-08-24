"""
Circular-neighbourhood peak detection for NanoLocz.

This module ports MATLAB ``peaks2D.m``.  It is similar to ``fast_peaks2d`` but
uses a circular neighbourhood mask for local maximum testing.

Output columns are:

    [x, y, height, prominence]

By default coordinates are MATLAB-style 1-based coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _sample_line(img: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    """Sample a line between two 0-based x/y coordinates."""
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    n = int(max(abs(end[0] - start[0]), abs(end[1] - start[1]))) + 1
    n = max(n, 2)
    xs = np.linspace(start[0], end[0], n)
    ys = np.linspace(start[1], end[1], n)
    return map_coordinates(np.asarray(img, dtype=np.float64), [ys, xs], order=1, mode="nearest")


def peaks2d(
    img: np.ndarray,
    thresh: float | None = None,
    ns: float = 1,
    min_prom: float = 0.0,
    *,
    matlab_indexing: bool = True,
) -> FloatArray:
    """Detect local maxima above a threshold inside a circular neighbourhood."""
    arr = np.asarray(img, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError("peaks2d expects a 2-D image")

    if thresh is not None:
        arr = arr * (arr > float(thresh))

    ns_int = int(np.round(float(ns) + 1.0))
    ns_int = max(ns_int, 1)
    nsi = ns_int - 1

    hx, hy = np.meshgrid(np.arange(-nsi, nsi + 1), np.arange(-nsi, nsi + 1), indexing="ij")
    H = (np.sqrt(np.maximum(-(hx**2) - (hy**2) + ns_int**2, 0.0)) > (ns_int / 2.0)).astype(float)

    target = int(np.round(((ns_int * 2 - 1) ** 2) / 2.0)) - 1  # MATLAB linear index -> zero-based.

    peak_xy: list[tuple[int, int]] = []
    peak_h: list[float] = []

    rows, cols = arr.shape
    for r in range(ns_int - 1, rows - ns_int):
        for c in range(ns_int - 1, cols - ns_int):
            if arr[r, c] == 0:
                continue
            patch = H * arr[r - nsi : r + nsi + 1, c - nsi : c + nsi + 1]
            flat_id = int(np.argmax(patch.ravel(order="F")))  # MATLAB linear indexing is column-major.
            if flat_id == target:
                peak_xy.append((c, r))
                peak_h.append(float(np.max(patch)))

    if not peak_xy:
        return np.empty((0, 4), dtype=np.float64)

    coords0 = np.asarray(peak_xy, dtype=np.float64)
    heights = np.asarray(peak_h, dtype=np.float64)

    if float(min_prom) > 0:
        prom = np.zeros(heights.size, dtype=np.float64)
        for j in range(coords0.shape[0]):
            distances = np.sqrt(np.sum((coords0[j] - coords0) ** 2, axis=1))
            order = np.argsort(distances)

            if heights[j] >= np.max(heights):
                prom[j] = heights[j]
            else:
                higher = heights[order] > heights[j]
                idxs = np.flatnonzero(higher)
                if idxs.size == 0:
                    prom[j] = heights[j]
                else:
                    nearest = order[idxs[0]]
                    vals = _sample_line(arr, coords0[nearest], coords0[j])
                    prom[j] = heights[j] - float(np.nanmin(vals))

        keep = prom > float(min_prom)
        coords0 = coords0[keep]
        heights = heights[keep]
        prom = prom[keep]
    else:
        prom = np.zeros(heights.size, dtype=np.float64)

    coords = coords0 + 1.0 if matlab_indexing else coords0
    return np.asarray(np.column_stack([coords, heights, prom]), dtype=np.float64)


# MATLAB-style alias.
peaks2D = peaks2d

__all__ = ["peaks2d", "peaks2D"]
