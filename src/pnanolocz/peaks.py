"""
Fast 2-D peak detection for NanoLocz-compatible AFM workflows.

This module ports MATLAB ``Fast_peaks2D.m``.  It detects local maxima above an
intensity threshold and optionally filters them by a simple prominence measure.

Coordinate convention
---------------------
Output coordinates are 0-based Python indices (``x`` indexes columns, ``y``
indexes rows), like the rest of pnanolocz:

    locs[:, 0] = x / column index
    locs[:, 1] = y / row index
    locs[:, 2] = peak height
    locs[:, 3] = prominence

The MATLAB original returns 1-based coordinates; add 1 to ``x`` and ``y`` when
comparing against MATLAB output.

Notes
-----
The maximum-filter, border-exclusion and prominence-sampling steps follow
MATLAB ``Fast_peaks2D.m``.  One deliberate deviation remains: the first-two /
last-three border exclusion is skipped for images smaller than 5 pixels in a
dimension, where the MATLAB index ranges would fall outside the image.

Prominence is measured along the line joining a peak to its nearest higher
neighbour.  The sampling follows MATLAB ``improfile``: ``max(|dx|, |dy|) + 1``
equally spaced points (``improfile.m`` derives the same count from the ceiling
of the coordinate differences, which are whole numbers here) interpolated with
nearest-neighbour, the ``improfile`` default.

Authors
-------
- MATLAB ``Fast_peaks2D.m``: George R. Heath (NanoLocz).
- Python port: Wang-1971.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _sample_line(
    img: FloatArray, x0: float, y0: float, x1: float, y1: float
) -> FloatArray:
    """Sample image values along a straight line, as MATLAB ``improfile`` does.

    ``x`` indexes columns and ``y`` indexes rows, matching the rest of
    pnanolocz.  One sample is taken per pixel along the longer axis of the line,
    and never fewer than two, so that a peak always yields a profile.  Values
    are read with nearest-neighbour interpolation, MATLAB's ``improfile``
    default.
    """
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    n = max(n, 2)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    line_profile = np.asarray(
        map_coordinates(
            np.asarray(img, dtype=np.float64),
            [ys, xs],
            order=0,
            mode="nearest",
        ),
        dtype=np.float64,
    )
    return line_profile


def fast_peaks2d(
    img: FloatArray,
    thresh: float,
    kernel_size: int,
    min_prom: float = 0.0,
) -> FloatArray:
    """Detect local maxima in a 2-D image.

    Parameters
    ----------
    img:
        2-D grayscale image.
    thresh:
        Minimum height threshold.
    kernel_size:
        Size of the local neighbourhood used for maximum filtering, matching
        the MATLAB ``Fast_peaks2D`` argument.  The effective maximum-filter
        window is ``kernel_size + 2`` pixels.
    min_prom:
        Optional minimum prominence.  If <= 0, every detected peak is returned
        with its prominence set to zero.

    Returns
    -------
    ndarray
        ``N x 4`` array of ``[x, y, height, prominence]`` with 0-based
        coordinates, or a 0-row array when no peaks are found.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("fast_peaks2d expects a 2-D image")

    # MATLAB: kernel_size = kernel_size + 2;
    #         ordfilt2(Img, kernel_size^2, true(kernel_size), 'zeros')
    window_size = max(int(kernel_size) + 2, 1)
    max_filtered = maximum_filter(
        arr, size=(window_size, window_size), mode="constant", cval=0.0
    )

    # A pixel is a peak when it is the maximum of its window and above thresh.
    maxima = (max_filtered == arr) & (arr > float(thresh))

    # MATLAB Nanolocz Fast_peaks2D.m excludes the first two and the last three
    # rows/columns, where the zero-padded maximum filter is not fully supported.
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

    # 0-based (x, y) peak coordinates; MATLAB's find returns 1-based ones.
    coords = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])

    if float(min_prom) > 0:
        prom = np.zeros(rows.size, dtype=np.float64)
        for j in range(rows.size):
            # Distance from peak j to every other peak, nearest first.
            d = np.sqrt(np.sum((coords[j, :2] - coords[:, :2]) ** 2, axis=1))
            order = np.argsort(d)

            # If this is the highest peak, MATLAB assigns its own height.
            if np.all(heights[j] >= heights):
                prom[j] = heights[j]
                continue

            # Nearest peak that is strictly higher than peak j.  Peak heights
            # are finite (a NaN pixel never passes ``arr > thresh``), so at
            # least one peak is strictly higher here.
            nearest_higher = int(order[heights[order] > heights[j]][0])

            # Prominence is the height of peak j above the lowest point on the
            # line joining it to that nearest higher peak.
            line_vals = _sample_line(
                arr,
                coords[nearest_higher, 0],
                coords[nearest_higher, 1],
                coords[j, 0],
                coords[j, 1],
            )
            prom[j] = heights[j] - float(np.nanmin(line_vals))

        keep = prom > float(min_prom)
        coords = coords[keep]
        heights = heights[keep]
        prom = prom[keep]
    else:
        prom = np.zeros(rows.size, dtype=np.float64)

    return np.asarray(np.column_stack([coords, heights, prom]), dtype=np.float64)


fast_peaks2d.__version__ = "0.1.0"  # type: ignore[attr-defined]

__all__ = ["fast_peaks2d"]
