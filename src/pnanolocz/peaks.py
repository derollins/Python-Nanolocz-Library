"""
Fast 2D peak detection for NanoLocz-compatible AFM workflows.

This module ports MATLAB ``Fast_peaks2D.m``.  It detects local maxima above an
intensity threshold and optionally filters them by a simple prominence measure.

Results are returned as a :class:`Peaks` record of 1-D arrays, one entry per
peak.  Coordinates are 0-based pixel indices (``x`` indexes columns, ``y``
indexes rows), like the rest of pnanolocz, so they can index the image
directly: ``img[peaks.y, peaks.x]``.  The MATLAB original returns 1-based
coordinates; add 1 to ``x`` and ``y`` when comparing against MATLAB output.

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

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.intp]]


@dataclass(frozen=True, slots=True)
class Peaks:
    """
    Local maxima found by :func:`fast_peaks2d`.

    All fields are 1-D arrays of the same length, with one entry per peak.

    Attributes
    ----------
    x : ndarray of intp
        Column index of each peak (0-based).
    y : ndarray of intp
        Row index of each peak (0-based).
    height : ndarray of float64
        Image value at each peak.
    prominence : ndarray of float64
        Prominence of each peak, or zero when ``min_prom`` was not used.
    """

    x: IntArray
    y: IntArray
    height: FloatArray
    prominence: FloatArray

    def __len__(self) -> int:
        """Return the number of peaks."""
        return int(self.x.size)


def _sample_line(
    img: FloatArray, x0: float, y0: float, x1: float, y1: float
) -> FloatArray:
    """
    Sample image values along a straight line, as MATLAB ``improfile`` does.

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
    return np.asarray(
        map_coordinates(img, [ys, xs], order=0, mode="nearest"), dtype=np.float64
    )


def _prominence(
    img: FloatArray, x: IntArray, y: IntArray, height: FloatArray
) -> FloatArray:
    """
    Return the prominence of each peak, as in MATLAB ``Fast_peaks2D.m``.

    Prominence is the height of a peak above the lowest point on the line to
    its nearest higher peak.  The highest peak has no higher neighbour, so its
    prominence is its own height.
    """
    prominence = np.empty(height.size, dtype=np.float64)
    highest = height.max()

    for j in range(height.size):
        if height[j] >= highest:
            prominence[j] = height[j]
            continue

        # Nearest strictly higher peak.  Peak heights are finite (a NaN pixel
        # never passes ``img > thresh``), so at least one exists here.
        distance = np.sqrt(((x - x[j]) ** 2 + (y - y[j]) ** 2).astype(np.float64))
        order = np.argsort(distance)
        nearest = int(order[height[order] > height[j]][0])

        profile = _sample_line(img, x[nearest], y[nearest], x[j], y[j])
        prominence[j] = height[j] - float(np.nanmin(profile))

    return prominence


def fast_peaks2d(
    img: FloatArray,
    thresh: float,
    kernel_size: int,
    min_prom: float = 0.0,
) -> Peaks:
    """
    Detect local maxima in a 2D image.

    Parameters
    ----------
    img : ndarray
        2D grayscale image.
    thresh : float
        Minimum height; only pixels strictly above it can be peaks.
    kernel_size : int
        Size of the local neighbourhood used for maximum filtering, matching
        the MATLAB ``Fast_peaks2D`` argument.  The effective maximum-filter
        window is ``kernel_size + 2`` pixels.
    min_prom : float, default 0.0
        Minimum prominence.  If <= 0, prominence is not computed and every
        peak is returned with a prominence of zero.

    Returns
    -------
    Peaks
        The detected peaks, in row-major order.  All fields are empty when no
        peaks are found.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"fast_peaks2d expects a 2D image, got shape {arr.shape}")

    # MATLAB: kernel_size = kernel_size + 2;
    #         ordfilt2(Img, kernel_size^2, true(kernel_size), 'zeros')
    window_size = max(int(kernel_size) + 2, 1)
    max_filtered = maximum_filter(
        arr, size=(window_size, window_size), mode="constant", cval=0.0
    )

    # A pixel is a peak when it is the maximum of its window and above thresh.
    maxima = (max_filtered == arr) & (arr > float(thresh))

    # MATLAB NanoLocz Fast_peaks2D.m excludes the first two and the last three
    # rows/columns, where the zero-padded maximum filter is not fully supported.
    if maxima.shape[0] >= 5:
        maxima[:2, :] = False
        maxima[-3:, :] = False
    if maxima.shape[1] >= 5:
        maxima[:, :2] = False
        maxima[:, -3:] = False

    y, x = np.nonzero(maxima)
    height = arr[y, x]

    if float(min_prom) > 0 and height.size > 0:
        prominence = _prominence(arr, x, y, height)
        keep = prominence > float(min_prom)
        return Peaks(x[keep], y[keep], height[keep], prominence[keep])

    return Peaks(x, y, height, np.zeros(height.size, dtype=np.float64))


fast_peaks2d.__version__ = "0.1.0"  # type: ignore[attr-defined]

__all__ = ["Peaks", "fast_peaks2d"]
