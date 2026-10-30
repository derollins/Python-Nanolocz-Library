"""
Normalized 2D cross-correlation: MATLAB-compatible ``normxcorr2``.

This module is an independent implementation of ``normxcorr2`` with MATLAB-compatible
output as used throughout NanoLocz, following the "fast normalized
cross-correlation" formulation of J. P. Lewis (1995).

Implementation
--------------
The numerator is a single ``scipy.signal.fftconvolve`` of the image with the
mean-subtracted, flipped template (O(N log N)).  The two denominator terms
(local sum and local sum of squares) are computed with separable running sums,
which are O(N) and need no transform, so each call performs one FFT instead of
the three used by the reference Python implementation
(https://github.com/Sabrewarrior/normxcorr2-python).

Scale handling
--------------
The correlation is normalized, so it should not depend on the units of the
input: the same image in metres, nanometres or volts gives the same peaks.
This requires the denominator test to be *relative* rather than absolute.  A
window whose variance is numerically indistinguishable from zero (a flat,
zero-padded region) would otherwise divide noise by noise and produce spurious
peaks.  Windows are therefore masked out when their denominator falls below
``sqrt(eps)`` of the largest denominator in the map, and any residual
``|C| > 1`` caused by round-off is zeroed, as MATLAB does.  Only those
near-flat windows can differ slightly between scalings.

References
----------
J. P. Lewis, "Fast Normalized Cross-Correlation", Vision Interface, 1995.

Output is designed to match ``normxcorr2`` from MATLAB's Image Processing
Toolbox (MATLAB is a registered trademark of The MathWorks, Inc.).  The FFT
reference implementation by Ujash Joshi / Benjamin Eltzner
(https://github.com/Sabrewarrior/normxcorr2-python) was used for comparison.

Authors
-------
- pnanolocz Python implementation: Wang-1971.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import fftconvolve

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

__all__ = ["normxcorr2"]


def _local_sum(a: FloatArray, m: int, n: int) -> FloatArray:
    """Sum every ``m``-by-``n`` window of zero-padded ``a`` on the "full" grid.

    Parameters
    ----------
    a:
        2D array to sum over.
    m, n:
        Window size in rows and columns.

    Returns
    -------
    FloatArray
        Array of shape ``(a.shape[0] + m - 1, a.shape[1] + n - 1)`` — the same
        shape as ``fftconvolve(a, np.ones((m, n)), mode="full")`` — computed
        with two O(N) running-sum passes instead of an FFT.
    """
    # Pad so that every window of the "full" correlation grid lies inside.
    padded = np.pad(a, ((m - 1, m - 1), (n - 1, n - 1)))

    # Sum down each column: row_sums[i] is the sum of rows i .. i+m-1.
    row_cumsum = np.concatenate(
        [np.zeros((1, padded.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0)],
        axis=0,
    )
    row_sums = row_cumsum[m:] - row_cumsum[:-m]

    # Sum across each row of that result to complete the 2D window sum.
    col_cumsum = np.concatenate(
        [
            np.zeros((row_sums.shape[0], 1), dtype=np.float64),
            np.cumsum(row_sums, axis=1),
        ],
        axis=1,
    )
    return np.asarray(col_cumsum[:, n:] - col_cumsum[:, :-n], dtype=np.float64)


def normxcorr2(template: FloatArray, image: FloatArray) -> FloatArray:
    """Normalized 2D cross-correlation, matching MATLAB ``normxcorr2``.

    Parameters
    ----------
    template:
        2D template / filter.  Must be no larger than ``image`` in either
        dimension, as in MATLAB.
    image:
        2D image to search.

    Returns
    -------
    FloatArray
        Normalized cross-correlation map of shape ``(H + h - 1, W + w - 1)``
        for an ``(H, W)`` image and ``(h, w)`` template, i.e. the 'full' output,
        as in MATLAB's ``normxcorr2``.  Values lie in ``[-1, 1]``; windows where
        the image is flat (correlation undefined) are set to 0.

    Raises
    ------
    ValueError
        If either input is not 2D, or the template is larger than the image.

    Notes
    -----
    As in MATLAB, an image with negative values is first shifted to a minimum
    of 0, which only affects windows that overlap the zero-padded border.
    Windows whose variance is at the round-off level (flat regions) are set
    to 0, so which of those near-flat border windows come back as 0 may differ
    from MATLAB.

    Non-finite input values (NaN, +/-Inf) are replaced by zero, whereas MATLAB
    propagates NaNs.  ``Detector.m`` zeroes NaN pixels before detection, so this
    only affects direct callers.  A constant template returns an all-zero map,
    where MATLAB raises an error.
    """
    template = np.asarray(template, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)

    if template.ndim != 2 or image.ndim != 2:
        raise ValueError("normxcorr2 expects 2D template and image arrays")

    t_rows, t_cols = template.shape
    if t_rows > image.shape[0] or t_cols > image.shape[1]:
        raise ValueError(
            "Template shape must be <= image shape in all dimensions, got "
            f"template {template.shape} and image {image.shape}"
        )

    template = np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # As MATLAB does, shift an image with negative values to a minimum of 0.
    # This only changes the partial-overlap (border) part of the map, where the
    # zero padding meets the data.
    image_min = float(image.min())
    if image_min < 0.0:
        image = image - image_min

    out_shape = (image.shape[0] + t_rows - 1, image.shape[1] + t_cols - 1)

    # Zero-mean template: the mean contributes nothing to the correlation, so
    # the numerator can be computed with a single convolution.  Flipping the
    # template turns that convolution into a cross-correlation.
    template_zm = template - float(np.mean(template))
    template_energy = float(np.sum(template_zm**2))

    if template_energy <= 0.0:
        # Constant template: there is no pattern to correlate with.
        return np.zeros(out_shape, dtype=np.float64)

    numerator = fftconvolve(image, template_zm[::-1, ::-1], mode="full")

    # Local energy of the image over each template-sized window.  The variance
    # of a window is E[x^2] - E[x]^2, clamped at zero because round-off can
    # push a flat window slightly negative.
    area = float(t_rows * t_cols)
    local_sum = _local_sum(image, t_rows, t_cols)
    local_sum_sq = _local_sum(image * image, t_rows, t_cols)
    local_energy = np.maximum(local_sum_sq - local_sum**2 / area, 0.0)

    denom = np.sqrt(local_energy * template_energy)

    # Relative mask: only trust windows whose variance is significant compared
    # with the strongest window in the map.  Relative (not absolute) keeps the
    # result independent of the units of the input data.
    rtol = float(np.sqrt(np.finfo(np.float64).eps))
    out = np.zeros(out_shape, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > rtol * float(denom.max()))

    # Round-off backstop: MATLAB zeroes any |C| that exceeds 1.
    out[np.abs(out) - 1.0 > rtol] = 0.0

    return out


normxcorr2.__version__ = "0.1.0"  # type: ignore[attr-defined]
