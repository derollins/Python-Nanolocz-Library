"""
Peak sharpening by even derivatives.

This module ports MATLAB ``sharpen.m`` by Thomas C. O'Haver, as used in
NanoLocz.  It also exposes ``fastsmooth`` and ``secderiv`` for compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def secderiv(x: np.ndarray, a: np.ndarray) -> FloatArray:
    """Second derivative with respect to x using a 3-point central difference."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(a, dtype=np.float64).ravel()

    if xx.size != yy.size:
        raise ValueError("x and a must have the same length")

    n = yy.size
    d = np.zeros_like(yy, dtype=np.float64)

    if n < 3:
        return d

    # MATLAB loop: for j = 2:n-2.  Python equivalent leaves the final interior
    # point untouched in the same spirit as the original implementation.
    for j in range(1, max(1, n - 2)):
        x1, x2, x3 = xx[j - 1], xx[j], xx[j + 1]
        denom1 = x3 - x2
        denom2 = x2 - x1
        denom3 = (x3 - x1) / 2.0
        if denom1 == 0 or denom2 == 0 or denom3 == 0:
            d[j] = 0.0
        else:
            d[j] = ((yy[j + 1] - yy[j]) / denom1 - (yy[j] - yy[j - 1]) / denom2) / denom3

    if n >= 2:
        d[0] = d[1]
        d[-1] = d[-2]

    return np.asarray(d, dtype=np.float64)


def _smooth_rect(y: np.ndarray, w: int, ends: int = 0) -> FloatArray:
    """Rectangular sliding average compatible with O'Haver fastsmooth."""
    data = np.asarray(y, dtype=np.float64).ravel()
    width = max(1, int(round(w)))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    out = np.convolve(data, kernel, mode="same")

    if int(ends) == 1 and width > 1:
        half = width // 2
        n = data.size
        for k in range(min(half, n)):
            out[k] = np.mean(data[: k + half + 1])
            out[n - k - 1] = np.mean(data[max(0, n - k - half - 1) :])

    return np.asarray(out, dtype=np.float64)


def fastsmooth(
    Y: np.ndarray,
    w: int,
    smooth_type: int = 1,
    ends: int = 0,
) -> FloatArray:
    """Smooth a vector with rectangular/triangular/pseudo-Gaussian passes."""
    out = np.asarray(Y, dtype=np.float64).ravel()
    stype = int(smooth_type)

    if stype == 1:
        out = _smooth_rect(out, w, ends)
    elif stype == 2:
        out = _smooth_rect(_smooth_rect(out, w, ends), w, ends)
    elif stype == 3:
        out = _smooth_rect(_smooth_rect(_smooth_rect(out, w, ends), w, ends), w, ends)
    elif stype == 4:
        out = _smooth_rect(_smooth_rect(_smooth_rect(_smooth_rect(out, w, ends), w, ends), w, ends), w, ends)
    elif stype == 5:
        out = _smooth_rect(out, w, ends)
        out = _smooth_rect(out, max(1, int(round(w * 1.2))), ends)
        out = _smooth_rect(out, max(1, int(round(w * 1.4))), ends)
        out = _smooth_rect(out, max(1, int(round(w * 1.6))), ends)
    else:
        out = _smooth_rect(out, w, ends)

    return np.asarray(out, dtype=np.float64)


def sharpen(
    x: np.ndarray,
    y: np.ndarray,
    factor1: float,
    factor2: float,
    smooth_width: int,
) -> FloatArray:
    """Sharpen peaks using second and fourth derivative weighting."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()

    d2 = secderiv(xx, yy)
    d4 = secderiv(xx, d2)

    sharpened = (
        yy
        - float(factor1) * fastsmooth(d2, smooth_width, 3)
        + float(factor2)
        * fastsmooth(
            fastsmooth(fastsmooth(d4, smooth_width), smooth_width, 3),
            smooth_width,
            3,
        )
    )

    return np.asarray(sharpened, dtype=np.float64)


__all__ = ["sharpen", "fastsmooth", "secderiv"]
