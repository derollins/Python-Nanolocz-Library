"""
Bidirectional scan-line shift correction for NanoLocz-compatible AFM images.

This module ports MATLAB ``LineShift.m``.  It estimates the lateral shift between
alternating scan lines using either cross-correlation or peak positions, then
shifts odd/even rows to reduce trace/retrace mismatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import shift as ndi_shift
from scipy.signal import correlate, find_peaks, windows
from scipy.optimize import curve_fit

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _gauss1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """MATLAB gauss1 model."""
    return a * np.exp(-((x - b) ** 2) / (c**2))


def _rmoutliers_mean(values: np.ndarray, z: float = 3.0) -> np.ndarray:
    """Simple mean/std outlier removal substitute for MATLAB rmoutliers(...,'mean')."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return vals
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    if sd == 0:
        return vals
    return vals[np.abs(vals - mu) <= z * sd]


def line_shift(im: np.ndarray, shift_type: str | float | int) -> tuple[float, FloatArray]:
    """Estimate and correct alternating-line lateral shift.

    Parameters
    ----------
    im:
        2-D image or 3-D image where the first channel/frame is used.
    shift_type:
        ``'peaks'``, ``'ccr'`` or a numeric shift value.

    Returns
    -------
    shift, result:
        Estimated shift and corrected image.
    """
    arr = np.asarray(im, dtype=np.float64)
    if arr.ndim == 3:
        img = arr[:, :, 0]
    elif arr.ndim == 2:
        img = arr
    else:
        raise ValueError("im must be 2D or 3D")

    rows, cols = img.shape

    if isinstance(shift_type, str):
        mode = shift_type.lower()
    else:
        mode = "manual"

    w = windows.gaussian(20, std=20 / 5)
    w = w / np.sum(w)

    shift_i = []
    s_vals = []

    for i in range(rows - 1):
        r = img[i + 1, :]
        rt = img[i, :]

        if mode == "ccr":
            corr = correlate(rt - np.mean(rt), r - np.mean(r), mode="full")
            lags = np.arange(-cols + 1, cols)
            keep = (lags >= -40) & (lags <= 40)
            corr = corr[keep]
            lags = lags[keep]
            local_shift = float(lags[int(np.argmax(corr))])

        elif mode == "peaks":
            rf = np.convolve(r, w, mode="same")
            rtf = np.convolve(rt, w, mode="same")
            peaks1, props1 = find_peaks(rf, prominence=0.2)
            peaks2, props2 = find_peaks(rtf, prominence=0.2)
            if peaks1.size and peaks2.size:
                id1 = int(np.argmax(props1["prominences"]))
                id2 = int(np.argmax(props2["prominences"]))
                local_shift = float(peaks2[id2] - peaks1[id1])
            else:
                local_shift = 0.0
        else:
            local_shift = float(shift_type)

        if (i + 1) % 2 == 0:
            local_shift = -local_shift

        shift_i.append(local_shift)
        s_vals.append(float(np.sum((rt + r) / (2.0 * rt.size))))

    shift_i_arr = np.asarray(shift_i, dtype=np.float64)
    s_arr = np.asarray(s_vals, dtype=np.float64)

    if mode in {"ccr", "peaks"}:
        P = np.percentile(s_arr, 50)
        candidates = shift_i_arr[s_arr > P]
        candidates = _rmoutliers_mean(candidates)

        if mode == "ccr":
            candidates = candidates[(candidates != 0) & (candidates != 40) & (candidates != -40)]

        if candidates.size == 0:
            shift = 0.0
        else:
            bins = np.arange(np.nanmin(candidates), np.nanmax(candidates) + 1)
            if bins.size <= 1:
                shift = -round(float(candidates[0]))
            else:
                hy, edges = np.histogram(candidates, bins=np.append(bins, bins[-1] + 1))
                x = bins.astype(float)
                try:
                    popt, _ = curve_fit(
                        _gauss1,
                        x,
                        hy.astype(float),
                        p0=[30, 0, 20],
                        bounds=([0, -20, 1], [1000, 20, 50]),
                        maxfev=10000,
                    )
                    shift = -round(float(popt[1]))
                except Exception:
                    shift = -round(float(x[int(np.argmax(hy))]))
    else:
        shift = float(shift_type)

    # Apply correction by shifting alternating rows.  This reproduces the
    # effective MATLAB crop behavior while avoiding imcrop edge ambiguity.
    result = np.zeros_like(img, dtype=np.float64)
    if shift == 0:
        result = img.astype(np.float64)
    else:
        for rr in range(rows):
            if shift > 0:
                row_shift = shift / 2.0 if rr % 2 == 0 else -shift / 2.0
            else:
                row_shift = shift / 2.0 if rr % 2 == 0 else -shift / 2.0
            result[rr, :] = ndi_shift(img[rr, :], shift=row_shift, order=1, mode="nearest")

    return float(shift), np.asarray(result, dtype=np.float64)


LineShift = line_shift

__all__ = ["line_shift", "LineShift"]
