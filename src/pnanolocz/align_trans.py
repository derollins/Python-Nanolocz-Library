"""
Translational alignment utilities for NanoLocz-compatible AFM workflows.

This module is a Python port of NanoLocz ``align_trans.m``.  It estimates
frame-wise X/Y translation between an image stack and a reference image using
either spatial normalized cross-correlation or FFT phase correlation.

Coordinate convention
---------------------
The returned shifts follow the MATLAB NanoLocz convention:

    x = horizontal / column shift
    y = vertical / row shift

The shifts represent the correction added to particle/localisation coordinates.
To physically translate an image in SciPy, use ``shift=(-y, -x)`` because
``scipy.ndimage.shift`` expects ``(row_shift, col_shift)``.

Stack convention
----------------
Python default is frame-first: ``(frames, rows, cols)``.  Use ``frame_axis=-1``
for MATLAB-style stacks ``(rows, cols, frames)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

try:
    from skimage.registration import phase_cross_correlation
except Exception:  # pragma: no cover - optional dependency guard
    phase_cross_correlation = None  # type: ignore[assignment]


FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _as_frame_first(img: np.ndarray, frame_axis: int = 0) -> tuple[FloatArray, bool, int]:
    """Convert a 2-D image or 3-D stack to frame-first layout."""
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0

    if arr.ndim != 3:
        raise ValueError("img must be 2D or 3D")

    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _nan_to_zero(arr: np.ndarray) -> FloatArray:
    """Replace NaN/Inf values with zero, matching the MATLAB preprocessing."""
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.isfinite(out)] = 0.0
    return np.asarray(out, dtype=np.float64)


def normxcorr2(template: np.ndarray, image: np.ndarray) -> FloatArray:
    """Normalized 2-D cross-correlation, matching MATLAB ``normxcorr2``.

    Re-exported from :mod:`pnanolocz.normxcorr2` for backward compatibility.
    See that module for full documentation and implementation details.
    """
    from pnanolocz.normxcorr2 import normxcorr2 as _impl

    return _impl(template, image)


def _peak_abs(corr: np.ndarray) -> tuple[int, int]:
    """Return the row/column index of the maximum absolute correlation."""
    idx = int(np.nanargmax(np.abs(corr)))
    return tuple(int(v) for v in np.unravel_index(idx, corr.shape))  # type: ignore[return-value]


def _subpixel_offset_from_zoom(
    corr: np.ndarray,
    y: int,
    x: int,
    *,
    half_window: int = 2,
    zoom_factor: int = 100,
    order: int = 1,
) -> tuple[float, float]:
    """Estimate sub-pixel offset by zooming a small correlation window.

    This follows the MATLAB pattern:

    ``clip_c = ccr(y-2:y+2, x-2:x+2)``
    ``c_zoom = imresize(clip_c, 100, ...)``

    Returns
    -------
    x_offset, y_offset:
        Sub-pixel offsets in correlation-map pixels.
    """
    y0 = max(0, int(y) - half_window)
    y1 = min(corr.shape[0], int(y) + half_window + 1)
    x0 = max(0, int(x) - half_window)
    x1 = min(corr.shape[1], int(x) + half_window + 1)

    if (y1 - y0) < 3 or (x1 - x0) < 3:
        return 0.0, 0.0

    clip = corr[y0:y1, x0:x1]
    zoomed = zoom(clip, zoom=zoom_factor, order=order)

    zy, zx = _peak_abs(zoomed)

    # MATLAB uses ``peak - size/2`` after imresize.  This is intentionally
    # kept close to that convention rather than using a parabola fit.
    x_offset = (float(zx) - (zoomed.shape[1] / 2.0)) / float(zoom_factor)
    y_offset = (float(zy) - (zoomed.shape[0] / 2.0)) / float(zoom_factor)

    return x_offset, y_offset


def _align_trans_cross_corr(
    stack: FloatArray,
    ref: FloatArray,
    pixel_shift: float,
    sub_pix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial normalized cross-correlation branch of ``align_trans``."""
    n_frames, rows, cols = stack.shape
    window = int(round(pixel_shift))

    x_peak = np.zeros(n_frames, dtype=np.float64)
    y_peak = np.zeros(n_frames, dtype=np.float64)
    x_offset = np.zeros(n_frames, dtype=np.float64)
    y_offset = np.zeros(n_frames, dtype=np.float64)

    # Relative local-window increments used when ``pixel_shift > 0``.
    xw = np.zeros(n_frames, dtype=np.float64)
    yw = np.zeros(n_frames, dtype=np.float64)

    for idx in range(n_frames):
        corr = normxcorr2(ref, stack[idx])
        corr_rows, corr_cols = corr.shape

        y0, x0 = _peak_abs(corr)

        # MATLAB optionally constrains frames after the first to a local search
        # window around the previous correlation peak.
        if pixel_shift > 0 and idx > 0:
            prev_y = int(round(y_peak[idx - 1]))
            prev_x = int(round(x_peak[idx - 1]))

            y_start = prev_y - window + 1
            y_end = prev_y + window
            x_start = prev_x - window + 1
            x_end = prev_x + window

            if 0 <= y_start < y_end <= corr_rows and 0 <= x_start < x_end <= corr_cols:
                local = corr[y_start:y_end, x_start:x_end]
                ly, lx = _peak_abs(local)

                # MATLAB's xw/yw are local offsets relative to the window center.
                xw[idx] = (lx + 1) - window
                yw[idx] = (ly + 1) - window

                x_peak[idx] = x_peak[0] + np.sum(xw[: idx + 1])
                y_peak[idx] = y_peak[0] + np.sum(yw[: idx + 1])
            else:
                x_peak[idx] = float(x0)
                y_peak[idx] = float(y0)
        else:
            x_peak[idx] = float(x0)
            y_peak[idx] = float(y0)

        if sub_pix:
            xo, yo = _subpixel_offset_from_zoom(
                corr,
                int(round(y_peak[idx])),
                int(round(x_peak[idx])),
                half_window=2,
                zoom_factor=100,
                order=1,
            )
            x_offset[idx] = xo
            y_offset[idx] = yo

    # MATLAB uses 1-based peak indices and subtracts image width/height.
    # Python peak indices are 0-based, so add 1 before subtracting.
    x = (x_peak + 1.0) + x_offset - float(cols)
    y = (y_peak + 1.0) + y_offset - float(rows)

    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _align_trans_fft(
    stack: FloatArray,
    ref: FloatArray,
    upsample_factor: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT phase-correlation branch of ``align_trans``.

    This uses scikit-image's Guizar-Sicairos phase cross-correlation
    implementation.  The sign is chosen to match NanoLocz MATLAB output:
    ``x = -net_col_shift`` and ``y = -net_row_shift``.
    """
    if phase_cross_correlation is None:
        raise ImportError("scikit-image is required for method='FFT cross'")

    n_frames = stack.shape[0]
    x = np.zeros(n_frames, dtype=np.float64)
    y = np.zeros(n_frames, dtype=np.float64)

    ref_clean = _nan_to_zero(ref)

    for idx in range(n_frames):
        moving = _nan_to_zero(stack[idx])
        shift, _, _ = phase_cross_correlation(
            ref_clean,
            moving,
            upsample_factor=upsample_factor,
            normalization=None,
        )
        # shift = (row_shift, col_shift) needed to align moving to reference.
        # MATLAB dftregistration branch returns x=-output(4), y=-output(3).
        y[idx] = -float(shift[0])
        x[idx] = -float(shift[1])

    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def align_trans(
    img: np.ndarray,
    ref: np.ndarray,
    pixel_shift: float,
    sub_pix: bool | int,
    method: str,
    *,
    frame_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate translation shifts for each frame in an image stack.

    Parameters
    ----------
    img:
        2-D image or 3-D stack.  Python default stack layout is
        ``(frames, rows, cols)``.
    ref:
        2-D reference image.
    pixel_shift:
        Maximum local-search drift.  If ``> 0``, frames after the first are
        searched around the previous peak.
    sub_pix:
        Enable sub-pixel refinement for the spatial cross-correlation branch.
    method:
        ``'Cross corr'`` or ``'FFT cross'``.
    frame_axis:
        Frame axis for 3-D stacks.  Use ``-1`` for MATLAB layout.

    Returns
    -------
    x, y:
        One shift per frame, using NanoLocz's horizontal/vertical convention.
    """
    stack, _, _ = _as_frame_first(img, frame_axis=frame_axis)
    stack = _nan_to_zero(stack)
    ref_arr = _nan_to_zero(ref)

    method_lc = method.lower()

    if method_lc == "cross corr":
        return _align_trans_cross_corr(
            stack,
            ref_arr,
            pixel_shift=float(pixel_shift),
            sub_pix=bool(sub_pix),
        )

    if method_lc == "fft cross":
        return _align_trans_fft(stack, ref_arr, upsample_factor=100)

    raise ValueError("method must be either 'Cross corr' or 'FFT cross'")


def align_trans_matlab(
    img: np.ndarray,
    ref: np.ndarray,
    pixel_shift: float,
    sub_pix: bool | int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-layout wrapper for ``align_trans``.

    Assumes ``img`` is shaped ``(rows, cols, frames)``.
    """
    return align_trans(
        img,
        ref,
        pixel_shift=pixel_shift,
        sub_pix=sub_pix,
        method=method,
        frame_axis=-1,
    )


__all__ = [
    "align_trans",
    "align_trans_matlab",
    "normxcorr2",
]
