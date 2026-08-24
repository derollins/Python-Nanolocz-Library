"""
AFM image leveling routines for NanoLocz-compatible Python workflows.

This module is a MATLAB-aligned port of NanoLocz ``level.m``.  It provides
background flattening methods for 2-D AFM images and frame-first 3-D image
stacks.  The public Python mask convention is intentionally different from the
numeric MATLAB mask convention:

    Python mask:  True  = excluded pixel
                  False = valid / included pixel

In the original MATLAB code, masks are usually numeric arrays where valid pixels
are ``1`` and excluded pixels are ``NaN``.  Internally this module converts the
Python exclusion mask into a finite-aware validity mask and then performs
NaN-outside computations to mimic MATLAB's ``omitnan`` behavior.

The main entry point is :func:`apply_level`.  The compatibility wrapper
:func:`level` is provided for code that prefers the MATLAB argument order.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

SMOOTHING_WINDOW = 10
LOG_FIT_BOUNDS = ([0.1, 0.01, 0.1], [1000.0, 20.0, 100.0])


FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]


def _validity_mask(
    arr: FloatArray,
    mask_excl: BoolArray | None,
    *,
    name: str = "mask",
) -> BoolArray:
    """Convert an exclusion mask into a finite-aware validity mask.

    Parameters
    ----------
    arr:
        A 2-D image frame.
    mask_excl:
        Optional boolean exclusion mask with the same shape as ``arr``.
        ``True`` means excluded; ``False`` means valid.
    name:
        Human-readable name used in validation errors.

    Returns
    -------
    ndarray of bool
        ``True`` for pixels that are valid for fitting, ``False`` for excluded
        or non-finite pixels.

    MATLAB alignment note
    ---------------------
    MATLAB NanoLocz masks use ``1`` for valid pixels and ``NaN`` for excluded
    pixels.  This helper produces the equivalent validity mask and always
    removes non-finite image values from fitting.
    """
    finite = np.isfinite(arr)

    if mask_excl is None:
        return np.asarray(finite, dtype=np.bool_)

    m_excl = np.asarray(mask_excl, dtype=np.bool_)
    if m_excl.shape != arr.shape:
        raise ValueError(
            f"{name} shape {m_excl.shape} must match image shape {arr.shape}"
        )

    return np.asarray((~m_excl) & finite, dtype=np.bool_)


def _polyfit_centered(
    x: np.ndarray, y: np.ndarray, order: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """Fit a polynomial using MATLAB ``polyfit(..., mu)`` style scaling.

    MATLAB's third ``polyfit`` output ``mu`` is ``[mean(x), std(x)]`` where
    ``std`` uses the sample standard deviation.  NumPy's default standard
    deviation is population-based, so we explicitly use ``ddof=1`` here.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size <= order:
        raise ValueError("not enough finite samples for polynomial fit")

    center = float(np.mean(x))
    scale = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    xs = (x - center) / scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*[Rr]ank.*")
        coeffs = np.polyfit(xs, y, order)

    return np.asarray(coeffs, dtype=np.float64), (center, scale)


def _polyval_centered(
    coeffs: np.ndarray,
    mu: tuple[float, float],
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate a polynomial fitted by :func:`_polyfit_centered`."""
    center, scale = mu
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    points = np.asarray(points, dtype=np.float64)
    return np.asarray(np.polyval(coeffs, (points - center) / scale), dtype=np.float64)


def _nanmedian_or_nan(values: np.ndarray) -> float:
    """Return ``nanmedian`` while suppressing all-NaN warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if np.isfinite(values).any():
            return float(np.nanmedian(values))
    return float("nan")


def _movmedian_centered(x: np.ndarray, window: int) -> np.ndarray:
    """Approximate MATLAB ``movmedian(x, window)`` for a 1-D vector.

    The implementation uses a centered, shrinking window at the boundaries.
    NaNs are omitted because NanoLocz usually uses these medians as robust
    background estimates rather than as NaN-propagating diagnostics.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()

    window = max(int(window), 1)
    left = window // 2
    right = window - left

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - left)
        end = min(n, i + right)
        out[i] = _nanmedian_or_nan(x[start:end])
    return out


def level_plane(
    img: FloatArray,
    mask: BoolArray | None,
    polyx: int,
    polyy: int,
) -> FloatArray:
    """Subtract a polynomial plane using masked column and row means.

    This mirrors the MATLAB ``case 'plane'`` logic:

    1. Compute column means over valid pixels and fit an X polynomial.
    2. Subtract the X background from the image.
    3. Compute row means on the partially leveled image and fit a Y polynomial.
    4. Subtract the Y background.

    Coordinates are 1-based to match MATLAB's ``1:numel(...)`` indexing.
    """
    arr = np.asarray(img, dtype=np.float64)
    valid = _validity_mask(arr, mask)

    if valid.sum() <= 5:
        return np.asarray(arr.copy(), dtype=np.float64)

    out = arr.copy()

    # X direction: column mean profile.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xp = np.nanmean(np.where(valid, arr, np.nan), axis=0)

    valid_cols = np.isfinite(xp)
    if valid_cols.sum() > polyx:
        cols = (np.nonzero(valid_cols)[0] + 1).astype(np.float64)
        try:
            coeffs_x, mu_x = _polyfit_centered(cols, xp[valid_cols], int(polyx))
            all_cols = np.arange(1, arr.shape[1] + 1, dtype=np.float64)
            out = out - _polyval_centered(coeffs_x, mu_x, all_cols)[None, :]
        except ValueError:
            pass

    # Y direction: row mean profile after X subtraction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yp = np.nanmean(np.where(valid, out, np.nan), axis=1)

    valid_rows = np.isfinite(yp)
    if valid_rows.sum() > polyy:
        rows = (np.nonzero(valid_rows)[0] + 1).astype(np.float64)
        try:
            coeffs_y, mu_y = _polyfit_centered(rows, yp[valid_rows], int(polyy))
            all_rows = np.arange(1, arr.shape[0] + 1, dtype=np.float64)
            out = out - _polyval_centered(coeffs_y, mu_y, all_rows)[:, None]
        except ValueError:
            pass

    return np.asarray(out, dtype=np.float64)


def level_line(
    img: FloatArray,
    mask: BoolArray | None,
    polyx: int,
    polyy: int,
) -> FloatArray:
    """Subtract row-wise and optional column-wise polynomial line trends.

    This implements the MATLAB ``case 'line'`` behavior.  The X stage fits a
    polynomial to each row using valid pixels.  Rows with too few valid pixels
    fall back to the median fitted row background.  The optional Y stage then
    fits a polynomial to each column of the partially leveled image.
    """
    arr = np.asarray(img, dtype=np.float64)
    valid = _validity_mask(arr, mask)
    out = arr.copy()

    rows, cols = arr.shape

    if polyx > 0:
        row_fits = np.full_like(arr, np.nan, dtype=np.float64)
        fitted_rows: list[int] = []
        fallback_rows: list[int] = []

        for rr in range(rows):
            pos = valid[rr, :]
            if int(pos.sum()) > int(polyx) + 8:
                x = (np.nonzero(pos)[0] + 1).astype(np.float64)
                y = arr[rr, pos]
                try:
                    coeffs, mu = _polyfit_centered(x, y, int(polyx))
                    grid = np.arange(1, cols + 1, dtype=np.float64)
                    fit = _polyval_centered(coeffs, mu, grid)
                    row_fits[rr, :] = fit
                    out[rr, :] = arr[rr, :] - fit
                    fitted_rows.append(rr)
                except ValueError:
                    fallback_rows.append(rr)
            else:
                fallback_rows.append(rr)

        if fitted_rows and fallback_rows:
            median_curve = np.nanmedian(row_fits[fitted_rows, :], axis=0)
            for rr in fallback_rows:
                out[rr, :] = arr[rr, :] - median_curve

    if polyy > 0:
        for cc in range(cols):
            yp = np.where(valid[:, cc], out[:, cc], np.nan)
            good = np.isfinite(yp)

            if good.sum() < int(polyy) + 1:
                # MATLAB has a slightly inconsistent fallback here.  Keeping the
                # column unchanged is safer and avoids underdetermined polyfits.
                out[:, cc] = arr[:, cc]
                continue

            y = yp[good]
            x = (np.nonzero(good)[0] + 1).astype(np.float64)
            try:
                coeffs, mu = _polyfit_centered(x, y, int(polyy))
                grid = np.arange(1, rows + 1, dtype=np.float64)
                out[:, cc] = out[:, cc] - _polyval_centered(coeffs, mu, grid)
            except ValueError:
                out[:, cc] = arr[:, cc]

    return np.asarray(out, dtype=np.float64)


def level_med_line(
    img: FloatArray,
    mask: BoolArray | None,
    polyx: float,
    polyy: int = 0,
) -> FloatArray:
    """Subtract a per-row median baseline and restore the global median.

    MATLAB NanoLocz uses ``polyx`` as a *strength* parameter for ``med_line``.
    If ``polyx > 0`` the row median is multiplied by ``polyx``; otherwise the
    row median is subtracted with strength 1.0.  This is why ``level_auto.m``
    sometimes calls ``med_line`` with ``polyx = 0.6``.
    """
    del polyy  # kept only for API parity

    arr = np.asarray(img, dtype=np.float64)
    valid = _validity_mask(arr, mask)
    masked = np.where(valid, arr, np.nan)
    bg = _nanmedian_or_nan(masked)

    strength = float(polyx) if float(polyx) > 0 else 1.0
    out = arr.copy()

    for rr in range(arr.shape[0]):
        pos = valid[rr, :]
        if int(pos.sum()) > 10:
            row_med = float(np.median(arr[rr, pos]))
            out[rr, :] = arr[rr, :] - strength * row_med + bg

    return np.asarray(out, dtype=np.float64)


def level_med_line_y(
    img: FloatArray,
    mask: BoolArray | None,
    polyx: int = 0,
    polyy: int = 0,
) -> FloatArray:
    """Subtract a per-column median baseline and restore the global median."""
    del polyx, polyy  # kept only for API parity

    arr = np.asarray(img, dtype=np.float64)
    valid = _validity_mask(arr, mask)
    bg = _nanmedian_or_nan(np.where(valid, arr, np.nan))
    out = arr.copy()

    for cc in range(arr.shape[1]):
        pos = valid[:, cc]
        if int(pos.sum()) > 10:
            col_med = float(np.median(arr[pos, cc]))
            out[:, cc] = arr[:, cc] - col_med + bg

    return np.asarray(out, dtype=np.float64)


def level_smed_line(
    img: FloatArray,
    mask: BoolArray | None,
    polyx: int = 0,
    polyy: int = 0,
    *,
    smoothing_window: int = SMOOTHING_WINDOW,
) -> FloatArray:
    """Subtract a smoothed per-row median baseline.

    MATLAB formula:

    ``r = img - (y1 - movmedian(y1, 10))``

    where ``y1`` is the row-wise median baseline plus the global background.
    """
    del polyx, polyy  # kept only for API parity

    arr = np.asarray(img, dtype=np.float64)
    valid = _validity_mask(arr, mask)
    bg = _nanmedian_or_nan(np.where(valid, arr, np.nan))

    row_baseline = np.empty(arr.shape[0], dtype=np.float64)
    for rr in range(arr.shape[0]):
        pos = valid[rr, :]
        if int(pos.sum()) > 10:
            row_baseline[rr] = float(np.median(arr[rr, pos])) + bg
        else:
            row_baseline[rr] = bg

    smooth_baseline = _movmedian_centered(row_baseline, smoothing_window)
    correction = row_baseline - smooth_baseline
    correction = np.where(np.isfinite(correction), correction, 0.0)

    return np.asarray(arr - correction[:, None], dtype=np.float64)


def level_mean_plane(
    img: FloatArray,
    mask: BoolArray | None,
    polyx: int = 0,
    polyy: int = 0,
) -> FloatArray:
    """Subtract the masked mean value from the image.

    This corresponds to MATLAB ``case 'mean_plane'``.  The polynomial arguments
    are ignored but kept for dispatcher compatibility.
    """
    del polyx, polyy

    arr = np.asarray(img, dtype=np.float64)
    valid = _validity_mask(arr, mask)
    masked = np.where(valid, arr, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        offset = float(np.nanmean(masked)) if np.isfinite(masked).any() else 0.0

    return np.asarray(arr - offset, dtype=np.float64)


def _log_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Model used by MATLAB ``fittype("a*log((c*x)+b)")``."""
    return a * np.log((c * x) + b)


def level_log_y(
    img: FloatArray,
    mask: BoolArray | None = None,
    polyx: int = 0,
    polyy: float = 1.0,
) -> FloatArray:
    """Subtract a fitted logarithmic trend along the Y direction.

    The MATLAB implementation wraps this method in ``try/catch`` because the
    nonlinear fit can fail.  This Python version follows the same principle and
    returns the original image unchanged if fitting is not possible.
    """
    del mask, polyx

    arr = np.asarray(img, dtype=np.float64)

    try:
        if float(polyy) == 0.0:
            return arr.copy()

        y = np.nanmean(arr, axis=1)
        y = y - np.nanmin(y)

        x = np.arange(1, y.size + 1, dtype=np.float64)
        x = x / float(polyy) / float(y.size) * 10.0

        y_fit = np.flip(y).astype(np.float64)
        xi = x.copy()

        pos = x < 5
        x_fit = x[pos]
        y_fit = y_fit[pos]

        if x_fit.size < 4 or not np.isfinite(y_fit).all():
            return arr.copy()

        params, _ = curve_fit(
            _log_model,
            x_fit,
            y_fit,
            p0=[5.0, 1.0, 2.0],
            bounds=LOG_FIT_BOUNDS,
            maxfev=10000,
        )

        trend = np.flip(_log_model(xi, *params))
        normal = arr - trend[:, None]
        reverse = arr - trend[::-1, None]
        normal_range = np.ptp(np.nanmean(normal, axis=1))
        reverse_range = np.ptp(np.nanmean(reverse, axis=1))
        selected = normal if normal_range <= reverse_range else reverse
        return np.asarray(selected, dtype=np.float64)

    except Exception:
        return np.asarray(arr.copy(), dtype=np.float64)


_METHODS = {
    "plane": level_plane,
    "line": level_line,
    "med_line": level_med_line,
    "med_line_y": level_med_line_y,
    "smed_line": level_smed_line,
    "mean_plane": level_mean_plane,
    "log_y": level_log_y,
}


def _prepare_stack_and_mask(
    img: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Convert 2-D/3-D inputs into a frame-first stack."""
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        stack = arr[np.newaxis, :, :]
        was_2d = True
    elif arr.ndim == 3:
        stack = arr
        was_2d = False
    else:
        raise ValueError("img must be 2D or frame-first 3D with shape (N, H, W)")

    if mask is None:
        return stack, None, was_2d

    mask_arr = np.asarray(mask, dtype=np.bool_)
    if mask_arr.ndim == 2:
        if mask_arr.shape != stack.shape[1:]:
            raise ValueError("2D mask must match image frame shape")
        mask_stack = np.broadcast_to(mask_arr, stack.shape)
    elif mask_arr.ndim == 3:
        if mask_arr.shape != stack.shape:
            raise ValueError("3D mask must match stack shape")
        mask_stack = mask_arr
    else:
        raise ValueError("mask must be 2D or frame-first 3D")

    return stack, np.asarray(mask_stack, dtype=np.bool_), was_2d


def apply_level(
    img: np.ndarray,
    polyx: float,
    polyy: float,
    method: str,
    mask: np.ndarray | None = None,
    *,
    smoothing_window: int = SMOOTHING_WINDOW,
) -> np.ndarray:
    """Apply a NanoLocz leveling method to a 2-D image or frame-first stack.

    Parameters
    ----------
    img:
        2-D image ``(H, W)`` or frame-first stack ``(N, H, W)``.
    polyx, polyy:
        Polynomial orders or method-specific parameters.  ``med_line`` uses
        ``polyx`` as a row-median strength, matching MATLAB.
    method:
        One of ``'plane'``, ``'line'``, ``'med_line'``, ``'med_line_y'``,
        ``'smed_line'``, ``'mean_plane'``, or ``'log_y'``.
    mask:
        Optional Python exclusion mask where ``True`` means excluded.
    smoothing_window:
        Moving-median window used only by ``smed_line``.

    Returns
    -------
    ndarray
        Leveled data with the same dimensionality as the input.
    """
    method_lc = method.lower()
    if method_lc not in _METHODS:
        raise ValueError(f"Unknown leveling method: {method!r}")

    stack, mask_stack, was_2d = _prepare_stack_and_mask(img, mask)
    out = np.empty_like(stack, dtype=np.float64)

    func = _METHODS[method_lc]
    for idx in range(stack.shape[0]):
        frame_mask = None if mask_stack is None else mask_stack[idx]
        if method_lc == "smed_line":
            out[idx] = func(
                stack[idx],
                frame_mask,
                polyx,
                polyy,
                smoothing_window=smoothing_window,
            )
        else:
            out[idx] = func(stack[idx], frame_mask, polyx, polyy)

    return np.asarray(out[0] if was_2d else out, dtype=np.float64)


def get_background(
    img: np.ndarray,
    polyx: float,
    polyy: float,
    method: str,
    mask: np.ndarray | None = None,
    *,
    smoothing_window: int = SMOOTHING_WINDOW,
) -> np.ndarray:
    """Return the background removed by :func:`apply_level`."""
    arr = np.asarray(img, dtype=np.float64)
    leveled = apply_level(
        arr,
        polyx=polyx,
        polyy=polyy,
        method=method,
        mask=mask,
        smoothing_window=smoothing_window,
    )
    return np.asarray(arr - leveled, dtype=np.float64)


get_background.__version__ = "0.1.0"  # type: ignore[attr-defined]


def level(
    img: np.ndarray,
    polyx: float,
    polyy: float,
    line_plane: str,
    imgt: np.ndarray | None = None,
) -> np.ndarray:
    """MATLAB-style compatibility wrapper around :func:`apply_level`.

    ``imgt`` must already follow the Python convention if it is boolean
    (``True = excluded``).  Numeric MATLAB-like masks can be converted with
    ``pnanolocz.thresholder.selection`` before calling this function.
    """
    return apply_level(img, polyx=polyx, polyy=polyy, method=line_plane, mask=imgt)


__all__ = [
    "apply_level",
    "get_background",
    "level",
    "level_plane",
    "level_line",
    "level_med_line",
    "level_med_line_y",
    "level_smed_line",
    "level_mean_plane",
    "level_log_y",
]
