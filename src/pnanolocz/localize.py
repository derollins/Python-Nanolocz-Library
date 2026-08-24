"""
Sub-pixel feature localization for NanoLocz-compatible AFM workflows.

This module ports MATLAB ``localize.m``.  It refines initial feature
coordinates with interpolation, 2-D Gaussian fitting, or spherical-cap fitting.

Coordinate convention
---------------------
NanoLocz localisation tables are MATLAB-origin tables.  By default ``locs`` is
treated as 1-based in x/y and frame columns.  Set ``matlab_indexing=False`` for
0-based Python coordinates.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import curve_fit

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _matlab_round(x: np.ndarray | float) -> np.ndarray:
    """MATLAB-style half-away-from-zero rounding."""
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def _as_frame_first(target: np.ndarray, frame_axis: int) -> tuple[FloatArray, bool, int]:
    """Convert target image/stack to frame-first layout."""
    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0
    if arr.ndim != 3:
        raise ValueError("target must be 2D or 3D")
    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _ensure_locs_cols(locs: np.ndarray, n_cols: int) -> FloatArray:
    """Pad localisation table to at least ``n_cols`` columns."""
    arr = np.asarray(locs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("locs must be a 2D table")
    if arr.shape[1] >= n_cols:
        return arr.copy()
    # MATLAB fills intermediate columns with zero when an assignment extends a
    # numeric matrix (for example, assigning Gaussian fit results to columns
    # 10 and 11).  Zero padding also keeps the localization table consumable by
    # LAFM_renderer, which rejects rows containing NaN.
    out = np.zeros((arr.shape[0], n_cols), dtype=np.float64)
    out[:, : arr.shape[1]] = arr
    return out


def _frame_index(row: np.ndarray, n_frames: int, matlab_indexing: bool) -> int:
    """Resolve frame index from locs column 5 if present."""
    if row.size < 5 or not np.isfinite(row[4]):
        return 0
    idx = int(_matlab_round(row[4]).item())
    if matlab_indexing:
        idx -= 1
    return int(np.clip(idx, 0, n_frames - 1))


def _interp_order(method: str) -> int:
    """Map MATLAB imresize method names to scipy zoom spline order."""
    method_lc = method.lower()
    if method_lc == "bilinear":
        return 1
    if method_lc == "bicubic":
        return 3
    if method_lc in {"lanczos2", "lanczos3"}:
        # SciPy zoom does not implement Lanczos; cubic is the closest built-in.
        return 3
    return 3


def _two_d_gaussian(coords: tuple[np.ndarray, np.ndarray], amp: float, x0: float, sx: float, y0: float, sy: float) -> np.ndarray:
    """2-D Gaussian model used by the Gaussian localization method."""
    X, Y = coords
    return (amp * np.exp(-(((X - x0) ** 2) / (2 * sx**2) + ((Y - y0) ** 2) / (2 * sy**2)))).ravel()


def _two_d_gauss_fit(Z: np.ndarray) -> np.ndarray:
    """Fit the MATLAB ``TwoDGaussFit`` model."""
    z = np.asarray(Z, dtype=np.float64)
    z = z - np.nanmin(z)

    mdata_size = z.shape[0] - 1
    coords = np.arange(-mdata_size / 2.0, mdata_size / 2.0 + 1.0)
    X, Y = np.meshgrid(coords, coords)

    p0 = np.array([1.0, 0.0, 3.0, 0.0, 3.0], dtype=np.float64)
    lb = np.array([0.05, -2.0, 0.5, -2.0, 0.5], dtype=np.float64)
    ub = np.array([40.0, 2.0, 40.0, 2.0, 40.0], dtype=np.float64)

    try:
        popt, _ = curve_fit(
            _two_d_gaussian,
            (X, Y),
            z.ravel(),
            p0=p0,
            bounds=(lb, ub),
            maxfev=10000,
        )
        return np.asarray(popt, dtype=np.float64)
    except Exception:
        return p0


def _sphere_fit(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic sphere fit ported from MATLAB ``sumith_fit``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()

    N = float(x.size)

    Sx, Sy, Sz = np.sum(x), np.sum(y), np.sum(z)
    Sxx, Syy, Szz = np.sum(x*x), np.sum(y*y), np.sum(z*z)
    Sxy, Sxz, Syz = np.sum(x*y), np.sum(x*z), np.sum(y*z)

    Sxxx, Syyy, Szzz = np.sum(x*x*x), np.sum(y*y*y), np.sum(z*z*z)
    Sxyy, Sxzz, Sxxy = np.sum(x*y*y), np.sum(x*z*z), np.sum(x*x*y)
    Sxxz, Syyz, Syzz = np.sum(x*x*z), np.sum(y*y*z), np.sum(y*z*z)

    A1 = Sxx + Syy + Szz

    a = 2*Sx*Sx - 2*N*Sxx
    b = 2*Sx*Sy - 2*N*Sxy
    c = 2*Sx*Sz - 2*N*Sxz
    d = -N*(Sxxx + Sxyy + Sxzz) + A1*Sx

    e = b
    f = 2*Sy*Sy - 2*N*Syy
    g = 2*Sy*Sz - 2*N*Syz
    h = -N*(Sxxy + Syyy + Syzz) + A1*Sy

    j = c
    k = g
    l = 2*Sz*Sz - 2*N*Szz
    m = -N*(Sxxz + Syyz + Szzz) + A1*Sz

    delta = a*(f*l - g*k) - e*(b*l - c*k) + j*(b*g - c*f)
    if abs(delta) < np.finfo(float).eps:
        return np.nan, np.nan, np.nan, np.nan

    xc = (d*(f*l-g*k) - h*(b*l-c*k) + m*(b*g-c*f)) / delta
    yc = (a*(h*l-m*g) - e*(d*l-m*c) + j*(d*g-h*c)) / delta
    zc = (a*(f*m-h*k) - e*(b*m-d*k) + j*(b*h-d*f)) / delta
    R = np.sqrt(xc*xc + yc*yc + zc*zc + (A1 - 2*(xc*Sx + yc*Sy + zc*Sz))/N)
    return float(xc), float(yc), float(zc), float(R)


def localize(
    target: np.ndarray,
    locs: np.ndarray,
    loc_method: str,
    pixperfeat: float,
    *,
    frame_axis: int = 0,
    matlab_indexing: bool = True,
) -> FloatArray:
    """Refine feature coordinates in an AFM image or stack."""
    stack, _, _ = _as_frame_first(target, frame_axis=frame_axis)
    out = _ensure_locs_cols(locs, 12)

    n_frames, rows, cols = stack.shape
    method = loc_method.lower()

    interpolation_methods = {
        "bicubic",
        "cvcubic",
        "bilinear",
        "lanczos3",
        "lanczos2",
    }
    if method in interpolation_methods:
        w = 3
        expand_factor = 10
        order = _interp_order(method)

        out[:, 0:2] = _matlab_round(out[:, 0:2])

        for jj in range(out.shape[0]):
            x = int(_matlab_round(out[jj, 0]).item())
            y = int(_matlab_round(out[jj, 1]).item())
            fidx = _frame_index(out[jj], n_frames, matlab_indexing)

            x0 = x - 1 if matlab_indexing else x
            y0 = y - 1 if matlab_indexing else y

            if y0 - w + 1 >= 0 and x0 - w + 1 >= 0 and y0 + w - 1 < rows and x0 + w - 1 < cols:
                clip = stack[fidx, y0 - w + 1 : y0 + w, x0 - w + 1 : x0 + w]
                if method == "cvcubic":
                    clip_z = cv2.resize(
                        clip,
                        (
                            clip.shape[1] * expand_factor,
                            clip.shape[0] * expand_factor,
                        ),
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    clip_z = zoom(clip, zoom=expand_factor, order=order)
                clip_z = clip_z[10:40, 10:40]
                iy, ix = np.unravel_index(int(np.nanargmax(clip_z)), clip_z.shape)
                out[jj, 0] = out[jj, 0] + ((ix + 1) - clip_z.shape[1] / 2.0) / expand_factor
                out[jj, 1] = out[jj, 1] + ((iy + 1) - clip_z.shape[0] / 2.0) / expand_factor
            else:
                out[jj, 0:2] = np.nan

    elif method == "gaussian":
        out[:, 0:2] = _matlab_round(out[:, 0:2])
        w = 2 if pixperfeat < 0.75 else 3

        for jj in range(out.shape[0]):
            x = int(_matlab_round(out[jj, 0]).item())
            y = int(_matlab_round(out[jj, 1]).item())
            fidx = _frame_index(out[jj], n_frames, matlab_indexing)
            x0 = x - 1 if matlab_indexing else x
            y0 = y - 1 if matlab_indexing else y

            if y0 - w + 1 > 1 and x0 - w + 1 > 1 and y0 + w - 1 < rows - 2 and x0 + w - 1 < cols - 2:
                clip = stack[fidx, y0 - w + 1 : y0 + w, x0 - w + 1 : x0 + w]
                params = _two_d_gauss_fit(clip)
                out[jj, 0] += params[1]
                out[jj, 1] += params[3]
                out[jj, 9] = (params[4] + params[2]) / 2.0
                out[jj, 10] = params[0]
            else:
                out[jj, 0:2] = np.nan

    elif method == "sphere":
        out[:, 0:2] = _matlab_round(out[:, 0:2])
        if pixperfeat < 0.5:
            w = 2
            const = 5
        else:
            w = 3
            const = 8

        for jj in range(out.shape[0]):
            x = int(_matlab_round(out[jj, 0]).item())
            y = int(_matlab_round(out[jj, 1]).item())
            fidx = _frame_index(out[jj], n_frames, matlab_indexing)
            x0 = x - 1 if matlab_indexing else x
            y0 = y - 1 if matlab_indexing else y

            if y0 - w + 1 > 1 and x0 - w + 1 > 1 and y0 + w - 1 < rows - 2 and x0 + w - 1 < cols - 2:
                clip = stack[fidx, y0 - w + 1 : y0 + w, x0 - w + 1 : x0 + w]
                clip = zoom(clip, zoom=3, order=3)
                rr, cc = np.indices(clip.shape)
                xc, yc, _, R = _sphere_fit(cc.ravel() + 1, rr.ravel() + 1, clip.ravel())
                out[jj, 0] += (xc - const) / 3.0
                out[jj, 1] += (yc - const) / 3.0
                out[jj, 9] = R
            else:
                out[jj, 0:2] = np.nan
    else:
        raise ValueError("loc_method must be interpolation, 'gaussian', or 'sphere'")

    return np.asarray(out, dtype=np.float64)


def localize_matlab(target: np.ndarray, locs: np.ndarray, loc_method: str, pixperfeat: float) -> FloatArray:
    """MATLAB-layout wrapper for ``localize``."""
    return localize(target, locs, loc_method, pixperfeat, frame_axis=-1, matlab_indexing=True)


__all__ = ["localize", "localize_matlab"]
