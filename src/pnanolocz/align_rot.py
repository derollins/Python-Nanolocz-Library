"""
Rotational alignment utilities for NanoLocz-compatible AFM workflows.

This module ports NanoLocz ``align_rot.m``.  Two methods are provided:

- ``'Rotation corr'``:
  Rotate the target image through an angle range and maximize Pearson
  correlation against the reference.

- ``'Polar Corr'``:
  Convert reference and target images to polar coordinates and estimate angular
  shift via normalized cross-correlation.

The implementation aims for algorithmic alignment with MATLAB NanoLocz.  It is
not expected to be bitwise identical because SciPy interpolation, rotation, and
correlation differ from MATLAB's image processing toolbox.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates, rotate, zoom
from scipy.optimize import minimize_scalar

try:
    from pnanolocz.align_trans import normxcorr2
except Exception:  # pragma: no cover - allows standalone use
    from align_trans import normxcorr2  # type: ignore


FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _clean_image(img: np.ndarray) -> FloatArray:
    """Convert an image to float and replace non-finite values with zero."""
    arr = np.asarray(img, dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = 0.0
    return np.asarray(arr, dtype=np.float64)


def _match_image_sizes(ref: np.ndarray, target: np.ndarray) -> tuple[FloatArray, FloatArray]:
    """Trim images from top/left so that both have identical shape.

    MATLAB ``align_rot.m`` removes extra leading rows/columns when the two input
    images do not have exactly the same size.  This helper reproduces that
    behavior closely.
    """
    r = _clean_image(ref)
    t = _clean_image(target)

    if r.shape[0] > t.shape[0]:
        r = r[r.shape[0] - t.shape[0] :, :]
    elif r.shape[0] < t.shape[0]:
        t = t[t.shape[0] - r.shape[0] :, :]

    if r.shape[1] > t.shape[1]:
        r = r[:, r.shape[1] - t.shape[1] :]
    elif r.shape[1] < t.shape[1]:
        t = t[:, t.shape[1] - r.shape[1] :]

    return np.asarray(r, dtype=np.float64), np.asarray(t, dtype=np.float64)


def _corr2(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient equivalent to MATLAB ``corr2``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0

    av = a[mask] - float(np.mean(a[mask]))
    bv = b[mask] - float(np.mean(b[mask]))

    denom = float(np.sqrt(np.sum(av**2) * np.sum(bv**2)))
    if denom <= np.finfo(float).eps:
        return 0.0

    return float(np.sum(av * bv) / denom)


def _rotation_corr(ref: np.ndarray, target: np.ndarray, angle_range: tuple[float, float]) -> float:
    """Direct rotational correlation branch of ``align_rot``."""
    min_angle, max_angle = float(angle_range[0]), float(angle_range[1])

    # MATLAB uses 0.5 degrees generally and 0.2 degrees for small ranges.
    step = 0.2 if (max_angle - min_angle) <= 20 else 0.5
    angles = np.arange(min_angle, max_angle + (0.5 * step), step, dtype=np.float64)

    ref_m, target_m = _match_image_sizes(ref, target)

    corrs = np.zeros(angles.size, dtype=np.float64)
    for idx, angle in enumerate(angles):
        rotated = rotate(
            target_m,
            angle=float(angle),
            reshape=False,
            order=3,  # bicubic-like interpolation
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        corrs[idx] = _corr2(rotated, ref_m)

    best_idx = int(np.argmax(corrs))
    best_angle = float(angles[best_idx])

    # MATLAB refines the discrete maximum using a spline interpolant and
    # fminsearch.  Here we use bounded scalar minimization on a cubic
    # interpolation around the best sampled angle.
    try:
        interp_kind = "cubic" if angles.size >= 4 else "linear"
        interpolant = interp1d(
            angles,
            corrs,
            kind=interp_kind,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        lo = max(min_angle, best_angle - step * 2.0)
        hi = min(max_angle, best_angle + step * 2.0)

        if hi > lo:
            result = minimize_scalar(
                lambda a: -float(interpolant(float(a))),
                bounds=(lo, hi),
                method="bounded",
            )
            if result.success and np.isfinite(result.x):
                best_angle = float(result.x)
    except Exception:
        pass

    return float(np.clip(best_angle, -360.0, 360.0))


def im_to_polar(
    image: np.ndarray,
    r_min: float,
    r_max: float,
    radial_samples: int,
    angular_samples: int,
) -> FloatArray:
    """Convert a rectangular image to a polar image.

    This vectorized function follows the MATLAB helper ``ImToPolar``:

    - Origin is at the image center.
    - ``r=1`` corresponds to half the image width/height scale.
    - Bilinear interpolation is used for non-integer source coordinates.
    """
    img = _clean_image(image)

    rows, cols = img.shape
    om = (rows + 1.0) / 2.0
    on = (cols + 1.0) / 2.0
    sx = (rows - 1.0) / 2.0
    sy = (cols - 1.0) / 2.0

    r = np.linspace(r_min, r_max, int(radial_samples), dtype=np.float64)
    theta = np.arange(int(angular_samples), dtype=np.float64) * (2.0 * np.pi / float(angular_samples))

    rr, tt = np.meshgrid(r, theta, indexing="ij")

    # Note: the MATLAB helper names the first image coordinate xR but indexes
    # image rows with it.  We preserve that convention for alignment parity.
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)

    row_coords = x * sx + (om - 1.0)  # convert MATLAB 1-based coordinate to Python 0-based
    col_coords = y * sy + (on - 1.0)

    polar = map_coordinates(
        img,
        [row_coords, col_coords],
        order=1,
        mode="nearest",
    )

    return np.asarray(polar, dtype=np.float64)


def _zoom_peak_offset(
    corr: np.ndarray,
    y: int,
    x: int,
    *,
    half_window: int = 3,
    zoom_factor: int = 100,
) -> tuple[float, float]:
    """Sub-pixel refinement by bicubic zoom of a local correlation window."""
    y0 = max(0, int(y) - half_window)
    y1 = min(corr.shape[0], int(y) + half_window + 1)
    x0 = max(0, int(x) - half_window)
    x1 = min(corr.shape[1], int(x) + half_window + 1)

    if (y1 - y0) < 3 or (x1 - x0) < 3:
        return 0.0, 0.0

    clip = corr[y0:y1, x0:x1]
    zoomed = zoom(clip, zoom=zoom_factor, order=3)

    peak = int(np.nanargmax(np.abs(zoomed)))
    yp, xp = np.unravel_index(peak, zoomed.shape)

    x_offset = (float(xp) - (zoomed.shape[1] / 2.0)) / float(zoom_factor)
    y_offset = (float(yp) - (zoomed.shape[0] / 2.0)) / float(zoom_factor)

    return x_offset, y_offset


def _polar_corr(ref: np.ndarray, target: np.ndarray, angle_range: tuple[float, float]) -> float:
    """Polar-coordinate correlation branch of ``align_rot``."""
    min_angle, max_angle = float(angle_range[0]), float(angle_range[1])

    angle_samples = 360
    r_min = 0.0
    r_max = 1.0
    radial_samples = int(round(min(ref.shape) / 3.0))
    radial_samples = max(radial_samples, 8)
    zoom_factor = 100

    ref_polar = im_to_polar(ref, r_min, r_max, radial_samples, angle_samples)
    target_polar = im_to_polar(target, r_min, r_max, radial_samples, angle_samples)

    # MATLAB creates [f1(:,ang/2:end), f1, f1(:,1:ang/2)] to make angular
    # wrap-around searchable.  Python slicing is adjusted for 0-based indexing.
    half = angle_samples // 2
    ref_tiled = np.concatenate(
        [ref_polar[:, half - 1 :], ref_polar, ref_polar[:, :half]],
        axis=1,
    )

    corr = normxcorr2(target_polar, ref_tiled)

    # MATLAB then keeps approximately the middle angular band.
    corr = corr[:, angle_samples : 2 * angle_samples]

    # Search only the requested angular range plus a 5-degree margin.
    margin = 5
    search_min = int(round(min_angle - margin + 180))
    search_max = int(round(max_angle + margin + 180))

    search_min = max(0, search_min)
    search_max = min(corr.shape[1] - 1, search_max)

    if search_max <= search_min:
        return 0.0

    corr_search = corr[:, search_min : search_max + 1].copy()

    # MATLAB suppresses a few border rows/columns before peak detection.
    if corr_search.shape[0] > 5:
        corr_search[4, :] = 0
    if corr_search.shape[1] > 0:
        corr_search[:, 0] = 0
    if corr_search.shape[0] > 2:
        corr_search[-2:, :] = 0
    if corr_search.shape[1] > 2:
        corr_search[:, -2:] = 0

    peak = int(np.nanargmax(np.abs(corr_search)))
    y_peak, x_peak = np.unravel_index(peak, corr_search.shape)

    # Map the local column back to an angle.  The -180 shift follows the
    # MATLAB indexing offset in ccr(:, range+180).
    rot_angle = float((search_min + x_peak) - 180)

    # MATLAB formula contains an additional small indexing offset.  Applying
    # a local zoom correction usually matters more than the exact integer
    # convention here.
    xo, _ = _zoom_peak_offset(
        corr_search,
        y_peak,
        x_peak,
        half_window=3,
        zoom_factor=zoom_factor,
    )
    rot_angle += xo

    return float(np.clip(rot_angle, -360.0, 360.0))


def align_rot(
    ref: np.ndarray,
    target: np.ndarray,
    angle_range: tuple[float, float] | list[float],
    method: str,
) -> float:
    """Estimate the rotation angle needed to align ``target`` to ``ref``.

    Parameters
    ----------
    ref:
        2-D reference image.
    target:
        2-D target image.
    angle_range:
        Two-element range ``(min_angle, max_angle)`` in degrees.
    method:
        ``'Rotation corr'`` or ``'Polar Corr'``.

    Returns
    -------
    float
        Estimated rotation angle in degrees.
    """
    if len(angle_range) != 2:
        raise ValueError("angle_range must contain two values")

    method_lc = method.lower()

    if method_lc == "rotation corr":
        return _rotation_corr(ref, target, (float(angle_range[0]), float(angle_range[1])))

    if method_lc == "polar corr":
        return _polar_corr(ref, target, (float(angle_range[0]), float(angle_range[1])))

    raise ValueError("method must be either 'Rotation corr' or 'Polar Corr'")


__all__ = [
    "align_rot",
    "im_to_polar",
]
