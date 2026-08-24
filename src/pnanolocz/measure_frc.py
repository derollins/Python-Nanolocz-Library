"""
Fourier Ring Correlation resolution measurement for NanoLocz.

This module ports MATLAB ``measureFRC.m``.  It reconstructs localization images,
splits frames randomly, computes FRC curves, and estimates resolution using the
1/7 threshold criterion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _radial_sum(img: np.ndarray) -> FloatArray:
    """Radial sum around FFT center."""
    arr = np.asarray(img)
    rows, cols = arr.shape
    center = np.floor((np.array([rows, cols]) + 1) / 2).astype(int)
    rs = np.zeros(int(np.ceil(rows / 2)) + 1, dtype=np.complex128)

    yy, xx = np.indices((rows, cols))
    d = np.sqrt((yy + 1 - center[0]) ** 2 + (xx + 1 - center[1]) ** 2)
    ind = np.round(d).astype(int)
    valid = ind < rs.size
    np.add.at(rs, ind[valid], arr[valid])
    return np.asarray(rs)


def _get_frc(img1: np.ndarray, img2: np.ndarray) -> FloatArray:
    """Compute raw FRC curve."""
    in1 = np.fft.fftshift(np.fft.fft2(img1))
    in2 = np.fft.fftshift(np.fft.fft2(img2))

    inc = in1 * np.conj(in2)
    frc_num = np.real(_radial_sum(inc))
    p1 = np.abs(in1) ** 2
    p2 = np.abs(in2) ** 2
    denom = np.sqrt(np.abs(_radial_sum(p1) * _radial_sum(p2)))

    frc = np.zeros_like(frc_num, dtype=np.float64)
    np.divide(frc_num, denom, out=frc, where=np.abs(denom) > np.finfo(float).eps)
    frc[~np.isfinite(frc)] = 0.0
    return np.asarray(frc, dtype=np.float64)


def _smooth_curve(curve: np.ndarray, size: int = 5) -> FloatArray:
    """Approximate MATLAB smooth(curve, 5)."""
    return np.asarray(uniform_filter1d(np.asarray(curve, dtype=np.float64), size=size, mode="nearest"), dtype=np.float64)


def _getxyt(frc_curve: np.ndarray, sz: tuple[int, int]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build q vector and 1/7 threshold curve."""
    frc = np.real(np.asarray(frc_curve, dtype=np.float64))
    frc = np.clip(frc, -1.0, 1.0)
    frc = _smooth_curve(frc, 5)
    q = np.arange(frc.size, dtype=np.float64) / float(sz[0])
    threshold = (1.0 / 7.0) * np.ones_like(frc)
    return q, frc, threshold


def _find_intersection(x: np.ndarray, y: np.ndarray, t: float) -> float:
    """Find first interpolated x above 0.25 where y <= threshold."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size < 2:
        return float(x[-1]) if x.size else np.nan

    step = (x[1] - x[0]) / 10.0
    if step <= 0:
        step = 1e-6
    xq = np.arange(x[0], x[-1] + step, step)
    yq = np.interp(xq, x, y)

    valid = xq > 0.25
    xqf = xq[valid]
    yqf = yq[valid]

    pos = yqf <= t
    if np.any(pos):
        return float(xqf[np.argmax(pos)])
    return float(np.max(x))


def measure_frc(
    locs: np.ndarray,
    pixpernm: float,
    runs: int,
    expand: float,
    *,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Measure average FRC resolution from localization data."""
    rng = np.random.default_rng(random_state)
    locs_arr = np.asarray(locs, dtype=np.float64).copy()

    img_gaus = 0.4
    locs_arr[:, 0] = locs_arr[:, 0] - np.nanmin(locs_arr[:, 0]) + 1
    locs_arr[:, 1] = locs_arr[:, 1] - np.nanmin(locs_arr[:, 1]) + 1

    explocs = np.column_stack([np.round(locs_arr[:, 0:2] * expand), locs_arr[:, 2:]])
    explocs = explocs[~np.any(np.isnan(explocs), axis=1)]

    image_size = (int(np.nanmax(explocs[:, 1])) + 5, int(np.nanmax(explocs[:, 0])) + 5)
    frames = np.unique(locs_arr[:, 4][np.isfinite(locs_arr[:, 4])])
    n_frames = len(frames)

    lafm_p = np.zeros((image_size[0], image_size[1], n_frames), dtype=np.float64)

    for i, frame in enumerate(frames):
        pos = explocs[:, 4] == frame
        sub = explocs[pos, :]
        for row in sub:
            y = int(row[1])
            x = int(row[0])
            if 1 <= y <= image_size[0] and 1 <= x <= image_size[1]:
                lafm_p[y - 1, x - 1, i] = 1.0

    # Smooth spatially, not across frames.
    lafm_p = gaussian_filter(lafm_p, sigma=(img_gaus * expand, img_gaus * expand, 0), mode="nearest")

    frc_curves = []
    resolutions = []
    q_final = None
    thresh_final = None

    for _ in range(int(runs)):
        k = lafm_p.shape[2]
        if k < 2:
            break
        perm = rng.permutation(k)
        rand_lafm = lafm_p[:, :, perm]
        split = int(round(k / 2))
        img1 = np.mean(rand_lafm[:, :, :split], axis=2)
        img2 = np.mean(rand_lafm[:, :, split:], axis=2)

        frc_curve = _get_frc(img1, img2)
        q, frc_curve, thresh = _getxyt(frc_curve, img1.shape)
        q = q * (expand * pixpernm)
        resx = _find_intersection(q, frc_curve, thresh[0])
        if resx != 0 and np.isfinite(resx):
            resolutions.append(1.0 / resx)
            frc_curves.append(frc_curve)
        q_final = q
        thresh_final = thresh

    if q_final is None or not frc_curves:
        return np.empty(0), np.empty(0), np.nan, np.nan

    frc_mat = np.column_stack(frc_curves)
    frc_mean = np.mean(frc_mat, axis=1)
    resx = _find_intersection(q_final, frc_mean, 1.0 / 7.0)
    av_resolution = 1.0 / resx if resx != 0 else np.nan
    sd_resolution = float(np.std(resolutions, ddof=1)) if len(resolutions) > 1 else 0.0

    return np.asarray(q_final), np.asarray(frc_mean), float(av_resolution), sd_resolution


measureFRC = measure_frc

__all__ = ["measure_frc", "measureFRC"]
