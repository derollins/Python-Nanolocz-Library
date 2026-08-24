"""
Weighted-region AFM leveling routines for NanoLocz-compatible Python workflows.

This module is a MATLAB-aligned port of NanoLocz ``level_weighted.m``.  It uses
connected valid regions from a mask to build region-weighted polynomial or
median background estimates.

Public mask convention
----------------------
The public Python convention is:

    True  = excluded pixel
    False = valid / included pixel

The original MATLAB function receives ``imgt`` where valid pixels are ``1`` and
excluded pixels are usually ``NaN``.  :func:`apply_level_weighted` converts the
Python exclusion mask into a validity mask before region detection.

Important MATLAB alignment details
----------------------------------
- Connected components use 8-connectivity, matching ``bwconncomp(mask, 8)``.
- Very small regions are discarded using ``floor(0.01 * rows * cols)`` area.
- Polynomial coordinates are 1-based, matching MATLAB's ``find`` and
  ``1:numel(...)`` coordinates.
- Region weights below or equal to 2% are set to zero and are not renormalized,
  matching ``W = W .* (W > 0.02)``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import ndimage

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]
IndexArray = np.ndarray[Any, np.dtype[np.int64]]


def _validity_mask(
    arr: FloatArray,
    mask_excl: BoolArray | None,
    *,
    name: str = "mask",
) -> BoolArray:
    """Convert a Python exclusion mask into a finite-aware validity mask."""
    finite = np.isfinite(arr)

    if mask_excl is None:
        return np.asarray(finite, dtype=np.bool_)

    m_excl = np.asarray(mask_excl, dtype=np.bool_)
    if m_excl.shape != arr.shape:
        raise ValueError(f"{name} shape {m_excl.shape} must match image shape {arr.shape}")

    return np.asarray((~m_excl) & finite, dtype=np.bool_)


def _polyfit_centered(x: np.ndarray, y: np.ndarray, order: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Fit a polynomial with MATLAB ``polyfit(..., mu)`` style scaling."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if x.size <= order:
        raise ValueError("not enough samples for polynomial fit")

    center = float(np.mean(x))
    scale = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*[Rr]ank.*")
        coeffs = np.polyfit((x - center) / scale, y, int(order))

    return np.asarray(coeffs, dtype=np.float64), (center, scale)


def _polyval_centered(coeffs: np.ndarray, mu: tuple[float, float], points: np.ndarray) -> np.ndarray:
    """Evaluate coefficients fitted with :func:`_polyfit_centered`."""
    center, scale = mu
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    points = np.asarray(points, dtype=np.float64)
    return np.asarray(np.polyval(coeffs, (points - center) / scale), dtype=np.float64)


def _find_regions(mask_valid: BoolArray, min_area: int | None = None) -> list[IndexArray]:
    """Find connected valid regions using MATLAB-like 8-connectivity.

    Parameters
    ----------
    mask_valid:
        Boolean validity mask where ``True`` means the pixel is included.
    min_area:
        Minimum component area.  If omitted, the MATLAB NanoLocz rule
        ``floor(0.01 * rows * cols)`` is used, with a lower bound of 1.

    Returns
    -------
    list of ndarray
        Flat row-major indices for each retained connected component.
    """
    structure = np.ones((3, 3), dtype=np.int8)
    labeled, num_features = ndimage.label(mask_valid, structure=structure)

    if num_features == 0:
        return []

    rows, cols = mask_valid.shape
    if min_area is None:
        min_area = max(1, int(np.floor(0.01 * rows * cols)))

    areas = ndimage.sum(mask_valid, labeled, index=np.arange(1, num_features + 1))

    regions: list[IndexArray] = []
    for label_id, area in zip(range(1, num_features + 1), areas, strict=False):
        if area < min_area:
            continue
        rr, cc = np.where(labeled == label_id)
        flat = np.ravel_multi_index((rr, cc), mask_valid.shape, order="C")
        regions.append(np.asarray(flat, dtype=np.int64))

    return regions


def level_weighted_plane(
    img: FloatArray,
    regions: list[IndexArray],
    polyx: int,
    polyy: int,
) -> FloatArray:
    """Subtract a region-weighted polynomial plane.

    Each connected region contributes an X polynomial fitted to its column mean
    profile and a Y polynomial fitted to its row mean profile.  These polynomial
    coefficients and their MATLAB ``mu`` scaling values are averaged using
    region-area weights.
    """
    img_f = np.asarray(img, dtype=np.float64)
    rows, cols = img_f.shape
    n_regions = len(regions)

    if n_regions == 0:
        return np.asarray(img_f.copy(), dtype=np.float64)

    weights_raw = np.zeros(n_regions, dtype=np.float64)

    x_coeffs: list[np.ndarray] = []
    x_mu: list[tuple[float, float]] = []
    y_coeffs: list[np.ndarray] = []
    y_mu: list[tuple[float, float]] = []

    for region_id, flat_idx in enumerate(regions):
        region_matrix = np.full((rows, cols), np.nan, dtype=np.float64)
        region_matrix.flat[flat_idx] = img_f.flat[flat_idx]
        weights_raw[region_id] = float(flat_idx.size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xp = np.nanmean(region_matrix, axis=0)
        valid_cols = np.isfinite(xp)
        if valid_cols.sum() > polyx:
            # MATLAB columns are 1-based: xl = 1:numel(xp)
            col_positions = (np.nonzero(valid_cols)[0] + 1).astype(np.float64)
            try:
                coeffs, mu = _polyfit_centered(col_positions, xp[valid_cols], int(polyx))
            except ValueError:
                coeffs, mu = np.zeros(int(polyx) + 1, dtype=np.float64), (0.0, 1.0)
        else:
            coeffs, mu = np.zeros(int(polyx) + 1, dtype=np.float64), (0.0, 1.0)
        x_coeffs.append(coeffs)
        x_mu.append(mu)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            yp = np.nanmean(region_matrix, axis=1)
        valid_rows = np.isfinite(yp)
        if valid_rows.sum() > polyy:
            # MATLAB rows are 1-based: yl = 1:numel(yp)
            row_positions = (np.nonzero(valid_rows)[0] + 1).astype(np.float64)
            try:
                coeffs, mu = _polyfit_centered(row_positions, yp[valid_rows], int(polyy))
            except ValueError:
                coeffs, mu = np.zeros(int(polyy) + 1, dtype=np.float64), (0.0, 1.0)
        else:
            coeffs, mu = np.zeros(int(polyy) + 1, dtype=np.float64), (0.0, 1.0)
        y_coeffs.append(coeffs)
        y_mu.append(mu)

    weights = weights_raw / (weights_raw.sum() if weights_raw.sum() > 0 else 1.0)
    weights = np.where(weights > 0.02, weights, 0.0)

    px = np.stack(x_coeffs, axis=1)
    py = np.stack(y_coeffs, axis=1)

    px_w = np.sum(px * weights[None, :], axis=1)
    py_w = np.sum(py * weights[None, :], axis=1)

    mux_w = (
        float(np.sum(np.array([mu[0] for mu in x_mu]) * weights)),
        float(np.sum(np.array([mu[1] for mu in x_mu]) * weights)),
    )
    muy_w = (
        float(np.sum(np.array([mu[0] for mu in y_mu]) * weights)),
        float(np.sum(np.array([mu[1] for mu in y_mu]) * weights)),
    )

    # MATLAB evaluates on 1-based coordinates.
    all_cols = np.arange(1, cols + 1, dtype=np.float64)
    all_rows = np.arange(1, rows + 1, dtype=np.float64)

    plane_x = _polyval_centered(px_w, mux_w, all_cols)[None, :]
    plane_y = _polyval_centered(py_w, muy_w, all_rows)[:, None]

    return np.asarray(img_f - (plane_x + plane_y), dtype=np.float64)


def level_weighted_line(
    img: FloatArray,
    regions: list[IndexArray],
    polyx: int,
    polyy: int,
) -> FloatArray:
    """Subtract region-weighted row and column polynomial line backgrounds."""
    img_f = np.asarray(img, dtype=np.float64)
    rows, cols = img_f.shape
    n_regions = len(regions)

    if n_regions == 0:
        return np.asarray(img_f.copy(), dtype=np.float64)

    out = img_f.copy()

    if polyx > 0:
        w_rows = np.zeros((rows, n_regions), dtype=np.float64)
        px_coeffs = np.zeros((rows, int(polyx) + 1, n_regions), dtype=np.float64)
        mux_center = np.zeros((rows, n_regions), dtype=np.float64)
        mux_scale = np.ones((rows, n_regions), dtype=np.float64)

        for region_id, flat_idx in enumerate(regions):
            region_matrix = np.full((rows, cols), np.nan, dtype=np.float64)
            region_matrix.flat[flat_idx] = img_f.flat[flat_idx]

            for rr in range(rows):
                pos = np.isfinite(region_matrix[rr, :])
                w_rows[rr, region_id] = float(pos.sum())

                if pos.sum() > polyx + 1:
                    # ``find(pos)`` in MATLAB is 1-based.
                    x = (np.nonzero(pos)[0] + 1).astype(np.float64)
                    y = img_f[rr, pos]
                    try:
                        coeffs, mu = _polyfit_centered(x, y, int(polyx))
                        px_coeffs[rr, :, region_id] = coeffs
                        mux_center[rr, region_id] = mu[0]
                        mux_scale[rr, region_id] = mu[1]
                    except ValueError:
                        pass

        denom = w_rows.sum(axis=1, keepdims=True)
        W = np.divide(w_rows, denom, out=np.zeros_like(w_rows), where=denom != 0)
        W = W * (W > 0.02)

        px_w = np.sum(px_coeffs * W[:, None, :], axis=2)
        px_w[:, -1] = 0.0  # MATLAB explicitly zeroes the constant term.

        mu_w_center = np.sum(mux_center * W, axis=1)
        mu_w_scale = np.sum(mux_scale * W, axis=1)

        grid = np.arange(1, cols + 1, dtype=np.float64)
        lines = np.zeros_like(img_f, dtype=np.float64)
        for rr in range(rows):
            lines[rr, :] = _polyval_centered(
                px_w[rr, :],
                (float(mu_w_center[rr]), float(mu_w_scale[rr]) or 1.0),
                grid,
            )

        lines = np.where(np.isfinite(lines), lines, 0.0)
        out = out - lines

    if polyy > 0:
        w_cols = np.zeros((cols, n_regions), dtype=np.float64)
        py_coeffs = np.zeros((cols, int(polyy) + 1, n_regions), dtype=np.float64)
        muy_center = np.zeros((cols, n_regions), dtype=np.float64)
        muy_scale = np.ones((cols, n_regions), dtype=np.float64)

        for region_id, flat_idx in enumerate(regions):
            region_matrix = np.full((rows, cols), np.nan, dtype=np.float64)
            region_matrix.flat[flat_idx] = img_f.flat[flat_idx]

            for cc in range(cols):
                pos = np.isfinite(region_matrix[:, cc])
                w_cols[cc, region_id] = float(pos.sum())

                if pos.sum() > polyy + 1:
                    y_coord = (np.nonzero(pos)[0] + 1).astype(np.float64)
                    values = img_f[pos, cc]
                    try:
                        coeffs, mu = _polyfit_centered(y_coord, values, int(polyy))
                        py_coeffs[cc, :, region_id] = coeffs
                        muy_center[cc, region_id] = mu[0]
                        muy_scale[cc, region_id] = mu[1]
                    except ValueError:
                        pass

        denom = w_cols.sum(axis=1, keepdims=True)
        W = np.divide(w_cols, denom, out=np.zeros_like(w_cols), where=denom != 0)
        W = W * (W > 0.02)

        py_w = np.sum(py_coeffs * W[:, None, :], axis=2)
        py_w[:, -1] = 0.0

        mu_w_center = np.sum(muy_center * W, axis=1)
        mu_w_scale = np.sum(muy_scale * W, axis=1)

        grid = np.arange(1, rows + 1, dtype=np.float64)
        lines = np.zeros_like(img_f, dtype=np.float64)
        for cc in range(cols):
            lines[:, cc] = _polyval_centered(
                py_w[cc, :],
                (float(mu_w_center[cc]), float(mu_w_scale[cc]) or 1.0),
                grid,
            )

        lines = np.where(np.isfinite(lines), lines, 0.0)
        out = out - lines

    return np.asarray(out, dtype=np.float64)


def level_weighted_med_line(img: FloatArray, regions: list[IndexArray]) -> FloatArray:
    """Subtract a region-weighted row-wise median baseline."""
    img_f = np.asarray(img, dtype=np.float64)
    rows, cols = img_f.shape
    n_regions = len(regions)

    if n_regions == 0:
        return np.asarray(img_f.copy(), dtype=np.float64)

    w = np.zeros((rows, n_regions), dtype=np.float64)
    y1 = np.zeros((rows, n_regions), dtype=np.float64)
    bg = np.zeros(n_regions, dtype=np.float64)

    for region_id, flat_idx in enumerate(regions):
        region_matrix = np.full((rows, cols), np.nan, dtype=np.float64)
        region_matrix.flat[flat_idx] = img_f.flat[flat_idx]
        bg[region_id] = float(np.nanmedian(region_matrix))

        for rr in range(rows):
            pos = np.isfinite(region_matrix[rr, :])
            w[rr, region_id] = float(pos.sum())

            if pos.sum() > 2:
                y1[rr, region_id] = float(np.median(img_f[rr, pos])) - bg[region_id]
            else:
                y1[rr, region_id] = -bg[region_id]

    denom = w.sum(axis=1, keepdims=True)
    W = np.divide(w, denom, out=np.zeros_like(w), where=denom != 0)
    yf = np.sum(W * y1, axis=1)

    zero_rows = denom[:, 0] == 0
    out = img_f.copy()
    out[~zero_rows, :] = img_f[~zero_rows, :] - yf[~zero_rows, None]

    return np.asarray(out, dtype=np.float64)


def level_weighted_med_line_y(img: FloatArray, regions: list[IndexArray]) -> FloatArray:
    """Subtract a region-weighted column-wise median baseline."""
    img_f = np.asarray(img, dtype=np.float64)
    rows, cols = img_f.shape
    n_regions = len(regions)

    if n_regions == 0:
        return np.asarray(img_f.copy(), dtype=np.float64)

    w = np.zeros((cols, n_regions), dtype=np.float64)
    y1 = np.zeros((cols, n_regions), dtype=np.float64)
    bg = np.zeros(n_regions, dtype=np.float64)

    for region_id, flat_idx in enumerate(regions):
        region_matrix = np.full((rows, cols), np.nan, dtype=np.float64)
        region_matrix.flat[flat_idx] = img_f.flat[flat_idx]
        bg[region_id] = float(np.nanmedian(region_matrix))

        for cc in range(cols):
            pos = np.isfinite(region_matrix[:, cc])
            w[cc, region_id] = float(pos.sum())

            if pos.sum() > 2:
                y1[cc, region_id] = float(np.median(img_f[pos, cc])) - bg[region_id]
            else:
                y1[cc, region_id] = -bg[region_id]

    denom = w.sum(axis=1, keepdims=True)
    W = np.divide(w, denom, out=np.zeros_like(w), where=denom != 0)
    yf = np.sum(W * y1, axis=1)

    zero_cols = denom[:, 0] == 0
    out = img_f.copy()
    out[:, ~zero_cols] = img_f[:, ~zero_cols] - yf[~zero_cols][None, :]

    return np.asarray(out, dtype=np.float64)


def _movmedian_centered(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving median used by MATLAB ``movmedian(yf(:,1), 10)``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    window = max(int(window), 1)
    left = window // 2
    right = window - left

    out = np.empty(n, dtype=np.float64)
    for idx in range(n):
        start = max(0, idx - left)
        end = min(n, idx + right)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[idx] = np.nanmedian(x[start:end])
    return np.asarray(out, dtype=np.float64)


def level_weighted_smed_line(
    img: FloatArray,
    regions: list[IndexArray],
    smoothing_window: int = 10,
) -> FloatArray:
    """Subtract a smoothed region-weighted row baseline.

    MATLAB formula:

    ``bg2 = movmedian(yf(:,1), 10)``
    ``r = img - (yf - bg2)``

    A previous Python version subtracted ``bg2`` directly.  That is not
    equivalent.  The corrected version subtracts the high-frequency row offset
    ``yf - bg2``.
    """
    img_f = np.asarray(img, dtype=np.float64)
    rows, cols = img_f.shape
    n_regions = len(regions)

    if n_regions == 0:
        return np.asarray(img_f.copy(), dtype=np.float64)

    w = np.zeros((rows, n_regions), dtype=np.float64)
    y1 = np.zeros((rows, n_regions), dtype=np.float64)
    bg = np.zeros(n_regions, dtype=np.float64)

    for region_id, flat_idx in enumerate(regions):
        region_matrix = np.full((rows, cols), np.nan, dtype=np.float64)
        region_matrix.flat[flat_idx] = img_f.flat[flat_idx]
        bg[region_id] = float(np.nanmedian(region_matrix))

        for rr in range(rows):
            pos = np.isfinite(region_matrix[rr, :])
            w[rr, region_id] = float(pos.sum())

            if pos.sum() > 2:
                y1[rr, region_id] = float(np.median(img_f[rr, pos]))
            else:
                y1[rr, region_id] = -bg[region_id]

    denom = w.sum(axis=1, keepdims=True)
    W = np.divide(w, denom, out=np.zeros_like(w), where=denom != 0)
    yf = np.sum(W * y1, axis=1)

    bg2 = _movmedian_centered(yf, smoothing_window)
    correction = yf - bg2
    correction = np.where(np.isfinite(correction), correction, 0.0)

    zero_rows = denom[:, 0] == 0
    out = img_f.copy()
    out[~zero_rows, :] = img_f[~zero_rows, :] - correction[~zero_rows, None]

    return np.asarray(out, dtype=np.float64)


def _prepare_stack_and_mask(
    img: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Convert 2-D/3-D image and mask inputs to frame-first stacks."""
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
            raise ValueError("3D mask must match image stack shape")
        mask_stack = mask_arr
    else:
        raise ValueError("mask must be 2D or frame-first 3D")

    return stack, np.asarray(mask_stack, dtype=np.bool_), was_2d


def apply_level_weighted(
    img: np.ndarray,
    polyx: int,
    polyy: int,
    method: str,
    mask: np.ndarray | None = None,
    smoothing_window: int = 10,
) -> np.ndarray:
    """Apply a region-weighted leveling method to an image or stack.

    Parameters
    ----------
    img:
        2-D image ``(H, W)`` or frame-first stack ``(N, H, W)``.
    polyx, polyy:
        Polynomial orders used by polynomial methods.
    method:
        One of ``'plane'``, ``'line'``, ``'med_line'``, ``'med_line_y'`` or
        ``'smed_line'``.
    mask:
        Optional Python exclusion mask where ``True`` means excluded.
    smoothing_window:
        Window used by ``smed_line``.
    """
    stack, mask_stack, was_2d = _prepare_stack_and_mask(img, mask)

    leveled_frames: list[np.ndarray] = []

    for frame_idx in range(stack.shape[0]):
        frame = stack[frame_idx]
        frame_mask = None if mask_stack is None else mask_stack[frame_idx]

        valid = _validity_mask(frame, frame_mask)
        rows, cols = frame.shape
        min_area = max(1, int(np.floor(0.01 * rows * cols)))
        regions = _find_regions(valid, min_area=min_area)

        method_lc = method.lower()
        if method_lc == "plane":
            leveled = level_weighted_plane(frame, regions, int(polyx), int(polyy))
        elif method_lc == "line":
            leveled = level_weighted_line(frame, regions, int(polyx), int(polyy))
        elif method_lc == "med_line":
            leveled = level_weighted_med_line(frame, regions)
        elif method_lc == "med_line_y":
            leveled = level_weighted_med_line_y(frame, regions)
        elif method_lc == "smed_line":
            leveled = level_weighted_smed_line(frame, regions, smoothing_window=smoothing_window)
        else:
            raise ValueError(f"Unknown weighted leveling method: {method!r}")

        leveled_frames.append(leveled)

    out = np.stack(leveled_frames, axis=0)
    return np.asarray(out[0] if was_2d else out, dtype=np.float64)


apply_level_weighted.__version__ = "0.2.0"  # type: ignore[attr-defined]


__all__ = [
    "apply_level_weighted",
    "level_weighted_plane",
    "level_weighted_line",
    "level_weighted_med_line",
    "level_weighted_med_line_y",
    "level_weighted_smed_line",
]
