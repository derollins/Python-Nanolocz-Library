"""
Point-cloud alignment utilities for NanoLocz-compatible LAFM workflows.

This module ports NanoLocz ``align_PtCloud.m``.  The MATLAB function uses:

- ``Fast_peaks2D`` to detect features in the LAFM reference image.
- ``localize`` to refine those features.
- MATLAB's ``pcregistericp`` to register per-frame point clouds.

Because ``Fast_peaks2D`` and ``localize`` are separate NanoLocz functions that
may be ported later, this Python implementation accepts them as required
callbacks.  ICP itself is implemented here with a lightweight point-to-point
nearest-neighbour rigid registration loop.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _to_grayscale(image: np.ndarray) -> FloatArray:
    """Normalize a 2-D/RGB LAFM image to grayscale float64."""
    arr = np.asarray(image, dtype=np.float64)

    max_value = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
    if max_value != 0.0:
        arr = arr / max_value

    if arr.ndim == 3:
        # MATLAB ``rgb2gray`` uses approximately these luminance weights.
        if arr.shape[-1] < 3:
            arr = arr[..., 0]
        else:
            arr = (
                0.2989 * arr[..., 0]
                + 0.5870 * arr[..., 1]
                + 0.1140 * arr[..., 2]
            )

    if arr.ndim != 2:
        raise ValueError("LAFM must be a 2D grayscale image or RGB image")

    arr[~np.isfinite(arr)] = 0.0
    return np.asarray(arr, dtype=np.float64)


def _ensure_columns(arr: np.ndarray, n_cols: int, fill: float = np.nan) -> FloatArray:
    """Pad a 2-D table to at least ``n_cols`` columns."""
    table = np.asarray(arr, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError("input table must be 2D")

    if table.shape[1] >= n_cols:
        return table.copy()

    out = np.full((table.shape[0], n_cols), fill, dtype=np.float64)
    out[:, : table.shape[1]] = table
    return np.asarray(out, dtype=np.float64)


def _best_fit_rigid_transform(source: FloatArray, target: FloatArray) -> tuple[np.ndarray, np.ndarray]:
    """Compute rigid transform that maps ``source`` points to ``target`` points.

    This is the standard Kabsch/SVD least-squares rigid alignment.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)

    src_centered = source - src_center
    tgt_centered = target - tgt_center

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Avoid reflections when the point layout is degenerate.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_center - (R @ src_center)

    return np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)


def _icp_point_to_point(
    moving: FloatArray,
    fixed: FloatArray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> tuple[FloatArray, float]:
    """Register ``moving`` to ``fixed`` using simple point-to-point ICP.

    Returns the transformed moving points and the final RMSE.  The algorithm is
    intentionally lightweight so that the NanoLocz port does not depend on
    Open3D or MATLAB's Computer Vision Toolbox.
    """
    src = np.asarray(moving, dtype=np.float64)
    dst = np.asarray(fixed, dtype=np.float64)

    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != dst.shape[1]:
        raise ValueError("moving and fixed point clouds must be NxD and MxD")

    if src.shape[0] < 3 or dst.shape[0] < 3:
        raise ValueError("ICP requires at least 3 points in each cloud")

    transformed = src.copy()
    tree = cKDTree(dst)

    previous_rmse = np.inf

    for _ in range(max_iterations):
        distances, indices = tree.query(transformed, k=1)
        matched = dst[indices]

        R, t = _best_fit_rigid_transform(transformed, matched)
        transformed = (R @ transformed.T).T + t

        rmse = float(np.sqrt(np.mean(distances**2)))
        if abs(previous_rmse - rmse) < tolerance:
            break
        previous_rmse = rmse

    distances, _ = tree.query(transformed, k=1)
    rmse = float(np.sqrt(np.mean(distances**2)))

    return np.asarray(transformed, dtype=np.float64), rmse


def align_ptcloud(
    locs: np.ndarray,
    lafm: np.ndarray,
    exp: float,
    *,
    fast_peaks2d_fn: Callable[[np.ndarray, float, float, float], np.ndarray],
    localize_fn: Callable[[np.ndarray, np.ndarray, str, float], np.ndarray],
    rmse_acceptance: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Align localisation point clouds to a LAFM-derived reference cloud.

    Parameters
    ----------
    locs:
        Localisation table.  Expected MATLAB columns are:
        ``x, y, z, ..., frame`` where frame is column 5.  In Python this means
        x/y/z are columns ``0:3`` and frame is column ``4``.
    lafm:
        Reference LAFM image, either grayscale or RGB.
    exp:
        Pixel size / scale factor used by MATLAB as ``Img_locs(:,1:2)/exp``.
    fast_peaks2d_fn:
        Callback equivalent to MATLAB ``Fast_peaks2D(LAFM_ref, 0.1, 2, 0)``.
    localize_fn:
        Callback equivalent to MATLAB ``localize(LAFM_ref, Img_locs, 'sphere', 1)``.
    rmse_acceptance:
        Only accept ICP-updated coordinates when ``rmse < rmse_acceptance``.

    Returns
    -------
    aligned_locs, rmse:
        Updated localisation table and one RMSE value per unique frame.
    """
    locs_arr = _ensure_columns(locs, 9)
    lafm_ref = _to_grayscale(lafm)

    if float(exp) == 0.0:
        raise ValueError("exp must be non-zero")

    img_locs = np.asarray(fast_peaks2d_fn(lafm_ref, 0.1, 2, 0), dtype=np.float64)
    img_locs = _ensure_columns(img_locs, 5)
    img_locs[:, 4] = 1.0

    img_locs = np.asarray(localize_fn(lafm_ref, img_locs, "sphere", 1), dtype=np.float64)
    img_locs = _ensure_columns(img_locs, 5)

    img_locs[:, 0:2] = img_locs[:, 0:2] / float(exp)

    fixed_cloud = np.asarray(img_locs[:, 0:3], dtype=np.float64)
    fixed_cloud = fixed_cloud[np.all(np.isfinite(fixed_cloud), axis=1)]

    if fixed_cloud.shape[0] < 3:
        raise ValueError("reference LAFM cloud has fewer than 3 finite points")

    aligned = locs_arr.copy()

    frames = np.unique(locs_arr[np.isfinite(locs_arr[:, 4]), 4])
    rmse = np.full(frames.shape[0], np.nan, dtype=np.float64)

    for idx, frame in enumerate(frames):
        pos = locs_arr[:, 4] == frame
        moving_cloud = np.asarray(locs_arr[pos, 0:3], dtype=np.float64)
        finite_rows = np.all(np.isfinite(moving_cloud), axis=1)

        if finite_rows.sum() < 3:
            continue

        try:
            registered, frame_rmse = _icp_point_to_point(
                moving_cloud[finite_rows],
                fixed_cloud,
                max_iterations=50,
                tolerance=1e-5,
            )

            rmse[idx] = frame_rmse

            # MATLAB only accepts the update for sufficiently low RMSE.
            if frame_rmse < float(rmse_acceptance):
                updated = moving_cloud.copy()
                updated[finite_rows, 0:2] = registered[:, 0:2]
                aligned[pos, 0:2] = updated[:, 0:2]

            aligned[pos, 8] = frame_rmse

        except Exception:
            # MATLAB uses try/catch and silently leaves the frame unchanged.
            continue

    return np.asarray(aligned, dtype=np.float64), np.asarray(rmse, dtype=np.float64)


# MATLAB-style alias with the original capitalization.
align_PtCloud = align_ptcloud


__all__ = [
    "align_ptcloud",
    "align_PtCloud",
]
