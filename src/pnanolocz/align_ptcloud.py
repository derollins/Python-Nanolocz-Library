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

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd  # type: ignore  # stubs optional; suppress import-untyped
from scipy.spatial import cKDTree

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.intp]]
Array = np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class PointCloudInput:
    """Frozen input record for point-cloud alignment."""

    xyz: FloatArray
    frame: IntArray
    particle_id: IntArray


@dataclass(frozen=True, slots=True)
class PointCloudResult:
    """Frozen result record for point-cloud alignment."""

    frame: IntArray
    particle_id: IntArray
    x: FloatArray
    y: FloatArray
    quality_frame: IntArray
    rmse: FloatArray


def _to_grayscale(image: Array) -> FloatArray:
    """Normalize a 2-D/RGB LAFM image to grayscale float64 without mutating it."""
    arr = np.array(image, dtype=np.float64, copy=True)

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


def _ensure_columns(arr: Array, n_cols: int, fill: float = np.nan) -> FloatArray:
    """Pad a 2-D table to at least ``n_cols`` columns."""
    table = np.asarray(arr, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError("input table must be 2D")

    if table.shape[1] >= n_cols:
        return table.copy()

    out = np.full((table.shape[0], n_cols), fill, dtype=np.float64)
    out[:, : table.shape[1]] = table
    return np.asarray(out, dtype=np.float64)


def _best_fit_rigid_transform(
    source: FloatArray, target: FloatArray
) -> tuple[Array, Array]:
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


def _align_ptcloud(
    data: PointCloudInput,
    lafm: Array,
    exp: float,
    *,
    fast_peaks2d_fn: Callable[[Array, float, float, float], Array],
    localize_fn: Callable[[Array, Array, str, float], Array],
    rmse_acceptance: float,
) -> PointCloudResult:
    """Private point-cloud stage; receives a frozen ``PointCloudInput``."""
    n_points = int(data.xyz.shape[0])
    if n_points == 0:
        return PointCloudResult(
            frame=data.frame,
            particle_id=data.particle_id,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            quality_frame=np.empty(0, dtype=np.intp),
            rmse=np.empty(0, dtype=np.float64),
        )

    exp_value = float(exp)
    acceptance = float(rmse_acceptance)
    xyz = np.asarray(data.xyz[:, :3], dtype=np.float64)
    lafm_ref = _to_grayscale(lafm)

    img_locs = np.asarray(fast_peaks2d_fn(lafm_ref, 0.1, 2, 0), dtype=np.float64)
    img_locs = _ensure_columns(img_locs, 5)
    img_locs[:, 4] = 1.0

    img_locs = np.asarray(
        localize_fn(lafm_ref, img_locs, "sphere", 1), dtype=np.float64
    )
    img_locs = _ensure_columns(img_locs, 5)
    img_locs[:, 0:2] = img_locs[:, 0:2] / exp_value

    fixed_cloud = np.asarray(img_locs[:, 0:3], dtype=np.float64)
    fixed_cloud = fixed_cloud[np.all(np.isfinite(fixed_cloud), axis=1)]

    if fixed_cloud.shape[0] < 3:
        raise ValueError("reference LAFM cloud has fewer than 3 finite points")

    out_x = xyz[:, 0].copy()
    out_y = xyz[:, 1].copy()
    z = xyz[:, 2]

    frames = np.unique(data.frame)
    rmse_values = np.full(frames.size, np.nan, dtype=np.float64)

    for idx, current_frame in enumerate(frames):
        frame_mask = data.frame == current_frame
        rows = np.flatnonzero(frame_mask)
        moving_cloud = np.column_stack(
            [out_x[frame_mask], out_y[frame_mask], z[frame_mask]]
        )
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
        except np.linalg.LinAlgError:
            continue

        rmse_values[idx] = frame_rmse

        if frame_rmse < acceptance:
            target_rows = rows[finite_rows]
            out_x[target_rows] = registered[:, 0]
            out_y[target_rows] = registered[:, 1]

    return PointCloudResult(
        frame=data.frame,
        particle_id=data.particle_id,
        x=np.asarray(out_x, dtype=np.float64),
        y=np.asarray(out_y, dtype=np.float64),
        quality_frame=frames,
        rmse=rmse_values,
    )


def align_ptcloud(
    xyz: FloatArray,
    frame: IntArray,
    particle_id: IntArray,
    lafm: Array,
    exp: float,
    *,
    fast_peaks2d_fn: Callable[[Array, float, float, float], Array],
    localize_fn: Callable[[Array, Array, str, float], Array],
    rmse_acceptance: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align localisation point clouds to a LAFM-derived reference cloud.

    Public adapter: NumPy arrays -> frozen :class:`PointCloudInput` -> private
    ICP -> named DataFrame results.

    Parameters
    ----------
    xyz : ndarray
        ``(N, 3)`` or wider array with x, y and z in columns 0, 1 and 2.
    frame : ndarray of int
        ``(N,)`` 0-based frame index for each localisation.
    particle_id : ndarray of int
        ``(N,)`` stable identifier for each localisation.
    lafm : ndarray
        Reference LAFM image, grayscale or RGB.
    exp : float
        Pixel size / scale factor.
    fast_peaks2d_fn, localize_fn : callable
        Reference-detection and localization callbacks.
    rmse_acceptance : float, default 20.0
        Accept ICP coordinates only when ``rmse < rmse_acceptance``.

    Returns
    -------
    aligned : pandas.DataFrame
        ``frame, particle_id, x, y`` with one row per input localisation.
    quality : pandas.DataFrame
        ``frame, rmse`` with one row per input frame.
    """
    xyz_arr = np.asarray(xyz, dtype=np.float64)
    if xyz_arr.ndim != 2 or xyz_arr.shape[1] < 3:
        raise ValueError("xyz must be an (N, 3) or wider array")

    frame_raw = np.asarray(frame)
    if frame_raw.ndim != 1 or frame_raw.shape[0] != xyz_arr.shape[0]:
        raise ValueError("frame must be a 1D array with one value per point")
    if not np.all(np.isfinite(frame_raw)):
        raise ValueError("frame values must be finite")

    frame_int = frame_raw.astype(np.intp)
    if not np.array_equal(frame_raw, frame_int):
        raise ValueError("frame values must be integers")
    if np.any(frame_int < 0):
        raise ValueError("frame values must be non-negative")

    ids_raw = np.asarray(particle_id)
    ids = ids_raw.astype(np.intp)
    if ids.ndim != 1 or ids.shape[0] != xyz_arr.shape[0]:
        raise ValueError("particle_id must be a 1D array with one value per point")
    if not np.array_equal(ids_raw, ids):
        raise ValueError("particle_id values must be integers")

    exp_value = float(exp)
    if not np.isfinite(exp_value) or exp_value == 0.0:
        raise ValueError("exp must be finite and non-zero")

    acceptance = float(rmse_acceptance)
    if not np.isfinite(acceptance) or acceptance <= 0.0:
        raise ValueError("rmse_acceptance must be finite and positive")

    data = PointCloudInput(
        xyz=np.asarray(xyz_arr[:, :3], dtype=np.float64),
        frame=np.asarray(frame_int, dtype=np.intp),
        particle_id=ids,
    )
    result = _align_ptcloud(
        data,
        lafm,
        exp_value,
        fast_peaks2d_fn=fast_peaks2d_fn,
        localize_fn=localize_fn,
        rmse_acceptance=acceptance,
    )

    aligned = pd.DataFrame(
        {
            "frame": result.frame,
            "particle_id": result.particle_id,
            "x": result.x,
            "y": result.y,
        }
    )
    quality = pd.DataFrame(
        {"frame": result.quality_frame, "rmse": result.rmse}
    )
    return aligned, quality


__all__ = ["align_ptcloud"]
