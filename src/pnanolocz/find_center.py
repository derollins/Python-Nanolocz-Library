"""
Rotational-center estimation utilities for NanoLocz-compatible workflows.

This module ports MATLAB ``FindCenterPositions.m`` and ``FindCenter_ptCloud.m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import rotate as scipy_rotate, zoom
from scipy.spatial import cKDTree

from pnanolocz.correlation import normxcorr2

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
Array = np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class CenterInput:
    """Frozen input record for center estimation."""

    image: FloatArray
    fold: int
    points: FloatArray | None = None
    align_exp: float = 1.0


@dataclass(frozen=True, slots=True)
class CenterResult:
    """Frozen result record for center estimation."""

    x: float
    y: float


def _center_result(values: Array | list[float]) -> CenterResult:
    """Convert a two-element coordinate vector to a frozen record."""
    arr = np.asarray(values, dtype=np.float64)
    return CenterResult(x=float(arr[0]), y=float(arr[1]))


def _peak_abs(corr: Array) -> tuple[int, int]:
    """Return row/column index of maximum absolute correlation."""
    idx = int(np.nanargmax(np.abs(corr)))
    return tuple(
        int(v) for v in np.unravel_index(idx, corr.shape)
    )  # type: ignore[return-value]


def _cc_align(img: Array, ref: Array, exp: float) -> Array:
    """Cross-correlation alignment helper from MATLAB ``ccAlign``."""
    c = normxcorr2(img, ref)
    ypeak, xpeak = _peak_abs(c)

    if exp > 1:
        w = 3
        c = c.copy()
        c[:4, :] = 0
        c[:, :4] = 0
        c[-4:, :] = 0
        c[:, -4:] = 0

        y0 = max(0, ypeak - w + 1)
        y1 = min(c.shape[0], ypeak + w)
        x0 = max(0, xpeak - w + 1)
        x1 = min(c.shape[1], xpeak + w)
        clip = c[y0:y1, x0:x1]

        if clip.size:
            c_zoom = zoom(clip, zoom=float(exp), order=3)
            y2, x2 = _peak_abs(c_zoom)
            corr_offset_2 = np.array(
                [
                    ((x2 + 1) - c_zoom.shape[1] / 2.0) / exp,
                    ((y2 + 1) - c_zoom.shape[0] / 2.0) / exp,
                ],
                dtype=np.float64,
            )
        else:
            corr_offset_2 = np.zeros(2, dtype=np.float64)

        return np.array(
            [
                (xpeak + 1) + corr_offset_2[0] - img.shape[1],
                (ypeak + 1) + corr_offset_2[1] - img.shape[0],
            ],
            dtype=np.float64,
        )

    return np.array(
        [(xpeak + 1) - img.shape[1], (ypeak + 1) - img.shape[0]],
        dtype=np.float64,
    )


def _find_center_positions(data: CenterInput) -> CenterResult:
    """Private center stage; receives a frozen :class:`CenterInput`."""
    img = np.asarray(data.image, dtype=np.float64)
    raw_fold = data.fold
    align_exp = float(data.align_exp)

    if img.ndim != 2:
        raise ValueError("input_img must be 2D")
    if isinstance(raw_fold, bool) or not isinstance(raw_fold, (int, np.integer)):
        raise ValueError("fold must be an integer")
    fold = int(raw_fold)
    if fold < 1:
        raise ValueError("fold must be >= 1")

    offsets = []
    for i in range(1, int(fold)):
        rotated = scipy_rotate(
            img,
            angle=float(i * 360.0 / fold),
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        offsets.append(_cc_align(rotated, img, float(align_exp)))

    if fold == 2:
        return _center_result(offsets[0] / 2.0)

    if fold > 2:
        return _center_result(np.sum(offsets, axis=0) / float(fold))

    # fold == 1: center of mass relative to the 0-based geometric centre.
    rows, cols = img.shape
    yy, xx = np.indices(img.shape, dtype=np.float64)
    total = float(np.sum(img))

    if abs(total) <= np.finfo(float).eps:
        return _center_result(np.zeros(2, dtype=np.float64))

    center_x = float(np.sum(img * xx) / total)
    center_y = float(np.sum(img * yy) / total)
    return _center_result(
        [
            center_x - (cols - 1) / 2.0,
            center_y - (rows - 1) / 2.0,
        ]
    )


def find_center_positions(
    fold: int, input_img: Array, align_exp: float = 1.0
) -> FloatArray:
    """Estimate rotational center from image rotations and cross-correlation.

    Public adapter: NumPy input -> frozen :class:`CenterInput` -> private
    computation -> NumPy coordinate vector.
    """
    data = CenterInput(
        image=np.asarray(input_img, dtype=np.float64),
        fold=fold,
        align_exp=align_exp,
    )
    result = _find_center_positions(data)
    return np.asarray([result.x, result.y], dtype=np.float64)


def _best_fit_rigid_transform(source: Array, target: Array) -> tuple[Array, Array]:
    """Kabsch rigid transform mapping source to target."""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    H = (src - src_c).T @ (tgt - tgt_c)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_c - R @ src_c
    return R, t


def _icp_translation(moving: Array, fixed: Array, max_iterations: int = 50) -> Array:
    """Simple ICP returning final translation vector."""
    mov = np.asarray(moving, dtype=np.float64).copy()
    fix = np.asarray(fixed, dtype=np.float64)
    tree = cKDTree(fix)
    total_R = np.eye(mov.shape[1])
    total_t = np.zeros(mov.shape[1], dtype=np.float64)

    prev = np.inf
    for _ in range(max_iterations):
        d, idx = tree.query(mov, k=1)
        matched = fix[idx]
        R, t = _best_fit_rigid_transform(mov, matched)
        mov = (R @ mov.T).T + t
        total_t = R @ total_t + t
        total_R = R @ total_R
        rmse = float(np.sqrt(np.mean(d**2)))
        if abs(prev - rmse) < 1e-6:
            break
        prev = rmse
    return total_t


def _find_center_ptcloud(data: CenterInput) -> CenterResult:
    """Private point-cloud center stage; receives a frozen ``CenterInput``."""
    img = np.asarray(data.image)
    if data.points is None:
        raise ValueError("points are required for find_center_ptcloud")
    points = np.asarray(data.points, dtype=np.float64)
    fold = data.fold

    if img.ndim != 2:
        raise ValueError("input_img must be 2D")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("locs must be Nx3 or wider")
    if points.shape[0] < 3:
        raise ValueError("locs must contain at least three points")
    if not np.all(np.isfinite(points[:, :3])):
        raise ValueError("locs must contain finite x, y and z values")
    if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)):
        raise ValueError("fold must be an integer")
    if int(fold) < 1:
        raise ValueError("fold must be >= 1")

    sd = img.shape
    locs_1 = np.column_stack(
        [
            points[:, 0] - (sd[1] - 1) / 2.0,
            points[:, 1] - (sd[0] - 1) / 2.0,
            points[:, 2],
        ]
    )

    translations = []
    for i in range(1, int(fold)):
        angle = i * 360.0 / float(fold)
        th = np.deg2rad(angle)
        R2 = np.array(
            [[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]],
            dtype=np.float64,
        )
        rotated = locs_1.copy()
        rotated[:, 0:2] = (R2 @ locs_1[:, 0:2].T).T
        t = _icp_translation(rotated[:, 0:3], locs_1[:, 0:3])
        translations.append(t[:2])

    translations_arr = np.asarray(translations, dtype=np.float64)

    if fold == 2:
        return _center_result(translations_arr[0] / 2.0)
    if fold > 2:
        return _center_result(
            np.sum(translations_arr, axis=0) / float(fold)
        )

    return _center_result(np.zeros(2, dtype=np.float64))


def find_center_ptcloud(fold: int, input_img: Array, locs: Array) -> FloatArray:
    """Estimate rotational symmetry center from a point cloud.

    Public adapter: NumPy inputs -> frozen :class:`CenterInput` -> private
    computation -> NumPy coordinate vector.
    """
    data = CenterInput(
        image=np.asarray(input_img, dtype=np.float64),
        fold=fold,
        points=np.asarray(locs, dtype=np.float64),
    )
    result = _find_center_ptcloud(data)
    return np.asarray([result.x, result.y], dtype=np.float64)


__all__ = [
    "find_center_positions",
    "find_center_ptcloud",
]
