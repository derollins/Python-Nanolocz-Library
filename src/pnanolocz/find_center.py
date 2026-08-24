"""
Rotational-center estimation utilities for NanoLocz-compatible workflows.

This module ports MATLAB ``FindCenterPositions.m`` and ``FindCenter_ptCloud.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import rotate as scipy_rotate, zoom
from scipy.spatial import cKDTree

try:
    from pnanolocz.align_trans import normxcorr2
except Exception:  # pragma: no cover - standalone use
    try:
        from align_trans import normxcorr2  # type: ignore
    except Exception:
        normxcorr2 = None  # type: ignore[assignment]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _peak_abs(corr: np.ndarray) -> tuple[int, int]:
    """Return row/column index of maximum absolute correlation."""
    idx = int(np.nanargmax(np.abs(corr)))
    return tuple(int(v) for v in np.unravel_index(idx, corr.shape))  # type: ignore[return-value]


def _cc_align(img: np.ndarray, ref: np.ndarray, exp: float) -> np.ndarray:
    """Cross-correlation alignment helper from MATLAB ``ccAlign``."""
    if normxcorr2 is None:
        raise ImportError("normxcorr2 is required; install/use pnanolocz.align_trans")

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

    return np.array([(xpeak + 1) - img.shape[1], (ypeak + 1) - img.shape[0]], dtype=np.float64)


def find_center_positions(fold: int, input_img: np.ndarray, align_exp: float = 1.0) -> FloatArray:
    """Estimate rotational center from image rotations and cross-correlation."""
    img = np.asarray(input_img, dtype=np.float64)

    if int(fold) < 1:
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
        return np.asarray(offsets[0] / 2.0, dtype=np.float64)

    if fold > 2:
        return np.asarray(np.sum(offsets, axis=0) / float(fold), dtype=np.float64)

    # fold == 1: center of mass shift relative to geometric center.
    rows, cols = img.shape
    X, Y = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    mean_img = float(np.mean(img))
    if mean_img == 0:
        return np.zeros(2, dtype=np.float64)

    center_x = float(np.mean(img * X) / mean_img)
    center_y = float(np.mean(img * Y) / mean_img)
    return np.asarray([center_x - cols / 2.0, center_y - rows / 2.0], dtype=np.float64)


def _best_fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _icp_translation(moving: np.ndarray, fixed: np.ndarray, max_iterations: int = 50) -> np.ndarray:
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


def find_center_ptcloud(fold: int, input_img: np.ndarray, locs: np.ndarray) -> FloatArray:
    """Estimate rotational symmetry center from a point cloud."""
    img = np.asarray(input_img)
    points = np.asarray(locs, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("locs must be Nx3 or wider")

    sd = img.shape
    locs_1 = np.column_stack(
        [
            points[:, 0] - sd[1] / 2.0,
            points[:, 1] - sd[0] / 2.0,
            points[:, 2],
        ]
    )

    translations = []
    for i in range(1, int(fold)):
        angle = i * 360.0 / float(fold)
        th = np.deg2rad(angle)
        R2 = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]], dtype=np.float64)
        rotated = locs_1.copy()
        rotated[:, 0:2] = (R2 @ locs_1[:, 0:2].T).T
        t = _icp_translation(rotated[:, 0:3], locs_1[:, 0:3])
        translations.append(t[:2])

    translations_arr = np.asarray(translations, dtype=np.float64)

    if fold == 2:
        return np.asarray(translations_arr[0] / 2.0, dtype=np.float64)
    if fold > 2:
        return np.asarray(np.sum(translations_arr, axis=0) / float(fold), dtype=np.float64)

    return np.zeros(2, dtype=np.float64)


FindCenterPositions = find_center_positions
FindCenter_ptCloud = find_center_ptcloud

__all__ = [
    "find_center_positions",
    "find_center_ptcloud",
    "FindCenterPositions",
    "FindCenter_ptCloud",
]
