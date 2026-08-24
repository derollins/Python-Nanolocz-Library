"""
Thick line/boundary intensity profile extraction.

This module ports MATLAB ``improfile_thick.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _sample_points(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample image at x/y coordinates using bilinear interpolation."""
    arr = np.asarray(img, dtype=np.float64)
    x = xy[:, 0]
    y = xy[:, 1]
    # MATLAB coordinates are 1-based in GUI workflows.  Python samples are
    # easier in 0-based coordinates; users can pass whatever convention is
    # consistent with their perimeter.  Here we treat input as 1-based to match
    # MATLAB and subtract one.
    return map_coordinates(arr, [y - 1.0, x - 1.0], order=1, mode="nearest")


def _sample_segment(img: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Sample a straight segment between two x/y points."""
    distance = float(np.linalg.norm(p2 - p1))
    n = max(2, int(round(distance)) + 1)
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    return _sample_points(img, np.column_stack([xs, ys]))


def improfile_thick(
    thickness: int,
    perimeter: np.ndarray,
    img: np.ndarray,
    shape: str,
) -> FloatArray:
    """Average intensity profiles over a boundary thickness."""
    per = np.asarray(perimeter, dtype=np.float64)
    arr = np.asarray(img, dtype=np.float64)

    if per.ndim != 2 or per.shape[1] < 2:
        raise ValueError("perimeter must be Nx2 coordinates")

    thick = int(round(thickness))
    shape_lc = shape.lower()

    if shape_lc == "circle":
        cp = np.mean(per[:, :2], axis=0)
        radius_old = float(np.mean(np.sqrt((per[:, 0] - cp[0]) ** 2 + (per[:, 1] - cp[1]) ** 2)))
        profiles = []

        for offset in range(-thick, thick + 1):
            radius_new = radius_old + offset
            if radius_new > 2 and radius_old != 0:
                vec = per[:, :2] - cp[None, :]
                perimeter_t = cp[None, :] + vec * (radius_new / radius_old)
                profiles.append(_sample_points(arr, perimeter_t))

        if not profiles:
            return np.empty(0, dtype=np.float64)

        return np.asarray(np.nanmean(np.stack(profiles, axis=1), axis=1), dtype=np.float64)

    if shape_lc == "freehand":
        # MATLAB GUI branch only sets app.LineThickness.Value = 1 and does not
        # compute a profile.  For Python, return an empty array explicitly.
        return np.empty(0, dtype=np.float64)

    cz_parts = []
    for kk in range(per.shape[0] - 1):
        p1 = per[kk, :2]
        p2 = per[kk + 1, :2]
        segment_vec = p2 - p1
        norm = float(np.linalg.norm(segment_vec))
        if norm == 0:
            continue
        perp_vec = np.array([-segment_vec[1], segment_vec[0]], dtype=np.float64) / norm

        profiles = []
        for offset in range(-thick, thick + 1):
            p1_off = p1 + offset * perp_vec
            p2_off = p2 + offset * perp_vec
            profiles.append(_sample_segment(arr, p1_off, p2_off))

        max_len = max(p.size for p in profiles)
        prof_mat = np.full((max_len, len(profiles)), np.nan, dtype=np.float64)
        for col, prof in enumerate(profiles):
            prof_mat[: prof.size, col] = prof
        average = np.nanmean(prof_mat, axis=1)

        if kk == per.shape[0] - 2:
            cz_parts.append(average)
        else:
            cz_parts.append(average[:-1])

    if not cz_parts:
        return np.empty(0, dtype=np.float64)

    return np.asarray(np.concatenate(cz_parts), dtype=np.float64)


__all__ = ["improfile_thick"]
