"""
Generate rotationally symmetric point-cloud copies.

Ports MATLAB ``sym_ptCloud.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def sym_ptcloud(fold: int, input_img: np.ndarray, locs: np.ndarray) -> FloatArray:
    """Return original plus rotated copies of x/y point coordinates."""
    img = np.asarray(input_img)
    points = np.asarray(locs, dtype=np.float64).copy()

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("locs must be an Nx2 or wider array")

    rows, cols = img.shape[:2]
    centered = points.copy()
    centered[:, 0] -= cols / 2.0
    centered[:, 1] -= rows / 2.0

    all_points = [centered]

    for i in range(1, int(fold)):
        angle = i * 360.0 / float(fold)
        th = np.deg2rad(angle)
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]], dtype=np.float64)
        rotated = centered.copy()
        rotated[:, 0:2] = (R @ centered[:, 0:2].T).T
        all_points.append(rotated)

    sym = np.vstack(all_points)
    sym[:, 0] += cols / 2.0
    sym[:, 1] += rows / 2.0

    return np.asarray(sym, dtype=np.float64)


# MATLAB-style alias.
sym_ptCloud = sym_ptcloud

__all__ = ["sym_ptcloud", "sym_ptCloud"]
