"""
Rotational symmetry tools for NanoLocz.

This module ports MATLAB ``rotation_sym.m`` and ``sym_ptCloud.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def rotation_sym(image: np.ndarray, fold: int) -> FloatArray:
    """Average an image with its rotated copies."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("rotation_sym expects a 2-D image")

    nfold = int(fold)
    if nfold <= 1:
        return arr.copy()

    acc = arr.copy()
    for i in range(2, nfold + 1):
        angle = (i - 1) * 360.0 / nfold
        acc += scipy_rotate(
            arr,
            angle=angle,
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

    return np.asarray(acc / nfold, dtype=np.float64)


def sym_ptcloud(fold: int, input_img: np.ndarray, locs: np.ndarray) -> FloatArray:
    """Generate rotationally symmetric copies of x/y point coordinates."""
    image = np.asarray(input_img)
    points = np.asarray(locs, dtype=np.float64).copy()

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("locs must be an Nx2 or wider array")

    center_x = image.shape[1] / 2.0
    center_y = image.shape[0] / 2.0

    base = points.copy()
    base[:, 0] -= center_x
    base[:, 1] -= center_y

    all_points = [base]
    for i in range(1, int(fold)):
        angle = i * 360.0 / float(fold)
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        rotated = base.copy()
        rotated[:, 0:2] = (R @ base[:, 0:2].T).T
        all_points.append(rotated)

    out = np.vstack(all_points)
    out[:, 0] += center_x
    out[:, 1] += center_y
    return np.asarray(out, dtype=np.float64)


# MATLAB-style aliases.
rotation_symmetry = rotation_sym
sym_ptCloud = sym_ptcloud

__all__ = ["rotation_sym", "rotation_symmetry", "sym_ptcloud", "sym_ptCloud"]
