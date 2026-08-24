"""
Particle clash prevention for AFM simulations.

This module ports MATLAB ``prevent_clash.m``.  It iteratively moves 3-D points
apart until all pairwise distances are at least ``diameter``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def prevent_clash(
    diameter: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    dp: float = 0.2,
    max_iterations: int = 10_000,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Move 3-D points apart until no pair is closer than ``diameter``."""
    x_out = np.asarray(x, dtype=np.float64).copy()
    y_out = np.asarray(y, dtype=np.float64).copy()
    z_out = np.asarray(z, dtype=np.float64).copy()

    if not (x_out.shape == y_out.shape == z_out.shape):
        raise ValueError("x, y, and z must have the same shape")

    for _ in range(int(max_iterations)):
        pts = np.column_stack([x_out.ravel(), y_out.ravel(), z_out.ravel()])
        if pts.shape[0] < 2:
            break

        dists = squareform(pdist(pts))
        too_close = np.tril(dists < float(diameter), k=-1)
        clash1, clash2 = np.where(too_close)

        if clash1.size == 0:
            break

        for i, j in zip(clash1, clash2, strict=False):
            dx = x_out.flat[i] - x_out.flat[j]
            dy = y_out.flat[i] - y_out.flat[j]
            dz = z_out.flat[i] - z_out.flat[j]

            if dx == 0 and dy == 0 and dz == 0:
                # Degenerate overlap: push deterministically along x.
                dx = float(diameter)

            x_out.flat[i] += dx * float(dp)
            x_out.flat[j] -= dx * float(dp)
            y_out.flat[i] += dy * float(dp)
            y_out.flat[j] -= dy * float(dp)
            z_out.flat[i] += dz * float(dp)
            z_out.flat[j] -= dz * float(dp)

    return (
        np.asarray(x_out, dtype=np.float64),
        np.asarray(y_out, dtype=np.float64),
        np.asarray(z_out, dtype=np.float64),
    )


__all__ = ["prevent_clash"]
