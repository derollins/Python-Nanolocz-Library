"""
Particle line-profile width analysis for NanoLocz-compatible AFM images.

This module ports MATLAB ``Lineprofiler.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _matlab_round(x: np.ndarray | float) -> np.ndarray:
    """MATLAB-style rounding."""
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def _width_from_profile(profile: np.ndarray, pleng: int, loctor: float) -> float:
    """Find the largest peak width near the profile center."""
    prof = np.asarray(profile, dtype=np.float64)
    prof = prof[prof != 0]
    if prof.size < 3:
        return 0.0

    peaks, props = find_peaks(prof, prominence=0.01)
    if peaks.size == 0:
        return 0.0

    widths = peak_widths(prof, peaks, rel_height=0.5)[0]
    locs = peaks + 1  # MATLAB-style one-based locations for center test.

    pos = (locs < pleng / 2.0 + pleng * loctor) & (locs > pleng / 2.0 - pleng * loctor)
    if not np.any(pos):
        return 0.0

    return float(np.max(widths[pos]))


def lineprofiler(
    A: np.ndarray,
    xy: np.ndarray,
    max_radius: float,
    directions: np.ndarray | list[int],
    width_ref: str = "local height",
) -> tuple[FloatArray, FloatArray, FloatArray, list[list[np.ndarray]]]:
    """Measure feature widths along x/y/diagonal profiles."""
    img = np.asarray(A, dtype=np.float64)
    coords = np.asarray(xy, dtype=np.float64)
    dirs = np.asarray(directions, dtype=bool).ravel()

    if img.ndim != 2:
        raise ValueError("A must be a 2D image")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("xy must be Nx2")
    if dirs.size < 4:
        raise ValueError("directions must have four flags")

    loctor = 0.2
    pleng = int(round(max_radius)) * 2
    n = coords.shape[0]

    Fwidth = np.zeros((4, n), dtype=np.float64)
    p: list[list[np.ndarray]] = []

    for j in range(n):
        x0 = int(_matlab_round(coords[j, 0]).item())
        y0 = int(_matlab_round(coords[j, 1]).item())

        px = np.zeros((pleng, 1), dtype=np.float64)
        py = np.zeros((pleng, 1), dtype=np.float64)
        for jj in range(pleng):
            px[jj, 0] = x0 - ((jj + 1) - pleng / 2.0)
            py[jj, 0] = y0 - ((jj + 1) - pleng / 2.0)
        py_flip = np.flipud(py)
        p.append([px, py, py_flip])

        prof_x = np.zeros(pleng, dtype=np.float64)
        prof_y = np.zeros(pleng, dtype=np.float64)
        prof_xy1 = np.zeros(pleng, dtype=np.float64)
        prof_xy2 = np.zeros(pleng, dtype=np.float64)

        for jj in range(pleng):
            xi = int(_matlab_round(px[jj, 0]).item())
            yi = int(_matlab_round(py[jj, 0]).item())
            yi2 = int(_matlab_round(py_flip[jj, 0]).item())

            if 1 <= xi <= img.shape[1] and dirs[0] and 1 <= y0 <= img.shape[0]:
                prof_x[jj] = img[y0 - 1, xi - 1]
            if 1 <= yi <= img.shape[0] and dirs[1] and 1 <= x0 <= img.shape[1]:
                prof_y[jj] = img[yi - 1, x0 - 1]
            if 1 <= xi <= img.shape[1] and 1 <= yi <= img.shape[0] and dirs[2]:
                prof_xy1[jj] = img[yi - 1, xi - 1]
            if 1 <= xi <= img.shape[1] and 1 <= yi2 <= img.shape[0] and dirs[3]:
                prof_xy2[jj] = img[yi2 - 1, xi - 1]

        if dirs[0]:
            Fwidth[0, j] = _width_from_profile(prof_x, pleng, loctor)
        if dirs[1]:
            Fwidth[1, j] = _width_from_profile(prof_y, pleng, loctor)
        if dirs[2]:
            Fwidth[2, j] = _width_from_profile(prof_xy1, pleng, loctor) * np.sqrt(2.0)
        if dirs[3]:
            Fwidth[3, j] = _width_from_profile(prof_xy2, pleng, loctor) * np.sqrt(2.0)

    Rmax = np.max(Fwidth, axis=0)
    Rmin = np.min(Fwidth, axis=0)
    Rmean = np.nanmean(Fwidth, axis=0)

    return np.asarray(Rmin), np.asarray(Rmax), np.asarray(Rmean), p


Lineprofiler = lineprofiler

__all__ = ["lineprofiler", "Lineprofiler"]
