"""
AFM topography simulation from atomic coordinates.

This module ports ``Mat_SimAFM.m``, ``Mat_SimAFM_dyn.m`` and
``Mat_SimAFM_spin.m`` from NanoLocz.  The model approximates AFM tip-sample
interaction with a spherical tip and conical extension.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _matlab_round(x: np.ndarray | float) -> np.ndarray:
    """MATLAB-style half-away-from-zero rounding."""
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def _prevent_clash_simple(coords: np.ndarray, min_dist: float = 2.8, iterations: int = 3) -> np.ndarray:
    """Small substitute for NanoLocz ``prevent_clash``.

    Points closer than ``min_dist`` are nudged apart in XY.  This is only used
    by the dynamic simulator and is intentionally conservative.
    """
    out = np.asarray(coords, dtype=np.float64).copy()
    for _ in range(iterations):
        for i in range(out.shape[0]):
            for j in range(i + 1, out.shape[0]):
                delta = out[j, :2] - out[i, :2]
                d = float(np.linalg.norm(delta))
                if 0 < d < min_dist:
                    push = (min_dist - d) / 2.0 * delta / d
                    out[i, :2] -= push
                    out[j, :2] += push
    return out


def _simulate_from_scaled_coords(coords_s: np.ndarray, r: float, angle: float, pix_per_ang: float, end_pos: np.ndarray, fspace_cone: float) -> FloatArray:
    """Core high-sampling NanoLocz AFM simulation loop."""
    rs = float(r) * float(pix_per_ang)
    tan_angle = np.tan(np.deg2rad(float(angle)))
    if tan_angle == 0:
        tan_angle = np.finfo(float).eps

    n_rows = int(end_pos[0, 1] - end_pos[0, 0] + 1)
    n_cols = int(end_pos[1, 1] - end_pos[1, 0] + 1)
    img = np.zeros((n_rows, n_cols), dtype=np.float64)

    grid = np.arange(-rs - fspace_cone, rs + fspace_cone + 1)
    dx, dy = np.meshgrid(grid, grid, indexing="ij")
    dx_flat = dx.ravel()
    dy_flat = dy.ravel()

    for atom in coords_s:
        offs_x = float(_matlab_round(atom[0]).item() - atom[0])
        offs_y = float(_matlab_round(atom[1]).item() - atom[1])

        dxl = dx_flat - offs_x
        dyl = dy_flat - offs_y

        radial_sq = (dxl / pix_per_ang) ** 2 + (dyl / pix_per_ang) ** 2
        sphere = np.sqrt(np.maximum(r * r - radial_sq, 0.0)) - r
        dh = sphere

        inside_sphere = dh > -r
        radial = np.sqrt(radial_sq)
        cone_h = atom[2] - r - ((radial - r) / tan_angle)
        h = np.where(inside_sphere, atom[2] + dh, cone_h)

        pos_x = _matlab_round(atom[0] + dxl - end_pos[0, 0]).astype(int)
        pos_y = _matlab_round(atom[1] + dyl - end_pos[1, 0]).astype(int)

        valid = (pos_x >= 0) & (pos_x < n_rows) & (pos_y >= 0) & (pos_y < n_cols)
        for px, py, hv in zip(pos_x[valid], pos_y[valid], h[valid], strict=False):
            if img[int(px), int(py)] < hv:
                img[int(px), int(py)] = hv

    return np.asarray(img, dtype=np.float64)


def mat_sim_afm(coords: np.ndarray, r: float, angle: float, pix_per_ang: float) -> FloatArray:
    """Simulate one AFM height image from atomic coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError("coords must be Nx3")

    rs = r * pix_per_ang
    max_z = float(np.max(coords[:, 2]))
    fspace = (max_z - r) * np.tan(np.deg2rad(angle)) * pix_per_ang + 1
    fspace_cone = fspace

    coords_s = np.column_stack([coords[:, 0] * pix_per_ang, coords[:, 1] * pix_per_ang, coords[:, 2]])

    end_pos = np.array(
        [
            [np.floor(np.min(coords[:, 0] * pix_per_ang) - rs - fspace), np.ceil(np.max(coords[:, 0] * pix_per_ang) + rs + fspace)],
            [np.floor(np.min(coords[:, 1] * pix_per_ang) - rs - fspace), np.ceil(np.max(coords[:, 1] * pix_per_ang) + rs + fspace)],
        ],
        dtype=np.float64,
    )

    return _simulate_from_scaled_coords(coords_s, r, angle, pix_per_ang, end_pos, fspace_cone)


def mat_sim_afm_dyn(
    coords: np.ndarray,
    r: float,
    angle: float,
    pix_per_ang: float,
    fluct_z: float,
    fluct_xy: float,
    n: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate a dynamic AFM image stack with Gaussian coordinate fluctuations."""
    coords = np.asarray(coords, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    rs = r * pix_per_ang
    max_z = float(np.max(coords[:, 2]))
    fspace = (max_z - r) * np.tan(np.deg2rad(angle)) * pix_per_ang + 1 + 5 * fluct_xy * pix_per_ang
    fspace_cone = (max_z - r) * np.tan(np.deg2rad(angle)) * pix_per_ang + 1

    end_pos = np.array(
        [
            [np.floor(np.min(coords[:, 0] * pix_per_ang) - rs - fspace), np.ceil(np.max(coords[:, 0] * pix_per_ang) + rs + fspace)],
            [np.floor(np.min(coords[:, 1] * pix_per_ang) - rs - fspace), np.ceil(np.max(coords[:, 1] * pix_per_ang) + rs + fspace)],
        ],
        dtype=np.float64,
    )

    frames = []
    for _ in range(int(n)):
        coords_s = np.column_stack(
            [
                coords[:, 0] * pix_per_ang + rng.normal(0, fluct_xy * pix_per_ang, coords.shape[0]),
                coords[:, 1] * pix_per_ang + rng.normal(0, fluct_xy * pix_per_ang, coords.shape[0]),
                coords[:, 2] + rng.normal(0, fluct_z, coords.shape[0]),
            ]
        )
        coords_s = _prevent_clash_simple(coords_s, min_dist=2.8)
        frames.append(_simulate_from_scaled_coords(coords_s, r, angle, pix_per_ang, end_pos, fspace_cone))

    return np.asarray(np.stack(frames, axis=0), dtype=np.float64)


def _rotate_coords(coords: np.ndarray, angle_deg: float, axis: str) -> np.ndarray:
    """Rotate coordinates around X/Y/Z using the MATLAB rotation conventions."""
    c = np.asarray(coords, dtype=np.float64).copy()
    th = np.deg2rad(angle_deg)
    ax = axis.upper()

    if ax == "X":
        y = c[:, 1] * np.cos(th) - c[:, 2] * np.sin(th)
        z = c[:, 1] * np.sin(th) + c[:, 2] * np.cos(th)
        c[:, 1], c[:, 2] = y, z
    elif ax == "Y":
        x = c[:, 0] * np.cos(th) + c[:, 2] * np.sin(th)
        z = c[:, 2] * np.cos(th) - c[:, 0] * np.sin(th)
        c[:, 0], c[:, 2] = x, z
    elif ax == "Z":
        x = c[:, 0] * np.cos(th) + c[:, 1] * np.sin(th)
        y = -c[:, 0] * np.sin(th) + c[:, 1] * np.cos(th)
        c[:, 0], c[:, 1] = x, y
    else:
        raise ValueError("axis must be 'X', 'Y', or 'Z'")

    return c


def mat_sim_afm_spin(
    coords: np.ndarray,
    r: float,
    angle: float,
    pix_per_ang: float,
    spin: np.ndarray,
    axis: str,
    z_thresh: float = 0.0,
) -> FloatArray:
    """Simulate AFM images for multiple rotated orientations."""
    coords = np.asarray(coords, dtype=np.float64)
    images = []

    for spin_angle in np.asarray(spin, dtype=np.float64).ravel():
        coords_sp = _rotate_coords(coords, float(spin_angle), axis)
        coords_sp[:, 2] = coords_sp[:, 2] - np.min(coords_sp[:, 2])

        if z_thresh > 0:
            threshold = z_thresh * (np.max(coords_sp[:, 2]) - np.min(coords_sp[:, 2])) + np.min(coords_sp[:, 2])
            coords_sp = coords_sp[coords_sp[:, 2] >= threshold, :3]
            coords_sp[:, 2] = coords_sp[:, 2] - threshold

        images.append(mat_sim_afm(coords_sp, r, angle, pix_per_ang))

    max_rows = max(img.shape[0] for img in images)
    max_cols = max(img.shape[1] for img in images)
    out = np.zeros((len(images), max_rows, max_cols), dtype=np.float64)

    center_row = int(np.ceil(max_rows / 2))
    center_col = int(np.ceil(max_cols / 2))

    for k, img in enumerate(images):
        start_row = center_row - img.shape[0] // 2
        start_col = center_col - img.shape[1] // 2
        out[k, start_row:start_row + img.shape[0], start_col:start_col + img.shape[1]] = img

    return np.asarray(out, dtype=np.float64)


# MATLAB-style aliases.
Mat_SimAFM = mat_sim_afm
Mat_SimAFM_dyn = mat_sim_afm_dyn
Mat_SimAFM_spin = mat_sim_afm_spin

__all__ = [
    "mat_sim_afm",
    "mat_sim_afm_dyn",
    "mat_sim_afm_spin",
    "Mat_SimAFM",
    "Mat_SimAFM_dyn",
    "Mat_SimAFM_spin",
]
