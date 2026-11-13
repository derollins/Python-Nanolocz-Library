"""
Iterative particle-stack alignment for NanoLocz-compatible AFM workflows.

This module ports NanoLocz ``align_iterate.m``.  It combines translation
alignment, rotation alignment, crop reconstruction through
:func:`~pnanolocz.construct_particle_stack.construct_particle_stack`, and
optional reference updating.

Conventions
-----------
- ``full_img`` is frame-first ``(frames, rows, cols)``.
- ``Particles.image`` is particle-first ``(particles, rows, cols)``.
- Localisation frames are 0-based.
- Images stay NumPy arrays; the particle record is a frozen dataclass.

The per-particle translation and rotation are estimated from a thresholded
copy of the particle stack/ref, but the transforms themselves are applied to
the original (unthresholded) stack.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd  # type: ignore  # stubs optional
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift

from pnanolocz.align import align_rot, align_trans
from pnanolocz.construct_particle_stack import (
    Particles,
    construct_particle_stack as _default_construct_particle_stack,
)

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
Array = np.ndarray[Any, Any]


def _as_stack(image: Array, *, name: str) -> FloatArray:
    """Return a 2-D image or frame-first 3-D stack as a 3-D float64 stack."""
    arr = np.asarray(image, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")

    return np.asarray(arr, dtype=np.float64)


def _as_image(image: Array, *, name: str) -> FloatArray:
    """Return a 2-D float64 image with non-finite values zeroed."""
    arr = np.asarray(image, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")

    return np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))


def _set_part_image_locs(
    part: Particles, image: Array, locs: Array
) -> Particles:
    """Return a new particle record with copies of the updated arrays."""
    return replace(
        part,
        image=np.asarray(image, dtype=np.float64).copy(),
        locs=np.asarray(locs, dtype=np.float64).copy(),
    )


def _ensure_locs_columns(locs: Array, n_cols: int) -> FloatArray:
    """Pad a localisation table with NaNs so that required columns exist."""
    arr = np.asarray(locs, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError("part.locs must be a 2D table")

    if arr.shape[1] >= n_cols:
        return arr.copy()

    padded = np.full((arr.shape[0], n_cols), np.nan, dtype=np.float64)
    padded[:, : arr.shape[1]] = arr
    return padded


def _translate_stack(stack: FloatArray, x: Array, y: Array) -> FloatArray:
    """Apply MATLAB ``imtranslate(frame, [-x, -y])`` to every frame."""
    out = np.empty_like(stack, dtype=np.float64)

    for idx in range(stack.shape[0]):
        out[idx] = scipy_shift(
            stack[idx],
            shift=(-float(y[idx]), -float(x[idx])),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    return np.asarray(out, dtype=np.float64)


def _rotate_stack(stack: FloatArray, angles: Array) -> FloatArray:
    """Apply MATLAB ``imrotate(frame, angle, 'crop')`` to every frame."""
    out = np.empty_like(stack, dtype=np.float64)

    for idx in range(stack.shape[0]):
        out[idx] = scipy_rotate(
            stack[idx],
            angle=float(angles[idx]),
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class IterateInput:
    """Frozen input record for iterative particle alignment."""

    full_image: FloatArray
    reference: FloatArray
    particles: Particles
    tran_iterations: int
    translat_method: str
    maxdrift: float
    rot_iterations: int
    rota_method: str
    maxang: float
    thresh_min: float
    autoupdateref: bool


@dataclass(frozen=True, slots=True)
class IterateResult:
    """Frozen result record for iterative particle alignment."""

    image: FloatArray
    locs: FloatArray
    reference: FloatArray


def _alignment_frame(locs: FloatArray) -> pd.DataFrame:
    """Build the public per-particle correction table."""
    frame = locs[:, 4]
    if not np.all(np.isfinite(frame)):
        raise ValueError("particle frame values must be finite")
    frame_int = np.rint(frame).astype(np.intp)
    if not np.array_equal(frame, frame_int):
        raise ValueError("particle frame values must be integers")
    if np.any(frame_int < 0):
        raise ValueError("particle frame values must be non-negative")
    return pd.DataFrame(
        {
            "particle_id": np.arange(locs.shape[0], dtype=np.intp),
            "frame": frame_int,
            "x": locs[:, 0],
            "y": locs[:, 1],
            "angle": locs[:, 7],
        }
    )


def _align_iterate(
    data: IterateInput,
    constructor: Callable[[Array, Array, Array, bool], Array],
) -> IterateResult:
    """Private iterative alignment stage; receives a frozen ``IterateInput``."""
    tran_iterations = data.tran_iterations
    rot_iterations = data.rot_iterations
    if isinstance(tran_iterations, bool) or not isinstance(
        tran_iterations, (int, np.integer)
    ):
        raise ValueError("tran_iterations must be an integer")
    if isinstance(rot_iterations, bool) or not isinstance(
        rot_iterations, (int, np.integer)
    ):
        raise ValueError("rot_iterations must be an integer")
    if int(tran_iterations) < 0 or int(rot_iterations) < 0:
        raise ValueError("iteration counts must be non-negative")
    if not np.isfinite(float(data.thresh_min)) or float(data.thresh_min) < 0.0:
        raise ValueError("thresh_min must be finite and non-negative")
    if int(rot_iterations) > 0:
        if not np.isfinite(float(data.maxang)) or float(data.maxang) <= 0.0:
            raise ValueError("maxang must be finite and positive")

    stack = _as_stack(data.particles.image, name="particle_images")
    locs = _ensure_locs_columns(data.particles.locs, 8)
    ref_current = _as_image(data.reference, name="ref")

    n_particles = stack.shape[0]
    if locs.shape[0] != n_particles:
        raise ValueError("particle_images and locs must have the same length")

    # A missing rotation column means "no rotation yet"; start it from zero so
    # the estimated angles can accumulate.
    if int(rot_iterations) > 0:
        locs[~np.isfinite(locs[:, 7]), 7] = 0.0

    part_current = _set_part_image_locs(data.particles, stack, locs)
    tran_count = 0
    rot_count = 0

    while tran_count < int(tran_iterations) or rot_count < int(rot_iterations):
        if float(data.thresh_min) > 0.0:
            measure_stack = np.where(
                stack > float(data.thresh_min), stack, 0.0
            ).astype(np.float64)
            measure_ref = np.where(
                ref_current > float(data.thresh_min), ref_current, 0.0
            ).astype(np.float64)
        else:
            measure_stack = stack
            measure_ref = ref_current

        if tran_count < int(tran_iterations):
            tran_count += 1

            translation = align_trans(
                measure_stack,
                measure_ref,
                pixel_shift=float(data.maxdrift),
                sub_pix=True,
                method=data.translat_method,
            )
            x = translation["dx"].to_numpy(dtype=np.float64)
            y = translation["dy"].to_numpy(dtype=np.float64)

            stack = _translate_stack(stack, x, y)
            locs[:, 0] = locs[:, 0] + x[: locs.shape[0]]
            locs[:, 1] = locs[:, 1] + y[: locs.shape[0]]

        if rot_count < int(rot_iterations):
            rot_count += 1

            # Rebuild the measurement image from the current stack so rotation
            # sees the translation that was just applied.
            if float(data.thresh_min) > 0.0:
                measure_stack = np.where(
                    stack > float(data.thresh_min), stack, 0.0
                ).astype(np.float64)
            else:
                measure_stack = stack

            rotation = align_rot(
                measure_ref,
                measure_stack,
                angle_range=(-float(data.maxang), float(data.maxang)),
                method=data.rota_method,
            )
            rot_angles = rotation["angle"].to_numpy(dtype=np.float64)

            if rot_angles.size != n_particles:
                raise ValueError("align_rot returned a different number of angles")

            stack = _rotate_stack(stack, rot_angles)
            locs[:, 7] = locs[:, 7] + rot_angles[: locs.shape[0]]

        part_current = _set_part_image_locs(part_current, stack, locs)
        rebuilt = constructor(
            data.full_image, part_current.image, part_current.locs, False
        )
        rebuilt_stack = _as_stack(rebuilt, name="construct_particle_stack result")

        if rebuilt_stack.shape[0] != n_particles:
            raise ValueError(
                "construct_particle_stack returned a different number of particles"
            )
        if rebuilt_stack.shape[1:] != stack.shape[1:]:
            raise ValueError("construct_particle_stack returned a different crop shape")

        stack = rebuilt_stack
        part_current = _set_part_image_locs(part_current, stack, locs)

        if bool(data.autoupdateref):
            ref_current = np.asarray(np.nanmean(stack, axis=0), dtype=np.float64)

    return IterateResult(
        image=np.asarray(part_current.image, dtype=np.float64),
        locs=np.asarray(part_current.locs, dtype=np.float64),
        reference=np.asarray(ref_current, dtype=np.float64),
    )


def align_iterate(
    full_img: Array,
    ref: Array,
    particle_images: Array,
    locs: Array,
    tran_iterations: int,
    translat_method: str,
    maxdrift: float,
    rot_iterations: int,
    rota_method: str,
    maxang: float,
    thresh_min: float,
    autoupdateref: bool,
    *,
    construct_particle_stack: (
        Callable[[Array, Array, Array, bool], Array] | None
    ) = None,
) -> tuple[FloatArray, pd.DataFrame, FloatArray]:
    """Iteratively align particle crops to a reference.

    Public adapter: NumPy inputs -> frozen :class:`Particles` inside a frozen
    :class:`IterateInput` -> private computation -> ``(stack, table, ref)``.

    Parameters
    ----------
    full_img : ndarray
        Original full image/stack used to rebuild particle crops.
    ref : ndarray
        2-D reference image.
    particle_images : ndarray
        2-D image or frame-first 3-D stack ``(particles, rows, cols)``.
    locs : ndarray
        Localisation table with x, y and frame in columns 0, 1 and 4 and the
        particle rotation angle in column 7.
    tran_iterations, rot_iterations : int
        Number of translation and rotation iterations.
    translat_method, rota_method : str
        Correlation method names passed to :func:`align_trans` and
        :func:`align_rot`.
    maxdrift, maxang, thresh_min : float
        Search limits and threshold as in the NanoLocz workflow.
    autoupdateref : bool
        Update the reference to the mean aligned particle image.

    Returns
    -------
    stack : ndarray
        Rebuilt particle stack ``(particles, rows, cols)``.
    alignment : pandas.DataFrame
        One row per particle with ``particle_id, frame, x, y, angle``.
    reference : ndarray
        Final 2-D reference image.
    """
    part = Particles(
        image=_as_stack(particle_images, name="particle_images"),
        locs=np.asarray(locs, dtype=np.float64),
    )
    data = IterateInput(
        full_image=np.asarray(full_img, dtype=np.float64),
        reference=np.asarray(ref, dtype=np.float64),
        particles=part,
        tran_iterations=tran_iterations,
        translat_method=translat_method,
        maxdrift=maxdrift,
        rot_iterations=rot_iterations,
        rota_method=rota_method,
        maxang=maxang,
        thresh_min=thresh_min,
        autoupdateref=bool(autoupdateref),
    )
    constructor = construct_particle_stack or _default_construct_particle_stack
    result = _align_iterate(data, constructor)
    return result.image, _alignment_frame(result.locs), result.reference


align_iterate.__version__ = "0.1.0"  # type: ignore[attr-defined]

__all__ = ["align_iterate"]
