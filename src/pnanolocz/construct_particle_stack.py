"""
Particle-stack construction for NanoLocz-compatible AFM workflows.

This module ports MATLAB ``ConstructParticleStack.m`` from NanoLocz-lib 2025.

Purpose
-------
Given a full AFM image/movie and a table of particle localisations, this function
extracts particle-centered crops, applies the per-particle rotation and
sub-pixel centering correction stored in ``Part.Locs``, and returns a particle
image stack.

Public Python conventions
-------------------------
All inputs and outputs are frame-first:

    image_target: (frames, rows, cols)
    Particles.image: (particles, rows, cols)
    result: (particles, rows, cols)

A 2-D ``Particles.image`` is treated as one particle, but the result is always
3-D so no particle is dropped.

Localisation table convention
-----------------------------
``Particles.locs`` follows the detector table layout in array form:

    column 0 -> x coordinate
    column 1 -> y coordinate
    column 4 -> 0-based frame index
    column 7 -> rotation angle (optional)

Frame indices are 0-based, matching the detector output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift


FloatArray = np.ndarray[Any, np.dtype[np.float64]]
Array = np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class Particles:
    """Frozen particle record: image stack plus localisation table.

    Attributes
    ----------
    image : ndarray
        Particle image stack, frame-first ``(particles, rows, cols)``.
    locs : ndarray
        Localisation table with x, y and frame in columns 0, 1 and 4 and the
        particle rotation angle in column 7.
    """

    image: FloatArray
    locs: FloatArray


@dataclass(frozen=True, slots=True)
class StackResult:
    """Frozen result record for particle-stack construction."""

    image: FloatArray


def _matlab_round(x: float | Array) -> Array:
    """Round values like MATLAB ``round`` for positive and negative numbers.

    NumPy uses bankers rounding for ``.5`` values, while MATLAB rounds halves
    away from zero.  Particle crop boundaries can land exactly on half pixels,
    so matching MATLAB rounding avoids systematic off-by-one crop differences.
    """
    arr = np.asarray(x, dtype=np.float64)
    return np.asarray(np.sign(arr) * np.floor(np.abs(arr) + 0.5), dtype=np.float64)


def _as_stack(image: Array, *, name: str = "image") -> FloatArray:
    """Return a 2-D image or frame-first 3-D stack as a 3-D float64 stack."""
    arr = np.asarray(image, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")

    return np.asarray(arr, dtype=np.float64)


def _pad_spatial_stack(stack: Array, pad_rows: int, pad_cols: int) -> FloatArray:
    """Pad only the spatial axes of a frame-first stack with zeros."""
    return np.asarray(
        np.pad(
            stack,
            ((0, 0), (int(pad_rows), int(pad_rows)), (int(pad_cols), int(pad_cols))),
            mode="constant",
            constant_values=0,
        ),
        dtype=np.float64,
    )


def _center_pad_or_crop(image: Array, target_shape: tuple[int, int]) -> FloatArray:
    """Center-pad or center-crop a 2-D image to ``target_shape``.

    MATLAB's fallback path pads a previous particle image when the crop from the
    full image fails.  This helper makes that fallback deterministic even when
    the target size is only known from the expected crop dimensions.
    """
    src = np.asarray(image, dtype=np.float64)
    target_rows, target_cols = int(target_shape[0]), int(target_shape[1])

    out = np.zeros((target_rows, target_cols), dtype=np.float64)

    src_rows, src_cols = src.shape

    copy_rows = min(src_rows, target_rows)
    copy_cols = min(src_cols, target_cols)

    src_r0 = max(0, (src_rows - copy_rows) // 2)
    src_c0 = max(0, (src_cols - copy_cols) // 2)
    dst_r0 = max(0, (target_rows - copy_rows) // 2)
    dst_c0 = max(0, (target_cols - copy_cols) // 2)

    out[dst_r0 : dst_r0 + copy_rows, dst_c0 : dst_c0 + copy_cols] = src[
        src_r0 : src_r0 + copy_rows,
        src_c0 : src_c0 + copy_cols,
    ]

    return np.asarray(out, dtype=np.float64)


def _transform_particle_crop(
    crop: Array,
    angle_degrees: float,
    shift_x: float,
    shift_y: float,
    *,
    quick: bool,
) -> FloatArray:
    """Apply particle rotation and sub-pixel centering correction.

    MATLAB behavior:
    - ``quick == 0`` tries a rigid transform with ``-angle`` and translation.
    - Fallback and quick mode use ``imrotate(..., angle, 'crop')`` followed by
      ``imtranslate(..., [shiftx, shifty])``.

    SciPy does not have a direct equivalent of MATLAB's centered
    ``rigidtform2d`` + ``affineOutputView`` workflow, so the precise branch uses
    a close centered approximation: rotate by ``-angle`` and then shift.
    """
    arr = np.asarray(crop, dtype=np.float64)

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if quick:
        rotated = scipy_rotate(
            arr,
            angle=float(angle_degrees),
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
    else:
        rotated = scipy_rotate(
            arr,
            angle=-float(angle_degrees),
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

    # MATLAB imtranslate uses [x, y].  SciPy shift uses (row, col), so the
    # translation order becomes (shift_y, shift_x).
    shifted = scipy_shift(
        rotated,
        shift=(float(shift_y), float(shift_x)),
        order=3 if not quick else 1,
        mode="constant",
        cval=0.0,
        prefilter=not quick,
    )

    return np.asarray(shifted, dtype=np.float64)


def _construct_particle_stack(
    image_target: Array,
    part: Particles,
    quick: bool = False,
    *,
    xtra: int = 20,
) -> StackResult:
    """Private particle-stack stage; receives a frozen :class:`Particles`.

    Parameters
    ----------
    image_target : ndarray
        Full AFM image or frame-first stack ``(frames, rows, cols)``.
    part : Particles
        Particle record with a frame-first image stack and a localisation
        table (x, y and frame in columns 0, 1 and 4, angle in column 7).
    quick : bool, default False
        Use the MATLAB quick transform branch.
    xtra : int, default 20
        Extra border used by the MATLAB crop construction.

    Returns
    -------
    ndarray of float64
        Particle image stack ``(particles, rows, cols)``.  A 2-D
        ``part.image`` is treated as one particle, but the result is always
        3-D so no particle is dropped.
    """
    image_stack = _as_stack(image_target, name="image_target")
    part_stack = _as_stack(part.image, name="part.image")
    locs = np.asarray(part.locs, dtype=np.float64)

    if locs.ndim != 2:
        raise ValueError("part.locs must be a 2D localisation table")
    if locs.shape[1] < 5:
        raise ValueError("part.locs must contain x, y and frame columns")
    if locs.shape[0] != part_stack.shape[0]:
        raise ValueError("part.locs rows must match part.image particles")

    extra = int(xtra)
    if extra < 0:
        raise ValueError("xtra must be non-negative")

    _, crop_rows, crop_cols = part_stack.shape

    # MATLAB pads Part.Image by xtra/2 before using it in the fallback path.
    part_fallback_stack = _pad_spatial_stack(
        part_stack,
        pad_rows=int(extra // 2),
        pad_cols=int(extra // 2),
    )

    # MATLAB pads the full movie by round(sd)+xtra in each spatial direction.
    pad_rows = int(_matlab_round(crop_rows).item()) + extra
    pad_cols = int(_matlab_round(crop_cols).item()) + extra
    image_padded = _pad_spatial_stack(image_stack, pad_rows=pad_rows, pad_cols=pad_cols)

    if crop_rows % 2 == 0:
        dy1, dy2 = 0.0, 1.0
    else:
        dy1, dy2 = 0.5, 0.5

    if crop_cols % 2 == 0:
        dx1, dx2 = 0.0, 1.0
    else:
        dx1, dx2 = 0.5, 0.5

    n_particles = int(locs.shape[0])
    crop_shape_with_extra = (int(crop_rows + 2 * extra), int(crop_cols + 2 * extra))
    result = np.zeros(
        (n_particles, crop_shape_with_extra[0], crop_shape_with_extra[1]),
        dtype=np.float64,
    )

    for idx in range(n_particles):
        x = float(locs[idx, 0])
        y = float(locs[idx, 1])

        frame_value = locs[idx, 4]
        if not np.isfinite(frame_value):
            raise ValueError("part.locs frame index must be finite")
        rounded_frame = float(_matlab_round(frame_value).item())
        if abs(float(frame_value) - rounded_frame) > 1e-9:
            raise ValueError("part.locs frame index must be an integer")
        frame_idx = int(rounded_frame)
        if frame_idx < 0:
            raise ValueError("part.locs frame index must be non-negative")

        angle = (
            float(locs[idx, 7])
            if locs.shape[1] >= 8 and np.isfinite(locs[idx, 7])
            else 0.0
        )

        # MATLAB 1-based inclusive crop bounds.
        clip_y1 = int(_matlab_round(y + crop_rows / 2.0 + dy1).item())
        clip_y2 = int(_matlab_round(y + crop_rows * 1.5 - dy2).item()) + 2 * extra
        clip_x1 = int(_matlab_round(x + crop_cols / 2.0 + dx1).item())
        clip_x2 = int(_matlab_round(x + crop_cols * 1.5 - dx2).item()) + 2 * extra

        # Convert to Python 0-based half-open bounds.
        y0 = clip_y1 - 1
        y1 = clip_y2
        x0 = clip_x1 - 1
        x1 = clip_x2

        try:
            if frame_idx < 0 or frame_idx >= image_padded.shape[0]:
                raise IndexError("frame index out of range")

            crop = image_padded[frame_idx, y0:y1, x0:x1]

            if crop.shape != crop_shape_with_extra:
                raise IndexError("crop has an unexpected shape")

        except (IndexError, ValueError):
            # MATLAB falls back to a padded version of Part.Image(:,:,i).  If
            # there are fewer fallback images than localisations, reuse the last
            # available fallback image.
            fallback_idx = min(idx, part_fallback_stack.shape[0] - 1)
            crop = _center_pad_or_crop(
                part_fallback_stack[fallback_idx], crop_shape_with_extra
            )

        shift_y = float(_matlab_round(y + crop_rows / 2.0 + dy1).item()) - (
            y + crop_rows / 2.0 + dy1
        )
        shift_x = float(_matlab_round(x + crop_cols / 2.0 + dx1).item()) - (
            x + crop_cols / 2.0 + dx1
        )

        result[idx] = _transform_particle_crop(
            crop,
            angle_degrees=angle,
            shift_x=shift_x,
            shift_y=shift_y,
            quick=bool(quick),
        )

    # MATLAB removes the extra border after transformation:
    # Result = Result(1+xtra:end-xtra, 1+xtra:end-xtra, :)
    if extra > 0:
        result = result[:, extra:-extra, extra:-extra]

    return StackResult(image=np.asarray(result, dtype=np.float64))


def construct_particle_stack(
    image_target: Array,
    particle_images: Array,
    locs: Array,
    quick: bool = False,
    *,
    xtra: int = 20,
) -> FloatArray:
    """Construct particle-centered image crops from an AFM image/movie.

    Public adapter: NumPy inputs -> frozen :class:`Particles` -> private
    computation -> NumPy particle stack.
    """
    part = Particles(
        image=_as_stack(particle_images, name="particle_images"),
        locs=np.asarray(locs, dtype=np.float64),
    )
    result = _construct_particle_stack(image_target, part, quick=quick, xtra=xtra)
    return result.image


__all__ = ["Particles", "construct_particle_stack"]
