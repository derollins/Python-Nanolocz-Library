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
The default Python convention is frame-first:

    ImageTarget: (frames, rows, cols)
    Part.Image:  (particles, rows, cols)
    Result:      (particles, rows, cols)

For MATLAB-style arrays, use :func:`construct_particle_stack_matlab`, which
expects and returns ``(rows, cols, frames/particles)``.

Localisation table convention
-----------------------------
``Part.Locs`` follows the MATLAB detector table layout:

    column 1 -> x coordinate       -> Python column 0
    column 2 -> y coordinate       -> Python column 1
    column 5 -> frame index        -> Python column 4
    column 8 -> rotation angle     -> Python column 7

By default, Python indexing is used for the frame column.  Set
``matlab_indexing=True`` when ``Part.Locs[:, 4]`` contains MATLAB 1-based frame
indices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift


FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass
class ParticleSet:
    """Small Python container matching MATLAB's ``Part`` structure."""

    image: FloatArray
    locs: FloatArray


def _matlab_round(x: float | np.ndarray) -> np.ndarray:
    """Round values like MATLAB ``round`` for positive and negative numbers.

    NumPy uses bankers rounding for ``.5`` values, while MATLAB rounds halves
    away from zero.  Particle crop boundaries can land exactly on half pixels,
    so matching MATLAB rounding avoids systematic off-by-one crop differences.
    """
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def _get_part_image(part: Any) -> FloatArray:
    """Read ``Part.Image`` from a dataclass, dict, or object."""
    if isinstance(part, dict):
        if "Image" in part:
            return np.asarray(part["Image"], dtype=np.float64)
        return np.asarray(part["image"], dtype=np.float64)

    if hasattr(part, "Image"):
        return np.asarray(getattr(part, "Image"), dtype=np.float64)

    return np.asarray(getattr(part, "image"), dtype=np.float64)


def _get_part_locs(part: Any) -> FloatArray:
    """Read ``Part.Locs`` from a dataclass, dict, or object."""
    if isinstance(part, dict):
        if "Locs" in part:
            return np.asarray(part["Locs"], dtype=np.float64)
        return np.asarray(part["locs"], dtype=np.float64)

    if hasattr(part, "Locs"):
        return np.asarray(getattr(part, "Locs"), dtype=np.float64)

    return np.asarray(getattr(part, "locs"), dtype=np.float64)


def _as_frame_first(stack: np.ndarray, frame_axis: int) -> tuple[FloatArray, bool, int]:
    """Convert a 2-D image or 3-D stack to frame-first layout."""
    arr = np.asarray(stack, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0

    if arr.ndim != 3:
        raise ValueError("expected a 2D image or 3D stack")

    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _restore_frame_axis(stack: np.ndarray, was_2d: bool, frame_axis: int) -> np.ndarray:
    """Restore a frame-first stack to the requested output axis convention."""
    if was_2d:
        return np.asarray(stack[0], dtype=np.float64)
    return np.asarray(np.moveaxis(stack, 0, frame_axis), dtype=np.float64)


def _pad_spatial_stack(stack: np.ndarray, pad_rows: int, pad_cols: int) -> FloatArray:
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


def _center_pad_or_crop(image: np.ndarray, target_shape: tuple[int, int]) -> FloatArray:
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
    crop: np.ndarray,
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


def construct_particle_stack(
    image_target: np.ndarray,
    part: Any,
    quick: bool | int = False,
    *,
    frame_axis: int = 0,
    part_frame_axis: int = 0,
    matlab_indexing: bool = True,
    xtra: int = 20,
) -> np.ndarray:
    """Construct particle-centered image crops from an AFM image/movie.

    Parameters
    ----------
    image_target:
        Full AFM image/movie.  Default layout is ``(frames, rows, cols)``.
    part:
        Particle container.  Accepted forms are:
        ``ParticleSet(image=..., locs=...)``, dict with ``image``/``locs`` or
        ``Image``/``Locs``, or an object with matching attributes.
    quick:
        If true, use the MATLAB quick transform branch.  If false, use the
        higher-order centered approximation of the rigid transform branch.
    frame_axis:
        Frame axis for ``image_target``.  Use ``-1`` for MATLAB layout.
    part_frame_axis:
        Particle/frame axis for ``Part.Image``.  Use ``-1`` for MATLAB layout.
    matlab_indexing:
        If true (the default), ``Part.Locs[:, 4]`` is interpreted as the
        1-based frame labels produced by :func:`pnanolocz.detector.detector`.
        Pass false explicitly for Python 0-based frame indices.
    xtra:
        Extra border used by the MATLAB function.  Default is 20.

    Returns
    -------
    ndarray
        Particle image stack in the same particle-axis convention as
        ``Part.Image``.
    """
    image_stack, _, _ = _as_frame_first(image_target, frame_axis=frame_axis)
    part_image_raw = _get_part_image(part)
    part_stack, part_was_2d, part_axis = _as_frame_first(part_image_raw, frame_axis=part_frame_axis)
    locs = _get_part_locs(part)

    if locs.ndim != 2:
        raise ValueError("Part.Locs must be a 2D localisation table")

    if locs.shape[1] < 5:
        raise ValueError("Part.Locs must contain at least x, y, and frame columns")

    # ``sd`` is captured before MATLAB pads Part.Image.
    _, crop_rows, crop_cols = part_stack.shape
    extra = int(xtra)

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
    result = np.zeros((n_particles, crop_shape_with_extra[0], crop_shape_with_extra[1]), dtype=np.float64)

    for idx in range(n_particles):
        x = float(locs[idx, 0])
        y = float(locs[idx, 1])

        frame_value = locs[idx, 4]
        if not np.isfinite(frame_value):
            frame_idx = 0
        else:
            frame_idx = int(_matlab_round(frame_value).item())
            if matlab_indexing:
                frame_idx -= 1

        angle = float(locs[idx, 7]) if locs.shape[1] >= 8 and np.isfinite(locs[idx, 7]) else 0.0

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

        except Exception:
            # MATLAB falls back to a padded version of Part.Image(:,:,i).  If
            # there are fewer fallback images than localisations, reuse the last
            # available fallback image.
            fallback_idx = min(idx, part_fallback_stack.shape[0] - 1)
            crop = _center_pad_or_crop(part_fallback_stack[fallback_idx], crop_shape_with_extra)

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

    return _restore_frame_axis(result, part_was_2d, part_axis)


def construct_particle_stack_matlab(
    image_target: np.ndarray,
    part: Any,
    quick: bool | int = False,
) -> np.ndarray:
    """MATLAB-layout wrapper for :func:`construct_particle_stack`.

    Assumes:
    - ``ImageTarget`` is shaped ``(rows, cols, frames)``.
    - ``Part.Image`` is shaped ``(rows, cols, particles)``.
    - ``Part.Locs[:, 4]`` contains MATLAB 1-based frame indices.
    - The returned result is shaped ``(rows, cols, particles)``.
    """
    return construct_particle_stack(
        image_target,
        part,
        quick=quick,
        frame_axis=-1,
        part_frame_axis=-1,
        matlab_indexing=True,
    )


# MATLAB-style alias with the original capitalization.
ConstructParticleStack = construct_particle_stack_matlab


__all__ = [
    "ParticleSet",
    "construct_particle_stack",
    "construct_particle_stack_matlab",
    "ConstructParticleStack",
]
