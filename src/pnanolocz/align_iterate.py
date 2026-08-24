"""
Iterative particle-stack alignment for NanoLocz-compatible AFM workflows.

This module ports NanoLocz ``align_iterate.m``.  The MATLAB version combines
translation alignment, rotation alignment, crop reconstruction through
``ConstructParticleStack``, and optional reference updating.

The companion Python port ``construct_particle_stack.py`` is used automatically
when it is importable.  You can still override that behavior by passing a custom
``construct_particle_stack`` callback, for example when testing or when using an
alternative crop reconstruction implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift

try:
    from pnanolocz.align_trans import align_trans
    from pnanolocz.align_rot import align_rot
except Exception:  # pragma: no cover - allows standalone use
    from align_trans import align_trans  # type: ignore
    from align_rot import align_rot  # type: ignore

try:
    from pnanolocz.construct_particle_stack import construct_particle_stack as _default_construct_particle_stack
except Exception:  # pragma: no cover - allows standalone use
    try:
        from construct_particle_stack import construct_particle_stack as _default_construct_particle_stack  # type: ignore
    except Exception:
        _default_construct_particle_stack = None  # type: ignore[assignment]


FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass
class Particles:
    """Container matching the MATLAB ``Part`` structure.

    Attributes
    ----------
    image:
        Particle image stack.  Python default is frame-first
        ``(particles, rows, cols)``.
    locs:
        Localisation table.  MATLAB columns 1, 2, and 8 correspond to Python
        columns 0, 1, and 7.
    """

    image: FloatArray
    locs: FloatArray


def _get_part_image(part: Any) -> FloatArray:
    """Read ``Part.Image`` from a dataclass, object, or dict."""
    if isinstance(part, dict):
        return np.asarray(part["Image"] if "Image" in part else part["image"], dtype=np.float64)
    return np.asarray(getattr(part, "Image", getattr(part, "image")), dtype=np.float64)


def _get_part_locs(part: Any) -> FloatArray:
    """Read ``Part.Locs`` from a dataclass, object, or dict."""
    if isinstance(part, dict):
        return np.asarray(part["Locs"] if "Locs" in part else part["locs"], dtype=np.float64)
    return np.asarray(getattr(part, "Locs", getattr(part, "locs")), dtype=np.float64)


def _set_part_image_locs(part: Any, image: np.ndarray, locs: np.ndarray) -> Any:
    """Return/update a particle container with new image and locs values."""
    image = np.asarray(image, dtype=np.float64)
    locs = np.asarray(locs, dtype=np.float64)

    if isinstance(part, Particles):
        return replace(part, image=image, locs=locs)

    if isinstance(part, dict):
        out = dict(part)
        if "Image" in out:
            out["Image"] = image
        else:
            out["image"] = image
        if "Locs" in out:
            out["Locs"] = locs
        else:
            out["locs"] = locs
        return out

    # Object-like MATLAB-style structure.  Update in-place and return it.
    if hasattr(part, "Image"):
        setattr(part, "Image", image)
    else:
        setattr(part, "image", image)

    if hasattr(part, "Locs"):
        setattr(part, "Locs", locs)
    else:
        setattr(part, "locs", locs)

    return part


def _as_frame_first(stack: np.ndarray, frame_axis: int = 0) -> tuple[FloatArray, bool, int]:
    """Convert a 2-D/3-D particle image stack to frame-first layout."""
    arr = np.asarray(stack, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0

    if arr.ndim != 3:
        raise ValueError("Part.Image must be 2D or 3D")

    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _restore_frame_axis(stack: np.ndarray, was_2d: bool, frame_axis: int) -> np.ndarray:
    """Restore a frame-first stack to its original axis convention."""
    if was_2d:
        return np.asarray(stack[0], dtype=np.float64)
    return np.asarray(np.moveaxis(stack, 0, frame_axis), dtype=np.float64)


def _ensure_locs_columns(locs: np.ndarray, n_cols: int) -> FloatArray:
    """Pad a localisation table with NaNs so that required columns exist."""
    arr = np.asarray(locs, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError("Part.Locs must be a 2D table")

    if arr.shape[1] >= n_cols:
        return arr.copy()

    padded = np.full((arr.shape[0], n_cols), np.nan, dtype=np.float64)
    padded[:, : arr.shape[1]] = arr
    return padded


def _translate_stack(stack: FloatArray, x: np.ndarray, y: np.ndarray) -> FloatArray:
    """Apply MATLAB ``imtranslate(frame, [-x, -y])`` to every frame.

    SciPy uses ``shift=(row_shift, col_shift)``, so the equivalent is
    ``shift=(-y, -x)``.
    """
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


def _rotate_stack(stack: FloatArray, angles: np.ndarray) -> FloatArray:
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


def align_iterate(
    full_img: np.ndarray,
    ref: np.ndarray,
    part: Any,
    tran_iterations: int,
    translat_method: str,
    maxdrift: float,
    rot_iterations: int,
    rota_method: str,
    maxang: float,
    thresh_min: float,
    autoupdateref: bool | int,
    *,
    frame_axis: int = 0,
    construct_particle_stack: Callable[[np.ndarray, Any, int], np.ndarray] | None = None,
) -> tuple[Any, np.ndarray]:
    """Iteratively align particle crops to a reference.

    Parameters
    ----------
    full_img:
        Original full image/stack used by ``ConstructParticleStack``.
    ref:
        2-D reference image.
    part:
        Particle container.  Accepted forms are ``Particles`` dataclass, dict
        with ``image``/``locs`` or ``Image``/``Locs``, or an object with matching
        attributes.
    tran_iterations:
        Number of translation iterations.
    translat_method:
        ``'Cross corr'`` or ``'FFT cross'``.
    maxdrift:
        Maximum allowed local translational drift.
    rot_iterations:
        Number of rotation iterations.
    rota_method:
        ``'Rotation corr'`` or ``'Polar Corr'``.
    maxang:
        Search range is ``[-maxang, +maxang]`` degrees.
    thresh_min:
        If positive, zero out pixels below this value before each alignment
        measurement, matching MATLAB ``Part.Image .* (Part.Image > Thresh_min)``.
    autoupdateref:
        If true, update ``ref`` to the average aligned particle image after each
        iteration.
    frame_axis:
        Axis containing particle frames in ``Part.Image``.  Use ``-1`` for
        MATLAB layout ``(rows, cols, particles)``.
    construct_particle_stack:
        Optional callback for the not-yet-ported ``ConstructParticleStack``.

    Returns
    -------
    part, ref:
        Updated particle container and final reference image.
    """
    image_raw = _get_part_image(part)
    locs = _ensure_locs_columns(_get_part_locs(part), 8)

    stack, was_2d, original_axis = _as_frame_first(image_raw, frame_axis=frame_axis)
    ref_current = np.asarray(ref, dtype=np.float64)

    n_particles = stack.shape[0]
    tran_count = 0
    rot_count = 0

    # Keep the caller's container synchronized between iterations.
    part_current = _set_part_image_locs(
        part,
        _restore_frame_axis(stack, was_2d, original_axis),
        locs,
    )

    while tran_count < int(tran_iterations) or rot_count < int(rot_iterations):
        if float(thresh_min) > 0:
            stack = stack * (stack > float(thresh_min))
            rf = ref_current * (ref_current > float(thresh_min))
        else:
            rf = ref_current

        if tran_count < int(tran_iterations):
            tran_count += 1

            x, y = align_trans(
                stack,
                rf,
                pixel_shift=float(maxdrift),
                sub_pix=True,
                method=translat_method,
                frame_axis=0,
            )

            stack = _translate_stack(stack, x, y)

            locs[:, 0] = locs[:, 0] + x[: locs.shape[0]]
            locs[:, 1] = locs[:, 1] + y[: locs.shape[0]]

        if rot_count < int(rot_iterations):
            rot_count += 1

            rot_angles = np.zeros(n_particles, dtype=np.float64)
            for idx in range(n_particles):
                rot_angles[idx] = align_rot(
                    rf,
                    stack[idx],
                    (-float(maxang), float(maxang)),
                    rota_method,
                )

            stack = _rotate_stack(stack, rot_angles)

            # MATLAB writes Part.Locs(:,8).  Python column index is 7.
            locs = _ensure_locs_columns(locs, 8)
            locs[:, 7] = locs[:, 7] + rot_angles[: locs.shape[0]]

        part_current = _set_part_image_locs(
            part_current,
            _restore_frame_axis(stack, was_2d, original_axis),
            locs,
        )

        # MATLAB calls ConstructParticleStack(full_img, Part, 0) after each
        # translation/rotation cycle.  The Python port now uses the companion
        # construct_particle_stack implementation by default when available,
        # while still allowing callers to inject a custom callback.
        constructor = construct_particle_stack
        if constructor is None:
            constructor = _default_construct_particle_stack

        if constructor is not None:
            try:
                # Preferred Python constructor accepts keyword arguments so it
                # can follow the same frame-axis convention as align_iterate.
                rebuilt = constructor(
                    full_img,
                    part_current,
                    0,
                    frame_axis=frame_axis,
                    part_frame_axis=frame_axis,
                    matlab_indexing=False,
                )
            except TypeError:
                # Backwards-compatible path for user-supplied callbacks with
                # the MATLAB-style 3-argument signature.
                rebuilt = constructor(full_img, part_current, 0)

            stack, was_2d, original_axis = _as_frame_first(rebuilt, frame_axis=frame_axis)
            part_current = _set_part_image_locs(
                part_current,
                _restore_frame_axis(stack, was_2d, original_axis),
                locs,
            )

        if bool(autoupdateref):
            ref_current = np.nanmean(stack, axis=0)

    return part_current, np.asarray(ref_current, dtype=np.float64)


def align_iterate_matlab(
    full_img: np.ndarray,
    ref: np.ndarray,
    part: Any,
    tran_iterations: int,
    translat_method: str,
    maxdrift: float,
    rot_iterations: int,
    rota_method: str,
    maxang: float,
    thresh_min: float,
    autoupdateref: bool | int,
    *,
    construct_particle_stack: Callable[[np.ndarray, Any, int], np.ndarray] | None = None,
) -> tuple[Any, np.ndarray]:
    """MATLAB-layout wrapper for ``align_iterate``.

    Assumes ``Part.Image`` is shaped ``(rows, cols, particles)``.
    """
    return align_iterate(
        full_img,
        ref,
        part,
        tran_iterations=tran_iterations,
        translat_method=translat_method,
        maxdrift=maxdrift,
        rot_iterations=rot_iterations,
        rota_method=rota_method,
        maxang=maxang,
        thresh_min=thresh_min,
        autoupdateref=autoupdateref,
        frame_axis=-1,
        construct_particle_stack=construct_particle_stack,
    )


__all__ = [
    "Particles",
    "align_iterate",
    "align_iterate_matlab",
]
