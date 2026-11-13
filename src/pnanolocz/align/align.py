"""
Image and movie alignment for NanoLocz-compatible AFM workflows.

This module ports NanoLocz ``align_trans.m``, ``align_rot.m`` and
``align_movie.m``.  It estimates per-frame translation, per-frame rotation,
and movie drift between an image/stack and a reference image.

Conventions
-----------
- Stacks are frame-first ``(frames, rows, cols)``; a 2D image is treated as a
  single frame.
- Frames are 0-based, matching the rest of pnanolocz and the detector table.
- ``dx`` is the column/horizontal displacement of each frame relative to the
  reference and ``dy`` the row/vertical displacement, following NanoLocz's
  x/y convention.  To correct an image, use ``shift=(-dy, -dx)`` with
  :func:`scipy.ndimage.shift`.

Outputs
-------
The public functions validate their inputs once, then return a
:class:`pandas.DataFrame` containing only the columns they compute:

    align_trans  -> frame, dx, dy
    align_rot    -> frame, angle
    align_movie  -> frame, dx, dy

Empty input returns an empty table with the same columns.  The computation
itself is done by private, array-based functions (``_align_translation``,
``_align_rotation``, ``_align_movie``) that return the frozen
:class:`Translation` / :class:`Rotation` records, so other code can reuse the
results without a DataFrame.  :func:`im_to_polar` is the one image-valued
operation and returns a NumPy array directly.

Notes
-----
The implementations follow the MATLAB NanoLocz algorithms, including the
``pixel_shift`` local-search window and the two rotation methods
(``'Rotation corr'`` and ``'Polar Corr'``).  The spatial sub-pixel refinement
uses a parabolic fit around the correlation peak instead of the MATLAB zoom
convention, and SciPy interpolation and optimisation are not bitwise identical
to MATLAB, so small numerical differences are expected.

When ``align_movie`` is given a cropped reference (``full_image=False``),
``ref_position`` describes where the crop sits in the full frame; the
correlation peak is found within the crop, so the crop's lower-right corner is
used as the effective frame size to keep ``dx``/``dy`` in full-frame
coordinates.

Authors
-------
- MATLAB ``align_trans.m``, ``align_rot.m``, ``align_movie.m``:
  George R. Heath (NanoLocz).
- Python port: Wang-1971.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd  # type: ignore  # stubs optional; suppress import-untyped
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, map_coordinates, rotate, zoom
from scipy.optimize import minimize_scalar
from skimage.registration import phase_cross_correlation

from pnanolocz.correlation import normxcorr2

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.intp]]
Array = np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class Translation:
    """
    Per-frame translation record.

    Attributes
    ----------
    frame : ndarray of intp
        0-based frame number.
    dx : ndarray of float64
        Horizontal (column) displacement of the frame.
    dy : ndarray of float64
        Vertical (row) displacement of the frame.
    """

    frame: IntArray
    dx: FloatArray
    dy: FloatArray

    def __len__(self) -> int:
        """Return the number of frames."""
        return int(self.frame.size)

    def as_dict(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return the record as a ``{name: array}`` dict."""
        return {"frame": self.frame, "dx": self.dx, "dy": self.dy}


@dataclass(frozen=True, slots=True)
class Rotation:
    """
    Per-frame rotation record.

    Attributes
    ----------
    frame : ndarray of intp
        0-based frame number.
    angle : ndarray of float64
        Estimated rotation angle in degrees.
    """

    frame: IntArray
    angle: FloatArray

    def __len__(self) -> int:
        """Return the number of frames."""
        return int(self.frame.size)

    def as_dict(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return the record as a ``{name: array}`` dict."""
        return {"frame": self.frame, "angle": self.angle}


TRANSLATION_COLUMNS = tuple(field.name for field in fields(Translation))
ROTATION_COLUMNS = tuple(field.name for field in fields(Rotation))


def _as_stack(img: Array, *, name: str = "img") -> FloatArray:
    """
    Return ``img`` as a frame-first ``(frames, rows, cols)`` float64 stack.

    A 2D image is promoted to a single-frame stack.  Non-finite values are
    replaced with zero, matching the MATLAB preprocessing.
    """
    arr = np.nan_to_num(
        np.asarray(img, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")

    return arr


def _as_image(img: Array, *, name: str = "ref") -> FloatArray:
    """Return ``img`` as a 2D float64 image with non-finite values zeroed."""
    arr = np.nan_to_num(
        np.asarray(img, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")

    return arr


def _method_key(method: str, allowed: set[str]) -> str:
    """Return the lower-case method name and validate it."""
    key = str(method).strip().lower()
    if key not in allowed:
        options = " or ".join(repr(item) for item in sorted(allowed))
        raise ValueError(f"method must be {options}, got {method!r}")
    return key


def _validate_angle_range(
    angle_range: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Validate and normalise a two-element angle range."""
    if len(angle_range) != 2:
        raise ValueError("angle_range must contain two values")

    low = float(angle_range[0])
    high = float(angle_range[1])

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("angle_range values must be finite")
    if high <= low:
        raise ValueError("angle_range must satisfy min < max")

    return low, high


def _validate_pixel_shift(pixel_shift: float) -> float:
    """Validate the local-search displacement limit."""
    value = float(pixel_shift)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("pixel_shift must be finite and non-negative")
    return value


def _validate_ref_position(
    ref_position: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Validate an optional ``(x, y, width, height)`` reference crop."""
    if ref_position is None:
        return None
    if len(ref_position) != 4:
        raise ValueError("ref_position must contain four values")
    position = (
        float(ref_position[0]),
        float(ref_position[1]),
        float(ref_position[2]),
        float(ref_position[3]),
    )
    if not all(np.isfinite(value) for value in position):
        raise ValueError("ref_position values must be finite")
    return position


def _peak_abs(corr: Array) -> tuple[int, int]:
    """Return the row/column index of the maximum absolute correlation."""
    idx = int(np.nanargmax(np.abs(corr)))
    row, col = np.unravel_index(idx, corr.shape)
    return int(row), int(col)


def _parabolic_offset(corr: Array, y: int, x: int) -> tuple[float, float]:
    """Refine an integer correlation peak with a 3x3 parabolic fit.

    This is the Python-native sub-pixel estimator: it is more accurate than the
    MATLAB zoom convention on broad correlation peaks, and does not depend on
    clipping a local window at the map border.  A peak on the border cannot be
    refined and returns zero offsets.
    """
    rows, cols = corr.shape
    if not (0 < x < cols - 1 and 0 < y < rows - 1):
        return 0.0, 0.0

    left = float(corr[y, x - 1])
    center = float(corr[y, x])
    right = float(corr[y, x + 1])
    up = float(corr[y - 1, x])
    down = float(corr[y + 1, x])

    denom_x = left - 2.0 * center + right
    denom_y = up - 2.0 * center + down

    x_offset = (
        0.5 * (left - right) / denom_x if abs(denom_x) > np.finfo(float).eps else 0.0
    )
    y_offset = (
        0.5 * (up - down) / denom_y if abs(denom_y) > np.finfo(float).eps else 0.0
    )

    return (
        float(np.clip(x_offset, -0.5, 0.5)),
        float(np.clip(y_offset, -0.5, 0.5)),
    )


def _zoom_peak_offset(
    corr: Array,
    y: int,
    x: int,
    *,
    half_window: int = 3,
    zoom_factor: int = 100,
) -> tuple[float, float]:
    """Refine a peak by bicubic zoom of a local correlation window."""
    y0 = max(0, int(y) - half_window)
    y1 = min(corr.shape[0], int(y) + half_window + 1)
    x0 = max(0, int(x) - half_window)
    x1 = min(corr.shape[1], int(x) + half_window + 1)

    if (y1 - y0) < 3 or (x1 - x0) < 3:
        return 0.0, 0.0

    clip = corr[y0:y1, x0:x1]
    zoomed = zoom(clip, zoom=zoom_factor, order=3)

    yp, xp = _peak_abs(zoomed)

    scale_x = (clip.shape[1] - 1) / (zoomed.shape[1] - 1)
    scale_y = (clip.shape[0] - 1) / (zoomed.shape[0] - 1)
    x_offset = float(xp) * scale_x - float(x - x0)
    y_offset = float(yp) * scale_y - float(y - y0)
    return x_offset, y_offset


def _match_image_sizes(
    ref: FloatArray,
    target: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Trim images from the top/left so both have identical shape."""
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)

    if r.shape[0] > t.shape[0]:
        r = r[r.shape[0] - t.shape[0] :, :]
    elif r.shape[0] < t.shape[0]:
        t = t[t.shape[0] - r.shape[0] :, :]

    if r.shape[1] > t.shape[1]:
        r = r[:, r.shape[1] - t.shape[1] :]
    elif r.shape[1] < t.shape[1]:
        t = t[:, t.shape[1] - r.shape[1] :]

    return np.asarray(r, dtype=np.float64), np.asarray(t, dtype=np.float64)


def _corr2(a: Array, b: Array) -> float:
    """Return the Pearson correlation coefficient, like MATLAB ``corr2``."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)

    av = aa - float(np.mean(aa))
    bv = bb - float(np.mean(bb))
    denom = float(np.sqrt(np.sum(av**2) * np.sum(bv**2)))

    if denom <= np.finfo(float).eps:
        return 0.0

    return float(np.sum(av * bv) / denom)


def _cross_corr_shifts(
    stack: FloatArray,
    ref: FloatArray,
    *,
    pixel_shift: float,
    sub_pix: bool,
    filt_cr: float = 0.0,
    window_offset: int = 0,
    effective_shape: tuple[float, float] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Return per-frame ``(x, y)`` displacements from normalized cross-correlation.

    Parameters
    ----------
    stack : ndarray
        Frame-first ``(frames, rows, cols)`` stack of cleaned images.
    ref : ndarray
        Cleaned 2D reference image.
    pixel_shift : float
        Maximum local-search drift.  When 0 the peak is searched in the whole
        correlation map.
    sub_pix : bool
        Refine the integer peak with :func:`_parabolic_offset`.
    filt_cr : float
        Gaussian sigma applied to each correlation map.  ``0`` disables it.
    window_offset : int
        Extra pixels added to the local-search half-window.  ``align_movie``
        uses 1 to match the MATLAB ``align_movie.m`` window.
    effective_shape : tuple of float, optional
        ``(rows, cols)`` frame size the peak positions are measured against.
        Defaults to the shape of ``stack``.  ``align_movie`` passes the size
        implied by ``ref_position`` when the reference is a crop.
    """
    n_frames = stack.shape[0]
    rows = float(stack.shape[1])
    cols = float(stack.shape[2])

    if effective_shape is not None:
        rows = float(effective_shape[0])
        cols = float(effective_shape[1])

    window = int(round(float(pixel_shift) + window_offset))

    x_peak = np.zeros(n_frames, dtype=np.float64)
    y_peak = np.zeros(n_frames, dtype=np.float64)
    x_offset = np.zeros(n_frames, dtype=np.float64)
    y_offset = np.zeros(n_frames, dtype=np.float64)
    xw = np.zeros(n_frames, dtype=np.float64)
    yw = np.zeros(n_frames, dtype=np.float64)

    for idx in range(n_frames):
        corr = normxcorr2(ref, stack[idx])

        if filt_cr > 0:
            corr = gaussian_filter(corr, sigma=float(filt_cr), mode="nearest")

        corr_rows, corr_cols = corr.shape
        y0, x0 = _peak_abs(corr)

        if pixel_shift > 0 and idx > 0:
            prev_y = int(round(y_peak[idx - 1]))
            prev_x = int(round(x_peak[idx - 1]))

            y_start = prev_y - window + 1
            y_end = prev_y + window
            x_start = prev_x - window + 1
            x_end = prev_x + window

            if 0 <= y_start < y_end <= corr_rows and 0 <= x_start < x_end <= corr_cols:
                local = corr[y_start:y_end, x_start:x_end]
                ly, lx = _peak_abs(local)

                xw[idx] = (lx + 1) - window
                yw[idx] = (ly + 1) - window

                x_peak[idx] = x_peak[0] + np.sum(xw[: idx + 1])
                y_peak[idx] = y_peak[0] + np.sum(yw[: idx + 1])
            else:
                x_peak[idx] = float(x0)
                y_peak[idx] = float(y0)
        else:
            x_peak[idx] = float(x0)
            y_peak[idx] = float(y0)

        if sub_pix:
            xo, yo = _parabolic_offset(
                corr,
                int(round(y_peak[idx])),
                int(round(x_peak[idx])),
            )
            x_offset[idx] = xo
            y_offset[idx] = yo

    x = (x_peak + 1.0) + x_offset - cols
    y = (y_peak + 1.0) + y_offset - rows
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


def _fft_shifts(
    stack: FloatArray,
    ref: FloatArray,
    *,
    upsample_factor: int = 100,
) -> tuple[FloatArray, FloatArray]:
    """Return per-frame ``(x, y)`` displacements from FFT phase correlation."""
    n_frames = stack.shape[0]
    x = np.zeros(n_frames, dtype=np.float64)
    y = np.zeros(n_frames, dtype=np.float64)

    for idx in range(n_frames):
        shift, _, _ = phase_cross_correlation(  # type: ignore[no-untyped-call]
            ref,
            stack[idx],
            upsample_factor=int(upsample_factor),
            normalization=None,
        )
        y[idx] = -float(shift[0])
        x[idx] = -float(shift[1])

    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _align_translation(
    stack: FloatArray,
    ref: FloatArray,
    *,
    pixel_shift: float,
    sub_pix: bool,
    method: str,
) -> Translation:
    """
    Estimate per-frame translation on cleaned arrays.

    Parameters
    ----------
    stack : ndarray
        Frame-first ``(frames, rows, cols)`` stack of finite float images.
    ref : ndarray
        Finite 2D reference image.
    pixel_shift : float
        Validated local-search drift limit.
    sub_pix : bool
        Enable sub-pixel refinement for the spatial correlation branch.
    method : {'cross corr', 'fft cross'}
        Lower-case, validated correlation method.

    Returns
    -------
    Translation
        One entry per frame, with displacements relative to ``ref``.
    """
    if method == "fft cross":
        x, y = _fft_shifts(stack, ref)
    else:
        x, y = _cross_corr_shifts(
            stack,
            ref,
            pixel_shift=pixel_shift,
            sub_pix=sub_pix,
        )

    return Translation(
        frame=np.arange(x.size, dtype=np.intp),
        dx=np.asarray(x, dtype=np.float64),
        dy=np.asarray(y, dtype=np.float64),
    )


def align_trans(
    img: Array,
    ref: Array,
    *,
    pixel_shift: float = 0.0,
    sub_pix: bool = True,
    method: str = "Cross corr",
) -> pd.DataFrame:
    """
    Estimate per-frame translation displacements relative to a reference.

    Parameters
    ----------
    img : ndarray
        2D image or frame-first 3D stack ``(frames, rows, cols)``.
    ref : ndarray
        2D reference image.
    pixel_shift : float, default 0.0
        Maximum local-search drift.  If > 0, frames after the first are
        searched around the previous correlation peak.
    sub_pix : bool, default True
        Enable sub-pixel refinement for the spatial cross-correlation branch.
    method : {'Cross corr', 'FFT cross'}, default 'Cross corr'
        Correlation method.

    Returns
    -------
    pandas.DataFrame
        One row per frame with columns ``frame, dx, dy`` (0-based frames).
        ``dx``/``dy`` are displacement of the frame relative to ``ref``.

    Notes
    -----
    The returned values are displacements, not corrections: to move a frame
    back onto the reference, pass ``shift=(-dy, -dx)`` to
    :func:`scipy.ndimage.shift`.
    """
    stack = _as_stack(img)
    reference = _as_image(ref)
    result = _align_translation(
        stack,
        reference,
        pixel_shift=_validate_pixel_shift(pixel_shift),
        sub_pix=bool(sub_pix),
        method=_method_key(method, {"cross corr", "fft cross"}),
    )
    return pd.DataFrame(result.as_dict())


def im_to_polar(
    image: Array,
    r_min: float,
    r_max: float,
    radial_samples: int,
    angular_samples: int,
) -> FloatArray:
    """
    Convert a rectangular image to a polar image.

    Parameters
    ----------
    image : ndarray
        2D image.  Non-finite values are replaced with zero.
    r_min : float
        Inner radius, in units of half the image size.
    r_max : float
        Outer radius, in units of half the image size.
    radial_samples : int
        Number of radial samples, i.e. rows in the returned image.
    angular_samples : int
        Number of angular samples, i.e. columns in the returned image.

    Returns
    -------
    ndarray
        Polar image with shape ``(radial_samples, angular_samples)``, where
        rows follow the radius from ``r_min`` to ``r_max`` and columns follow
        the angle from 0 to 2 pi.

    Raises
    ------
    ValueError
        If ``image`` is not 2D, the sample counts are not positive, or either
        radius is not finite.

    Notes
    -----
    Adapted from the polar unwrapping used by NanoLocz ``align_rot.m``
    (George R. Heath); the Python port is by Wang-1971.
    """
    img = _as_image(image, name="image")

    if int(radial_samples) < 1 or int(angular_samples) < 1:
        raise ValueError("radial_samples and angular_samples must be positive")
    if not np.isfinite(float(r_min)) or not np.isfinite(float(r_max)):
        raise ValueError("r_min and r_max must be finite")

    rows, cols = img.shape
    om = (rows + 1.0) / 2.0
    on = (cols + 1.0) / 2.0
    sx = (rows - 1.0) / 2.0
    sy = (cols - 1.0) / 2.0

    r = np.linspace(
        float(r_min),
        float(r_max),
        int(radial_samples),
        dtype=np.float64,
    )
    theta = np.arange(int(angular_samples), dtype=np.float64) * (
        2.0 * np.pi / float(angular_samples)
    )

    rr, tt = np.meshgrid(r, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)

    row_coords = x * sx + (om - 1.0)
    col_coords = y * sy + (on - 1.0)

    polar = map_coordinates(
        img,
        [row_coords, col_coords],
        order=1,
        mode="nearest",
    )
    return np.asarray(polar, dtype=np.float64)


def _rotation_corr(
    ref: FloatArray,
    target: FloatArray,
    angle_range: tuple[float, float],
) -> float:
    """Direct rotational-correlation branch of ``align_rot``."""
    min_angle, max_angle = float(angle_range[0]), float(angle_range[1])
    step = 0.2 if (max_angle - min_angle) <= 20 else 0.5
    angles = np.arange(min_angle, max_angle + (0.5 * step), step, dtype=np.float64)

    ref_m, target_m = _match_image_sizes(ref, target)

    corrs = np.zeros(angles.size, dtype=np.float64)
    for idx, angle in enumerate(angles):
        rotated = rotate(
            target_m,
            angle=float(angle),
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        corrs[idx] = _corr2(rotated, ref_m)

    best_idx = int(np.argmax(corrs))
    best_angle = float(angles[best_idx])

    # Refine the discrete maximum on a cubic interpolation of the scan.
    interp_kind = "cubic" if angles.size >= 4 else "linear"
    interpolant = interp1d(
        angles,
        corrs,
        kind=interp_kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )

    lo = max(min_angle, best_angle - step * 2.0)
    hi = min(max_angle, best_angle + step * 2.0)

    if hi > lo:
        result = minimize_scalar(
            lambda a: -float(interpolant(float(a))),
            bounds=(lo, hi),
            method="bounded",
        )
        if result.success and np.isfinite(result.x):
            best_angle = float(result.x)

    return float(np.clip(best_angle, -360.0, 360.0))


def _polar_corr(
    ref: FloatArray,
    target: FloatArray,
    angle_range: tuple[float, float],
) -> float:
    """Polar-coordinate correlation branch of ``align_rot``."""
    min_angle, max_angle = float(angle_range[0]), float(angle_range[1])

    angle_samples = 360
    r_min = 0.0
    r_max = 1.0
    radial_samples = max(int(round(min(ref.shape) / 3.0)), 8)
    zoom_factor = 100

    ref_polar = im_to_polar(ref, r_min, r_max, radial_samples, angle_samples)
    target_polar = im_to_polar(target, r_min, r_max, radial_samples, angle_samples)

    half = angle_samples // 2
    ref_tiled = np.concatenate(
        [ref_polar[:, half - 1 :], ref_polar, ref_polar[:, :half]],
        axis=1,
    )

    corr = normxcorr2(target_polar, ref_tiled)
    corr = corr[:, angle_samples : 2 * angle_samples]

    margin = 5
    search_min = int(round(min_angle - margin + 180))
    search_max = int(round(max_angle + margin + 180))
    search_min = max(0, search_min)
    search_max = min(corr.shape[1] - 1, search_max)

    if search_max <= search_min:
        return 0.0

    corr_search = corr[:, search_min : search_max + 1].copy()

    if corr_search.shape[0] > 5:
        corr_search[4, :] = 0
    if corr_search.shape[1] > 0:
        corr_search[:, 0] = 0
    if corr_search.shape[0] > 2:
        corr_search[-2:, :] = 0
    if corr_search.shape[1] > 2:
        corr_search[:, -2:] = 0

    y_peak, x_peak = _peak_abs(corr_search)
    rot_angle = float((search_min + x_peak) - 180)

    xo, _ = _zoom_peak_offset(
        corr_search,
        y_peak,
        x_peak,
        half_window=3,
        zoom_factor=zoom_factor,
    )
    rot_angle += xo

    return float(np.clip(rot_angle, -360.0, 360.0))


def _align_rotation(
    ref: FloatArray,
    target: FloatArray,
    *,
    angle_range: tuple[float, float],
    method: str,
) -> Rotation:
    """
    Estimate per-frame rotation on cleaned arrays.

    Parameters
    ----------
    ref : ndarray
        Finite 2D reference image.
    target : ndarray
        Frame-first ``(frames, rows, cols)`` stack of finite float images.
    angle_range : tuple of float
        Validated ``(min_angle, max_angle)`` search range in degrees.
    method : {'rotation corr', 'polar corr'}
        Lower-case, validated rotation method.

    Returns
    -------
    Rotation
        One entry per target frame.
    """
    angles = np.array(
        [
            _rotation_corr(ref, target[idx], angle_range)
            if method == "rotation corr"
            else _polar_corr(ref, target[idx], angle_range)
            for idx in range(target.shape[0])
        ],
        dtype=np.float64,
    )
    return Rotation(
        frame=np.arange(angles.size, dtype=np.intp),
        angle=np.asarray(angles, dtype=np.float64),
    )


def align_rot(
    ref: Array,
    target: Array,
    *,
    angle_range: tuple[float, float] | list[float] = (-180.0, 180.0),
    method: str = "Rotation corr",
) -> pd.DataFrame:
    """
    Estimate rotation angles between a reference and one or more targets.

    Parameters
    ----------
    ref : ndarray
        2D reference image.
    target : ndarray
        2D target image or frame-first 3D stack ``(frames, rows, cols)``.
    angle_range : pair of float, default (-180.0, 180.0)
        Search range ``(min_angle, max_angle)`` in degrees.
    method : {'Rotation corr', 'Polar Corr'}, default 'Rotation corr'
        Rotation estimation method.

    Returns
    -------
    pandas.DataFrame
        One row per target frame with columns ``frame, angle`` (0-based
        frames and degrees).  ``angle`` is the rotation to apply to
        ``target`` to align it with ``ref``; it has the opposite sign to the
        rotation already applied to the target.
    """
    reference = _as_image(ref)
    stack = _as_stack(target, name="target")
    result = _align_rotation(
        reference,
        stack,
        angle_range=_validate_angle_range(angle_range),
        method=_method_key(method, {"rotation corr", "polar corr"}),
    )
    return pd.DataFrame(result.as_dict())


def _align_movie(
    stack: FloatArray,
    ref: FloatArray,
    *,
    pixel_shift: float,
    sub_pix: bool,
    filt_cr: float,
    method: str,
    effective_shape: tuple[float, float],
) -> Translation:
    """
    Estimate movie drift on cleaned arrays.

    Parameters
    ----------
    stack : ndarray
        Frame-first ``(frames, rows, cols)`` stack of finite float images.
    ref : ndarray
        Finite 2D reference image.
    pixel_shift : float
        Validated local-search drift limit.
    sub_pix : bool
        Enable sub-pixel refinement for the spatial correlation branch.
    filt_cr : float
        Gaussian sigma applied to the correlation maps.
    method : {'cross corr', 'fft cross'}
        Lower-case, validated correlation method.
    effective_shape : tuple of float
        ``(rows, cols)`` frame size used to express the displacements.

    Returns
    -------
    Translation
        One entry per frame, with displacements relative to ``ref``.
    """
    if method == "fft cross":
        x, y = _fft_shifts(stack, ref)
    else:
        x, y = _cross_corr_shifts(
            stack,
            ref,
            pixel_shift=pixel_shift,
            sub_pix=sub_pix,
            filt_cr=filt_cr,
            window_offset=1,
            effective_shape=effective_shape,
        )

    return Translation(
        frame=np.arange(x.size, dtype=np.intp),
        dx=np.asarray(x, dtype=np.float64),
        dy=np.asarray(y, dtype=np.float64),
    )


def align_movie(
    image_target: Array,
    ref: Array | None = None,
    *,
    ref_position: tuple[float, float, float, float] | None = None,
    pixel_shift: float = 0.0,
    full_image: bool = True,
    sub_pix: bool = True,
    filt_cr: float = 0.0,
    method: str = "Cross corr",
) -> pd.DataFrame:
    """
    Estimate translational drift for a movie or image stack.

    Parameters
    ----------
    image_target : ndarray
        2D image or frame-first 3D stack ``(frames, rows, cols)``.
    ref : ndarray, optional
        2D reference image.  Required when ``full_image=True``.  When
        ``full_image=False`` and ``ref`` is omitted, the first frame is used.
    ref_position : tuple of four floats, optional
        ``(x, y, width, height)`` of the reference crop in the full frame.
        Only used when ``full_image=False``.  The correlation peak is found
        within the crop, so the effective frame size is taken from the crop's
        lower-right corner to keep ``dx``/``dy`` in full-frame coordinates;
        this is the coordinate offset introduced by NanoLocz's ``ref.position``.
    pixel_shift : float, default 0.0
        Maximum allowed local displacement.  Must be finite and non-negative.
        Not used for ``'FFT cross'``.
    full_image : bool, default True
        If true, ``ref`` is used directly as the reference image.  If false,
        the reference is a crop (``ref`` or the first frame) and
        ``ref_position`` may describe its position in the full frame.
    sub_pix : bool, default True
        Enable sub-pixel refinement for ``'Cross corr'``.
    filt_cr : float, default 0.0
        Gaussian sigma for smoothing the correlation map (spatial only).
    method : {'Cross corr', 'FFT cross'}, default 'Cross corr'
        Correlation method.

    Returns
    -------
    pandas.DataFrame
        One row per frame with columns ``frame, dx, dy`` (0-based frames).
        ``dx``/``dy`` are displacement of the frame relative to the reference.
    """
    stack = _as_stack(image_target, name="image_target")
    shift_limit = _validate_pixel_shift(pixel_shift)

    rows = float(stack.shape[1])
    cols = float(stack.shape[2])

    if full_image:
        if ref is None:
            raise ValueError("ref is required when full_image=True")
        ref_img = _as_image(ref)
    else:
        if ref is None:
            if stack.shape[0] == 0:
                raise ValueError("image_target has no frames to use as a reference")
            ref_img = _as_image(stack[0], name="ref")
        else:
            ref_img = _as_image(ref)
        position = _validate_ref_position(ref_position)
        if position is not None:
            cols = float(round(position[0] + position[2]))
            rows = float(round(position[1] + position[3]))

    result = _align_movie(
        stack,
        ref_img,
        pixel_shift=shift_limit,
        sub_pix=bool(sub_pix),
        filt_cr=float(filt_cr),
        method=_method_key(method, {"cross corr", "fft cross"}),
        effective_shape=(rows, cols),
    )
    return pd.DataFrame(result.as_dict())


align_trans.__version__ = "0.1.0"  # type: ignore[attr-defined]
align_rot.__version__ = "0.1.0"  # type: ignore[attr-defined]
align_movie.__version__ = "0.1.0"  # type: ignore[attr-defined]
im_to_polar.__version__ = "0.1.0"  # type: ignore[attr-defined]

__all__ = [
    "ROTATION_COLUMNS",
    "TRANSLATION_COLUMNS",
    "Rotation",
    "Translation",
    "align_movie",
    "align_rot",
    "align_trans",
    "im_to_polar",
]
