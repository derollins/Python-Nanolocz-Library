"""
Movie/stack translational alignment for NanoLocz-compatible AFM workflows.

This module ports NanoLocz ``align_movie.m``.  It is very similar to
``align_trans`` but supports NanoLocz reference objects that may contain
``image`` and ``position`` fields and optionally smooths the correlation map.

Python default stack convention is frame-first ``(frames, rows, cols)``.
Use ``frame_axis=-1`` for MATLAB-style ``(rows, cols, frames)`` input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from pnanolocz.align_trans import normxcorr2, _subpixel_offset_from_zoom
except Exception:  # pragma: no cover - allows standalone use
    from align_trans import normxcorr2, _subpixel_offset_from_zoom  # type: ignore

try:
    from skimage.registration import phase_cross_correlation
except Exception:  # pragma: no cover - optional dependency
    phase_cross_correlation = None  # type: ignore[assignment]


FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _as_frame_first(img: np.ndarray, frame_axis: int = 0) -> tuple[FloatArray, bool, int]:
    """Convert a 2-D image or 3-D stack to frame-first layout."""
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0

    if arr.ndim != 3:
        raise ValueError("ImageTarget must be 2D or 3D")

    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _get_ref_field(ref: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict-like or object-like NanoLocz reference."""
    if isinstance(ref, dict):
        return ref.get(name, default)
    return getattr(ref, name, default)


def _resolve_reference(
    image_target: FloatArray,
    ref: Any,
    full_image: bool | int,
) -> tuple[FloatArray, tuple[float, float, float, float] | None]:
    """Resolve MATLAB ``ref.image`` / ``ref.position`` behavior."""
    if bool(full_image):
        return np.asarray(ref, dtype=np.float64), None

    ref_image = _get_ref_field(ref, "image", None)
    ref_position = _get_ref_field(ref, "position", None)

    if ref_image is None:
        ref_image = image_target[0]

    if ref_position is not None:
        ref_position = tuple(float(v) for v in ref_position)

    return np.asarray(ref_image, dtype=np.float64), ref_position


def _peak_abs(corr: np.ndarray) -> tuple[int, int]:
    """Return row/column index of the maximum absolute correlation."""
    idx = int(np.nanargmax(np.abs(corr)))
    return tuple(int(v) for v in np.unravel_index(idx, corr.shape))  # type: ignore[return-value]


def align_movie(
    image_target: np.ndarray,
    ref: Any,
    pixel_shift: float,
    full_image: bool | int,
    sub_pix: bool | int,
    filt_cr: float,
    *,
    method: str = "Cross corr",
    frame_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate translational drift for a movie/image stack.

    Parameters
    ----------
    image_target:
        2-D image or 3-D stack.
    ref:
        Either a 2-D reference image, or an object/dict with ``image`` and
        optionally ``position`` fields.
    pixel_shift:
        Maximum allowed local shift.  MATLAB uses ``window = round(pixel_shift+1)``.
        Not used for ``'FFT cross'`` method.
    full_image:
        If true, ``ref`` itself is used as the reference image.  If false,
        ``ref.image`` is attempted; if unavailable, the first frame is used.
    sub_pix:
        Enable sub-pixel refinement (spatial cross-correlation only).
        ``'FFT cross'`` always returns sub-pixel via built-in upsampling.
    filt_cr:
        Gaussian smoothing sigma for the correlation map (spatial only).
    method:
        ``'Cross corr'`` (default) or ``'FFT cross'`` (phase correlation).
    frame_axis:
        Frame axis for 3-D input.  Use ``-1`` for MATLAB layout.

    Returns
    -------
    x, y:
        Estimated shifts using NanoLocz's horizontal/vertical convention.
    """
    stack, _, _ = _as_frame_first(image_target, frame_axis=frame_axis)
    stack = np.asarray(stack, dtype=np.float64)
    stack[~np.isfinite(stack)] = 0.0

    ref_img, ref_position = _resolve_reference(stack, ref, bool(full_image))
    ref_img = np.asarray(ref_img, dtype=np.float64)
    ref_img[~np.isfinite(ref_img)] = 0.0

    n_frames, rows, cols = stack.shape
    method_lc = method.strip().lower()

    x_peak = np.zeros(n_frames, dtype=np.float64)
    y_peak = np.zeros(n_frames, dtype=np.float64)
    x_offset = np.zeros(n_frames, dtype=np.float64)
    y_offset = np.zeros(n_frames, dtype=np.float64)

    if method_lc == "fft cross":
        if phase_cross_correlation is None:
            raise ImportError(
                "scikit-image is required for method='FFT cross'. "
                "Install with: pip install scikit-image"
            )

        for idx in range(n_frames):
            shift, _error, _phasediff = phase_cross_correlation(
                ref_img,
                stack[idx],
                upsample_factor=100,
                normalization=None,
            )
            # shift = (row_shift, col_shift) needed to align moving to reference.
            # Negate to match NanoLocz convention: correction = -measured_shift.
            x_peak[idx] = -float(shift[1])
            y_peak[idx] = -float(shift[0])

        # FFT already sub-pixel; offset stays at 0.
        x = x_peak.copy()
        y = y_peak.copy()

    else:
        window = int(round(float(pixel_shift) + 1.0))
        xw = np.zeros(n_frames, dtype=np.float64)
        yw = np.zeros(n_frames, dtype=np.float64)

        for idx in range(n_frames):
            corr = normxcorr2(ref_img, stack[idx])

            if float(filt_cr) > 0:
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

            if bool(sub_pix):
                xo, yo = _subpixel_offset_from_zoom(
                    corr,
                    int(round(y_peak[idx])),
                    int(round(x_peak[idx])),
                    half_window=2,
                    zoom_factor=100,
                    order=3,
                )
                x_offset[idx] = xo
                y_offset[idx] = yo

        # MATLAB optionally adjusts sz from ref.position when operating on a crop.
        effective_cols = float(cols)
        effective_rows = float(rows)

        if not bool(full_image) and ref_position is not None and len(ref_position) >= 4:
            effective_cols = float(round(ref_position[0] + ref_position[2]))
            effective_rows = float(round(ref_position[1] + ref_position[3]))

        x = (x_peak + 1.0) + x_offset - effective_cols
        y = (y_peak + 1.0) + y_offset - effective_rows

    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def align_movie_matlab(
    image_target: np.ndarray,
    ref: Any,
    pixel_shift: float,
    full_image: bool | int,
    sub_pix: bool | int,
    filt_cr: float,
    *,
    method: str = "Cross corr",
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-layout wrapper for ``align_movie``."""
    return align_movie(
        image_target,
        ref,
        pixel_shift=pixel_shift,
        full_image=full_image,
        sub_pix=sub_pix,
        filt_cr=filt_cr,
        method=method,
        frame_axis=-1,
    )


__all__ = [
    "align_movie",
    "align_movie_matlab",
]
