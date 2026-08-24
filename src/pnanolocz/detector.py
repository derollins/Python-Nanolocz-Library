"""
Particle detection for NanoLocz-compatible AFM movies.

This module ports NanoLocz ``Detector.m``.  It supports two detection modes:

- ``'Peak picker'``: direct peak detection on each image frame.
- ``'ccr'``: normalized cross-correlation against a reference image, with
  optional rotational freedom over a list of candidate angles.

Output convention
-----------------
The returned localisation table has at least 8 columns, matching MATLAB:

    col 1 -> x coordinate      -> Python index 0
    col 2 -> y coordinate      -> Python index 1
    col 3 -> z height          -> Python index 2
    col 4 -> correlation value -> Python index 3
    col 5 -> frame number      -> Python index 4
    col 8 -> best angle        -> Python index 7

Frame numbers are MATLAB-style 1-based by default.  Coordinates are also kept
MATLAB-style to stay compatible with the rest of the NanoLocz port.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, rotate as scipy_rotate, zoom

try:
    from pnanolocz.fast_peaks2d import fast_peaks2d
except Exception:  # pragma: no cover - standalone fallback
    from fast_peaks2d import fast_peaks2d  # type: ignore

try:
    from pnanolocz.align_trans import normxcorr2
except Exception:  # pragma: no cover - standalone fallback
    from align_trans import normxcorr2  # type: ignore


FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _as_frame_first(img: np.ndarray, frame_axis: int) -> tuple[FloatArray, bool, int]:
    """Convert a 2-D image or 3-D stack to frame-first layout."""
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0

    if arr.ndim != 3:
        raise ValueError("img must be 2D or 3D")

    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _matlab_round(x: float | np.ndarray) -> np.ndarray:
    """MATLAB-like rounding: halves away from zero."""
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def _ensure_8_cols(locs: np.ndarray) -> FloatArray:
    """Pad localisation rows to at least 8 columns."""
    arr = np.asarray(locs, dtype=np.float64)

    if arr.size == 0:
        return np.empty((0, 8), dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] >= 8:
        return arr.copy()

    out = np.zeros((arr.shape[0], 8), dtype=np.float64)
    out[:, : arr.shape[1]] = arr
    return out


def _crop_corr_edges(ccr: np.ndarray, ref_shape: tuple[int, int], *, rotated: bool) -> np.ndarray:
    """Crop correlation map when ``ex_edge == 1``.

    MATLAB uses slightly different endpoints for the rotational and
    non-rotational CCR branches.  This helper approximates the same exclusion of
    border responses where the template is not fully supported by the image.
    """
    r_rows, r_cols = ref_shape

    if ccr.ndim == 2:
        ccr_work = ccr[:, :, np.newaxis]
    else:
        ccr_work = ccr

    if rotated:
        y0 = int(round(r_rows))
        y1 = ccr_work.shape[0] - int(round(r_rows)) - 1
        x0 = int(round(r_cols))
        x1 = ccr_work.shape[1] - int(round(r_cols)) - 1
    else:
        y0 = int(round(r_rows)) - 1
        y1 = ccr_work.shape[0] - int(round(r_rows))
        x0 = int(round(r_cols)) - 1
        x1 = ccr_work.shape[1] - int(round(r_cols))

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = max(y0 + 1, min(ccr_work.shape[0], y1))
    x1 = max(x0 + 1, min(ccr_work.shape[1], x1))

    out = ccr_work[y0:y1, x0:x1, :]
    return out[:, :, 0] if ccr.ndim == 2 else out


def detector(
    img: np.ndarray,
    method: str,
    ref: np.ndarray | float | int,
    filt_img: float,
    filt_ccr: float,
    min_thresh: float,
    ex_edge: bool | int,
    fastdetect: bool | int,
    angles: Iterable[float] | None = None,
    *,
    frame_axis: int = 0,
    matlab_frame_numbers: bool = True,
) -> FloatArray:
    """Detect particles in an AFM image stack.

    Parameters
    ----------
    img:
        2-D image or 3-D movie stack.  Python default stack layout is
        ``(frames, rows, cols)``.
    method:
        ``'Peak picker'`` or ``'ccr'``.
    ref:
        For ``'ccr'``, a reference image.  For ``'Peak picker'``, MATLAB passes
        this value through to ``Fast_peaks2D`` as the kernel size.
    filt_img:
        Gaussian sigma applied to input images before detection.
    filt_ccr:
        Gaussian sigma applied to correlation maps.
    min_thresh:
        Minimum peak or correlation threshold.
    ex_edge:
        If true, exclude correlation edge regions.
    fastdetect:
        If true, resize image and reference by 1/2 for faster detection.
    angles:
        Optional rotation angle list for CCR template rotation.
    frame_axis:
        Frame axis for 3-D input.  Use ``-1`` for MATLAB layout.
    matlab_frame_numbers:
        If true, output frame column is 1-based.  If false, it is 0-based.

    Returns
    -------
    ndarray
        ``N x 8`` localisation table, or ``[[nan, ..., nan]]`` if no particles
        are detected.
    """
    stack, _, _ = _as_frame_first(img, frame_axis=frame_axis)
    stack = np.asarray(stack, dtype=np.float64)
    stack[~np.isfinite(stack)] = 0.0

    method_lc = method.strip().lower()

    if angles is None:
        angle_array = np.array([0.0], dtype=np.float64)
    else:
        angle_array = np.asarray(list(angles), dtype=np.float64)
        if angle_array.size == 0:
            angle_array = np.array([0.0], dtype=np.float64)

    rotfreedom = not (angle_array.size == 1 and float(angle_array[0]) == 0.0)

    ref_img: np.ndarray | None
    if method_lc == "ccr":
        ref_img = np.asarray(ref, dtype=np.float64)
        ref_img[~np.isfinite(ref_img)] = 0.0
    else:
        ref_img = None

    if bool(fastdetect):
        stack = zoom(stack, zoom=(1.0, 0.5, 0.5), order=1)
        if ref_img is not None:
            ref_img = zoom(ref_img, zoom=0.5, order=1)
        ns = 3
    else:
        if ref_img is None:
            try:
                ns = int(round(float(ref)))
            except Exception:
                ns = 3
        else:
            ns = int(round(min(ref_img.shape) / 2.0))
            if ns >= 50:
                ns = 50

    if float(filt_img) > 0:
        img_g = gaussian_filter(stack, sigma=(0.0, float(filt_img), float(filt_img)), mode="nearest")
    else:
        img_g = stack

    all_locs: list[np.ndarray] = []

    for frame_idx in range(stack.shape[0]):
        locs_i = np.empty((0, 8), dtype=np.float64)
        frame_img = img_g[frame_idx]

        if method_lc == "peak picker":
            locs_raw = fast_peaks2d(frame_img, min_thresh, ns, matlab_indexing=True)
            locs_i = _ensure_8_cols(locs_raw)
            if locs_i.shape[0] > 0:
                locs_i[:, 7] = 0.0

        elif method_lc == "ccr":
            if ref_img is None:
                raise ValueError("ref image is required for method='ccr'")

            if rotfreedom:
                corr_stack = []
                for angle in angle_array:
                    rotated_ref = scipy_rotate(
                        ref_img,
                        angle=float(angle),
                        reshape=False,
                        order=3,
                        mode="constant",
                        cval=0.0,
                        prefilter=True,
                    )
                    corr_stack.append(normxcorr2(rotated_ref, frame_img))

                ccr = np.stack(corr_stack, axis=2)

                if float(filt_ccr) > 0:
                    ccr = gaussian_filter(ccr, sigma=(float(filt_ccr), float(filt_ccr), 0.0), mode="nearest")

                if bool(ex_edge):
                    ccr = _crop_corr_edges(ccr, ref_img.shape, rotated=True)

                val = np.nanmax(ccr, axis=2)
                ids = np.nanargmax(ccr, axis=2)

                locs_raw = fast_peaks2d(val, min_thresh, ns, matlab_indexing=True)
                locs_i = _ensure_8_cols(locs_raw)

                if locs_i.shape[0] > 0:
                    for row_idx in range(locs_i.shape[0]):
                        yy = int(_matlab_round(locs_i[row_idx, 1]).item()) - 1
                        xx = int(_matlab_round(locs_i[row_idx, 0]).item()) - 1
                        yy = int(np.clip(yy, 0, ids.shape[0] - 1))
                        xx = int(np.clip(xx, 0, ids.shape[1] - 1))
                        best_angle_idx = int(ids[yy, xx])
                        locs_i[row_idx, 7] = -float(angle_array[best_angle_idx])

            else:
                ccr = normxcorr2(ref_img, frame_img)

                if float(filt_ccr) > 0:
                    ccr = gaussian_filter(ccr, sigma=float(filt_ccr), mode="nearest")

                if bool(ex_edge):
                    ccr = _crop_corr_edges(ccr, ref_img.shape, rotated=False)

                locs_raw = fast_peaks2d(ccr, min_thresh, ns, matlab_indexing=True)
                locs_i = _ensure_8_cols(locs_raw)
                if locs_i.shape[0] > 0:
                    locs_i[:, 7] = 0.0
        else:
            raise ValueError("method must be 'Peak picker' or 'ccr'")

        if locs_i.shape[0] == 0:
            continue

        if method_lc == "ccr" and ref_img is not None:
            if bool(ex_edge):
                locs_i[:, 1] = locs_i[:, 1] + ref_img.shape[0] / 2.0
                locs_i[:, 0] = locs_i[:, 0] + ref_img.shape[1] / 2.0
            else:
                locs_i[:, 1] = locs_i[:, 1] - ref_img.shape[0] / 2.0 + 0.5
                locs_i[:, 0] = locs_i[:, 0] - ref_img.shape[1] / 2.0 + 0.5
            locs_i[:, 3] = locs_i[:, 2]

        # Height lookup from original unfiltered image stack.
        heights = np.zeros(locs_i.shape[0], dtype=np.float64)
        for row_idx in range(locs_i.shape[0]):
            rr = int(_matlab_round(locs_i[row_idx, 1]).item()) - 1
            cc = int(_matlab_round(locs_i[row_idx, 0]).item()) - 1
            if 0 <= rr < stack.shape[1] and 0 <= cc < stack.shape[2]:
                heights[row_idx] = stack[frame_idx, rr, cc]
        locs_i[:, 2] = heights

        locs_i[:, 4] = (frame_idx + 1) if matlab_frame_numbers else frame_idx
        all_locs.append(locs_i)

    if not all_locs:
        return np.full((1, 8), np.nan, dtype=np.float64)

    locs = np.vstack(all_locs)

    if bool(fastdetect):
        locs[:, 0:2] *= 2.0

    return np.asarray(locs, dtype=np.float64)


# MATLAB-style alias.
Detector = detector


__all__ = [
    "detector",
    "Detector",
]
