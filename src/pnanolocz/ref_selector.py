"""
Reference crop selection for NanoLocz.

This module ports the processing part of MATLAB ``ref_selector.m``.  The MATLAB
version is GUI-driven; the Python version accepts an explicit rectangle and can
optionally use a lightweight Matplotlib selector.

Rectangle convention
--------------------
``rect = (x, y, width, height)`` in MATLAB-style 1-based coordinates by default.
Set ``matlab_indexing=False`` for 0-based Python rectangles.

Stack convention
----------------
Python stacks are frame-first ``(frames, rows, cols)`` by default.  Pass
``frame_axis=-1`` for MATLAB-style ``(rows, cols, frames)`` input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift

try:
    from .find_center import find_center_positions
    from .symmetry import rotation_sym
except Exception:  # pragma: no cover
    from find_center import find_center_positions  # type: ignore
    from symmetry import rotation_sym  # type: ignore


def _as_frame_first(stack: np.ndarray, frame_axis: int) -> np.ndarray:
    """Convert 2-D/3-D input to frame-first."""
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return np.moveaxis(arr, frame_axis, 0)
    raise ValueError("d must be 2-D or 3-D")


def _crop_rect(img: np.ndarray, rect: tuple[float, float, float, float], matlab_indexing: bool) -> np.ndarray:
    """Crop using a MATLAB-style rectangle."""
    x, y, w, h = [int(round(v)) for v in rect]
    if matlab_indexing:
        x -= 1
        y -= 1
    return np.asarray(img[y : y + h + 1, x : x + w + 1], dtype=np.float64)


def ref_selector(
    d: np.ndarray,
    center: str | None = None,
    fold: int | None = None,
    *,
    rect: tuple[float, float, float, float] | None = None,
    frame_axis: int = 0,
    matlab_indexing: bool = True,
) -> np.ndarray:
    """Crop and optionally center/symmetrize a reference image.

    For a 3-D input, the first frame along ``frame_axis`` is used.
    """
    stack = _as_frame_first(d, frame_axis=frame_axis)
    dg = gaussian_filter(stack, sigma=(0, 0.7, 0.7))
    d1 = dg[0]

    if rect is None:
        raise ValueError("Python ref_selector requires rect=(x, y, width, height); GUI drawing is not implemented here")

    ref = _crop_rect(d1, rect, matlab_indexing=matlab_indexing)

    if center == "yes" and fold is not None and fold > 1:
        trans = find_center_positions(int(fold), ref, 1)
        # MATLAB imtranslate(ref, -center_translation).  SciPy uses row/col.
        ref = ndi_shift(ref, shift=(-float(trans[1]), -float(trans[0])), order=1, mode="constant", cval=0.0)

    if fold is not None and fold > 1:
        ref = rotation_sym(ref, int(fold))

    return np.asarray(ref, dtype=np.float64)


__all__ = ["ref_selector"]
