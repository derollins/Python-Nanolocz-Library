"""
Horizontal scar/artifact filling for AFM images.

This module ports MATLAB ``scar_fill.m``.  The algorithm detects thin horizontal
scar masks from vertical gradient sign changes, removes short fragments, expands
the mask along rows, and fills masked pixels by interpolation/inpainting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import remove_small_objects
from skimage.restoration import inpaint

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def scar_fill(
    A: np.ndarray,
    thresh: float,
    thresh_h: float,
    min_length: int,
) -> FloatArray:
    """Fill thin horizontal scars/artifacts in a 2-D AFM image."""
    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("scar_fill expects a 2-D image")

    # MATLAB: Af = medfilt2(A, [1, 10])
    Af = median_filter(arr, size=(1, 10), mode="nearest")

    # MATLAB: [~, grad] = gradient(Af) returns the gradient along columns as the
    # second output.  In NumPy, gradient returns [row_gradient, col_gradient].
    _, grad = np.gradient(Af)

    At = grad > float(thresh)
    At2 = grad < -float(thresh)

    # MATLAB shifts the negative-gradient mask down by two rows, then combines
    # it with the positive-gradient mask to identify paired scar edges.
    At2_shift = np.vstack([At2[2:, :], np.zeros((2, At2.shape[1]), dtype=bool)])
    scar_mask = At2_shift & At
    scar_mask = np.vstack([np.zeros((1, scar_mask.shape[1]), dtype=bool), scar_mask])
    scar_mask = scar_mask[:-1, :]

    scar_mask = remove_small_objects(
        scar_mask.astype(bool), max_size=3, connectivity=2
    )

    # MATLAB: medfilt2(At, [1, min_length*2]) > 0
    width = max(1, int(min_length) * 2)
    scar_mask = median_filter(scar_mask.astype(float), size=(1, width), mode="nearest") > 0
    scar_mask = scar_mask & (arr > float(thresh_h))

    if not np.any(scar_mask):
        return arr.copy()

    # MATLAB regionfill solves a smooth interpolation inside the masked region.
    try:
        filled = inpaint.inpaint_biharmonic(arr, scar_mask, channel_axis=None)
    except TypeError:
        filled = inpaint.inpaint_biharmonic(arr, scar_mask, multichannel=False)

    return np.asarray(filled, dtype=np.float64)


__all__ = ["scar_fill"]
