"""
Central padding and stacking helpers for NanoLocz.

This module ports MATLAB ``pad_stacker.m``.  The function centrally pads two
2-D arrays to the same spatial size and stacks them into a 3-D array.

Python convention
-----------------
The default output is frame-first:

    C[0, :, :] = padded A
    C[1, :, :] = padded B

Set ``matlab_layout=True`` to return MATLAB-style ``(rows, cols, 2)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _center_pad_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> FloatArray:
    """Center-pad a 2-D array with zeros to ``target_shape``.

    If the difference in a dimension is odd, the extra pixel is placed at the
    end/post side, matching the intent of the MATLAB fallback path.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("pad_stacker expects 2-D arrays")

    target_rows, target_cols = target_shape
    if target_rows < a.shape[0] or target_cols < a.shape[1]:
        raise ValueError("target shape must be at least as large as input")

    pad_rows = target_rows - a.shape[0]
    pad_cols = target_cols - a.shape[1]

    before_rows = pad_rows // 2
    after_rows = pad_rows - before_rows
    before_cols = pad_cols // 2
    after_cols = pad_cols - before_cols

    return np.asarray(
        np.pad(
            a,
            ((before_rows, after_rows), (before_cols, after_cols)),
            mode="constant",
            constant_values=0,
        ),
        dtype=np.float64,
    )


def pad_stacker(
    A: np.ndarray,
    B: np.ndarray,
    *,
    matlab_layout: bool = False,
) -> FloatArray:
    """Centrally pad two matrices and stack them.

    Parameters
    ----------
    A, B:
        2-D numeric arrays.
    matlab_layout:
        If true, return ``(rows, cols, 2)``.  Otherwise return
        frame-first ``(2, rows, cols)``.

    Returns
    -------
    ndarray
        Padded and stacked output.
    """
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must both be 2-D arrays")

    target_shape = (max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1]))

    padded_a = _center_pad_to_shape(a, target_shape)
    padded_b = _center_pad_to_shape(b, target_shape)

    if matlab_layout:
        return np.asarray(np.stack([padded_a, padded_b], axis=-1), dtype=np.float64)

    return np.asarray(np.stack([padded_a, padded_b], axis=0), dtype=np.float64)


__all__ = ["pad_stacker"]
