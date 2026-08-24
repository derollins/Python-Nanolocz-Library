"""
Parachute-artifact simulation for NanoLocz-compatible AFM images.

This module ports NanoLocz ``add_para.m``.  The routine adds a synthetic
line-scanning/parachuting artifact to a 2-D AFM image by detecting steep positive
x-gradient transitions and then forcing subsequent pixels to decay by a fixed
gradient until the original line catches up again.

Direction handling matches MATLAB:
- ``'trace'``: process rows left-to-right.
- ``'retrace'``: flip image left/right, process, then flip back.
- ``'bi-directional'``: flip every second row before processing, then flip
  those rows back after processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def add_para(
    img: np.ndarray,
    para_grad: float,
    direction: str = "trace",
) -> FloatArray:
    """Apply synthetic parachute scanning artifacts to a 2-D image.

    Parameters
    ----------
    img:
        Input 2-D AFM height image.
    para_grad:
        Gradient/slope of the artificial parachute artifact.
    direction:
        ``'trace'``, ``'retrace'``, or ``'bi-directional'``.

    Returns
    -------
    ndarray
        Image with simulated parachute artifact added.

    MATLAB alignment notes
    ----------------------
    The MATLAB implementation uses ``conv2(Li, [-1, 0, 1], 'same')`` for each
    row and starts a descending artifact when the local gradient exceeds
    ``2 * para_grad``.  This Python version keeps the same stateful row-walking
    logic, including the while-loop behavior that updates the line in-place.
    """
    arr = np.asarray(img, dtype=np.float64).copy()

    if arr.ndim != 2:
        raise ValueError("add_para expects a 2-D image")

    direction_lc = direction.strip().lower()

    if direction_lc == "trace":
        work = arr.copy()
    elif direction_lc == "retrace":
        work = np.fliplr(arr)
    elif direction_lc in {"bi-directional", "bidirectional", "bi directional"}:
        work = arr.copy()
        work[1::2, :] = np.fliplr(work[1::2, :])
    else:
        raise ValueError("direction must be 'trace', 'retrace', or 'bi-directional'")

    result = work.copy()
    sobel_x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

    for row_idx in range(work.shape[0]):
        original_line = work[row_idx, :]
        line = original_line.copy()

        # MATLAB conv2(..., 'same') on a row vector is equivalent to 1-D
        # convolution with zero padding and same output length.
        x_gradient = np.convolve(original_line, sobel_x, mode="same")
        trigger = x_gradient > (2.0 * float(para_grad))
        if trigger.size:
            trigger[0] = False

        trigger_positions = np.flatnonzero(trigger)

        for k in trigger_positions:
            # MATLAB uses 1-based k and initializes count = k - 1.  The Python
            # equivalent starts at the previous zero-based index.
            count = int(k) - 1

            while count < (original_line.size - 1) and line[count] >= original_line[count]:
                count += 1
                line[count] = line[count - 1] - float(para_grad)

            if count < (original_line.size - 1):
                line[count] = original_line[count]

        result[row_idx, :] = line

    if direction_lc == "trace":
        return np.asarray(result, dtype=np.float64)

    if direction_lc == "retrace":
        return np.asarray(np.fliplr(result), dtype=np.float64)

    # bi-directional: flip every second row back.
    result[1::2, :] = np.fliplr(result[1::2, :])
    return np.asarray(result, dtype=np.float64)


__all__ = ["add_para"]
