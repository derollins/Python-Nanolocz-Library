"""
Floating-point TIFF exporter for NanoLocz.

This module ports MATLAB ``tiff_exporter.m``.  It writes 2-D images or 3-D image
stacks as 32-bit floating-point TIFF files, optionally setting X/Y resolution
tags from pixels-per-nanometre.

The default 3-D layout is frame-first ``(frames, rows, cols)``, matching the
leveling APIs.  Pass ``frame_axis=-1`` for MATLAB-style ``(rows, cols, frames)``
input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

FloatArray = np.ndarray[Any, np.dtype[np.float32]]


def _as_frame_first(image_sequence: np.ndarray, frame_axis: int) -> tuple[np.ndarray, bool, int]:
    """Convert 2-D/3-D data to frame-first layout."""
    arr = np.asarray(image_sequence, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0
    if arr.ndim != 3:
        raise ValueError("image_sequence must be 2-D or 3-D")
    axis = int(frame_axis) % 3
    return np.moveaxis(arr, axis, 0), False, axis


def tiff_exporter(
    image_sequence: np.ndarray,
    full_file_name: str | Path,
    pix_per_nm: float | None = None,
    *,
    frame_axis: int = 0,
) -> Path:
    """Write a 2-D image or 3-D stack as a 32-bit floating-point TIFF.

    ``frame_axis`` identifies the frame dimension for a 3-D input.
    """
    frames, _, _ = _as_frame_first(image_sequence, frame_axis=frame_axis)
    path = Path(full_file_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {"Software": "Python pnanolocz"}
    kwargs: dict[str, Any] = {
        "photometric": "minisblack",
        "metadata": metadata,
    }

    if pix_per_nm is not None:
        # TIFF resolution is stored as rational pixels per unit.  MATLAB only
        # writes XResolution/YResolution without an explicit unit, so this is a
        # faithful numeric tag equivalent.
        kwargs["resolution"] = (float(pix_per_nm), float(pix_per_nm))

    tifffile.imwrite(path, frames.astype(np.float32), **kwargs)
    return path


__all__ = ["tiff_exporter"]
