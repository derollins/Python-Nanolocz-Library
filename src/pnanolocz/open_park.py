"""
Park AFM TIFF reader.

This module ports MATLAB ``open_park.m``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile


def _bytes_to_double(data: bytes | bytearray | np.ndarray, start_1based: int, end_1based: int) -> float:
    """Read MATLAB-style 1-based byte slice as little-endian double."""
    if isinstance(data, np.ndarray):
        raw = bytes(np.asarray(data, dtype=np.uint8).ravel())
    else:
        raw = bytes(data)
    start = start_1based - 1
    end = end_1based
    chunk = raw[start:end]
    if len(chunk) < 8:
        return float("nan")
    return float(np.frombuffer(chunk[:8], dtype="<f8")[0])


def open_park(afm_file: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read Park AFM TIFF data and basic scaling metadata."""
    with tifffile.TiffFile(afm_file) as tif:
        page = tif.pages[0]
        im = page.asarray()
        tags = page.tags

        # MATLAB uses UnknownTags(4).Value.  Park files commonly store the
        # parameter byte block in a private TIFF tag; choose the fourth private
        # tag if available, otherwise fall back to the largest byte-like tag.
        private_tags = [tag for tag in tags.values() if int(tag.code) >= 32768]
        param = None
        if len(private_tags) >= 4:
            param = private_tags[3].value
        else:
            byte_candidates = [tag.value for tag in private_tags if isinstance(tag.value, (bytes, bytearray))]
            if byte_candidates:
                param = max(byte_candidates, key=len)

    meta: dict[str, Any] = {}

    if param is not None:
        meta["scale"] = _bytes_to_double(param, 141, 148)
        meta["frameAcqTime"] = _bytes_to_double(param, 173, 180) * im.shape[0]
        gain = _bytes_to_double(param, 221, 228)
    else:
        meta["scale"] = np.nan
        meta["frameAcqTime"] = np.nan
        gain = 1.0

    image = -float(gain) * np.asarray(im, dtype=np.float64)
    return image, meta


__all__ = ["open_park"]
