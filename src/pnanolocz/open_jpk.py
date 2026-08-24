"""
JPK classic TIFF reader helpers.

This module ports MATLAB ``open_JPK_info.m`` and ``open_JPK_image.m`` using
`tifffile`.  JPK stores channel metadata in private TIFF tags; the implementation
reads the same tag IDs used by the MATLAB code when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile


def _decode(value: Any) -> Any:
    """Decode TIFF tag values."""
    if isinstance(value, bytes):
        return value.decode(errors="replace").rstrip("\x00")
    if isinstance(value, tuple) and len(value) == 1:
        return _decode(value[0])
    return value


def _tag(page: tifffile.TiffPage, tag_id: int, default: Any = np.nan) -> Any:
    """Return a TIFF tag value by numeric ID."""
    try:
        return _decode(page.tags[tag_id].value)
    except Exception:
        return default


def _channel_type_to_scale_ids(page: tifffile.TiffPage) -> tuple[int, int]:
    """Replicate MATLAB's scaling-tag selection logic for JPK TIFF channels."""
    typ = str(_tag(page, 32897, "")).lower()

    if typ in {"nominal", "voltsamplitude"}:
        return 33028, 33029

    if typ in {"force", "calibrated", "distanceamplitude"}:
        return 33076, 33077

    if typ == "volts":
        detector = str(_tag(page, 32848, "")).lower()
        if detector in {
            "capacitivesensorxposition",
            "servodacy",
            "servodacx",
            "capacitivesensoryposition",
        }:
            return 33028, 33029
        return 32980, 32981

    return 32980, 32981


def open_jpk_image(
    afm_file: str | Path,
    ch: int,
    *,
    matlab_indexing: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Read one JPK TIFF channel and return raw image plus multiplier/offset."""
    page_index = int(ch) - 1 if matlab_indexing else int(ch)

    with tifffile.TiffFile(afm_file) as tif:
        page = tif.pages[page_index]
        m_id, off_id = _channel_type_to_scale_ids(page)
        multi = float(_tag(page, m_id, 1.0))
        offs = float(_tag(page, off_id, 0.0))
        im = page.asarray()

    return np.asarray(im), multi, offs


def open_jpk_info(afm_file: str | Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Read JPK TIFF scan metadata and channel descriptors."""
    with tifffile.TiffFile(afm_file) as tif:
        pages = list(tif.pages)
        first = pages[0]

        info = {
            "x_Origin": _tag(first, 32832),
            "y_Origin": _tag(first, 32833),
            "x_scan_length": _tag(first, 32834),
            "y_scan_length": _tag(first, 32835),
            "x_scan_pixels": _tag(first, 32838),
            "y_scan_pixels": _tag(first, 32839),
            "Reference_Amp": _tag(first, 32821),
            "Set_Amplitude": _tag(first, 32822),
            "Oscillation_Freq": _tag(first, 32823),
            "Scan_Rate": _tag(first, 32841),
        }

        channels: list[dict[str, str]] = []
        for page in pages[1:]:
            channel_name = str(_tag(page, 32850, "Channel"))
            descriptor = str(_tag(page, 32851, ""))
            tokens = descriptor.split()

            trace_type = "Trace"
            for k, token in enumerate(tokens):
                if token == "retrace":
                    is_retrace = (k + 2 < len(tokens)) and tokens[k + 2].lower() == "true"
                    is_smooth = (k - 1 >= 0) and tokens[k - 1] == "Smooth"

                    if is_retrace:
                        trace_type = "Bi-Directional Raw" if is_smooth else "ReTrace"
                    else:
                        trace_type = "Bi-Directional Smooth" if is_smooth else "Trace"
                    break

            channels.append({"Channel_name": channel_name, "Trace_type": trace_type})

    return info, channels


# MATLAB-style aliases.
open_JPK_image = open_jpk_image
open_JPK_info = open_jpk_info

__all__ = [
    "open_jpk_image",
    "open_jpk_info",
    "open_JPK_image",
    "open_JPK_info",
]
