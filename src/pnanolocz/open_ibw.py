"""
Igor Binary Wave (IBW) reader for NanoLocz.

This module ports the public behavior of MATLAB ``open_IBW.m``.  It first tries
to use the optional ``igor2`` package for robust IBW parsing.  If unavailable,
it uses a small fallback parser for common numeric IBW v5/v2 waves.

The function returns:

    image, metadata

where image is a 2-D AFM channel and metadata contains ScanSize, x/y pixels and
frame acquisition timing when those note fields are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import struct

import numpy as np


def _parse_notes(notes: str) -> dict[str, Any]:
    """Extract NanoLocz-relevant fields from Igor wave notes."""
    meta: dict[str, Any] = {}
    patterns = {
        "ScanSize": r"ScanSize:\s*([+-]?\d+(?:\.\d+)?)",
        "yPixel": r"ScanPoints:\s*([+-]?\d+(?:\.\d+)?)",
        "xPixel": r"ScanLines:\s*([+-]?\d+(?:\.\d+)?)",
        "ScanRate": r"ScanRate:\s*([+-]?\d+(?:\.\d+)?)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, notes)
        if m:
            meta[key] = float(m.group(1))

    if "ScanRate" in meta and "xPixel" in meta:
        meta["frameAcqTime"] = float(meta["ScanRate"]) * float(meta["xPixel"])

    return meta


def _load_with_igor2(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Use igor2 when installed."""
    from igor2 import binarywave  # type: ignore

    data = binarywave.load(str(path))
    wave = data["wave"]
    arr = np.asarray(wave["wData"], dtype=np.float64)
    notes_raw = wave.get("note", b"")
    if isinstance(notes_raw, bytes):
        notes = notes_raw.decode(errors="replace")
    else:
        notes = str(notes_raw)

    meta = _parse_notes(notes)
    meta["WaveNotes"] = notes

    if arr.ndim >= 2:
        img = np.asarray(arr[:, :, 0] if arr.ndim > 2 else arr, dtype=np.float64).T
    else:
        img = np.asarray(arr, dtype=np.float64)

    meta.setdefault("xPixel", img.shape[1] if img.ndim == 2 else img.size)
    meta.setdefault("yPixel", img.shape[0] if img.ndim == 2 else 1)

    return img, meta


def _fallback_read_ibw(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimal fallback parser for common little-endian numeric IBW files."""
    raw = path.read_bytes()

    # Detect endian/version from the first int16.
    candidates = [("<", struct.unpack("<h", raw[:2])[0]), (">", struct.unpack(">h", raw[:2])[0])]
    endian, version = next(((e, v) for e, v in candidates if v in (2, 5)), candidates[0])

    if version not in (2, 5):
        raise ValueError("Only IBW versions 2 and 5 are supported by fallback parser")

    if version == 5:
        # BinHeader5 is 64 bytes. WaveHeader5 starts at 64 bytes.
        wfm_size = struct.unpack(endian + "l", raw[4:8])[0]
        formula_size = struct.unpack(endian + "l", raw[8:12])[0]
        note_size = struct.unpack(endian + "l", raw[12:16])[0]

        wave_header = raw[64:384]
        wave_type = struct.unpack(endian + "h", wave_header[0:2])[0]
        npnts = struct.unpack(endian + "l", wave_header[12:16])[0]

        # nDim[4] in WaveHeader5 is commonly located at byte 76.
        dims = struct.unpack(endian + "4l", wave_header[76:92])
        dims_used = [d for d in dims if d > 0]

        data_offset = 64 + 320
        data_bytes = wfm_size - 320
        note_offset = data_offset + max(data_bytes, 0) + formula_size
    else:
        wfm_size = struct.unpack(endian + "L", raw[2:6])[0]
        note_size = struct.unpack(endian + "L", raw[6:10])[0]

        wave_header = raw[16:126]
        wave_type = struct.unpack(endian + "H", wave_header[0:2])[0]
        npnts = struct.unpack(endian + "L", wave_header[6:10])[0]
        dims_used = [npnts]
        data_offset = 16 + 110
        data_bytes = wfm_size - 110
        note_offset = data_offset + max(data_bytes, 0) + 16

    type_map = {
        2: "f4",
        4: "f8",
        8: "i1",
        16: "i2",
        32: "i4",
        64 + 8: "u1",
        64 + 16: "u2",
        64 + 32: "u4",
    }
    base_type = wave_type - 1 if wave_type % 2 else wave_type
    dtype_code = type_map.get(base_type)
    if dtype_code is None:
        raise ValueError(f"Unsupported IBW numeric type: {wave_type}")

    dtype = np.dtype(endian + dtype_code)
    n_values = int(npnts)
    data = np.frombuffer(raw[data_offset : data_offset + n_values * dtype.itemsize], dtype=dtype, count=n_values)

    if len(dims_used) >= 2 and np.prod(dims_used) <= data.size:
        arr = data[: int(np.prod(dims_used))].reshape(tuple(dims_used), order="F")
        img = np.asarray(arr[:, :, 0] if arr.ndim > 2 else arr, dtype=np.float64).T
    else:
        img = np.asarray(data, dtype=np.float64)

    note_bytes = raw[note_offset : note_offset + int(note_size)] if note_size else b""
    notes = note_bytes.decode(errors="replace")
    meta = _parse_notes(notes)
    meta["WaveNotes"] = notes
    meta.setdefault("xPixel", img.shape[1] if img.ndim == 2 else img.size)
    meta.setdefault("yPixel", img.shape[0] if img.ndim == 2 else 1)

    return img, meta


def open_ibw(filename: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an Igor Binary Wave AFM image."""
    path = Path(filename)
    try:
        return _load_with_igor2(path)
    except Exception:
        return _fallback_read_ibw(path)


# MATLAB-style alias.
open_IBW = open_ibw

__all__ = ["open_ibw", "open_IBW"]
