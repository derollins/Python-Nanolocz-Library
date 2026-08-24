"""
NanoScope SPM reader for NanoLocz.

This module ports the practical public behavior of MATLAB ``open_nanoscope.m``.
The NanoScope format varies across software versions, so this reader implements
a tolerant header parser and binary-image loader for common Bruker/DI ``.spm``
files.

It returns:

    image, metadata

The image is scaled according to the best parsed Z scale.  Height-like channels
are generally returned in nanometres when the file's Z scale is in SI units, but
vendor files differ; always verify against known data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np


def _first_float(text: str, default: float = np.nan) -> float:
    """Extract the first floating point number from text."""
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(m.group(0)) if m else float(default)


def _numbers(text: str) -> list[float]:
    """Extract all floats from text."""
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def _read_header(path: Path) -> tuple[list[str], int]:
    """Read NanoScope text header until file-list end."""
    raw = path.read_bytes()
    marker = b"\\*File list end"
    pos = raw.find(marker)
    if pos == -1:
        # Fallback: locate first long run of non-text/binary data.
        pos = raw.find(b"\x00\x00\x00\x00")
        if pos == -1:
            pos = min(len(raw), 65536)
    end = raw.find(b"\n", pos)
    if end == -1:
        end = pos
    header_bytes = raw[: end + 1]
    header_text = header_bytes.decode("latin-1", errors="replace")
    return header_text.splitlines(), end + 1


def _parse_header(lines: list[str]) -> dict[str, Any]:
    """Parse relevant NanoScope header fields."""
    info: dict[str, Any] = {
        "channels": [],
        "channel_offsets": [],
        "z_scales": [],
        "samps": None,
        "line_no": None,
        "ScanSize": np.nan,
        "Scan_Rate": np.nan,
        "Time": "",
    }

    for line in lines[:5]:
        if len(line) >= 8 and not info["Time"]:
            info["Time"] = line[7:].strip()

    current_channel = None
    for line in lines:
        lower = line.lower()

        if "image data:" in lower:
            # Channel names are often quoted.
            q = re.findall(r'"([^"]+)"', line)
            current_channel = q[0] if q else line.split(":", 1)[-1].strip()
            info["channels"].append(current_channel)

        if "data offset" in lower:
            info["channel_offsets"].append(int(_first_float(line, 0)))

        if "samps" in lower and info["samps"] is None:
            nums = _numbers(line)
            if nums:
                info["samps"] = int(nums[-1])

        if "lines:" in lower and info["line_no"] is None:
            nums = _numbers(line)
            if nums:
                info["line_no"] = int(nums[-1])

        if "\\scan size:" in lower or "scan size:" in lower:
            nums = _numbers(line)
            if nums:
                # MATLAB chooses a value >100 when present, otherwise next.
                candidates = [n for n in nums if n > 0]
                if candidates:
                    info["ScanSize"] = candidates[1] if len(candidates) > 1 and candidates[0] <= 100 else candidates[0]

        if "scan rate:" in lower:
            nums = _numbers(line)
            if nums:
                info["Scan_Rate"] = nums[0]

        if "@2:z scale:" in lower or "z scale:" in lower or "@sens. zsens:" in lower:
            nums = _numbers(line)
            if nums:
                info["z_scales"].append(nums[-1])

        if "aspect ratio:" in lower:
            nums = _numbers(line)
            if nums:
                info["AspectRatio"] = nums[0]

        if "capture direction:" in lower:
            info["Direction"] = "Down" if ": d" in lower else "Up"

    if not info["channels"] and info["channel_offsets"]:
        info["channels"] = [f"Channel {i+1}" for i in range(len(info["channel_offsets"]))]

    return info


def open_nanoscope(
    fname: str | Path,
    channel: str = "Height",
    error: int | bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one channel from a NanoScope/Bruker SPM file."""
    path = Path(fname)
    lines, _ = _read_header(path)
    info = _parse_header(lines)

    channels = list(info.get("channels", []))
    offsets = list(info.get("channel_offsets", []))
    if not offsets:
        raise ValueError("No NanoScope data offsets found")

    if channel in channels:
        idx = channels.index(channel)
    else:
        # Tolerant contains search.
        idx = next((i for i, name in enumerate(channels) if channel.lower() in str(name).lower()), 0)

    idx = min(idx, len(offsets) - 1)

    samples = int(info["samps"] or 0)
    lines_n = int(info["line_no"] or 0)
    if samples <= 0 or lines_n <= 0:
        raise ValueError("Could not parse NanoScope image dimensions")

    offset = int(offsets[idx])
    raw = path.read_bytes()

    # NanoScope image data is commonly int16.  If there are not enough int16
    # values, try int32 as fallback.
    count = samples * lines_n
    data16 = np.frombuffer(raw[offset : offset + count * 2], dtype="<i2", count=count)
    if data16.size == count:
        image = data16.reshape((lines_n, samples))
    else:
        data32 = np.frombuffer(raw[offset : offset + count * 4], dtype="<i4", count=count)
        if data32.size != count:
            raise EOFError("Not enough binary data for NanoScope image")
        image = data32.reshape((lines_n, samples))

    z_scales = info.get("z_scales", [])
    if z_scales:
        scale = float(z_scales[min(idx, len(z_scales) - 1)]) / (2**16)
    else:
        scale = 1.0

    img = np.asarray(image, dtype=np.float64) * scale

    # Populate MATLAB-like metadata names.
    meta = dict(info)
    meta["line_no"] = lines_n
    meta["channels"] = channels
    meta["channel"] = channels[idx] if channels else channel
    meta["ScanSize"] = info.get("ScanSize", np.nan)
    meta["Scan_Rate"] = info.get("Scan_Rate", np.nan)

    return img, meta


__all__ = ["open_nanoscope"]
