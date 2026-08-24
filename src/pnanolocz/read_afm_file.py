"""
High-level AFM file reader dispatch for NanoLocz.

This module ports the public behavior of MATLAB ``ReadAFMFile.m``.  It routes
common AFM file extensions to the corresponding low-level readers and returns:

    image_stack, image_info

Python output convention
------------------------
The returned image stack is frame-first ``(frames, rows, cols)``.  A single
image is promoted to one frame.

Supported formats depend on which low-level reader modules are installed:
JPK TIFF, JPK HDF5, NanoScope SPM, IBW, NHF, GWY, HDF5, ASD, TIFF/images, and
folders containing a sequence of supported files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile

try:
    from .time_elapsed import time_elapsed
except Exception:  # pragma: no cover
    from time_elapsed import time_elapsed  # type: ignore

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


_IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif"}
_TABLE_EXTS = {".csv", ".txt", ".dat", ".tsv", ".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"}


def _as_frame_first(image: np.ndarray, source_frame_axis: int = -1) -> FloatArray:
    """Promote image data to frame-first layout."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return np.asarray(np.moveaxis(arr, source_frame_axis, 0), dtype=np.float64)
    raise ValueError("image data must be 2-D or 3-D")


def _select_frames(stack: np.ndarray, frames: int | str = "all") -> FloatArray:
    """Apply MATLAB-style frames argument."""
    if isinstance(frames, str) and frames.lower() == "all":
        return np.asarray(stack, dtype=np.float64)

    n = int(frames)
    return np.asarray(stack[: min(n, stack.shape[0])], dtype=np.float64)


def _read_tiff_or_image(path: Path) -> tuple[FloatArray, dict[str, Any]]:
    """Read TIFF or common image formats."""
    arr = tifffile.imread(path)
    stack = _as_frame_first(arr, source_frame_axis=0 if np.asarray(arr).ndim == 3 and np.asarray(arr).shape[-1] in (3, 4) else -1)

    # If RGB/RGBA image was interpreted as frames, convert to grayscale instead.
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = np.asarray(arr, dtype=np.float64)
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        stack = gray[np.newaxis, :, :]

    info = {
        "Channel": "Image",
        "Channels": ["Image"],
        "n": int(stack.shape[0]),
        "yPixels": int(stack.shape[1]),
        "xPixels": int(stack.shape[2]),
        "ScanSize": np.nan,
        "PixelPerNm": np.nan,
        "ScanSpeed": np.nan,
        "LineSpeed": np.nan,
        "time": np.arange(stack.shape[0], dtype=np.float64),
    }
    return stack, info


def _read_one_file(path: Path, channel: str = "Height") -> tuple[FloatArray, dict[str, Any]]:
    """Read one AFM file using extension dispatch."""
    suffix = path.suffix.lower()

    if suffix == ".jpk":
        try:
            from .open_jpk import open_jpk_image, open_jpk_info
        except Exception:  # pragma: no cover
            from open_jpk import open_jpk_image, open_jpk_info  # type: ignore

        info_raw, channels = open_jpk_info(path)
        channel_names = [f"{c['Channel_name']} - {c['Trace_type']}" for c in channels]
        if channel in channel_names:
            ch_idx = channel_names.index(channel) + 2  # MATLAB pages start at 2 for channels.
        else:
            ch_idx = 2
        im, multi, offs = open_jpk_image(path, ch_idx, matlab_indexing=True)
        stack = ((np.flip(im) * multi) + offs)[np.newaxis, :, :]
        if "Height" in channel:
            stack = stack * 1e9

        scan_size = float(info_raw.get("x_scan_length", np.nan)) * 1e9
        xpix = int(info_raw.get("x_scan_pixels", stack.shape[2]) or stack.shape[2])
        ypix = int(info_raw.get("y_scan_pixels", stack.shape[1]) or stack.shape[1])

        info = {
            **info_raw,
            "Channels": channel_names,
            "Channel": channel if channel in channel_names else (channel_names[0] if channel_names else "Height"),
            "n": 1,
            "ScanSize": scan_size,
            "yPixels": ypix,
            "xPixels": xpix,
            "PixelPerNm": xpix / scan_size if scan_size else np.nan,
            "ScanSpeed": np.nan,
            "LineSpeed": info_raw.get("Scan_Rate", np.nan),
            "time": np.array([0.0]),
        }
        return np.asarray(stack, dtype=np.float64), info

    if suffix in {".h5-jpk", ".jpk-force-map"} or path.name.lower().endswith(".h5-jpk"):
        try:
            from .open_h5jpk import open_h5jpk
        except Exception:  # pragma: no cover
            from open_h5jpk import open_h5jpk  # type: ignore

        im, info = open_h5jpk(path, channel, matlab_layout=False)
        stack = _as_frame_first(im, source_frame_axis=0)
        info = dict(info)
        info["n"] = stack.shape[0]
        info["Channel"] = info.get("channel", channel)
        info["Channels"] = info.get("channels", [])
        info["ScanSize"] = float(info.get("xsize", np.nan)) * 1e9 if np.isfinite(info.get("xsize", np.nan)) else np.nan
        info["xPixels"] = info.get("xPixel", stack.shape[2])
        info["yPixels"] = info.get("yPixel", stack.shape[1])
        info["PixelPerNm"] = info["xPixels"] / info["ScanSize"] if info.get("ScanSize") else np.nan
        info["time"] = np.arange(stack.shape[0], dtype=np.float64)
        return stack, info

    if suffix == ".spm":
        try:
            from .open_nanoscope import open_nanoscope
        except Exception:  # pragma: no cover
            from open_nanoscope import open_nanoscope  # type: ignore

        im, info_raw = open_nanoscope(path, channel, False)
        stack = im[np.newaxis, :, :]
        info = dict(info_raw)
        info["Channel"] = info.get("channel", channel)
        info["Channels"] = info.get("channels", [])
        info["n"] = 1
        info["yPixels"] = stack.shape[1]
        info["xPixels"] = stack.shape[2]
        info["PixelPerNm"] = info["xPixels"] / info["ScanSize"] if info.get("ScanSize") else np.nan
        info["time"] = np.array([0.0])
        return stack, info

    if suffix == ".ibw":
        try:
            from .open_ibw import open_ibw
        except Exception:  # pragma: no cover
            from open_ibw import open_ibw  # type: ignore

        im, info_raw = open_ibw(path)
        stack = np.flip(im[np.newaxis, :, :] * 1e9)
        scan_size = float(info_raw.get("ScanSize", np.nan)) / 1e-9 if "ScanSize" in info_raw else np.nan
        info = dict(info_raw)
        info.update({
            "Channel": "Height",
            "Channels": ["Height"],
            "n": 1,
            "ScanSize": scan_size,
            "yPixels": stack.shape[1],
            "xPixels": stack.shape[2],
            "PixelPerNm": stack.shape[2] / scan_size if scan_size else np.nan,
            "time": np.array([0.0]),
        })
        return stack, info

    if suffix == ".nhf":
        try:
            from .open_nhf import open_nhf
        except Exception:  # pragma: no cover
            from open_nhf import open_nhf  # type: ignore

        im, info_raw = open_nhf(path, channel)
        stack = im[np.newaxis, :, :] * 1e9
        scan_size = float(info_raw.get("ScanSize", np.nan)) / 1e-9 if "ScanSize" in info_raw else np.nan
        info = dict(info_raw)
        info.update({
            "Channel": info.get("channel", channel),
            "Channels": info.get("channels", []),
            "n": 1,
            "ScanSize": scan_size,
            "yPixels": stack.shape[1],
            "xPixels": stack.shape[2],
            "PixelPerNm": stack.shape[2] / scan_size if scan_size else np.nan,
            "time": np.array([0.0]),
        })
        return stack, info

    if suffix == ".gwy":
        try:
            from .open_gwychannel import open_gwychannel
        except Exception:  # pragma: no cover
            from open_gwychannel import open_gwychannel  # type: ignore

        images, info_raw = open_gwychannel(path)
        channels = info_raw.get("channels", [])
        idx = channels.index(channel) if channel in channels else 0
        stack = np.asarray(images[idx], dtype=np.float64)[np.newaxis, :, :] * 1e9
        scan_size = float(info_raw.get("xreal", np.nan)) / 1e-9 if "xreal" in info_raw else np.nan
        info = dict(info_raw)
        info.update({
            "Channel": channels[idx] if channels else channel,
            "Channels": channels,
            "n": 1,
            "ScanSize": scan_size,
            "yPixels": stack.shape[1],
            "xPixels": stack.shape[2],
            "PixelPerNm": stack.shape[2] / scan_size if scan_size else np.nan,
            "time": np.array([0.0]),
        })
        return stack, info

    if suffix in {".h5", ".hdf5"}:
        # Try H5-JPK first, then generic H5 loader if installed.
        try:
            from .open_h5jpk import open_h5jpk
            im, info = open_h5jpk(path, channel, matlab_layout=False)
            stack = _as_frame_first(im, source_frame_axis=0)
            info = dict(info)
            info["n"] = stack.shape[0]
            info["Channel"] = info.get("channel", channel)
            info["Channels"] = info.get("channels", [])
            info["time"] = np.arange(stack.shape[0], dtype=np.float64)
            return stack, info
        except Exception:
            try:
                from .open_h5 import open_h5
            except Exception:  # pragma: no cover
                from open_h5 import open_h5  # type: ignore
            data = open_h5(path)
            return np.empty((0, 0, 0)), {"data": data, "Channel": channel, "Channels": []}

    if suffix == ".asd":
        try:
            from .open_asd import open_asd
        except Exception:  # pragma: no cover
            from open_asd import open_asd  # type: ignore
        im, info = open_asd(path, channel, matlab_layout=False)
        return _as_frame_first(im, source_frame_axis=0), dict(info)

    if suffix in _IMAGE_EXTS:
        return _read_tiff_or_image(path)

    raise ValueError(f"Unsupported AFM file type: {suffix}")


def _folder_files(folder: Path) -> list[Path]:
    """Return sorted readable files from a folder."""
    files = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
    return sorted(files, key=lambda p: p.name)


def read_afm_file(
    image_name: str | Path,
    channel: str = "Height",
    frames: int | str = "all",
) -> tuple[FloatArray, dict[str, Any]]:
    """Read AFM image data from a file or folder."""
    path = Path(image_name)

    if path.is_dir():
        files = _folder_files(path)
        if not files:
            return np.empty((0, 0, 0), dtype=np.float64), {"n": 0, "Channel": channel, "Channels": []}

        if isinstance(frames, str) and frames.lower() == "all":
            selected = files
        else:
            selected = files[: int(frames)]

        stacks = []
        first_info: dict[str, Any] | None = None
        times: list[float] = []

        for idx, file in enumerate(selected):
            stack, info = _read_one_file(file, channel)
            if first_info is None:
                first_info = dict(info)
            stacks.append(stack[0])
            times.append(float(idx))

        out = np.stack(stacks, axis=0)
        info = first_info or {}
        info["n"] = out.shape[0]
        info["time"] = np.asarray(times, dtype=np.float64)
        info["Channel"] = info.get("Channel", channel)
        return np.asarray(out, dtype=np.float64), info

    stack, info = _read_one_file(path, channel)
    stack = _select_frames(stack, frames)
    info = dict(info)
    info["n"] = stack.shape[0]
    if "time" not in info:
        info["time"] = np.arange(stack.shape[0], dtype=np.float64)
    return np.asarray(stack, dtype=np.float64), info


# MATLAB-style alias.
ReadAFMFile = read_afm_file

__all__ = ["read_afm_file", "ReadAFMFile"]
