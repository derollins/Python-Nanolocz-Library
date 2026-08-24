"""
ARIS/NanoRacer HDF5 reader for NanoLocz-compatible AFM data.

This module ports the public behavior of MATLAB ``open_ARIS.m``.  ARIS files are
HDF5 files with frame groups and channel-specific ``Image`` datasets.

The returned image stack uses Python frame-first layout by default:
``(frames, rows, cols)``.  Set ``matlab_layout=True`` for
``(rows, cols, frames)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_attr(value: Any) -> Any:
    """Decode HDF5 byte/string attributes."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return np.array([v.decode(errors="replace") for v in value])
    return value


def _visit_groups(f: h5py.File) -> list[str]:
    """Return all group paths in an HDF5 file."""
    groups: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Group):
            groups.append("/" + name if not name.startswith("/") else name)

    f.visititems(visitor)
    return groups


def _find_frame_groups(f: h5py.File) -> list[str]:
    """Find groups whose basename looks like ``Frame N``."""
    groups = _visit_groups(f)
    frame_groups = []
    for g in groups:
        base = g.rstrip("/").split("/")[-1]
        if base.lower().startswith("frame "):
            frame_groups.append(g)

    def frame_num(path: str) -> int:
        base = path.rstrip("/").split("/")[-1]
        try:
            return int(base.split("Frame ")[-1])
        except Exception:
            return 0

    return sorted(frame_groups, key=frame_num)


def _find_channels(f: h5py.File) -> list[str]:
    """Find available channel names under DataSetInfo or frame groups."""
    channels: list[str] = []

    if "/DataSetInfo/Global/Channels" in f:
        for key in f["/DataSetInfo/Global/Channels"].keys():
            channels.append(str(key))

    if channels:
        return channels

    # Fallback: inspect the first frame group for children containing Image.
    frames = _find_frame_groups(f)
    if frames:
        grp = f[frames[0]]
        for key, val in grp.items():
            if isinstance(val, h5py.Group) and "Image" in val:
                channels.append(str(key))
    return channels


def _read_first_attr(f: h5py.File, paths: list[str], attr_name: str, default: Any = None) -> Any:
    """Read the first matching HDF5 attribute."""
    for path in paths:
        try:
            if path in f and attr_name in f[path].attrs:
                return _decode_attr(f[path].attrs[attr_name])
        except Exception:
            pass
    return default


def open_aris(
    filename: str | Path,
    ch: str | None = None,
    *,
    matlab_layout: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read ARIS/NanoRacer HDF5 image data.

    Parameters
    ----------
    filename:
        ARIS HDF5 file path.
    ch:
        Requested channel name.  If absent or unavailable, the first available
        channel is used.
    matlab_layout:
        If true, return ``(rows, cols, frames)``.

    Returns
    -------
    image_stack, metadata
        Image stack and metadata dictionary.
    """
    meta: dict[str, Any] = {}

    with h5py.File(filename, "r") as f:
        frame_groups = _find_frame_groups(f)
        if not frame_groups:
            raise ValueError("No ARIS frame groups found")

        channels = _find_channels(f)
        meta["channels"] = channels

        if ch is None or ch not in channels:
            channel = channels[0] if channels else str(ch)
        else:
            channel = ch
        meta["channel"] = channel

        # Metadata best-effort.  ARIS layouts vary across versions.
        meta["xPixel"] = _read_first_attr(f, ["/DataSetInfo/Global"], "ScanPoints", None)
        meta["yPixel"] = _read_first_attr(f, ["/DataSetInfo/Global"], "ScanLines", None)
        meta["frameAcqTime"] = _read_first_attr(f, ["/DataSetInfo/Global"], "TimePerFrame", None)
        meta["numberofFrames"] = len(frame_groups)

        scale_paths = [
            f"/DataSetInfo/Global/Channels/{channel}/ImageDims",
            "/DataSetInfo/Global/Channels/HeightTrace/ImageDims",
            "/DataSetInfo/Global/Channels/HeightRetrace/ImageDims",
        ]
        scale_attr = _read_first_attr(f, scale_paths, "DimScaling", None)
        try:
            meta["scale"] = float(np.max(scale_attr))
        except Exception:
            meta["scale"] = np.nan

        frames = []
        for g in frame_groups:
            dataset_path = f"{g}/{channel}/Image"
            if dataset_path not in f:
                # Fallback: locate first child with Image.
                found = None
                for key, val in f[g].items():
                    if isinstance(val, h5py.Group) and "Image" in val:
                        found = f"{g}/{key}/Image"
                        break
                if found is None:
                    continue
                dataset_path = found

            data = np.asarray(f[dataset_path][()], dtype=np.float64).T
            frames.append(data)

        if not frames:
            raise ValueError("No ARIS image frames could be read")

    im = np.stack(frames, axis=0)
    im[~np.isfinite(im)] = 0.0

    if matlab_layout:
        im = np.moveaxis(im, 0, -1)

    return np.asarray(im, dtype=np.float64), meta


open_ARIS = open_aris

__all__ = ["open_aris", "open_ARIS"]
