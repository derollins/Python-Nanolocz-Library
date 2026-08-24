"""
NanoHybrid/NHF HDF5 reader.

This module ports MATLAB ``open_NHF.m``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return value


def _attrs(obj: Any) -> dict[str, Any]:
    return {str(k): _decode(v) for k, v in obj.attrs.items()}


def open_nhf(filename: str | Path, ch: str = "Topography") -> tuple[np.ndarray, dict[str, Any]]:
    """Read one channel from a NanoHybrid/NHF HDF5 file."""
    trace = 0
    meta: dict[str, Any] = {}

    with h5py.File(filename, "r") as f:
        root_group = next(v for v in f.values() if isinstance(v, h5py.Group))
        root_attrs = _attrs(root_group)

        meta["ScanSize"] = root_attrs.get("image_size_x", np.nan)
        meta["xPixel"] = int(root_attrs.get("image_points_per_line", 0))
        meta["yPixel"] = int(root_attrs.get("image_number_of_lines", root_attrs.get("image_number_of_lines_aquired", 0)))
        meta["linerate"] = root_attrs.get("image_line_rate", np.nan)
        meta["frameAcqTime"] = float(meta["yPixel"]) / float(meta["linerate"]) if meta["linerate"] else np.nan

        group_name = f"/measurement_0/segment_{trace}"
        if group_name not in f:
            # Fallback: find a segment group.
            candidates = []
            f.visititems(lambda name, obj: candidates.append("/" + name) if isinstance(obj, h5py.Group) and name.endswith(f"segment_{trace}") else None)
            if not candidates:
                raise ValueError("NHF segment group not found")
            group_name = candidates[0]

        grp = f[group_name]
        channels: list[str] = []
        dataset_name = None

        for name, ds in grp.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            attrs = _attrs(ds)
            cname = str(attrs.get("name", name))
            channels.append(cname)
            if cname == ch:
                dataset_name = name

        if dataset_name is None:
            if not channels:
                raise ValueError("no datasets/channels found in NHF file")
            dataset_name = list(grp.keys())[0]

        ds = grp[dataset_name]
        attrs = _attrs(ds)
        image_data = np.asarray(ds[()], dtype=np.float64)

        meta["channels"] = channels
        meta["channel"] = str(attrs.get("name", dataset_name))
        meta["channel_units"] = attrs.get("base_calibration_unit", "")
        cali_min = float(attrs.get("base_calibration_min", 0.0))
        cali_max = float(attrs.get("base_calibration_max", 1.0))

    xpix = int(meta["xPixel"])
    ypix = int(meta["yPixel"])
    im = image_data.reshape((xpix, ypix)).T

    bit = 2**31
    cali_factor = (cali_max - cali_min) / float(bit * 2)
    im = (im.astype(float) + bit) * cali_factor + cali_min
    im = np.flip(im, axis=0)

    return np.asarray(im, dtype=np.float64), meta


# MATLAB-style alias.
open_NHF = open_nhf

__all__ = ["open_nhf", "open_NHF"]
