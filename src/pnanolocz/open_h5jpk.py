"""
JPK HDF5 AFM reader.

This module ports MATLAB ``open_h5jpk.m``.  It reads JPK HDF5 channel groups,
extracts channel names, scaling metadata, scan parameters, and an image stack.

Default Python output layout is frame-first ``(frames, rows, cols)``.  Set
``matlab_layout=True`` for ``(rows, cols, frames)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _decode(value: Any) -> Any:
    """Decode HDF5 scalar bytes/arrays into Python values."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value.item())
        if value.dtype.kind == "S":
            return np.array([v.decode(errors="replace") for v in value.ravel()])
    return value


def _attrs(group: h5py.Group) -> dict[str, Any]:
    """Return decoded HDF5 attributes."""
    return {str(k): _decode(v) for k, v in group.attrs.items()}


def _first_group(file: h5py.File) -> h5py.Group:
    """Return the first top-level group."""
    for obj in file.values():
        if isinstance(obj, h5py.Group):
            return obj
    raise ValueError("no HDF5 groups found")


def _channel_groups(root: h5py.Group) -> list[h5py.Group]:
    """Find channel groups under the first/root group."""
    groups = [v for v in root.values() if isinstance(v, h5py.Group)]
    # MATLAB subtracts one group from count, usually metadata/position group.
    return [g for g in groups if "channel.fancy-name" in g.attrs or any(isinstance(v, h5py.Dataset) for v in g.values())]


def _build_channel_name(group: h5py.Group) -> str:
    """Build NanoLocz channel label ``FancyName-trace/retrace``."""
    a = _attrs(group)
    fancy = str(a.get("channel.fancy-name", group.name.rsplit("/", 1)[-1]))
    retrace = str(a.get("retrace", "false")).lower()
    suffix = "-retrace" if retrace == "true" else "-trace"
    return fancy + suffix


def _get_attr_float(attrs: dict[str, Any], name: str, default: float = 0.0) -> float:
    """Read an attribute as float."""
    try:
        return float(attrs[name])
    except Exception:
        return float(default)


def _resolve_scaling(attrs: dict[str, Any]) -> tuple[float, float, bool]:
    """Replicate MATLAB scaling selection for JPK HDF5 channels."""
    offs = _get_attr_float(attrs, "encoder.scaling.offset", 0.0)
    multi = _get_attr_float(attrs, "encoder.scaling.multiplier", 1.0)
    net = False

    if "net-encoder.scaling.multiplier" in attrs:
        offs = _get_attr_float(attrs, "net-encoder.scaling.offset", offs)
        multi = _get_attr_float(attrs, "net-encoder.scaling.multiplier", multi)
        net = True

    elif "conversion.calibrated.scaling.multiplier" in attrs:
        cal_multi = _get_attr_float(attrs, "conversion.calibrated.scaling.multiplier", 1.0)
        cal_offs = _get_attr_float(attrs, "conversion.calibrated.scaling.offset", 0.0)
        nom_multi = _get_attr_float(attrs, "conversion.nominal.scaling.multiplier", 1.0)
        nom_offs = _get_attr_float(attrs, "conversion.nominal.scaling.offset", 0.0)
        enc_multi = _get_attr_float(attrs, "encoder.scaling.multiplier", 1.0)
        enc_offs = _get_attr_float(attrs, "encoder.scaling.offset", 0.0)

        multi = cal_multi * nom_multi * enc_multi
        offs = enc_offs * nom_multi * cal_multi + nom_offs * cal_multi + cal_offs
        net = True

    elif "conversion.distanceamplitude.scaling.multiplier" in attrs:
        dist_multi = _get_attr_float(attrs, "conversion.distanceamplitude.scaling.multiplier", 1.0)
        dist_offs = _get_attr_float(attrs, "conversion.distanceamplitude.scaling.offset", 0.0)
        amp_multi = _get_attr_float(attrs, "conversion.voltsamplitude.scaling.multiplier", 1.0)
        amp_offs = _get_attr_float(attrs, "conversion.voltsamplitude.scaling.offset", 0.0)
        enc_multi = _get_attr_float(attrs, "encoder.scaling.multiplier", 1.0)
        enc_offs = _get_attr_float(attrs, "encoder.scaling.offset", 0.0)

        multi = dist_multi * amp_multi * enc_multi
        offs = enc_offs * amp_multi * dist_multi + amp_offs * dist_multi + dist_offs
        net = True

    return float(multi), float(offs), bool(net)


def open_h5jpk(
    filename: str | Path,
    ch: str = "Height-trace",
    *,
    matlab_layout: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a JPK HDF5 image stack."""
    meta: dict[str, Any] = {}

    with h5py.File(filename, "r") as f:
        root = _first_group(f)
        root_attrs = _attrs(root)
        groups = _channel_groups(root)

        if not groups:
            raise ValueError("no JPK channel groups found")

        channel_names = [_build_channel_name(g) for g in groups]
        meta["channels"] = channel_names
        meta["LineRate"] = _get_attr_float(root_attrs, "timing-settings.scanRate", np.nan)
        meta["xPixel"] = int(_get_attr_float(root_attrs, "position-pattern.grid.ilength", 0))
        meta["yPixel"] = int(_get_attr_float(root_attrs, "position-pattern.grid.jlength", 0))
        meta["xsize"] = _get_attr_float(root_attrs, "position-pattern.grid.ulength", np.nan)
        meta["ysize"] = _get_attr_float(root_attrs, "position-pattern.grid.vlength", np.nan)

        if ch in channel_names:
            idx = channel_names.index(ch)
            meta["channel"] = ch
        else:
            fallback = "Height-trace" if "Height-trace" in channel_names else channel_names[0]
            idx = channel_names.index(fallback)
            meta["channel"] = fallback

        group = groups[idx]
        attrs = _attrs(group)
        multi, offs, net = _resolve_scaling(attrs)

        datasets = [v for v in group.values() if isinstance(v, h5py.Dataset)]
        if not datasets:
            raise ValueError(f"channel group {group.name!r} contains no datasets")

        img_data = np.asarray(datasets[0][()], dtype=np.float64)

    xpix = int(meta["xPixel"]) if meta["xPixel"] else None
    ypix = int(meta["yPixel"]) if meta["yPixel"] else None

    if img_data.ndim == 2 and xpix and ypix:
        n_frames = img_data.shape[0] - 1 if img_data.shape[0] > 1 else img_data.shape[0]
        frames = []
        for i in range(n_frames):
            frame = img_data[i, :].reshape((xpix, ypix)).T
            frames.append(frame)
        im = np.stack(frames, axis=0)
    elif img_data.ndim == 3:
        im = img_data
    else:
        raise ValueError("unsupported JPK image dataset shape")

    meta["numberofFrames"] = im.shape[0]

    im = (np.flip(im) * multi) + offs
    if meta.get("channel") in {"Height-retrace", "Height-trace"} and net:
        im = im * 1e9

    if matlab_layout:
        im = np.moveaxis(im, 0, -1)

    return np.asarray(im, dtype=np.float64), meta


__all__ = ["open_h5jpk"]
