"""
HDF5 reader for NanoLocz-compatible data files.

This module ports MATLAB ``open_h5.m``.  It recursively reads HDF5 groups and
datasets into nested Python dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _read_group(group: h5py.Group) -> dict[str, Any]:
    """Recursively read an HDF5 group."""
    out: dict[str, Any] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if isinstance(data, bytes):
                data = data.decode(errors="replace")
            out[key] = data
        elif isinstance(item, h5py.Group):
            out[key] = _read_group(item)
    return out


def open_h5(filename: str | Path) -> dict[str, Any]:
    """Read all datasets from an HDF5 file into a nested dictionary."""
    with h5py.File(filename, "r") as f:
        return _read_group(f)


__all__ = ["open_h5"]
