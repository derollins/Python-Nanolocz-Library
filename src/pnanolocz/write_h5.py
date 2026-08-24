"""
HDF5 writer for NanoLocz data structures.

This module ports MATLAB ``write_h5.m``.  It writes nested dictionaries,
dataclasses, simple objects, lists/tuples, strings, booleans and numeric arrays
to an HDF5 file.

MATLAB behavior retained
------------------------
- Struct fields become HDF5 groups/datasets.
- Character/cell string data are written as strings.
- Logical arrays are stored as uint8 for compatibility.
- Existing datasets are overwritten silently.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


def _to_mapping(data: Any) -> Mapping[str, Any]:
    """Convert common Python data containers to a mapping."""
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, Mapping):
        return data
    if hasattr(data, "__dict__"):
        return vars(data)
    raise TypeError("Data must be a mapping, dataclass, or object with __dict__")


def _prepare_dataset_value(value: Any) -> tuple[Any, Any | None]:
    """Convert a Python value to something h5py can write.

    Returns
    -------
    data, dtype
        ``dtype`` is optional and is used mainly for variable-length strings.
    """
    # MATLAB converts char to string.
    if isinstance(value, str):
        return value, h5py.string_dtype(encoding="utf-8")

    if isinstance(value, bytes):
        return value.decode(errors="replace"), h5py.string_dtype(encoding="utf-8")

    if isinstance(value, (list, tuple)):
        # Cell arrays of strings become string arrays.  Numeric lists become
        # normal numeric arrays.
        if all(isinstance(v, (str, bytes)) for v in value):
            decoded = [v.decode(errors="replace") if isinstance(v, bytes) else str(v) for v in value]
            return np.asarray(decoded, dtype=object), h5py.string_dtype(encoding="utf-8")
        return _prepare_dataset_value(np.asarray(value))

    arr = np.asarray(value)

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8), None

    if arr.dtype.kind in {"U", "S", "O"}:
        # Object arrays are only safe if string-like.
        flat = arr.ravel()
        if all(isinstance(v, (str, bytes, np.str_, np.bytes_)) for v in flat):
            decoded = [
                bytes(v).decode(errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
                for v in flat
            ]
            decoded_arr = np.asarray(decoded, dtype=object).reshape(arr.shape)
            return decoded_arr, h5py.string_dtype(encoding="utf-8")

        # Fallback: store repr strings rather than failing silently.  This keeps
        # the file inspectable and avoids losing field names.
        repr_arr = np.asarray([repr(v) for v in flat], dtype=object).reshape(arr.shape)
        return repr_arr, h5py.string_dtype(encoding="utf-8")

    return arr, None


def _write_value(group: h5py.Group, name: str, value: Any) -> None:
    """Write one field into a group, recursing into nested mappings."""
    if value is None:
        # HDF5 has no native None.  Store an empty string marker to preserve the
        # field name.
        data, dtype = "", h5py.string_dtype(encoding="utf-8")
    elif is_dataclass(value) or isinstance(value, Mapping) or hasattr(value, "__dict__") and not isinstance(value, np.ndarray):
        submap = _to_mapping(value)
        if name in group and isinstance(group[name], h5py.Dataset):
            del group[name]
        subgroup = group.require_group(name)
        for key, subvalue in submap.items():
            _write_value(subgroup, str(key), subvalue)
        return
    else:
        data, dtype = _prepare_dataset_value(value)

    if name in group:
        del group[name]

    if dtype is not None:
        group.create_dataset(name, data=data, dtype=dtype)
    else:
        group.create_dataset(name, data=data)


def write_h5(filepath: str | Path, data: Any) -> Path:
    """Write a NanoLocz-style data structure to an HDF5 file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    root_map = _to_mapping(data)

    with h5py.File(path, "a") as h5:
        for key, value in root_map.items():
            _write_value(h5, str(key), value)

    return path


__all__ = ["write_h5"]
