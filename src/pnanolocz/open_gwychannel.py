"""
Gwyddion GWY channel reader.

This module ports MATLAB ``open_gwychannel.m``.  It reads GWY containers and
returns a list of channel arrays plus metadata.

The GWY binary format is compact but idiosyncratic; this implementation follows
the same component/object recursion used in the MATLAB reader and is intended
for common ``GwyDataField`` channels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


def _read_c_string(fid: BinaryIO) -> str:
    """Read a null-terminated ASCII string."""
    data = bytearray()
    while True:
        b = fid.read(1)
        if not b or b == b"\x00":
            break
        data.extend(b)
    return data.decode("latin-1", errors="replace")


def _read_obj(fid: BinaryIO) -> tuple[str, int]:
    """Read GWY object type and size."""
    typ = _read_c_string(fid)
    size_bytes = fid.read(4)
    if len(size_bytes) != 4:
        raise EOFError
    size = int(np.frombuffer(size_bytes, dtype="<u4")[0])
    return typ, size


def _read_component(fid: BinaryIO) -> tuple[str, str, Any]:
    """Read one GWY component."""
    name = _read_c_string(fid)
    type_raw = fid.read(1)
    if not type_raw:
        raise EOFError
    typ_char = type_raw.decode("latin-1")
    typ = typ_char.lower()

    if typ_char.isupper():
        rep = int(np.frombuffer(fid.read(4), dtype="<u4")[0])
    else:
        rep = 1

    if typ == "b":
        ret = np.frombuffer(fid.read(rep), dtype=np.uint8)
    elif typ == "c":
        ret = fid.read(rep).decode("latin-1", errors="replace")
    elif typ == "i":
        ret = np.frombuffer(fid.read(4 * rep), dtype="<i4")
    elif typ == "q":
        ret = np.frombuffer(fid.read(8 * rep), dtype="<i8")
    elif typ == "d":
        ret = np.frombuffer(fid.read(8 * rep), dtype="<f8")
    elif typ == "s":
        ret = [_read_c_string(fid) for _ in range(rep)]
    elif typ == "o":
        ret = None
    else:
        ret = None

    if isinstance(ret, np.ndarray) and ret.size == 1:
        ret = ret[0].item()

    return name, typ, ret


def _read_datafield(fid: BinaryIO, size: int) -> dict[str, Any]:
    """Read a GWY GwyDataField object payload."""
    start = fid.tell()
    field: dict[str, Any] = {}

    while fid.tell() < start + size:
        name, typ, ret = _read_component(fid)
        if typ != "o":
            field[name] = ret
        else:
            _, sub_size = _read_obj(fid)
            fid.seek(sub_size, 1)

    if "data" in field and "xres" in field and "yres" in field:
        xres = int(field["xres"])
        yres = int(field["yres"])
        data = np.asarray(field["data"], dtype=np.float64)
        if data.size == xres * yres:
            field["data"] = data.reshape((xres, yres)).T

    return field


def open_gwychannel(filename: str | Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Read channels from a Gwyddion ``.gwy`` file."""
    channels: list[dict[str, Any]] = []
    titles: list[str] = []

    with open(filename, "rb") as fid:
        magic = fid.read(4).decode("latin-1", errors="replace")
        if magic != "GWYP":
            raise ValueError("not a GWY file")

        obj_type, obj_size = _read_obj(fid)
        if obj_type != "GwyContainer":
            raise ValueError("GWY root object is not GwyContainer")

        container_end = fid.tell() + obj_size

        while fid.tell() < container_end:
            try:
                name, typ, ret = _read_component(fid)
            except EOFError:
                break

            if typ == "o":
                obj_name, size = _read_obj(fid)
                if obj_name == "GwyDataField" and name.endswith("/data"):
                    channels.append(_read_datafield(fid, size))
                    titles.append("")
                else:
                    fid.seek(size, 1)
            elif typ == "s" and name.endswith("/title"):
                title = ret[0] if isinstance(ret, list) and ret else str(ret)
                if titles:
                    titles[-1] = title

    images = [np.asarray(ch["data"], dtype=np.float64) for ch in channels if "data" in ch]
    meta: dict[str, Any] = {}
    if channels:
        meta = {k: v for k, v in channels[0].items() if k != "data"}
    meta["channels"] = titles if titles else [f"Channel {i+1}" for i in range(len(images))]

    return images, meta


__all__ = ["open_gwychannel"]
