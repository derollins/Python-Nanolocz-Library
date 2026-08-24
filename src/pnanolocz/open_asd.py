"""
ASD file reader for NanoLocz-compatible HS-AFM data.

This module ports MATLAB ``open_asd.m``.  ASD files contain a binary header and
12-bit HS-AFM image frames.  The returned image stack uses Python frame-first
layout by default: ``(frames, rows, cols)``.

Use ``matlab_layout=True`` to return ``(rows, cols, frames)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _read_scalar(f, dtype: str):
    """Read one little-endian scalar."""
    arr = np.fromfile(f, dtype=np.dtype(dtype).newbyteorder("<"), count=1)
    if arr.size == 0:
        raise EOFError("unexpected end of ASD file")
    return arr[0].item()


def _read_chars(f, n: int) -> str:
    """Read n characters and decode as latin-1."""
    if n <= 0:
        return ""
    data = np.fromfile(f, dtype=np.uint8, count=int(n))
    return bytes(data).decode("latin-1", errors="replace")


def open_asd(
    filename: str | Path,
    ch: str = "Height",
    *,
    matlab_layout: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an ASD file and return image stack plus header dictionary."""
    header: dict[str, Any] = {}

    with open(filename, "rb") as f:
        int_fields = [
            "fileVersion", "fileHeaderSize", "frameHeaderSize", "encNumber",
            "operationNameSize", "commentSize", "dataTypeCh1", "dataTypeCh2",
            "numberFramesRecorded", "numberFramesCurrent", "scanDirection",
            "fileName", "xPixel", "yPixel", "xScanRange", "yScanRange",
        ]
        for field in int_fields:
            header[field] = int(_read_scalar(f, "int32"))

        header["avgFlag"] = bool(_read_scalar(f, "bool"))
        header["avgNumber"] = int(_read_scalar(f, "int32"))

        for field in ["yearRec", "monthRec", "dayRec", "hourRec", "minuteRec", "secondRec", "xRoundDeg", "yRoundDeg"]:
            header[field] = int(_read_scalar(f, "int32"))

        header["frameAcqTime"] = float(_read_scalar(f, "float32"))
        header["sensorSens"] = float(_read_scalar(f, "float32"))
        header["phaseSens"] = float(_read_scalar(f, "float32"))

        header["offset"] = int(_read_scalar(f, "int32"))
        f.seek(12, 1)

        header["machineNum"] = int(_read_scalar(f, "int32"))
        header["adRange"] = int(_read_scalar(f, "int32"))
        header["adRes"] = int(_read_scalar(f, "int32"))

        for field in ["xMaxScanRange", "yMaxScanRange", "xExtCoef", "yExtCoef", "zExtCoef", "zDriveGain"]:
            header[field] = float(_read_scalar(f, "float32"))

        header["operName"] = _read_chars(f, header["operationNameSize"])
        header["comment"] = _read_chars(f, header["commentSize"])
        header["AA"] = header["operName"]
        header["BB"] = header["comment"]

        if header["dataTypeCh2"] > 0:
            n_raw = header["numberFramesCurrent"] * 2
        else:
            n_raw = header["numberFramesCurrent"]

        ypix = int(header["yPixel"])
        xpix = int(header["xPixel"])
        pre = np.zeros((n_raw, ypix, xpix), dtype=np.float64)

        frame_meta: dict[str, list[Any]] = {
            "frameNumber": [],
            "frameMaxData": [],
            "frameMinData": [],
            "xOffset": [],
            "dataType": [],
            "xTilt": [],
            "yTilt": [],
            "flagLaserIr": [],
        }

        for k in range(n_raw):
            frame_meta["frameNumber"].append(int(_read_scalar(f, "int32")))
            frame_meta["frameMaxData"].append(int(_read_scalar(f, "int16")))
            frame_meta["frameMinData"].append(int(_read_scalar(f, "int16")))
            frame_meta["xOffset"].append(int(_read_scalar(f, "int16")))
            frame_meta["dataType"].append(int(_read_scalar(f, "int16")))
            frame_meta["xTilt"].append(float(_read_scalar(f, "float32")))
            frame_meta["yTilt"].append(float(_read_scalar(f, "float32")))

            skip = int(header["frameHeaderSize"]) - 21
            if skip > 0:
                f.seek(skip, 1)
            frame_meta["flagLaserIr"].append(False)

            sub = np.fromfile(f, dtype=np.dtype("int16").newbyteorder("<"), count=xpix * ypix)
            if sub.size != xpix * ypix:
                raise EOFError("unexpected end of ASD image data")
            pre[k] = sub.reshape((xpix, ypix)).T

        header.update(frame_meta)

    im = -pre / 205.0 * float(header["zExtCoef"])
    im = np.flip(im)

    if header["dataTypeCh2"] > 0:
        header["channels"] = ["Height", "Phase"]
        if str(ch).lower() == "phase":
            im = im[header["numberFramesCurrent"] : n_raw]
            header["Ch"] = "Phase"
        else:
            im = im[: header["numberFramesCurrent"]]
            header["Ch"] = "Height"
    else:
        header["channels"] = ["Height"]
        im = im[: header["numberFramesCurrent"]]
        header["Ch"] = "Height"

    if matlab_layout:
        im = np.moveaxis(im, 0, -1)

    return np.asarray(im, dtype=np.float64), header


__all__ = ["open_asd"]
