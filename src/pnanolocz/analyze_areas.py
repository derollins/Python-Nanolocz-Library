"""
Region/mask analysis tools for NanoLocz-compatible AFM workflows.

This module ports NanoLocz ``AnalyzeAreas.m``.  It analyzes labelled regions in
2-D images or 3-D image stacks and returns two pandas DataFrames:

- ``T1``: per-object measurements.
- ``T2``: per-frame summary metrics.

Python conventions
------------------
Default stack layout is frame-first:

    M:   (frames, rows, cols)
    img: (frames, rows, cols)

Use :func:`analyze_areas_matlab` for MATLAB-style layout:

    M:   (rows, cols, frames)
    img: (rows, cols, frames)

The input mask ``M`` is interpreted as a validity/foreground mask where non-zero
or ``True`` pixels belong to measured regions.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


# Mapping from MATLAB regionprops names to internal normalized names.  Some
# properties are computed manually to keep MATLAB-style column names.
_PROP_ALIASES = {
    "area": "Area",
    "centroid": "Centroid",
    "orientation": "Orientation",
    "circularity": "Circularity",
    "perimeter": "Perimeter",
    "minintensity": "MinIntensity",
    "maxintensity": "MaxIntensity",
    "meanintensity": "MeanIntensity",
    "pixelvalues": "PixelValues",
}


def _normalize_props(props: Iterable[str]) -> list[str]:
    """Normalize MATLAB property names while preserving output order."""
    out: list[str] = []
    for prop in props:
        key = str(prop).replace(" ", "").lower()
        normalized = _PROP_ALIASES.get(key, str(prop))
        if normalized not in out:
            out.append(normalized)
    return out


def _as_frame_first(arr: np.ndarray, frame_axis: int) -> tuple[np.ndarray, bool, int]:
    """Convert a 2-D or 3-D array to frame-first layout."""
    array = np.asarray(arr)

    if array.ndim == 2:
        return array[np.newaxis, :, :], True, 0

    if array.ndim != 3:
        raise ValueError("input must be 2D or 3D")

    axis = int(frame_axis) % 3
    return np.moveaxis(array, axis, 0), False, axis


def _scale_for_frame(scale: float | Iterable[float], frame_idx: int) -> float:
    """Return the pixel scale for a frame, matching MATLAB scalar/vector logic."""
    scale_arr = np.asarray(scale, dtype=np.float64).ravel()

    if scale_arr.size == 0:
        return 1.0

    if scale_arr[0] == 0:
        scale_arr[0] = 1.0

    if scale_arr.size > 1:
        idx = min(frame_idx, scale_arr.size - 1)
        return float(scale_arr[idx])

    return float(scale_arr[0])


def _time_for_frame(
    time: float | Iterable[float] | None,
    speed: float | None,
    frame_idx: int,
) -> float:
    """Return frame time in seconds, with MATLAB-like fallbacks."""
    if time is None:
        if speed not in (None, 0):
            return float(frame_idx + 1) / float(speed)
        return float(frame_idx + 1)

    time_arr = np.asarray(time, dtype=np.float64).ravel()

    if time_arr.size == 0:
        return float(frame_idx + 1)

    if time_arr.size == 1 and time_arr[0] == 0:
        return float(frame_idx + 1)

    if frame_idx < time_arr.size:
        return float(time_arr[frame_idx])

    if speed not in (None, 0):
        return float(frame_idx + 1) / float(speed)

    return float(frame_idx + 1)


def _safe_rms(values: np.ndarray, center: float) -> float:
    """Compute root-mean-square deviation from ``center``."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0 or not np.isfinite(center):
        return np.nan

    return float(np.sqrt(np.mean((vals - center) ** 2)))


def _region_rows_for_frame(
    mask_frame: np.ndarray,
    image_frame: np.ndarray,
    props: list[str],
    *,
    frame_number: int,
    frame_time: float,
    scale_nm: float,
) -> list[dict[str, Any]]:
    """Compute per-object rows for one frame."""
    foreground = np.asarray(mask_frame).astype(bool)
    intensity = np.asarray(image_frame, dtype=np.float64)

    labels = label(foreground, connectivity=2)
    regions = regionprops(labels, intensity_image=intensity)

    rows: list[dict[str, Any]] = []

    for region in regions:
        row: dict[str, Any] = {
            "Frame": frame_number,
            "Time (s)": frame_time,
        }

        pixel_values = intensity[tuple(region.coords.T)]
        finite_pixel_values = pixel_values[np.isfinite(pixel_values)]

        if "Area" in props:
            area_px = float(region.area)
            row["Area"] = area_px * (scale_nm * scale_nm)
            row["Radius"] = np.sqrt(area_px / np.pi) * scale_nm

        if "Centroid" in props:
            # skimage returns (row, col); MATLAB Centroid is (x, y).
            row["x"] = float(region.centroid[1])
            row["y"] = float(region.centroid[0])

        if "Orientation" in props:
            row["Orientation"] = float(region.orientation)

        if "Circularity" in props:
            # MATLAB regionprops circularity is 4*pi*Area/Perimeter^2.
            perim = float(region.perimeter)
            row["Circularity"] = (
                float(4.0 * np.pi * region.area / (perim**2))
                if perim > 0
                else np.nan
            )

        if "Perimeter" in props:
            row["Perimeter"] = float(region.perimeter) * scale_nm

        if "MinIntensity" in props:
            row["MinIntensity"] = (
                float(np.nanmin(finite_pixel_values)) if finite_pixel_values.size else np.nan
            )

        if "MaxIntensity" in props:
            row["MaxIntensity"] = (
                float(np.nanmax(finite_pixel_values)) if finite_pixel_values.size else np.nan
            )

        if "MeanIntensity" in props:
            row["MeanIntensity"] = (
                float(np.nanmean(finite_pixel_values)) if finite_pixel_values.size else np.nan
            )

        if "PixelValues" in props:
            row["PixelValues"] = finite_pixel_values.copy()

        if "Area" in props and "MeanIntensity" in props:
            row["Vol. (nm3)"] = row["Area"] * row["MeanIntensity"]

        if "PixelValues" in props and "MeanIntensity" in props:
            row["RMS"] = _safe_rms(finite_pixel_values, row.get("MeanIntensity", np.nan))

        rows.append(row)

    return rows


def _summary_for_frame(
    mask_frame: np.ndarray,
    image_frame: np.ndarray,
    *,
    frame_number: int,
    frame_time: float,
    scale_nm: float,
) -> dict[str, float]:
    """Compute MATLAB-style per-frame summary metrics."""
    mask = np.asarray(mask_frame).astype(float)
    image = np.asarray(image_frame, dtype=np.float64)

    rows, cols = image.shape
    foreground = mask >= 0.5
    background = ~foreground

    n_foreground = int(np.sum(foreground))
    n_background = int(np.sum(background))

    area_percent = float(np.sum(foreground) / float(rows * cols) * 100.0)

    mask_values = image[foreground]
    background_values = image[background]

    mean_mask = float(np.nanmean(mask_values)) if n_foreground > 0 else np.nan
    mean_background = (
        float(np.nanmean(background_values)) if n_background > 0 else np.nan
    )

    rms_mask = _safe_rms(mask_values, mean_mask)
    rms_background = _safe_rms(background_values, mean_background)

    volume = (
        mean_mask * n_foreground * (scale_nm * scale_nm)
        if n_foreground > 0 and np.isfinite(mean_mask)
        else np.nan
    )

    return {
        "Frame": frame_number,
        "Time (s)": frame_time,
        "Area (%)": area_percent,
        "Mean Mask (nm)": mean_mask,
        "RMS Mask (nm)": rms_mask,
        "Mean Background (nm)": mean_background,
        "RMS background (nm)": rms_background,
        "Vol. (nm3)": volume,
    }


def analyze_areas(
    M: np.ndarray,
    img: np.ndarray,
    props: Iterable[str],
    scale: float | Iterable[float] = 1.0,
    time: float | Iterable[float] | None = None,
    speed: float | None = None,
    *,
    frame_axis: int = 0,
    drop_pixel_values: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze segmented regions and per-frame mask statistics.

    Parameters
    ----------
    M:
        2-D or 3-D region mask.  Non-zero pixels define foreground regions.
    img:
        2-D or 3-D grayscale image data corresponding to ``M``.
    props:
        MATLAB-style region property names, for example ``['Area',
        'Centroid', 'MeanIntensity', 'PixelValues']``.
    scale:
        Pixel size in nm.  May be scalar or one value per frame.
    time:
        Time vector in seconds.  If unavailable, frame numbers are used.
    speed:
        Frame rate used as a fallback if ``time`` is too short.
    frame_axis:
        Frame axis for 3-D inputs.  Use ``-1`` for MATLAB layout.
    drop_pixel_values:
        If true, remove the large ``PixelValues`` object column from ``T1``,
        matching MATLAB's final ``removevars`` call.

    Returns
    -------
    T1, T2:
        Per-object and per-frame summary tables as pandas DataFrames.
    """
    mask_stack, mask_was_2d, _ = _as_frame_first(M, frame_axis=frame_axis)
    image_stack, image_was_2d, _ = _as_frame_first(img, frame_axis=frame_axis)

    if mask_stack.shape != image_stack.shape:
        raise ValueError("M and img must have matching shapes after frame-axis normalization")

    requested_props = _normalize_props(props)
    n_frames = image_stack.shape[0]

    object_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for frame_idx in range(n_frames):
        scale_nm = _scale_for_frame(scale, frame_idx)
        frame_time = _time_for_frame(time, speed, frame_idx)
        frame_number = frame_idx + 1  # MATLAB-style frame numbering in outputs.

        object_rows.extend(
            _region_rows_for_frame(
                mask_stack[frame_idx],
                image_stack[frame_idx],
                requested_props,
                frame_number=frame_number,
                frame_time=frame_time,
                scale_nm=scale_nm,
            )
        )

        summary_rows.append(
            _summary_for_frame(
                mask_stack[frame_idx],
                image_stack[frame_idx],
                frame_number=frame_number,
                frame_time=frame_time,
                scale_nm=scale_nm,
            )
        )

    T1 = pd.DataFrame(object_rows)
    T2 = pd.DataFrame(summary_rows)

    if not T1.empty:
        if "Track id" not in T1.columns:
            T1["Track id"] = 0

        # MATLAB removes PixelValues after using them for RMS.
        if drop_pixel_values and "PixelValues" in T1.columns:
            T1 = T1.drop(columns=["PixelValues"])

        # Prefer MATLAB-like ordering for common columns.
        preferred = [
            "Frame",
            "Time (s)",
            "Area",
            "x",
            "y",
            "RMS",
            "Radius",
            "Vol. (nm3)",
            "Track id",
        ]
        ordered = [col for col in preferred if col in T1.columns]
        ordered.extend([col for col in T1.columns if col not in ordered])
        T1 = T1.loc[:, ordered]

    # Preserve MATLAB summary column order.
    summary_order = [
        "Frame",
        "Time (s)",
        "Area (%)",
        "Mean Mask (nm)",
        "RMS Mask (nm)",
        "Mean Background (nm)",
        "RMS background (nm)",
        "Vol. (nm3)",
    ]
    T2 = T2.loc[:, summary_order]

    return T1, T2


def analyze_areas_matlab(
    M: np.ndarray,
    img: np.ndarray,
    props: Iterable[str],
    scale: float | Iterable[float] = 1.0,
    time: float | Iterable[float] | None = None,
    speed: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MATLAB-layout wrapper for :func:`analyze_areas`.

    Assumes 3-D arrays are shaped ``(rows, cols, frames)``.
    """
    return analyze_areas(
        M,
        img,
        props,
        scale=scale,
        time=time,
        speed=speed,
        frame_axis=-1,
    )


# MATLAB-style alias.
AnalyzeAreas = analyze_areas_matlab


__all__ = [
    "analyze_areas",
    "analyze_areas_matlab",
    "AnalyzeAreas",
]
