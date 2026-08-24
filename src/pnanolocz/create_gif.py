"""
GIF/PNG export utilities for NanoLocz-compatible AFM image stacks.

This updated version uses ``draw_labels_on_image`` from ``draw_labels.py`` when
labels are requested, matching NanoLocz ``CreateGif.m`` more closely than the
previous simplified label overlay.

Python default stack convention is frame-first ``(frames, rows, cols)``.
Use :func:`create_gif_matlab` for MATLAB-style ``(rows, cols, frames)`` stacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from pnanolocz.afm_colormap import afm_colormap
except Exception:  # pragma: no cover - allows standalone use
    try:
        from afm_colormap import afm_colormap  # type: ignore
    except Exception:
        afm_colormap = None  # type: ignore[assignment]

try:
    from pnanolocz.draw_labels import draw_labels_on_image
except Exception:  # pragma: no cover - allows standalone use
    try:
        from draw_labels import draw_labels_on_image  # type: ignore
    except Exception:
        draw_labels_on_image = None  # type: ignore[assignment]


FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _as_frame_first(stack: np.ndarray, frame_axis: int) -> tuple[FloatArray, bool, int]:
    """Convert a 2-D image or 3-D stack to frame-first layout."""
    arr = np.asarray(stack, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0

    if arr.ndim != 3:
        raise ValueError("stack must be 2D or 3D")

    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _default_colormap() -> np.ndarray:
    """Return NanoLocz's default-ish AFM colormap."""
    if afm_colormap is not None:
        try:
            return np.asarray(afm_colormap("AFM gold"), dtype=np.float64)
        except Exception:
            pass

    try:
        import matplotlib.pyplot as plt

        return np.asarray(plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3], dtype=np.float64)
    except Exception:
        gray = np.linspace(0, 1, 256)
        return np.column_stack([gray, gray, gray]).astype(np.float64)


def _normalize_cmap(cmap: np.ndarray | None) -> np.ndarray:
    """Normalize colormap to an ``N x 3`` float array."""
    if cmap is None:
        cm = _default_colormap()
    else:
        cm = np.asarray(cmap, dtype=np.float64)

    if cm.ndim != 2 or cm.shape[1] not in (3, 4):
        raise ValueError("cmap must have shape (N, 3) or (N, 4)")

    if cm.shape[1] == 4:
        cm = cm[:, :3]

    if np.nanmax(cm) > 1.0:
        cm = cm / 255.0

    return np.clip(cm, 0.0, 1.0)


def _contrast_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Apply MATLAB-style percentile contrast stretching to uint8."""
    arr = np.asarray(frame, dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.5))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    clipped = np.clip(arr, low, high)
    norm = 255.0 * (clipped - low) / (high - low)
    norm[~np.isfinite(norm)] = 0.0

    return np.asarray(np.clip(norm, 0, 255), dtype=np.uint8)


def _apply_colormap(gray_uint8: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """Map a uint8 grayscale image through an RGB colormap."""
    cm = _normalize_cmap(cmap)
    idx = np.asarray(gray_uint8, dtype=np.uint8)
    lut_idx = np.round(idx.astype(np.float64) / 255.0 * (cm.shape[0] - 1)).astype(np.int64)
    rgb = cm[lut_idx, :3]
    return np.asarray(np.round(rgb * 255.0), dtype=np.uint8)


def create_gif(
    stack: np.ndarray,
    output_base_path: str | Path,
    image_info: Any = None,
    labels: bool = False,
    cmap: np.ndarray | None = None,
    *,
    frame_axis: int = 0,
    delay_time: float = 0.1,
    label_settings: Any = None,
    scalebar: bool = True,
    timescale: bool = True,
) -> Path:
    """Export an AFM image stack as GIF or a single frame as PNG.

    Parameters
    ----------
    stack:
        2-D image or 3-D image stack.  Default 3-D layout is
        ``(frames, rows, cols)``.
    output_base_path:
        Output path without extension.  ``.gif`` or ``.png`` is added
        automatically.
    image_info:
        Optional metadata used for labels.  Expected fields include
        ``PixelPerNm``, ``time``, and ``n``.
    labels:
        If true, draw scale bar and/or timestamp labels.
    cmap:
        Optional RGB colormap array.  If omitted, ``AFM gold`` is used when
        available.
    frame_axis:
        Frame axis for 3-D input.  Use ``-1`` for MATLAB layout.
    delay_time:
        GIF frame delay in seconds.
    label_settings:
        Optional settings object/dict compatible with ``draw_labels.py``.
    scalebar, timescale:
        Toggle scale bar and timestamp independently when ``labels=True``.

    Returns
    -------
    Path
        Path to the saved GIF or PNG.
    """
    frames, _, _ = _as_frame_first(stack, frame_axis=frame_axis)
    cm = _normalize_cmap(cmap)

    output_base = Path(output_base_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    rgb_frames: list[np.ndarray] = []

    for frame_idx in range(frames.shape[0]):
        gray = _contrast_to_uint8(frames[frame_idx])
        rgb = _apply_colormap(gray, cm)

        if labels and image_info is not None and draw_labels_on_image is not None:
            rgb = draw_labels_on_image(
                rgb,
                frame=frame_idx,
                image_info=image_info,
                scalebar=scalebar,
                timescale=timescale,
                settings=label_settings,
            )

        rgb_frames.append(rgb)

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("Pillow is required for create_gif") from exc

    if len(rgb_frames) == 1:
        out_path = output_base.with_suffix(".png")
        Image.fromarray(rgb_frames[0]).save(out_path)
        return out_path

    out_path = output_base.with_suffix(".gif")
    pil_frames = [Image.fromarray(frame) for frame in rgb_frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(round(delay_time * 1000.0)),
    )

    return out_path


def create_gif_matlab(
    stack: np.ndarray,
    output_base_path: str | Path,
    image_info: Any = None,
    labels: bool = False,
    cmap: np.ndarray | None = None,
) -> Path:
    """MATLAB-layout wrapper for :func:`create_gif`.

    Assumes 3-D stacks are shaped ``(rows, cols, frames)``.
    """
    return create_gif(
        stack,
        output_base_path,
        image_info=image_info,
        labels=labels,
        cmap=cmap,
        frame_axis=-1,
    )


# MATLAB-style alias.
CreateGif = create_gif_matlab


__all__ = [
    "create_gif",
    "create_gif_matlab",
    "CreateGif",
]
