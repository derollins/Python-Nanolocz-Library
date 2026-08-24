"""
AFM colormap utilities for NanoLocz-compatible Python workflows.

This module ports NanoLocz ``afm_colormap.m``.  The MATLAB function loads
``AFM_luts_full.mat`` and returns one of several AFM-specific colormap matrices.
The Python port loads a lightweight NumPy archive ``afm_luts_full.npz`` by
default.  A MATLAB ``.mat`` file can still be supplied explicitly.

Usage
-----
>>> from pnanolocz.afm_colormap import afm_colormap
>>> cmap = afm_colormap("AFM gold")

The returned value is a NumPy array with shape ``(N, 3)`` or ``(N, 4)`` and
values in the ``0..1`` range.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


_NAME_TO_FIELD = {
    "afm brown": "AFM_brown",
    "afm dark gold": "AFM_Dark_Gold",
    "afm gold": "AFM_Gold",
    "orange hot": "AFM_orange",
    "fire": "AFM_fire",
    "rainbow": "Rainbow",
    "inferno": "inferno",
    "magma": "magma",
    "plasma": "plasma",
    "viridis": "viridis",
    "lafm color": "LAFMcolor",
}


def _normalize_colormap(arr: np.ndarray) -> FloatArray:
    """Return a colormap array in floating-point 0..1 RGB/RGBA format."""
    cmap = np.asarray(arr, dtype=np.float64)

    if cmap.ndim != 2 or cmap.shape[1] not in (3, 4):
        raise ValueError("colormap must have shape (N, 3) or (N, 4)")

    if np.nanmax(cmap) > 1.0:
        cmap = cmap / 255.0

    cmap = np.clip(cmap, 0.0, 1.0)
    return np.asarray(cmap, dtype=np.float64)


def _default_lut_path() -> Path:
    """Return the default package-local LUT archive path."""
    return Path(__file__).with_name("afm_luts_full.npz")


@lru_cache(maxsize=8)
def _load_luts_cached(path_str: str) -> dict[str, FloatArray]:
    """Load LUT arrays from ``.npz`` or MATLAB ``.mat`` files.

    Loading is cached because colormaps are often requested repeatedly during
    rendering.
    """
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"LUT file not found: {path}")

    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return {key: _normalize_colormap(data[key]) for key in data.files}

    if path.suffix.lower() == ".mat":
        try:
            import scipy.io as sio
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError("scipy is required to load MATLAB .mat LUT files") from exc

        mat = sio.loadmat(path)
        out: dict[str, FloatArray] = {}
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[1] in (3, 4):
                out[key] = _normalize_colormap(arr)
        return out

    raise ValueError("LUT path must be a .npz or .mat file")


def load_afm_luts(lut_path: str | Path | None = None) -> dict[str, FloatArray]:
    """Load all available AFM LUTs.

    Parameters
    ----------
    lut_path:
        Optional path to ``afm_luts_full.npz`` or ``AFM_luts_full.mat``.  If not
        supplied, the package-local ``afm_luts_full.npz`` is used.
    """
    path = Path(lut_path) if lut_path is not None else _default_lut_path()
    return _load_luts_cached(str(path.resolve()))


def afm_colormap(
    map_name: str,
    *,
    lut_path: str | Path | None = None,
    as_mpl: bool = False,
    name: str | None = None,
) -> FloatArray | Any:
    """Return a NanoLocz AFM colormap by name.

    Parameters
    ----------
    map_name:
        MATLAB-compatible name such as ``'AFM gold'``, ``'fire'``,
        ``'viridis'`` or ``'LAFM color'``.
    lut_path:
        Optional path to a ``.npz`` or ``.mat`` LUT file.
    as_mpl:
        If true, return a ``matplotlib.colors.ListedColormap`` instead of a raw
        NumPy array.
    name:
        Optional Matplotlib colormap name when ``as_mpl=True``.

    Returns
    -------
    ndarray or ListedColormap
        Colormap as a normalized array unless ``as_mpl=True``.

    MATLAB alignment notes
    ----------------------
    Unknown names fall back to Matplotlib's built-in colormaps where possible.
    If that also fails, ``viridis`` is returned, mirroring MATLAB's fallback to
    a default colormap.
    """
    normalized_name = " ".join(map_name.strip().lower().split())

    luts = {}
    try:
        luts = load_afm_luts(lut_path)
    except FileNotFoundError:
        # A user may still request a built-in Matplotlib colormap without having
        # the NanoLocz LUT archive installed.
        luts = {}

    field = _NAME_TO_FIELD.get(normalized_name)
    if field is not None and field in luts:
        cmap = luts[field]
    else:
        try:
            import matplotlib.pyplot as plt

            mpl_cmap = plt.get_cmap(normalized_name)
            cmap = mpl_cmap(np.linspace(0.0, 1.0, 256))[:, :3]
        except Exception:
            if "viridis" in luts:
                cmap = luts["viridis"]
            else:
                # Last-resort fallback that does not require the LUT archive.
                cmap = np.column_stack(
                    [
                        np.linspace(0.267, 0.993, 256),
                        np.linspace(0.004, 0.906, 256),
                        np.linspace(0.329, 0.144, 256),
                    ]
                )

    cmap = _normalize_colormap(cmap)

    if as_mpl:
        try:
            from matplotlib.colors import ListedColormap
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError("matplotlib is required when as_mpl=True") from exc
        return ListedColormap(cmap, name=name or normalized_name.replace(" ", "_"))

    return cmap


def apply_afm_colormap(ax: Any, map_name: str, *, lut_path: str | Path | None = None) -> Any:
    """Apply an AFM colormap to a Matplotlib Axes image.

    This is the closest Python analogue to MATLAB's ``colormap(c)`` call.  It
    updates every image artist attached to ``ax`` and returns the Matplotlib
    colormap object.
    """
    mpl_cmap = afm_colormap(map_name, lut_path=lut_path, as_mpl=True)

    for image_artist in getattr(ax, "images", []):
        image_artist.set_cmap(mpl_cmap)

    return mpl_cmap


__all__ = [
    "afm_colormap",
    "apply_afm_colormap",
    "load_afm_luts",
]
