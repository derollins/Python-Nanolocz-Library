"""
LAFM rendering utilities for NanoLocz-compatible localization AFM workflows.

This module ports ``LAFM_renderer.m``, ``LAFM_Movie_renderer.m`` and a small
Matplotlib analogue of ``LAFM_plotter.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from pnanolocz.afm_colormap import afm_colormap
except Exception:  # pragma: no cover
    try:
        from afm_colormap import afm_colormap  # type: ignore
    except Exception:
        afm_colormap = None  # type: ignore[assignment]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _resolve_colormap(fullcolormap: Any) -> FloatArray:
    """Return an RGB colormap array."""
    if isinstance(fullcolormap, str):
        if afm_colormap is not None:
            try:
                return np.asarray(afm_colormap(fullcolormap), dtype=np.float64)[:, :3]
            except Exception:
                pass
        import matplotlib.pyplot as plt
        return np.asarray(plt.get_cmap(fullcolormap)(np.linspace(0, 1, 256))[:, :3], dtype=np.float64)

    cmap = np.asarray(fullcolormap, dtype=np.float64)
    if cmap.ndim != 2 or cmap.shape[1] < 3:
        raise ValueError("fullcolormap must be a name or an Nx3/Nx4 array")
    cmap = cmap[:, :3]
    if np.nanmax(cmap) > 1:
        cmap = cmap / 255.0
    return np.asarray(np.clip(cmap, 0, 1), dtype=np.float64)


def _round_significant(values: np.ndarray | float, digits: int = 3) -> np.ndarray:
    """Round to significant figures, similar to MATLAB round(...,'significant')."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr)
    finite_nonzero = np.isfinite(arr) & (arr != 0)
    selected = arr[finite_nonzero]
    decimals = digits - np.floor(np.log10(np.abs(selected))).astype(int) - 1
    scale = np.power(10.0, decimals.astype(np.float64))
    scaled = selected * scale
    out[finite_nonzero] = (
        np.sign(scaled) * np.floor(np.abs(scaled) + 0.5) / scale
    )
    out[np.isposinf(arr)] = np.inf
    out[np.isneginf(arr)] = -np.inf
    out[np.isnan(arr)] = np.nan
    return out


def _exclude_outliers_mean(values: np.ndarray, z: float = 3.0) -> np.ndarray:
    """Approximate MATLAB rmoutliers(values,'mean')."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return vals
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    if sd == 0:
        return vals
    return vals[np.abs(vals - mu) <= z * sd]


def _prepare_locs(locs: np.ndarray, expand: float, side: bool = False) -> tuple[FloatArray, tuple[int, int]]:
    """Shift localizations to positive coordinates and expand to pixel grid."""
    locs_arr = np.asarray(locs, dtype=np.float64).copy()
    locs_arr[:, 0] = locs_arr[:, 0] - np.nanmin(locs_arr[:, 0]) + 1
    locs_arr[:, 1] = locs_arr[:, 1] - np.nanmin(locs_arr[:, 1]) + 1

    if side:
        explocs = np.column_stack([np.round(locs_arr[:, 1:3] * expand), locs_arr[:, 2:]])
    else:
        explocs = np.column_stack([np.round(locs_arr[:, 0:2] * expand), locs_arr[:, 2:]])

    explocs = explocs[~np.any(np.isnan(explocs), axis=1)]
    image_size = (int(np.nanmax(explocs[:, 1])) + 5, int(np.nanmax(explocs[:, 0])) + 5)
    return np.asarray(explocs, dtype=np.float64), image_size


def _color_limits(explocs: np.ndarray, colorlimits: np.ndarray, mode: str) -> tuple[float, float]:
    """Resolve z color limits from mode."""
    limits = np.asarray(colorlimits, dtype=np.float64).copy()
    mode_lc = str(mode).lower()

    if mode_lc == "max min":
        limits[0] = _round_significant(np.nanmin(explocs[:, 2]), 3).item()
        limits[1] = _round_significant(np.nanmax(explocs[:, 2]), 3).item()
    elif mode_lc == "exc outliers":
        B = _exclude_outliers_mean(explocs[:, 2])
        limits[0] = _round_significant(np.nanmin(B), 3).item()
        limits[1] = _round_significant(np.nanmax(B), 3).item()
    elif mode_lc == "manual":
        pass
    else:
        raise ValueError("colorlimit_mode must be 'Max Min', 'Exc outliers', or 'Manual'")

    if limits[1] == limits[0]:
        limits[1] = limits[0] + 1.0

    return float(limits[0]), float(limits[1])


def _render_subset(
    locs_subset: np.ndarray,
    color_index: np.ndarray,
    colormap: np.ndarray,
    image_size: tuple[int, int],
    img_gus: float,
    expand: float,
    prob: bool,
) -> FloatArray:
    """Render one LAFM image/movie frame."""
    sigma = float(img_gus) * float(expand) / 2.0

    correction = np.zeros((5, 5), dtype=np.float64)
    correction[2, 2] = 1.0
    correction = gaussian_filter(
        correction,
        sigma=sigma,
        mode="nearest",
        truncate=2.0,
    )
    correction_max = float(np.max(correction)) if np.max(correction) != 0 else 1.0

    if prob:
        out = np.zeros(image_size, dtype=np.float64)
    else:
        out = np.zeros((image_size[0], image_size[1], 3), dtype=np.float64)

    n_colors = colormap.shape[0]

    for i in range(n_colors):
        render = np.zeros(image_size, dtype=np.float64)

        if i == 0:
            pos = color_index < 2
        elif i > n_colors - 2:
            pos = color_index > n_colors - 1
        else:
            pos = color_index == (i + 1)

        sub = locs_subset[pos, :]
        if sub.size == 0:
            continue

        for row in sub:
            x = int(row[0])
            y = int(row[1])
            if 1 <= y <= image_size[0] and 1 <= x <= image_size[1]:
                # MATLAB uses assignment, not accumulation: multiple
                # localizations rounded to the same render pixel form one
                # occupied point.
                render[y - 1, x - 1] = 1.0

        render = gaussian_filter(
            render,
            sigma=sigma,
            mode="nearest",
            truncate=2.0,
        ) / correction_max

        if prob:
            out += render
        else:
            color = colormap[i, :3]
            for channel in range(3):
                out[:, :, channel] += render * color[channel]

    return np.asarray(out, dtype=np.float64)


def lafm_renderer(
    locs: np.ndarray,
    img_gus: float,
    expand: float,
    fullcolormap: Any,
    prob: bool | int,
    colorlimits: np.ndarray | list[float],
    colorlimit_mode: str,
) -> tuple[FloatArray, np.ndarray]:
    """Render a static LAFM image from localizations."""
    cmap = _resolve_colormap(fullcolormap)
    explocs, image_size = _prepare_locs(locs, expand, side=False)

    zmin, zmax = _color_limits(explocs, np.asarray(colorlimits, dtype=np.float64), colorlimit_mode)
    zlims = np.asarray([zmin, zmax], dtype=np.float64)

    xp = np.linspace(zmin, zmax, cmap.shape[0])
    color_index = np.round(np.interp(explocs[:, 2], xp, np.arange(1, cmap.shape[0] + 1), left=1, right=cmap.shape[0]))

    image = _render_subset(
        explocs,
        color_index,
        cmap,
        image_size,
        img_gus=img_gus,
        expand=expand,
        prob=bool(prob),
    )

    return np.asarray(image, dtype=np.float64), zlims


def lafm_movie_renderer(
    locs: np.ndarray,
    img_gus: float,
    expand: float,
    fullcolormap: Any,
    prob: bool | int,
    colorlimits: np.ndarray | list[float],
    colorlimit_mode: str,
    window: float,
    slide: float,
) -> tuple[list[FloatArray], np.ndarray, np.ndarray]:
    """Render a LAFM movie from localizations over frame windows."""
    locs_arr = np.asarray(locs, dtype=np.float64)
    total_parts = float(np.nanmax(locs_arr[:, 4]))

    if slide > 0:
        n = int(round((total_parts - window) / slide) + 1)
    else:
        n = int(round(total_parts / window))
    n = max(n, 1)

    cmap = _resolve_colormap(fullcolormap)
    explocs, image_size = _prepare_locs(locs_arr, expand, side=False)
    zmin, zmax = _color_limits(explocs, np.asarray(colorlimits, dtype=np.float64), colorlimit_mode)
    zlims = np.asarray([zmin, zmax], dtype=np.float64)

    xp = np.linspace(zmin, zmax, cmap.shape[0])
    color_index = np.round(np.interp(explocs[:, 2], xp, np.arange(1, cmap.shape[0] + 1), left=1, right=cmap.shape[0]))

    frames: list[FloatArray] = []
    times = np.full(n, np.nan, dtype=np.float64)

    for jj in range(1, n + 1):
        if slide > 0:
            pos = (explocs[:, 4] > (jj - 1) * slide) & (explocs[:, 4] <= window + (jj - 1) * slide)
        else:
            pos = (explocs[:, 4] > (jj - 1) * window) & (explocs[:, 4] <= jj * window)

        movlocs = explocs[pos, :]
        mov_color = color_index[pos]

        if movlocs.size and movlocs.shape[1] >= 7:
            times[jj - 1] = np.nanmin(movlocs[:, 6])

        frames.append(
            _render_subset(
                movlocs,
                mov_color,
                cmap,
                image_size,
                img_gus=img_gus,
                expand=expand,
                prob=bool(prob),
            )
        )

    return frames, zlims, times


def lafm_plotter(lafm_full: np.ndarray) -> Any:
    """Interactive Matplotlib viewer similar to MATLAB ``LAFM_plotter``."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    t = np.asarray(lafm_full, dtype=np.float64)
    mx = float(np.nanmax(t)) if np.isfinite(t).any() else 1.0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img_artist = ax.imshow(t, cmap="gray", vmin=0, vmax=mx)

    ax_lower = plt.axes([0.2, 0.05, 0.25, 0.03])
    ax_upper = plt.axes([0.55, 0.05, 0.25, 0.03])
    s_lower = Slider(ax_lower, "Lower", 0.0, 1.0, valinit=0.0)
    s_upper = Slider(ax_upper, "Upper", 0.0, 1.0, valinit=1.0)

    def update(_val: float) -> None:
        lower = s_lower.val
        upper = max(s_upper.val, lower + 0.01)
        if upper != s_upper.val:
            s_upper.set_val(upper)
        img_artist.set_data((t - lower * mx) / (upper * mx - lower * mx))
        fig.canvas.draw_idle()

    s_lower.on_changed(update)
    s_upper.on_changed(update)

    return fig


LAFM_renderer = lafm_renderer
LAFM_Movie_renderer = lafm_movie_renderer
LAFM_plotter = lafm_plotter

__all__ = [
    "lafm_renderer",
    "lafm_movie_renderer",
    "lafm_plotter",
    "LAFM_renderer",
    "LAFM_Movie_renderer",
    "LAFM_plotter",
]
