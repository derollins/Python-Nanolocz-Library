"""
Matplotlib label overlays for NanoLocz-compatible image display.

This module ports NanoLocz ``drawLabels.m``.  It overlays an optional scale bar
and optional time stamp onto a Matplotlib Axes.

The MATLAB version draws directly into an axes object.  The Python version does
the same with Matplotlib primitives and accepts either dict-like or object-like
``ImageInfo`` and ``Settings`` inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get(obj: Any, path: str, default: Any = None) -> Any:
    """Read nested dict/object values using dotted paths."""
    current = obj
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


def _set_default_settings(img_width: int) -> dict[str, Any]:
    """Return NanoLocz-like default label settings."""
    ref_width = img_width * 1.5
    base_scale_font = 25.0
    base_time_font = 25.0

    return {
        "label": {
            "ScaleBar": {
                "position": "Bottom Right",
                "offimage": False,
                "font": int(round(base_scale_font * (img_width / ref_width))),
                "color": "w",
                "baronly": False,
            },
            "TimeStamp": {
                "position": "Top Left",
                "offimage_2": False,
                "font": int(round(base_time_font * (img_width / ref_width))),
                "color": "w",
                "decimalPlaces": 2,
                "showUnits": True,
                "format": "seconds",
            },
        }
    }


def _frame_value(values: Any, frame: int, default: float = 1.0) -> float:
    """Return scalar or per-frame metadata value using MATLAB-style frame index."""
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size >= frame:
            return float(arr[frame - 1])
        if arr.size > 0:
            return float(arr[0])
    except Exception:
        pass
    return float(default)


def _format_time(
    time_value: float,
    fmt: str,
    decimals: int,
    show_units: bool,
) -> str:
    """Format a time stamp using NanoLocz options."""
    fmt_lc = fmt.lower()

    if fmt_lc == "seconds":
        text = f"{time_value:.{decimals}f}"
        return f"{text} s" if show_units else text

    if fmt_lc == "milliseconds":
        text = f"{time_value * 1000.0:.{decimals}f}"
        return f"{text} ms" if show_units else text

    if fmt_lc in {"minutes:seconds", "min:sec"}:
        minutes = int(np.floor(time_value / 60.0))
        seconds = int(np.floor(time_value % 60.0))
        text = f"{minutes:02d}:{seconds:02d}"
        return f"{text} min" if show_units else text

    if fmt_lc == "hh:mm:ss":
        hours = int(np.floor(time_value / 3600.0))
        minutes = int(np.floor((time_value % 3600.0) / 60.0))
        seconds = int(np.floor(time_value % 60.0))
        text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{text} h" if show_units else text

    text = f"{time_value:.{decimals}f}"
    return f"{text} s" if show_units else text


def _position_coords(
    position: str,
    rows: int,
    cols: int,
    *,
    offimage: bool,
    for_bar: bool,
    bar_length_pix: float = 0.0,
) -> tuple[Any, Any, str]:
    """Compute label coordinates for a position string."""
    pos = position

    if for_bar:
        if pos == "Top Right":
            x = [round(cols * 0.95) - bar_length_pix, round(cols * 0.95)]
            y = [-rows * 0.07, -rows * 0.07] if offimage else list(np.round([rows * 0.05, rows * 0.05]))
            align = "center"
        elif pos == "Top Left":
            x = [round(cols * 0.05), round(cols * 0.05) + bar_length_pix]
            y = [-rows * 0.07, -rows * 0.07] if offimage else list(np.round([rows * 0.05, rows * 0.05]))
            align = "center"
        elif pos == "Bottom Left":
            x = [round(cols * 0.05), round(cols * 0.05) + bar_length_pix]
            y = [rows + rows * 0.07, rows + rows * 0.07] if offimage else list(np.round([rows * 0.95, rows * 0.95]))
            align = "center"
        else:
            x = [round(cols * 0.95) - bar_length_pix, round(cols * 0.95)]
            y = [rows + rows * 0.07, rows + rows * 0.07] if offimage else list(np.round([rows * 0.95, rows * 0.95]))
            align = "center"
        return x, y, align

    if pos == "Top Right":
        x = round(cols * 0.97)
        y = -rows * 0.05 if offimage else round(rows * 0.05)
        align = "right"
    elif pos == "Top Left":
        x = round(cols * 0.03)
        y = -rows * 0.05 if offimage else round(rows * 0.05)
        align = "left"
    elif pos == "Bottom Left":
        x = round(cols * 0.03)
        y = rows + rows * 0.05 if offimage else round(rows * 0.95)
        align = "left"
    else:
        x = round(cols * 0.97)
        y = rows + rows * 0.05 if offimage else round(rows * 0.95)
        align = "right"

    return x, y, align


def draw_labels(
    ax: Any,
    img: np.ndarray,
    frame: int,
    image_info: Any,
    scalebar: bool = True,
    timescale: bool = True,
    settings: Any | None = None,
) -> None:
    """Draw scale bar and timestamp labels on a Matplotlib Axes.

    Parameters
    ----------
    ax:
        Matplotlib Axes object.
    img:
        Displayed image or stack.  Only spatial dimensions are used.
    frame:
        MATLAB-style 1-based frame index.
    image_info:
        Object/dict with ``PixelPerNm``, ``time`` and ``n`` fields.
    scalebar, timescale:
        Toggle scale bar and time stamp.
    settings:
        Optional NanoLocz-style nested settings object/dict.
    """
    arr = np.asarray(img)
    rows, cols = arr.shape[:2]

    if settings is None:
        settings = _set_default_settings(cols)

    if scalebar:
        try:
            px_per_nm = _frame_value(_get(image_info, "PixelPerNm", 1.0), frame, 1.0)
            if px_per_nm == 0:
                px_per_nm = 1.0

            sb_nm = cols / px_per_nm
            # Python approximation of MATLAB round(..., 1, "significant").
            scale_bar = 0.0 if sb_nm == 0 else round(sb_nm / 5.0, -int(np.floor(np.log10(abs(sb_nm / 5.0)))))
            scale_bar_pix = scale_bar * px_per_nm

            pos = _get(settings, "label.ScaleBar.position", "Bottom Right")
            offimage = bool(_get(settings, "label.ScaleBar.offimage", False))
            font_size = int(_get(settings, "label.ScaleBar.font", 12))
            color = _get(settings, "label.ScaleBar.color", "w")
            baronly = bool(_get(settings, "label.ScaleBar.baronly", False))

            x, y, _ = _position_coords(
                pos,
                rows,
                cols,
                offimage=offimage,
                for_bar=True,
                bar_length_pix=scale_bar_pix,
            )

            ax.plot(x, y, linewidth=2, color=color)

            if offimage:
                if "Top" in pos:
                    ax.set_ylim(rows, -rows * 0.1)
                else:
                    ax.set_ylim(rows + rows * 0.1, 0)

            if not baronly:
                ss = 0.02 if rows < cols * 0.8 else 0.0
                label_y = y[0] - rows * (0.03 + ss)
                if scale_bar > 1000:
                    label_text = f"{scale_bar / 1000:g}µm"
                else:
                    label_text = f"{scale_bar:g}nm"

                ax.text(
                    float(np.mean(x)),
                    label_y,
                    label_text,
                    fontweight="bold",
                    fontsize=font_size,
                    color=color,
                    ha="center",
                )
        except Exception:
            pass

    n_frames = int(_frame_value(_get(image_info, "n", 1), 1, 1))
    if n_frames > 1 and timescale:
        try:
            time_value = _frame_value(_get(image_info, "time", [0.0]), frame, 0.0)

            pos = _get(settings, "label.TimeStamp.position", "Top Left")
            offimage = bool(_get(settings, "label.TimeStamp.offimage_2", False))
            font_size = int(_get(settings, "label.TimeStamp.font", 12))
            color = _get(settings, "label.TimeStamp.color", "w")
            decimals = int(_get(settings, "label.TimeStamp.decimalPlaces", 2))
            show_units = bool(_get(settings, "label.TimeStamp.showUnits", True))
            time_format = _get(settings, "label.TimeStamp.format", "seconds")

            x, y, align = _position_coords(
                pos,
                rows,
                cols,
                offimage=offimage,
                for_bar=False,
            )

            if offimage:
                if "Top" in pos:
                    ax.set_ylim(rows, -rows * 0.1)
                else:
                    ax.set_ylim(rows + rows * 0.1, 0)

            time_text = _format_time(time_value, time_format, decimals, show_units)
            ax.text(
                x,
                y,
                time_text,
                fontweight="bold",
                fontsize=font_size,
                color=color,
                ha=align,
            )
        except Exception:
            pass


# MATLAB-style alias.
drawLabels = draw_labels


__all__ = ["draw_labels", "drawLabels"]
