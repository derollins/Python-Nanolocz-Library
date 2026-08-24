"""
Interactive LAFM plotter for NanoLocz-compatible Python workflows.

This module ports NanoLocz ``LAFM_plotter.m`` using Matplotlib widgets.
It displays an image and provides lower/upper sliders that adjust the displayed
dynamic range.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def lafm_plotter(lafm_full: np.ndarray) -> Any:
    """Display a LAFM image with interactive lower/upper range sliders.

    Parameters
    ----------
    lafm_full:
        2-D probability map or rendered image.

    Returns
    -------
    tuple
        ``(fig, ax, image_artist, lower_slider, upper_slider)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    t = np.asarray(lafm_full, dtype=np.float64)
    mx = float(np.nanmax(t)) if np.isfinite(t).any() else 1.0
    if mx == 0:
        mx = 1.0

    lower_init = 0.0
    upper_init = 1.0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)

    image_artist = ax.imshow(
        (t - lower_init * mx) / (upper_init * mx - lower_init * mx),
        cmap="gray",
        interpolation="nearest",
    )
    ax.set_axis_off()

    ax_lower = fig.add_axes([0.20, 0.04, 0.25, 0.03])
    ax_upper = fig.add_axes([0.55, 0.04, 0.25, 0.03])

    lower_slider = Slider(ax_lower, "Lower", 0.0, 1.0, valinit=lower_init)
    upper_slider = Slider(ax_upper, "Upper", 0.0, 1.0, valinit=upper_init)

    def update(_value: float) -> None:
        lower = float(lower_slider.val)
        upper = float(upper_slider.val)

        if upper <= lower:
            upper = min(1.0, lower + 0.01)
            upper_slider.set_val(upper)

        image_artist.set_data((t - lower * mx) / (upper * mx - lower * mx))
        fig.canvas.draw_idle()

    lower_slider.on_changed(update)
    upper_slider.on_changed(update)

    plt.show(block=False)

    return fig, ax, image_artist, lower_slider, upper_slider


# MATLAB-style alias.
LAFM_plotter = lafm_plotter


__all__ = ["lafm_plotter", "LAFM_plotter"]
