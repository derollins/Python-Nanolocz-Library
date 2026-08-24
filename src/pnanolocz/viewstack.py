"""
Interactive image-stack viewer for NanoLocz.

This module ports the user-facing behavior of MATLAB ``viewstack.m`` using
Matplotlib.  It displays a 3-D stack slice-by-slice and supports:

- slider navigation
- mouse wheel navigation
- left/right arrow navigation
- up/down arrow contrast adjustment
- optional mask overlay
- optional label drawing callback

Python stack convention
-----------------------
The default input layout is frame-first ``(frames, rows, cols)``, matching the
rest of the Python NanoLocz port.  Set ``frame_axis=-1`` or use
``viewstack_matlab`` for MATLAB-style ``(rows, cols, frames)`` arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

FloatArray = np.ndarray


LabelCallback = Callable[[Any, np.ndarray, int, Any], None]


@dataclass
class ViewStackHandle:
    """Container for the Matplotlib objects created by :func:`viewstack`."""
    figure: Any
    axes: Any
    image_artist: Any
    slider: Any
    stack: np.ndarray
    mask: np.ndarray | None
    current_slice: int
    clim: tuple[float, float]


def _as_frame_first(stack: np.ndarray, frame_axis: int) -> np.ndarray:
    """Convert a 2-D or 3-D image stack to frame-first layout."""
    arr = np.asarray(stack, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("stack must be 2-D or 3-D")

    return np.asarray(np.moveaxis(arr, int(frame_axis) % 3, 0), dtype=np.float64)


def _initial_clim(frame: np.ndarray, low_percentile: float = 1, high_percentile: float = 99) -> tuple[float, float]:
    """Compute the initial contrast range from percentiles."""
    data = np.asarray(frame, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0

    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        if low >= high:
            high = low + 1.0

    return low, high


def _resolve_cmap(colormap: str | np.ndarray | None) -> Any:
    """Resolve a Matplotlib colormap or custom LUT."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if colormap is None:
        return "gray"
    if isinstance(colormap, str):
        return colormap

    cmap = np.asarray(colormap, dtype=np.float64)
    if cmap.ndim != 2 or cmap.shape[1] < 3:
        raise ValueError("colormap array must be Nx3 or Nx4")

    if np.nanmax(cmap) > 1:
        cmap = cmap / 255.0

    return ListedColormap(cmap[:, :3])


def viewstack(
    stack: np.ndarray | None = None,
    info: Any | None = None,
    mask: np.ndarray | None = None,
    *,
    frame_axis: int = 0,
    colormap: str | np.ndarray | None = "gray",
    draw_labels_callback: LabelCallback | None = None,
    show_labels: bool = False,
    show_mask: bool = False,
    mask_alpha: float = 0.4,
    show: bool = True,
) -> ViewStackHandle:
    """Display an interactive image-stack viewer.

    Parameters
    ----------
    stack:
        2-D image or 3-D image stack.  If omitted, a random demo stack is used.
    info:
        Optional label metadata passed to ``draw_labels_callback``.
    mask:
        Optional mask stack with the same frame convention as ``stack``.  Mask
        pixels greater than zero are drawn as a green overlay when ``show_mask``
        is true.
    frame_axis:
        Axis containing frames in ``stack``.  Python default is ``0``.
    colormap:
        Matplotlib colormap name or custom RGB LUT.
    draw_labels_callback:
        Optional callable with signature ``callback(ax, frame, slice_index, info)``.
        This replaces MATLAB's direct call to ``drawLabels`` while keeping the
        viewer decoupled from a specific label table schema.
    show_labels:
        Whether labels are drawn initially.
    show_mask:
        Whether the mask overlay is drawn initially.
    mask_alpha:
        Overlay opacity.
    show:
        If true, call ``plt.show(block=False)`` before returning.

    Returns
    -------
    ViewStackHandle
        Handle containing the Matplotlib figure and viewer state.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons, Slider

    if stack is None:
        rng = np.random.default_rng(0)
        stack_arr = rng.random((50, 256, 256))
    else:
        stack_arr = _as_frame_first(stack, frame_axis=frame_axis)

    mask_arr = None
    if mask is not None:
        mask_arr = _as_frame_first(mask, frame_axis=frame_axis)
        if mask_arr.shape != stack_arr.shape:
            raise ValueError("mask must have the same shape as stack")

    n_slices = stack_arr.shape[0]
    current = 0
    clim = _initial_clim(stack_arr[current])

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18, right=0.82)

    image_artist = ax.imshow(
        stack_arr[current],
        cmap=_resolve_cmap(colormap),
        vmin=clim[0],
        vmax=clim[1],
        origin="upper",
    )
    ax.set_axis_off()
    ax.set_title(f"Slice {current + 1} of {n_slices}")
    fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)

    slider_ax = fig.add_axes([0.20, 0.05, 0.55, 0.04])
    slider = Slider(
        slider_ax,
        "Slice",
        1,
        max(n_slices, 1),
        valinit=1,
        valstep=1 if n_slices > 1 else None,
    )

    check_ax = fig.add_axes([0.84, 0.75, 0.13, 0.12])
    check = CheckButtons(
        check_ax,
        ["Labels", "Mask"],
        [bool(show_labels and draw_labels_callback is not None), bool(show_mask and mask_arr is not None)],
    )

    overlay_artists: list[Any] = []
    label_artists_before: set[Any] = set()

    handle = ViewStackHandle(
        figure=fig,
        axes=ax,
        image_artist=image_artist,
        slider=slider,
        stack=stack_arr,
        mask=mask_arr,
        current_slice=current,
        clim=clim,
    )

    def _clear_overlays() -> None:
        """Remove mask and user label overlay artists from the axes."""
        nonlocal overlay_artists
        for artist in overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        overlay_artists = []

        # Remove artists tagged by this viewer.  Label callbacks can also tag
        # custom artists with ``artist.set_gid('viewstack_label')``.
        for artist in list(ax.get_children()):
            try:
                if artist.get_gid() == "viewstack_label":
                    artist.remove()
            except Exception:
                pass

    def _draw_optional_overlays() -> None:
        """Draw labels and mask for the current frame."""
        current_state = check.get_status()
        draw_labels = bool(current_state[0])
        draw_mask = bool(current_state[1])

        if draw_mask and mask_arr is not None:
            m = mask_arr[handle.current_slice] > 0
            rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float64)
            rgba[..., 1] = 1.0
            rgba[..., 3] = float(mask_alpha) * m.astype(float)
            overlay = ax.imshow(rgba, origin="upper")
            overlay.set_gid("viewstack_mask")
            overlay_artists.append(overlay)

        if draw_labels and draw_labels_callback is not None and info is not None:
            # The callback is responsible for drawing labels.  To make cleanup
            # easy, callbacks should set gid='viewstack_label' on any artists
            # they add.  Even if they do not, the next image redraw remains safe.
            draw_labels_callback(ax, stack_arr[handle.current_slice], handle.current_slice + 1, info)

    def _redraw(slice_index: int | None = None) -> None:
        """Update displayed image and overlays."""
        if slice_index is not None:
            handle.current_slice = int(np.clip(slice_index, 0, n_slices - 1))

        image_artist.set_data(stack_arr[handle.current_slice])
        image_artist.set_clim(*handle.clim)
        ax.set_title(f"Slice {handle.current_slice + 1} of {n_slices}")

        # Keep the slider synchronized without recursively fighting its callback.
        if int(slider.val) != handle.current_slice + 1:
            slider.set_val(handle.current_slice + 1)

        _clear_overlays()
        _draw_optional_overlays()
        fig.canvas.draw_idle()

    def _slider_changed(value: float) -> None:
        """Slider callback."""
        _redraw(int(round(value)) - 1)

    def _scroll(event: Any) -> None:
        """Mouse-wheel slice navigation."""
        if event.step > 0:
            _redraw(handle.current_slice + 1)
        elif event.step < 0:
            _redraw(handle.current_slice - 1)

    def _key(event: Any) -> None:
        """Keyboard navigation and contrast adjustment."""
        low, high = handle.clim
        rng = high - low
        if rng <= 0:
            rng = 1.0

        if event.key == "right":
            _redraw(handle.current_slice + 1)
        elif event.key == "left":
            _redraw(handle.current_slice - 1)
        elif event.key == "up":
            # Raise contrast window.
            handle.clim = (low - 0.05 * rng, high - 0.05 * rng)
            _redraw()
        elif event.key == "down":
            # Lower contrast window.
            handle.clim = (low + 0.05 * rng, high + 0.05 * rng)
            _redraw()
        elif event.key == "shift+up":
            # Narrow contrast range.
            handle.clim = (low + 0.05 * rng, high - 0.05 * rng)
            if handle.clim[0] >= handle.clim[1]:
                handle.clim = (low, high)
            _redraw()
        elif event.key == "shift+down":
            # Widen contrast range.
            handle.clim = (low - 0.05 * rng, high + 0.05 * rng)
            _redraw()

    def _check_changed(_label: str) -> None:
        """Checkbox callback."""
        _redraw()

    slider.on_changed(_slider_changed)
    fig.canvas.mpl_connect("scroll_event", _scroll)
    fig.canvas.mpl_connect("key_press_event", _key)
    check.on_clicked(_check_changed)

    _redraw(current)

    if show:
        plt.show(block=False)

    return handle


def viewstack_matlab(
    stack: np.ndarray | None = None,
    info: Any | None = None,
    mask: np.ndarray | None = None,
    **kwargs: Any,
) -> ViewStackHandle:
    """MATLAB-layout wrapper for ``viewstack``."""
    return viewstack(stack, info=info, mask=mask, frame_axis=-1, **kwargs)


__all__ = ["ViewStackHandle", "viewstack", "viewstack_matlab"]
