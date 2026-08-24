"""
Movie/image-stack filtering utilities for NanoLocz-compatible AFM workflows.

This module ports MATLAB ``filter_movie.m``.  It applies one or two sequential
filters to a 3-D image stack.

Python default stack convention is frame-first ``(frames, rows, cols)``.
Use ``frame_axis=-1`` for MATLAB-style ``(rows, cols, frames)`` data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import convolve, gaussian_filter, uniform_filter1d
from scipy.signal import butter, filtfilt, wiener
from skimage import restoration
from skimage.exposure import rescale_intensity

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _as_frame_first(stack: np.ndarray, frame_axis: int) -> tuple[FloatArray, bool, int]:
    """Convert 2-D or 3-D input to frame-first layout."""
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0
    if arr.ndim != 3:
        raise ValueError("target must be 2D or 3D")
    axis = int(frame_axis) % 3
    return np.asarray(np.moveaxis(arr, axis, 0), dtype=np.float64), False, axis


def _restore(stack: np.ndarray, was_2d: bool, frame_axis: int) -> np.ndarray:
    """Restore stack layout after processing."""
    if was_2d:
        return np.asarray(stack[0], dtype=np.float64)
    return np.asarray(np.moveaxis(stack, 0, frame_axis), dtype=np.float64)


def _sphere_kernel(strength: float, disk_only: bool = False) -> FloatArray:
    """Construct MATLAB-like sphere/disk smoothing kernels."""
    P = float(strength) + 1.0
    offs = round(P) - P
    coords = np.arange(-P, P + 1)
    hx, hy = np.meshgrid(coords, coords, indexing="ij")
    hx = hx - offs
    hy = hy - offs

    if disk_only:
        H = (np.sqrt(np.maximum(-(hx**2) - (hy**2) + P**2, 0.0)) > (P / 2.0)).astype(float)
    else:
        H = np.sqrt(np.maximum(-(hx**2) - (hy**2) + P**2, 0.0))

    s = float(np.sum(H))
    if s > 0:
        H = H / s

    if offs > 0 and H.shape[0] > 1 and H.shape[1] > 1:
        H = H[1:, 1:]

    return np.asarray(H, dtype=np.float64)


def _row_butter_filter(frame: np.ndarray, strength: Any, mode: str) -> FloatArray:
    """Apply a first-order Butterworth filter to each row."""
    img = np.asarray(frame, dtype=np.float64)
    out = np.empty_like(img)

    try:
        if mode == "low":
            Wn = float(strength) / 0.5
            b, a = butter(1, Wn, btype="low")
        elif mode == "high":
            Wn = float(strength) / 0.5
            b, a = butter(1, Wn, btype="high")
        else:
            low, high = float(strength[0]), float(strength[1])
            b, a = butter(1, [low / 0.5, high / 0.5], btype="bandpass")
    except Exception:
        if mode == "band":
            b, a = butter(1, [0.01 / 0.5, 0.49 / 0.5], btype="bandpass")
        else:
            b, a = butter(1, 0.49 / 0.5, btype="high")

    for rr in range(img.shape[0]):
        try:
            out[rr, :] = filtfilt(b, a, img[rr, :])
        except Exception:
            out[rr, :] = img[rr, :]

    return np.asarray(out, dtype=np.float64)


def _high_pass_fft(frame: np.ndarray, strength: float) -> FloatArray:
    """Frequency-domain Gaussian high-pass filter."""
    img = np.asarray(frame, dtype=np.float64)
    mx, ny = img.shape
    ft = np.fft.fft2(img)
    shifted = np.fft.fftshift(ft)

    yy, xx = np.indices((mx, ny))
    ps = mx / 2.0
    qs = ny / 2.0
    distance = np.sqrt((yy - ps) ** 2 + (xx - qs) ** 2)
    low_filter = 1.0 - np.exp(-(distance**2) / (2.0 * float(strength)))

    filtered = shifted * low_filter
    return np.asarray(np.abs(np.fft.ifft2(np.fft.ifftshift(filtered))), dtype=np.float64)


def _peak_sharp_line(x: np.ndarray, y: np.ndarray, factor: float) -> np.ndarray:
    """Small Python substitute for the NanoLocz ``sharpen`` row operation."""
    yy = np.asarray(y, dtype=np.float64)
    smooth = gaussian_filter(yy, sigma=1.0, mode="nearest")
    return yy + float(factor) * (yy - smooth)


def _apply_single_filter(stack: FloatArray, filt: str, strength: Any) -> FloatArray:
    """Apply one MATLAB-named filter to a frame-first stack."""
    if strength is None:
        return stack

    try:
        strength_scalar = float(np.asarray(strength).ravel()[0])
    except Exception:
        strength_scalar = 0.0

    if strength_scalar <= 0:
        return stack

    name = str(filt)

    if name == "Gaussian":
        return np.asarray(gaussian_filter(stack, sigma=(0, strength_scalar, strength_scalar)), dtype=np.float64)

    if name == "-Gaussian":
        return np.asarray(stack - gaussian_filter(stack, sigma=(0, strength_scalar, strength_scalar)), dtype=np.float64)

    if name == "Wiener":
        if strength_scalar < 2:
            return stack
        size = int(round(strength_scalar))
        return np.asarray(np.stack([wiener(frame, (size, size)) for frame in stack], axis=0), dtype=np.float64)

    if name == "Non-local-means":
        out = []
        for frame in stack:
            sigma_est = np.std(frame)
            out.append(restoration.denoise_nl_means(frame, h=strength_scalar, fast_mode=True, channel_axis=None))
        return np.asarray(out, dtype=np.float64)

    if name == "Disk":
        kernel = _sphere_kernel(strength_scalar, disk_only=True)
        return np.asarray(np.stack([convolve(frame, kernel, mode="nearest") for frame in stack], axis=0), dtype=np.float64)

    if name == "Sphere":
        kernel = _sphere_kernel(strength_scalar, disk_only=False)
        return np.asarray(np.stack([convolve(frame, kernel, mode="nearest") for frame in stack], axis=0), dtype=np.float64)

    if name == "ButterLP":
        return np.asarray(np.stack([_row_butter_filter(frame, strength_scalar, "low") for frame in stack], axis=0), dtype=np.float64)

    if name == "ButterHP":
        return np.asarray(np.stack([_row_butter_filter(frame, strength_scalar, "high") for frame in stack], axis=0), dtype=np.float64)

    if name == "ButterBP":
        return np.asarray(np.stack([_row_butter_filter(frame, strength, "band") for frame in stack], axis=0), dtype=np.float64)

    if name == "Laplacian":
        lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        smooth = gaussian_filter(stack, sigma=(0, 0.6, 0.6))
        lap = np.stack([convolve(frame, lap_kernel, mode="nearest") for frame in smooth], axis=0)
        lap_rs = rescale_intensity(lap, out_range=(0.0, 1.0))
        return np.asarray(rescale_intensity(-strength_scalar * lap_rs + stack, out_range=(0.0, 1.0)), dtype=np.float64)

    if name == "-Average":
        d_av = np.nanmean(stack, axis=0)
        return np.asarray(stack - d_av[None, :, :] * strength_scalar, dtype=np.float64)

    if name == "Deconv":
        kernel = _sphere_kernel(strength_scalar, disk_only=False)
        out = []
        for frame in stack:
            try:
                out.append(restoration.richardson_lucy(frame, kernel, num_iter=1))
            except TypeError:
                out.append(restoration.richardson_lucy(frame, kernel, iterations=1))
        return np.asarray(out, dtype=np.float64)

    if name == "Peak sharp":
        out = stack.copy()
        x = np.arange(stack.shape[2])
        for j in range(stack.shape[0]):
            for rr in range(stack.shape[1]):
                out[j, rr, :] = _peak_sharp_line(x, stack[j, rr, :], strength_scalar)
        return np.asarray(out, dtype=np.float64)

    if name == "High-pass":
        return np.asarray(np.stack([_high_pass_fft(frame, strength_scalar) for frame in stack], axis=0), dtype=np.float64)

    if name == "RAvg":
        size = max(1, int(round(strength_scalar)))
        return np.asarray(uniform_filter1d(stack, size=size, axis=0, mode="nearest"), dtype=np.float64)

    # Unknown filter name: return unchanged, matching a forgiving GUI workflow.
    return stack


def filter_movie(
    target: np.ndarray,
    filt1: str,
    strength1: Any,
    filt2: str | None = None,
    strength2: Any | None = None,
    *,
    frame_axis: int = 0,
) -> np.ndarray:
    """Apply one or two NanoLocz filters to an AFM movie stack."""
    stack, was_2d, axis = _as_frame_first(target, frame_axis=frame_axis)
    out = _apply_single_filter(stack, filt1, strength1)
    if filt2 is not None:
        out = _apply_single_filter(out, filt2, strength2)
    return _restore(out, was_2d, axis)


def filter_movie_matlab(target: np.ndarray, filt1: str, strength1: Any, filt2: str | None = None, strength2: Any | None = None) -> np.ndarray:
    """MATLAB-layout wrapper for ``filter_movie``."""
    return filter_movie(target, filt1, strength1, filt2, strength2, frame_axis=-1)


__all__ = ["filter_movie", "filter_movie_matlab"]
