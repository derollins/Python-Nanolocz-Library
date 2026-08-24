"""
FFT-based dominant period detection for AFM line profiles.

This module ports MATLAB ``fft_line_analysis.m``.  It detrends a line profile,
applies a Hann window, computes the single-sided FFT amplitude spectrum, detects
dominant peaks, and reports the corresponding real-space periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import detrend, find_peaks, peak_widths, windows

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass
class FFTLineAnalysisResult:
    """Dominant periods extracted from a line profile."""

    periods: FloatArray
    period_error: FloatArray
    amps: FloatArray
    freqs: FloatArray
    spectrum_frequency: FloatArray
    spectrum_amplitude: FloatArray


def fft_line_analysis(
    x: np.ndarray,
    y: np.ndarray,
    do_plot: bool = True,
    *,
    max_peaks: int = 5,
) -> FFTLineAnalysisResult:
    """Detect dominant periods in an AFM line profile.

    Parameters
    ----------
    x, y:
        Coordinate and height/intensity vectors.
    do_plot:
        If true, create a Matplotlib FFT plot.
    max_peaks:
        Number of strongest non-DC peaks to retain.

    Returns
    -------
    FFTLineAnalysisResult
        Dataclass containing periods, errors, amplitudes, frequencies and the
        full single-sided FFT spectrum.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    if x_arr.size != y_arr.size or x_arr.size < 4:
        raise ValueError("x and y must have the same length and at least 4 samples")

    n = y_arr.size
    dx = float(np.nanmean(np.diff(x_arr)))
    if not np.isfinite(dx) or dx == 0:
        raise ValueError("x must be monotonically spaced enough to estimate dx")

    y_proc = detrend(y_arr, type="linear")
    y_proc = y_proc * windows.hann(n, sym=True)

    Y = np.fft.fft(y_proc)
    p2 = np.abs(Y / n)
    p1 = p2[: n // 2 + 1].copy()
    if p1.size > 2:
        p1[1:-1] *= 2.0

    f = np.arange(0, n // 2 + 1, dtype=np.float64) / (n * dx)

    min_peak_height = 0.1 * float(np.nanmax(p1)) if p1.size else 0.0
    min_peak_distance_freq = 1.0 / (float(np.nanmax(x_arr)) - float(np.nanmin(x_arr)))

    # Convert MinPeakDistance in frequency units to sample units.
    df = float(np.nanmean(np.diff(f))) if f.size > 1 else 1.0
    distance_samples = max(1, int(round(min_peak_distance_freq / df)))

    peaks, props = find_peaks(p1, height=min_peak_height, distance=distance_samples)

    if peaks.size == 0:
        raise RuntimeError("No meaningful FFT peaks found")

    # Approximate MATLAB findpeaks halfheight widths.
    width_result = peak_widths(p1, peaks, rel_height=0.5)
    widths_samples = width_result[0]
    widths_freq = widths_samples * df

    pk_amp = np.asarray(props["peak_heights"], dtype=np.float64)
    pk_freq = f[peaks]

    # Remove the lowest-frequency artifact / line-length component.
    if pk_freq.size > 0:
        idx_min = int(np.argmin(pk_freq))
        pk_freq = np.delete(pk_freq, idx_min)
        pk_amp = np.delete(pk_amp, idx_min)
        widths_freq = np.delete(widths_freq, idx_min)

    if pk_freq.size == 0:
        raise RuntimeError("No meaningful non-DC FFT peaks found")

    periods = 1.0 / pk_freq
    freq_err = widths_freq / 2.0
    period_err = periods * (freq_err / pk_freq)

    order = np.argsort(pk_amp)[::-1]
    order = order[: min(max_peaks, order.size)]

    pk_amp = pk_amp[order]
    periods = periods[order]
    period_err = period_err[order]
    pk_freq = pk_freq[order]

    if do_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(f, p1, linewidth=1.2)
        ax.set_xlabel("Spatial Frequency (1/nm)")
        ax.set_ylabel("Amplitude")
        ax.set_title("FFT Amplitude Spectrum")
        ax.grid(True)

        for freq, period, err, amp in zip(pk_freq, periods, period_err, pk_amp, strict=False):
            ax.axvline(freq, linestyle="--", linewidth=1)
            ax.text(freq, amp, f"{period:.2f} ± {err:.2f} nm", ha="center", va="bottom")

    return FFTLineAnalysisResult(
        periods=np.asarray(periods, dtype=np.float64),
        period_error=np.asarray(period_err, dtype=np.float64),
        amps=np.asarray(pk_amp, dtype=np.float64),
        freqs=np.asarray(pk_freq, dtype=np.float64),
        spectrum_frequency=np.asarray(f, dtype=np.float64),
        spectrum_amplitude=np.asarray(p1, dtype=np.float64),
    )


__all__ = ["FFTLineAnalysisResult", "fft_line_analysis"]
