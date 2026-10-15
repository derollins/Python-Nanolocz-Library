"""
Particle detection for NanoLocz-compatible AFM movies.

This module ports NanoLocz ``Detector.m``.  It supports two detection modes:

- ``'Peak picker'``: direct peak detection on each image frame.
- ``'ccr'``: normalized cross-correlation against a reference image, with
  optional rotational freedom over a list of candidate angles.

Output convention
-----------------
Results are returned as a :class:`pandas.DataFrame` with one row per detection
and the columns listed in :data:`DETECTOR_COLUMNS`:

    x            -> x coordinate, 0-based column index
    y            -> y coordinate, 0-based row index
    z            -> height from the unfiltered image
    correlation  -> correlation for ``'ccr'``; prominence (always 0) for
                    ``'Peak picker'``
    frame        -> 0-based frame number (integer)
    angle        -> best rotation angle in degrees (0 without ``angles``)

These correspond to columns 1-5 and 8 of the MATLAB ``Detector.m`` output.
Coordinates and frame numbers are 0-based so they can index arrays directly;
add 1 when comparing with MATLAB output.  Detections come out frame by frame,
in :func:`numpy.nonzero` (row-major) order within each frame.  When nothing is
detected an empty (0-row) table with the same columns is returned.

The detection itself is done by :func:`_detect_locs`, which works on plain
arrays and returns a :class:`Detections` record; :func:`detector` validates the
inputs and turns that record into the DataFrame.  :func:`limit_per_frame` keeps
the best ``n`` detections in each frame.

Notes
-----
Intentional deviations from MATLAB:

- An empty result is a 0-row table rather than a scalar ``NaN``.
- Non-finite values in ``img`` are replaced with 0, as MATLAB does.  Non-finite
  values in ``ref`` are also replaced with 0, where MATLAB would propagate them.
- ``ex_edge`` and ``fastdetect`` are booleans, and 3D input is always
  frame-first ``(frames, rows, cols)``.

Authors
-------
- MATLAB ``Detector.m``: George R. Heath (NanoLocz).
- Python port: Wang-1971.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd  # type: ignore  # stubs optional; suppress import-untyped
from scipy.ndimage import gaussian_filter, rotate, zoom

from pnanolocz.correlation import normxcorr2
from pnanolocz.peaks import fast_peaks2d

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.intp]]


@dataclass(frozen=True, slots=True)
class Detections:
    """
    Particle detections, one entry per detection in every field.

    Attributes
    ----------
    x, y : ndarray of float64
        Position, as 0-based column (``x``) and row (``y``) coordinates.
    z : ndarray of float64
        Height from the unfiltered image at the detection.
    correlation : ndarray of float64
        Correlation for ``'ccr'``; prominence (always 0) for ``'Peak picker'``.
    frame : ndarray of intp
        0-based frame number.
    angle : ndarray of float64
        Best rotation angle in degrees (0 without ``angles``).
    """

    x: FloatArray
    y: FloatArray
    z: FloatArray
    correlation: FloatArray
    frame: IntArray
    angle: FloatArray

    def __len__(self) -> int:
        """Return the number of detections."""
        return int(self.x.size)

    def as_dict(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return the fields as a ``{name: array}`` dict, e.g. for a DataFrame."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Column order of the detector table, taken from the Detections fields.
DETECTOR_COLUMNS = tuple(f.name for f in fields(Detections))


def _as_stack(img: FloatArray) -> FloatArray:
    """Return ``img`` as a frame-first ``(N, H, W)`` stack, promoting a 2D image."""
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise ValueError(f"img must be 2D or 3D, got shape {arr.shape}")

    return arr


def _concat_detections(parts: list[Detections]) -> Detections:
    """Join per-frame detections into one record; empty arrays if ``parts`` is empty."""
    if not parts:
        empty = np.empty(0, dtype=np.float64)
        return Detections(empty, empty, empty, empty, np.empty(0, dtype=np.intp), empty)
    return Detections(
        **{
            name: np.concatenate([getattr(p, name) for p in parts])
            for name in DETECTOR_COLUMNS
        }
    )


def _lookup_heights(frame: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    """Read the raw image value under each localisation.

    The value comes from the nearest pixel centre: pixel centres sit at integer
    coordinates, so the index is ``floor(coord + 0.5)``.  A localisation counts
    as inside the frame only when its centre is within half a pixel of a real
    pixel centre, so a centre exactly half a pixel outside an edge
    (``coord == -0.5``) or past the last centre reads 0.  Re-centred CCR
    detections land on exact half pixels, so this rule decides the value.
    """
    n_rows, n_cols = frame.shape
    inside = (y > -0.5) & (y < n_rows - 0.5) & (x > -0.5) & (x < n_cols - 0.5)
    rows = np.floor(y[inside] + 0.5).astype(np.intp)
    cols = np.floor(x[inside] + 0.5).astype(np.intp)

    z = np.zeros(x.shape, dtype=np.float64)
    z[inside] = frame[rows, cols]
    return z


def _crop_corr_edges(
    ccr: FloatArray, ref_shape: tuple[int, int], *, rotated: bool
) -> FloatArray:
    """Crop the correlation map to the region where the template fits fully.

    Matches MATLAB ``Detector.m`` exactly.  ``ccr`` from
    :func:`~pnanolocz.correlation.normxcorr2` has shape ``image + ref - 1``,
    and MATLAB crops it with the 1-based ranges ``round(R)+1 : end-R-1`` when
    the reference was rotated and ``round(R) : end-R`` otherwise, where ``R =
    size(ref, 1)`` in each dimension.  Shifting those ranges to 0-based Python
    slices gives ``[R : M-R-1]`` and ``[R-1 : M-R]`` below.  The final clamping
    only guards degenerate sizes, where the MATLAB ranges would be empty or
    reversed.
    """
    ref_rows = int(ref_shape[0])
    ref_cols = int(ref_shape[1])
    n_rows, n_cols = ccr.shape[0], ccr.shape[1]

    if rotated:
        y0, y1 = ref_rows, n_rows - ref_rows - 1
        x0, x1 = ref_cols, n_cols - ref_cols - 1
    else:
        y0, y1 = ref_rows - 1, n_rows - ref_rows
        x0, x1 = ref_cols - 1, n_cols - ref_cols

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = max(y0 + 1, min(n_rows, y1))
    x1 = max(x0 + 1, min(n_cols, x1))

    # Works for 2D maps and for (rows, cols, angles) stacks alike.
    return np.asarray(ccr[y0:y1, x0:x1, ...], dtype=np.float64)


def limit_per_frame(
    detections: pd.DataFrame, n: int, by: str = "correlation"
) -> pd.DataFrame:
    """
    Keep the ``n`` highest-scoring detections in each frame.

    Parameters
    ----------
    detections : pandas.DataFrame
        Detection table with a ``frame`` column, e.g. from :func:`detector`.
    n : int
        Maximum number of detections to keep per frame.
    by : str, default "correlation"
        Column to rank by, highest first.  Use ``"z"`` for ``'Peak picker'``
        results, whose ``correlation`` column is all zeros.

    Returns
    -------
    pandas.DataFrame
        The selected rows in their original order, with their original index
        labels.  ``detections`` itself is not modified.  Ties keep the earlier
        row, and NaN scores rank last.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    KeyError
        If ``by`` or ``frame`` is not a column of ``detections``.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    top = (
        detections.sort_values(by, ascending=False, kind="stable")
        .groupby("frame")
        .head(n)
    )
    return top.sort_index()


def _detect_locs(
    stack: FloatArray,
    method: Literal["peak picker", "ccr"],
    ref_img: FloatArray | None,
    peak_kernel_size: int,
    filt_img: float,
    filt_ccr: float,
    min_thresh: float,
    ex_edge: bool,
    fastdetect: bool,
    angle_array: FloatArray,
) -> Detections:
    """Run the detection on a validated ``(N, H, W)`` stack.

    Inputs are already cleaned by :func:`detector`: ``stack`` has no non-finite
    values, ``ref_img`` is a 2D template for ``'ccr'`` (``None`` otherwise),
    ``peak_kernel_size`` is the Fast_peaks2D neighbourhood and ``angle_array``
    is non-empty.  Returns every detection, frame by frame.
    """
    # Template rotation is only needed for a non-zero angle list.
    rotfreedom = not (angle_array.size == 1 and float(angle_array[0]) == 0.0)

    if fastdetect:
        # MATLAB: imresize(img, 1/2, 'bilinear'); frames are one per plane, so
        # only the spatial axes are resized here.
        stack = zoom(stack, zoom=(1.0, 0.5, 0.5), order=1)
        if ref_img is not None:
            ref_img = zoom(ref_img, zoom=0.5, order=1)

    if filt_img > 0:
        img_g = gaussian_filter(
            stack, sigma=(0.0, float(filt_img), float(filt_img)), mode="nearest"
        )
    else:
        img_g = stack

    per_frame: list[Detections] = []

    for frame_idx in range(stack.shape[0]):
        frame_img = img_g[frame_idx]

        if method == "peak picker":
            peaks = fast_peaks2d(frame_img, min_thresh, peak_kernel_size)
            x = peaks.x.astype(np.float64)
            y = peaks.y.astype(np.float64)
            correlation = peaks.prominence
            angle = np.zeros(len(peaks), dtype=np.float64)

        else:
            assert ref_img is not None
            if rotfreedom:
                # Correlate every rotated copy of the reference with the frame,
                # then keep the best-matching angle at each pixel.
                corr_stack = [
                    normxcorr2(
                        rotate(
                            ref_img,
                            angle=float(a),
                            reshape=False,
                            order=3,
                            mode="constant",
                            cval=0.0,
                            prefilter=True,
                        ),
                        frame_img,
                    )
                    for a in angle_array
                ]
                ccr = np.stack(corr_stack, axis=2)
                sigma: tuple[float, ...] = (float(filt_ccr), float(filt_ccr), 0.0)
            else:
                ccr = normxcorr2(ref_img, frame_img)
                sigma = (float(filt_ccr), float(filt_ccr))

            if filt_ccr > 0:
                ccr = gaussian_filter(ccr, sigma=sigma, mode="nearest")

            if ex_edge:
                ccr = _crop_corr_edges(ccr, ref_img.shape, rotated=rotfreedom)

            if rotfreedom:
                best_correlation = np.max(ccr, axis=2)
                best_angle_index = np.argmax(ccr, axis=2)
                peaks = fast_peaks2d(best_correlation, min_thresh, peak_kernel_size)
                # MATLAB looks the winning angle up at the raw peak position,
                # before the re-centring below; peaks are integer pixel indices.
                angle = -angle_array[best_angle_index[peaks.y, peaks.x]]
            else:
                peaks = fast_peaks2d(ccr, min_thresh, peak_kernel_size)
                angle = np.zeros(len(peaks), dtype=np.float64)

            # On a correlation map the peak "height" is the correlation value.
            correlation = peaks.height

            # Re-centre the peak positions on the frame.  MATLAB uses a
            # different convention for the ex_edge branch, kept here for parity.
            ref_rows, ref_cols = ref_img.shape
            if ex_edge:
                y = peaks.y + ref_rows / 2.0
                x = peaks.x + ref_cols / 2.0
            else:
                y = peaks.y - ref_rows / 2.0 + 0.5
                x = peaks.x - ref_cols / 2.0 + 0.5

        if len(peaks) == 0:
            continue

        # Z is the raw height at the localisation, read before any rescaling.
        z = _lookup_heights(stack[frame_idx], x, y)
        if fastdetect:
            # Positions were found on the half-resolution stack.
            x, y = x * 2.0, y * 2.0

        per_frame.append(
            Detections(
                x=x,
                y=y,
                z=z,
                correlation=correlation,
                frame=np.full(len(peaks), frame_idx, dtype=np.intp),
                angle=angle,
            )
        )

    return _concat_detections(per_frame)


def detector(
    img: FloatArray,
    method: str,
    ref: FloatArray | float | int,
    filt_img: float,
    filt_ccr: float,
    min_thresh: float,
    ex_edge: bool,
    fastdetect: bool,
    angles: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    Detect particles in an AFM image stack.

    Parameters
    ----------
    img : ndarray
        2D image or 3D movie stack in frame-first ``(frames, rows, cols)``
        order.
    method : {'Peak picker', 'ccr'}
        Detection method.
    ref : ndarray or int
        For ``'ccr'``, the 2D reference image that is correlated with every
        frame.  For ``'Peak picker'``, the peak-detection neighbourhood size
        passed to :func:`~pnanolocz.peaks.fast_peaks2d`.
    filt_img : float
        Gaussian sigma applied to the input frames before detection.  Use 0 for
        no filtering.
    filt_ccr : float
        Gaussian sigma applied to correlation maps.  Use 0 for no filtering.
    min_thresh : float
        Minimum peak or correlation value.
    ex_edge : bool
        If true, exclude correlation edge regions where the reference is not
        fully supported by the frame.  As in MATLAB ``Detector.m``, positions
        are then reported half a pixel from the ``ex_edge=False`` position
        (+0.5 px without ``angles``, -0.5 px with them).
    fastdetect : bool
        If true, halve the resolution of the frames (and reference) for a
        faster, less precise detection.
    angles : iterable of float, optional
        Rotation angles in degrees for the CCR template.  Rotation is skipped
        when this is empty or a single zero angle.

    Returns
    -------
    pandas.DataFrame
        One row per detection with the columns in :data:`DETECTOR_COLUMNS`,
        0-based.  Empty (``res.empty``) when nothing is detected.

    Raises
    ------
    ValueError
        If ``method`` is not recognised, ``img`` is not 2D or 3D, ``ref`` is
        not a 2D image for ``'ccr'``, or ``ref`` is not numeric for
        ``'Peak picker'``.
    """
    stack = np.nan_to_num(_as_stack(img), nan=0.0, posinf=0.0, neginf=0.0)

    method_lc = method.strip().lower()
    if method_lc not in {"peak picker", "ccr"}:
        raise ValueError("method must be 'Peak picker' or 'ccr'")

    if angles is None:
        angle_array = np.array([0.0], dtype=np.float64)
    else:
        angle_array = np.asarray(list(angles), dtype=np.float64)
        if angle_array.size == 0:
            angle_array = np.array([0.0], dtype=np.float64)

    ref_img: FloatArray | None = None
    if method_lc == "ccr":
        ref_arr = np.asarray(ref, dtype=np.float64)
        if ref_arr.ndim != 2:
            raise ValueError("ref must be a 2D reference image for method='ccr'")
        ref_img = np.nan_to_num(ref_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # MATLAB Detector.m: ns = 3 for fastdetect, otherwise half the smallest
    # reference dimension capped at 50.  It is the neighbourhood that
    # Fast_peaks2D searches the correlation map with.  The 'Peak picker'
    # branch passes the caller's ref through as the neighbourhood size.
    # Here the neighbourhood size is called peak_kernel_size.
    if method_lc == "peak picker":
        try:
            peak_kernel_size = int(np.floor(float(ref) + 0.5))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ref must be a number (the peak neighbourhood size) for "
                "method='Peak picker'"
            ) from exc
    elif fastdetect:
        peak_kernel_size = 3
    else:
        assert ref_img is not None
        peak_kernel_size = min(int(np.floor(min(ref_img.shape) / 2.0 + 0.5)), 50)

    locs = _detect_locs(
        stack,
        "peak picker" if method_lc == "peak picker" else "ccr",
        ref_img,
        peak_kernel_size,
        filt_img,
        filt_ccr,
        min_thresh,
        ex_edge,
        fastdetect,
        angle_array,
    )
    return pd.DataFrame(locs.as_dict())


detector.__version__ = "0.1.0"  # type: ignore[attr-defined]
limit_per_frame.__version__ = "0.1.0"  # type: ignore[attr-defined]

__all__ = [
    "DETECTOR_COLUMNS",
    "detector",
    "limit_per_frame",
]
