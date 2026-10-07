"""
Particle detection for NanoLocz-compatible AFM movies.

This module ports NanoLocz ``Detector.m``.  It supports two detection modes:

- ``'Peak picker'``: direct peak detection on each image frame.
- ``'ccr'``: normalized cross-correlation against a reference image, with
  optional rotational freedom over a list of candidate angles.

Output convention
-----------------
Results are returned as a :class:`pandas.DataFrame` with the columns listed in
:data:`DETECTOR_COLUMNS`, which follows the NanoLocz ``Part Locs`` table (the
detector output as the NanoLocz GUI exports it: ``x, y, Height, ccr, Frame,
id, Track id, Angle``):

    x            -> x coordinate, 0-based column index
    y            -> y coordinate, 0-based row index
    z            -> height from the unfiltered image
    correlation  -> correlation for ``'ccr'``; prominence for ``'Peak picker'``
    frame        -> 0-based frame number
    id           -> 0-based index of the detection within this table
    track_id     -> track number from particle tracking; NaN until then
    angle        -> best rotation angle in degrees (``'ccr'`` with ``angles``)
    kept         -> True for a detection the caller wants to keep

The first eight columns are the NanoLocz ``Part Locs`` layout.  ``id`` and
``track_id`` are the two columns NanoLocz fills *after* detection: ``id``
numbers the particles that survive the GUI's threshold/track filters
(``filter_Partlocs``), with ``NaN`` for the ones that were dropped, and
``track_id`` comes from ``track_particles``.  This port assigns ``id`` to every
detection, 0-based in row order, so that a caller (or GUI) can refer to one
stably, and leaves ``track_id`` as ``NaN`` until tracking is ported.

``kept`` is the ninth column and a pnanolocz addition: a real boolean saying
whether a detection should be used.  It starts ``True``;
``max_particles_per_frame`` (``'ccr'`` only) sets it to ``False`` for the
detections that lose the per-frame correlation ranking, and a caller or GUI can
set it for any row without re-running detection and without renumbering ``id``.
It plays the role of the 0/1 ``IncludeImages`` mask that the NanoLocz GUI keeps
*outside* the table, so ``res[res["kept"]]`` is the curated table and
``res["id"]`` stays a stable handle for the detections that were dropped.

Every index is Python-native: coordinates and frame numbers are 0-based so that
they can index arrays directly, ``id`` counts from 0, and detections come out in
:func:`numpy.nonzero` order (row-major) rather than MATLAB's 1-based numbering.
Add 1 when comparing with MATLAB output.  When nothing is detected an empty
(0-row) table is returned, so ``res.empty`` is ``True`` and no caller has to
filter out a phantom row.

Notes
-----
Intentional deviations from MATLAB:

- An empty result is a 0-row table rather than a scalar ``NaN``.
- Non-finite values in ``img`` are replaced with 0, as MATLAB does.  Non-finite
  values in ``ref`` are also replaced with 0, where MATLAB would propagate them.
- ``ex_edge`` and ``fastdetect`` are booleans, and 3-D input is always
  frame-first ``(frames, rows, cols)``.
- ``id`` is assigned by the detector (0-based, one per row), where MATLAB
  renumbers the surviving particles 1..N after filtering, and ``track_id`` is
  ``NaN`` where MATLAB leaves 0.
- ``kept`` and ``max_particles_per_frame`` have no MATLAB equivalent.

Authors
-------
- MATLAB ``Detector.m``: George R. Heath (NanoLocz).
- Python port: Wang-1971.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd  # type: ignore  # stubs optional; suppress import-untyped
from scipy.ndimage import gaussian_filter, rotate, zoom

from pnanolocz.correlation import normxcorr2
from pnanolocz.peaks import fast_peaks2d

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

DETECTOR_COLUMNS = (
    "x",
    "y",
    "z",
    "correlation",
    "frame",
    "id",
    "track_id",
    "angle",
    "kept",
)


def _as_stack(img: FloatArray) -> FloatArray:
    """Return ``img`` as a frame-first ``(N, H, W)`` stack of floats.

    A 2-D image is promoted to a single-frame stack; a 3-D array is assumed to
    be ``(frames, rows, cols)`` already.  The array is not copied, so callers
    must not mutate the result if they need to keep ``img`` unchanged.
    """
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise ValueError(f"img must be 2D or 3D, got shape {arr.shape}")

    return arr


def _localisation_table(peaks: FloatArray) -> FloatArray:
    """Expand the ``(N, 4)`` :func:`~pnanolocz.peaks.fast_peaks2d` output.

    ``peaks`` is ``[x, y, height, prominence]``; the returned table has the
    :data:`DETECTOR_COLUMNS` layout with the ``z``, ``frame``, ``id``,
    ``track_id``, ``angle`` and ``kept`` columns left at zero for the caller to
    fill in.
    """
    locs = np.zeros((peaks.shape[0], len(DETECTOR_COLUMNS)), dtype=np.float64)
    locs[:, : peaks.shape[1]] = peaks
    return locs


def _per_frame_keep_mask(
    locs: FloatArray, max_per_frame: int
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Mask the ``max_per_frame`` highest-correlation detections of each frame.

    Used for ``detector(..., max_particles_per_frame=N)``.  The mask is applied
    to the ``kept`` column, so no row is dropped.  Rows keep their original order
    and equal correlations are broken by that order, so the result is
    deterministic.
    """
    scores = locs[:, 3]
    frames = locs[:, 4]
    keep = np.zeros(locs.shape[0], dtype=bool)

    for frame in np.unique(frames):
        rows = np.flatnonzero(frames == frame)
        ranked = rows[np.argsort(-scores[rows], kind="stable")]
        keep[ranked[:max_per_frame]] = True

    return keep


def _lookup_heights(stack: FloatArray, frame_idx: int, locs: FloatArray) -> FloatArray:
    """Read the raw image value under each localisation.

    The value comes from the nearest pixel centre: pixel centres sit at integer
    coordinates, so the index is ``floor(coord + 0.5)``.  A localisation counts
    as inside the frame only when its centre is within half a pixel of a real
    pixel centre, so a centre exactly half a pixel outside an edge
    (``coord == -0.5``) or past the last centre reads 0.  Re-centred CCR
    detections land on exact half pixels, so this rule decides the value.
    """
    heights = np.zeros(locs.shape[0], dtype=np.float64)
    n_rows = stack.shape[1]
    n_cols = stack.shape[2]

    for row_idx in range(locs.shape[0]):
        y = float(locs[row_idx, 1])
        x = float(locs[row_idx, 0])
        if -0.5 < y < n_rows - 0.5 and -0.5 < x < n_cols - 0.5:
            row = int(np.floor(y + 0.5))
            col = int(np.floor(x + 0.5))
            heights[row_idx] = stack[frame_idx, row, col]

    return heights


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

    was_2d = ccr.ndim == 2
    ccr_work = ccr[:, :, np.newaxis] if was_2d else ccr
    n_rows, n_cols = ccr_work.shape[0], ccr_work.shape[1]

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

    cropped = ccr_work[y0:y1, x0:x1, :]
    return np.asarray(cropped[:, :, 0] if was_2d else cropped, dtype=np.float64)


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
    max_particles_per_frame: int | bool | None = None,
) -> pd.DataFrame:
    """Detect particles in an AFM image stack.

    Parameters
    ----------
    img:
        2-D image or 3-D movie stack in frame-first ``(frames, rows, cols)``
        order.
    method:
        ``'Peak picker'`` or ``'ccr'``.
    ref:
        For ``'ccr'``, the 2-D reference image that is correlated with every
        frame.  For ``'Peak picker'``, the peak-detection neighbourhood size
        passed to :func:`~pnanolocz.peaks.fast_peaks2d`.
    filt_img:
        Gaussian sigma applied to the input frames before detection.  Use 0 for
        no filtering.
    filt_ccr:
        Gaussian sigma applied to correlation maps.  Use 0 for no filtering.
    min_thresh:
        Minimum peak or correlation value.
    ex_edge:
        If true, exclude correlation edge regions where the reference is not
        fully supported by the frame.
    fastdetect:
        If true, halve the resolution of the frames (and reference) for a
        faster, less precise detection.
    angles:
        Optional rotation angles in degrees for the CCR template.  Rotation is
        skipped when this is empty or a single zero angle.
    max_particles_per_frame:
        Optional per-frame limit for the ``'ccr'`` method, off by default
        (``None`` or ``False``).  Set it to ``True`` to rank with the default
        count of 1 per frame, or to a positive integer to choose the count.  The
        detections with the highest ``correlation`` in each frame keep
        ``kept=True`` and the rest are marked ``kept=False``; no row is dropped,
        so ``res[res["kept"]]`` is the limited table.  ``'Peak picker'`` has no
        correlation to rank by -- its ``correlation`` column is the prominence,
        which is 0 because ``Detector.m`` calls ``Fast_peaks2D`` without
        ``min_prom`` -- so it is rejected for that method.

    Returns
    -------
    pandas.DataFrame
        Localisation table with the columns given by :data:`DETECTOR_COLUMNS`,
        0-based, and one row per detection.  ``id`` is a stable 0-based handle
        for each detection and ``kept`` starts ``True`` for all of them.  The
        frame is empty (``len(res) == 0``, ``res.empty``) when nothing is
        detected.

    Raises
    ------
    ValueError
        If ``method`` is not recognised, ``img`` is not 2-D or 3-D, ``ref`` is
        not a 2-D image for ``'ccr'``, ``ref`` is not numeric for
        ``'Peak picker'``, ``max_particles_per_frame`` is not positive, or
        ``max_particles_per_frame`` is given for ``'Peak picker'``.
    """
    stack = np.nan_to_num(_as_stack(img), nan=0.0, posinf=0.0, neginf=0.0)

    method_lc = method.strip().lower()
    if method_lc not in {"peak picker", "ccr"}:
        raise ValueError("method must be 'Peak picker' or 'ccr'")

    # The per-frame limit is opt-in: None/False keep every detection, True uses
    # the default count of 1, and an integer sets the count.
    max_per_frame: int | None
    if max_particles_per_frame is None:
        max_per_frame = None
    elif isinstance(max_particles_per_frame, (bool, np.bool_)):
        max_per_frame = 1 if bool(max_particles_per_frame) else None
    else:
        max_per_frame = int(max_particles_per_frame)
        if max_per_frame < 1:
            raise ValueError("max_particles_per_frame must be a positive integer")

    if max_per_frame is not None and method_lc == "peak picker":
        raise ValueError(
            "max_particles_per_frame ranks detections by correlation and is "
            "only supported for method='ccr'"
        )

    if angles is None:
        angle_array = np.array([0.0], dtype=np.float64)
    else:
        angle_array = np.asarray(list(angles), dtype=np.float64)
        if angle_array.size == 0:
            angle_array = np.array([0.0], dtype=np.float64)

    # Template rotation is only needed for a non-zero angle list.
    rotfreedom = not (angle_array.size == 1 and float(angle_array[0]) == 0.0)

    ref_img: FloatArray | None = None
    if method_lc == "ccr":
        ref_arr = np.asarray(ref, dtype=np.float64)
        if ref_arr.ndim != 2:
            raise ValueError("ref must be a 2-D reference image for method='ccr'")
        ref_img = np.nan_to_num(ref_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # MATLAB Detector.m: ns = 3 for fastdetect, otherwise half the smallest
    # reference dimension capped at 50.  It is the neighbourhood that
    # Fast_peaks2D searches the correlation map with.  The 'Peak picker'
    # branch passes the caller's ref through as the neighbourhood size.
    if method_lc == "peak picker":
        try:
            peak_kernel_size = int(np.floor(float(ref) + 0.5))
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

    all_locs: list[FloatArray] = []

    for frame_idx in range(stack.shape[0]):
        frame_img = img_g[frame_idx]
        locs_i = np.empty((0, len(DETECTOR_COLUMNS)), dtype=np.float64)

        if method_lc == "peak picker":
            locs_i = _localisation_table(
                fast_peaks2d(frame_img, min_thresh, peak_kernel_size)
            )

        elif rotfreedom:
            assert ref_img is not None
            # Correlate every rotated copy of the reference with the frame,
            # then keep the best-matching angle at each pixel.
            corr_stack = []
            for angle in angle_array:
                rotated_ref = rotate(
                    ref_img,
                    angle=float(angle),
                    reshape=False,
                    order=3,
                    mode="constant",
                    cval=0.0,
                    prefilter=True,
                )
                corr_stack.append(normxcorr2(rotated_ref, frame_img))
            ccr = np.stack(corr_stack, axis=2)

            if filt_ccr > 0:
                ccr = gaussian_filter(
                    ccr, sigma=(float(filt_ccr), float(filt_ccr), 0.0), mode="nearest"
                )

            if ex_edge:
                ccr = _crop_corr_edges(ccr, ref_img.shape, rotated=True)

            best_correlation = np.max(ccr, axis=2)
            best_angle_index = np.argmax(ccr, axis=2)

            locs_i = _localisation_table(
                fast_peaks2d(best_correlation, min_thresh, peak_kernel_size)
            )

            # MATLAB looks the winning angle up with the raw (uncentred) peak
            # coordinates, so use the columns before the re-centring below; the
            # clip only guards the nearest-pixel rounding at the map edge.
            for row_idx in range(locs_i.shape[0]):
                row = int(
                    np.clip(
                        np.floor(locs_i[row_idx, 1] + 0.5),
                        0,
                        best_angle_index.shape[0] - 1,
                    )
                )
                col = int(
                    np.clip(
                        np.floor(locs_i[row_idx, 0] + 0.5),
                        0,
                        best_angle_index.shape[1] - 1,
                    )
                )
                angle_index = int(best_angle_index[row, col])
                locs_i[row_idx, 7] = -float(angle_array[angle_index])

        else:
            assert ref_img is not None
            ccr = normxcorr2(ref_img, frame_img)

            if filt_ccr > 0:
                ccr = gaussian_filter(ccr, sigma=float(filt_ccr), mode="nearest")

            if ex_edge:
                ccr = _crop_corr_edges(ccr, ref_img.shape, rotated=False)

            locs_i = _localisation_table(
                fast_peaks2d(ccr, min_thresh, peak_kernel_size)
            )
            locs_i[:, 7] = 0.0

        if locs_i.shape[0] == 0:
            continue

        if method_lc == "ccr":
            assert ref_img is not None
            # Column 4 of the MATLAB table holds the correlation value, not the
            # height of the localisation.
            locs_i[:, 3] = locs_i[:, 2]

            # Re-centre the peak positions on the frame.  MATLAB uses a
            # different convention for the ex_edge branch, kept here for parity.
            if ex_edge:
                locs_i[:, 1] += ref_img.shape[0] / 2.0
                locs_i[:, 0] += ref_img.shape[1] / 2.0
            else:
                locs_i[:, 1] = locs_i[:, 1] - ref_img.shape[0] / 2.0 + 0.5
                locs_i[:, 0] = locs_i[:, 0] - ref_img.shape[1] / 2.0 + 0.5

        # Z is the raw height at the localisation in the unfiltered stack.
        locs_i[:, 2] = _lookup_heights(stack, frame_idx, locs_i)

        locs_i[:, 4] = frame_idx
        all_locs.append(locs_i)

    if all_locs:
        locs = np.vstack(all_locs)
        if fastdetect:
            # Positions were found on the half-resolution stack.
            locs[:, 0:2] *= 2.0
    else:
        locs = np.empty((0, len(DETECTOR_COLUMNS)), dtype=np.float64)

    # The per-frame limit marks detections rather than dropping them: every row
    # stays in the table and 'kept' says which ones won their frame's ranking.
    kept = np.ones(locs.shape[0], dtype=bool)
    if max_per_frame is not None:
        kept = _per_frame_keep_mask(locs, max_per_frame)

    # Stable per-detection identifier (NanoLocz 'Part Locs' column 'id') and a
    # placeholder for particle tracking ('Track id', NaN until it runs).
    locs[:, 5] = np.arange(locs.shape[0], dtype=np.float64)
    locs[:, 6] = np.nan

    table = pd.DataFrame(locs, columns=list(DETECTOR_COLUMNS))
    # 'kept' is the curation flag: True for every fresh detection, set to False
    # by the per-frame limit or by a caller/GUI, so that a row can be ignored
    # without re-running detection or renumbering 'id'.
    table["kept"] = kept
    return table


detector.__version__ = "0.1.0"  # type: ignore[attr-defined]

__all__ = [
    "DETECTOR_COLUMNS",
    "detector",
]
