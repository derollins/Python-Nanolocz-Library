"""Tests for pnanolocz.detector, pnanolocz.peaks and pnanolocz.correlation.

These modules port MATLAB ``Detector.m``, ``Fast_peaks2D.m`` and the MATLAB
built-in ``normxcorr2`` respectively.

Parity note
-----------
A committed MATLAB reference fixture for the detector is not available yet, so
the parity claim is currently backed by hand-computable synthetic cases, a
brute-force reference implementation of ``normxcorr2``, unit-invariance checks
and structural checks on the shared real AFM fixture.  Golden fixtures should be
added once the MATLAB export step has been run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter, rotate

from pnanolocz.correlation import normxcorr2
from pnanolocz.detector import (
    DETECTOR_COLUMNS,
    _crop_corr_edges,
    _lookup_heights,
    _per_frame_keep_mask,
    detector,
)
from pnanolocz.peaks import _sample_line, fast_peaks2d

# -------------------------------
# Helpers
# -------------------------------


def _blob_stack(
    n_frames: int = 3,
    shape: tuple[int, int] = (64, 64),
    center: tuple[int, int] = (32, 32),
    sigma: float = 1.0,
    jitter: int = 0,
) -> np.ndarray:
    """Return a frame-first stack with one Gaussian blob per frame.

    ``jitter`` moves the blob by an increasing offset from frame to frame so
    that tests catch frame indexing and per-frame bookkeeping mistakes.
    """
    img = np.zeros((n_frames, *shape), dtype=np.float64)
    for frame in range(n_frames):
        img[frame, center[0] + frame * jitter, center[1] + frame * jitter] = 1.0
        img[frame] = gaussian_filter(img[frame], sigma=sigma)
    return img


def _run_peak_picker(img: np.ndarray, **kwargs: object) -> pd.DataFrame:
    """Run the detector in 'Peak picker' mode with convenient defaults."""
    params: dict[str, object] = {
        "method": "Peak picker",
        "ref": 3,
        "filt_img": 0.0,
        "filt_ccr": 0.0,
        "min_thresh": 0.05,
        "ex_edge": False,
        "fastdetect": False,
    }
    params.update(kwargs)
    return detector(img, **params)  # type: ignore[arg-type]


def _run_ccr(img: np.ndarray, ref: np.ndarray, **kwargs: object) -> pd.DataFrame:
    """Run the detector in 'ccr' mode with convenient defaults."""
    params: dict[str, object] = {
        "method": "ccr",
        "ref": ref,
        "filt_img": 0.0,
        "filt_ccr": 0.0,
        "min_thresh": 0.5,
        "ex_edge": False,
        "fastdetect": False,
    }
    params.update(kwargs)
    return detector(img, **params)  # type: ignore[arg-type]


# -------------------------------
# detector - Peak picker
# -------------------------------


def test_columns_match_nanolocz_part_locs_layout():
    """Column names and order follow the NanoLocz 'Part Locs' export.

    ``savedata.m`` exports ``x, y, Height, ccr, Frame, id, Track id, Angle``;
    ``id`` and ``Track id`` are filled after detection, so they are exposed here
    as ``id`` and ``track_id``.  ``kept`` is appended as a pnanolocz addition for
    interactive curation.
    """
    assert DETECTOR_COLUMNS == (
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


def test_peak_picker_returns_named_dataframe():
    """Peak picker returns a DataFrame with the documented column names."""
    res = _run_peak_picker(_blob_stack())

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == list(DETECTOR_COLUMNS)
    assert len(res) == 3
    # 0-based coordinates and frame numbers.
    assert res["x"].tolist() == pytest.approx([32.0, 32.0, 32.0])
    assert res["y"].tolist() == pytest.approx([32.0, 32.0, 32.0])
    assert res["frame"].tolist() == [0.0, 1.0, 2.0]
    # 'Peak picker' puts prominence (here 0, the default) in 'correlation'.
    assert res["correlation"].tolist() == [0.0, 0.0, 0.0]
    assert res["angle"].tolist() == [0.0, 0.0, 0.0]
    # Each detection gets a stable 0-based id; tracking is not ported yet.
    assert res["id"].tolist() == [0.0, 1.0, 2.0]
    assert res["track_id"].isna().all()
    # 'kept' starts True, for a caller or GUI to curate later.
    assert res["kept"].tolist() == [True, True, True]


def test_peak_picker_tracks_jittered_blob():
    """Each frame is detected at its own blob position."""
    res = _run_peak_picker(_blob_stack(n_frames=3, jitter=4))

    assert res["x"].tolist() == pytest.approx([32.0, 36.0, 40.0])
    assert res["y"].tolist() == pytest.approx([32.0, 36.0, 40.0])
    assert res["frame"].tolist() == [0.0, 1.0, 2.0]
    # ids run across the whole table, not per frame
    assert res["id"].tolist() == [0.0, 1.0, 2.0]


def test_peak_picker_height_from_unfiltered_image():
    """Z must be looked up from the unfiltered stack even when filt_img > 0."""
    img = _blob_stack()
    res = _run_peak_picker(img, filt_img=0.5)

    raw_center = float(img[0, 32, 32])
    assert res["z"].tolist() == pytest.approx([raw_center] * 3)


def test_peak_picker_skips_frames_without_detections():
    """Frames with no peak contribute no rows, and frames keep their index."""
    img = np.zeros((3, 64, 64), dtype=np.float64)
    img[1] = _blob_stack(n_frames=1)[0]

    res = _run_peak_picker(img)

    assert len(res) == 1
    assert res["frame"].tolist() == [1.0]


def test_peak_picker_fastdetect_locates_blob():
    """Fast-detect finds the blob and rescales the position back to full size."""
    res = _run_peak_picker(_blob_stack(n_frames=1), fastdetect=True)

    assert len(res) == 1
    assert res["x"].iloc[0] == pytest.approx(32.0, abs=1.0)
    assert res["y"].iloc[0] == pytest.approx(32.0, abs=1.0)
    assert np.isfinite(res["z"]).all()


def test_empty_detection_returns_empty_table():
    """No detections yield a 0-row table, not a phantom NaN localisation."""
    res = _run_peak_picker(np.zeros((2, 64, 64)), min_thresh=0.1)

    assert isinstance(res, pd.DataFrame)
    assert res.empty
    assert len(res) == 0
    assert list(res.columns) == list(DETECTOR_COLUMNS)


def test_2d_input_is_promoted_to_one_frame():
    """A 2-D input is treated as a single frame of a stack."""
    res = _run_peak_picker(_blob_stack(n_frames=1)[0])

    assert len(res) == 1
    assert res["frame"].tolist() == [0.0]


def test_input_array_not_mutated():
    """Detector never modifies the caller's input or reference arrays."""
    img = _blob_stack()
    img[0, 0, 0] = np.nan
    before = img.copy()

    _run_peak_picker(img)
    np.testing.assert_array_equal(img, before)

    ref = np.ones((12, 12))
    ref[0, 0] = np.nan
    ref_before = ref.copy()
    _run_ccr(_blob_stack(n_frames=1), ref)
    np.testing.assert_array_equal(ref, ref_before)


# -------------------------------
# detector - validation
# -------------------------------


def test_invalid_method_raises():
    """Unknown method names raise ValueError."""
    with pytest.raises(ValueError, match="method must be"):
        _run_peak_picker(_blob_stack(n_frames=1), method="bogus")


def test_invalid_ndim_raises():
    """Inputs that are neither 2-D nor 3-D raise ValueError."""
    with pytest.raises(ValueError, match="img must be 2D or 3D"):
        _run_peak_picker(np.zeros((2, 2, 2, 2)))


def test_ccr_scalar_ref_raises():
    """CCR with a non-image reference raises ValueError."""
    with pytest.raises(ValueError, match="2-D reference image"):
        _run_ccr(_blob_stack(n_frames=1), np.asarray(5.0))


def test_peak_picker_non_numeric_ref_raises():
    """A non-numeric peak-picker neighbourhood size raises ValueError."""
    with pytest.raises(ValueError, match="must be a number"):
        _run_peak_picker(_blob_stack(n_frames=1), ref=np.zeros((4, 4)))


# -------------------------------
# detector - CCR
# -------------------------------


def test_ccr_self_reference_correlation_one():
    """Self-reference CCR gives unit correlation and angle 0."""
    img = _blob_stack(n_frames=1)
    res = _run_ccr(img, img[0].copy())

    assert len(res) == 1
    assert res["correlation"].tolist() == pytest.approx([1.0], abs=1e-6)
    assert res["angle"].tolist() == [0.0]


def _l_shaped_template(size: int = 40) -> np.ndarray:
    """Return an 'L' that looks different after every 90 degree rotation.

    A symmetric bar cannot detect a rotation-direction error: rotating it +90 or
    -90 gives the same image.
    """
    template = np.zeros((size, size))
    template[12:28, 12:17] = 1.0  # vertical stem
    template[23:28, 12:26] = 1.0  # horizontal foot
    return template


@pytest.mark.parametrize(
    ("angle_deg", "angles", "wrong_angle"),
    [
        (90.0, [-90.0, 0.0, 90.0], -90.0),
        (-90.0, [-90.0, 0.0, 90.0], 90.0),
        (180.0, [0.0, 180.0], 0.0),
    ],
)
def test_ccr_rotation_picks_matching_angle(angle_deg, angles, wrong_angle):
    """The reported angle is the rotation that matched, with the right sign.

    The template is an 'L', which looks different after every 90 degree
    rotation, so a reversed rotation direction cannot pass this test: the
    mirrored orientation scores well below the match (asserted below), whereas a
    symmetric bar would score the same either way.
    """
    template = _l_shaped_template()
    img = np.zeros((1, 40, 40))
    img[0] = rotate(
        template, angle=angle_deg, reshape=False, order=3, mode="constant", cval=0.0
    )

    res = _run_ccr(img, template, angles=angles)
    mirrored = _run_ccr(img, template, angles=[wrong_angle], min_thresh=0.0)

    assert len(res) >= 1
    best = res.loc[res["correlation"].idxmax()]
    assert best["angle"] == pytest.approx(-angle_deg)
    assert best["correlation"] > 0.9
    assert mirrored["correlation"].max() < 0.8


def test_ccr_ex_edge_offset_matches_matlab_convention():
    """ex_edge=True/False differ by exactly the 0.5 px of Detector.m."""
    img = _blob_stack(n_frames=1)
    ref = img[0, 26:38, 26:38].copy()  # 12x12 reference around the blob

    no_edge = _run_ccr(img, ref)
    with_edge = _run_ccr(img, ref, ex_edge=True)

    assert len(no_edge) == len(with_edge) == 1
    assert with_edge["x"].iloc[0] - no_edge["x"].iloc[0] == pytest.approx(0.5, abs=0.01)
    assert with_edge["y"].iloc[0] - no_edge["y"].iloc[0] == pytest.approx(0.5, abs=0.01)
    # ex_edge=True centres the detection on the blob's own 0-based pixel
    assert with_edge["x"].iloc[0] == pytest.approx(32.0)
    assert with_edge["y"].iloc[0] == pytest.approx(32.0)


def test_ccr_rotational_with_filtering_and_edge_crop():
    """The rotated branch runs with filtering and edge exclusion enabled."""
    img = _blob_stack(n_frames=1)
    ref = img[0, 26:38, 26:38].copy()

    res = _run_ccr(
        img,
        ref,
        filt_img=0.5,
        filt_ccr=0.5,
        min_thresh=0.3,
        ex_edge=True,
        angles=[-90.0, 0.0, 90.0],
    )

    assert len(res) >= 1
    assert res["angle"].abs().max() <= 90.0
    assert np.isfinite(res[["x", "y", "z", "correlation"]].to_numpy()).all()


def test_ccr_filtering_smooths_the_correlation_map():
    """filt_ccr smooths the map in the non-rotated branch, lowering the peak."""
    img = _blob_stack(n_frames=1)
    ref = img[0, 26:38, 26:38].copy()

    plain = _run_ccr(img, ref)
    filtered = _run_ccr(img, ref, filt_ccr=1.0)

    assert len(filtered) >= 1
    assert np.isfinite(filtered["correlation"]).all()
    assert filtered["correlation"].max() < plain["correlation"].max()


def test_ccr_fastdetect_matches_full_resolution_location():
    """Fast-detect locates the same particle as the full-resolution path."""
    img = _blob_stack(n_frames=1)
    ref = img[0, 26:38, 26:38].copy()

    full = _run_ccr(img, ref)
    fast = _run_ccr(img, ref, fastdetect=True)

    assert len(full) == len(fast) == 1
    assert fast["x"].iloc[0] == pytest.approx(full["x"].iloc[0], abs=1.0)
    assert fast["y"].iloc[0] == pytest.approx(full["y"].iloc[0], abs=1.0)


def test_ccr_detects_particles_in_small_valued_data():
    """Correlation is unit free: data in metres must still be detected.

    An absolute denominator floor used to zero the whole correlation map for
    data whose values are far below 1, so the detector found nothing.
    """
    img = _blob_stack(n_frames=1) * 1e-9  # nanometres expressed in metres
    ref = img[0, 26:38, 26:38].copy()

    res = _run_ccr(img, ref)

    raw_center = float(_blob_stack(n_frames=1)[0, 32, 32])

    assert len(res) == 1
    assert res["correlation"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert res["z"].iloc[0] == pytest.approx(1e-9 * raw_center)


# -------------------------------
# detector - per-frame limit and curation flag
# -------------------------------


def test_kept_column_exists_for_empty_table():
    """An empty result still carries every documented column, 'kept' included."""
    res = _run_peak_picker(np.zeros((2, 64, 64)), min_thresh=0.1)

    assert res.empty
    assert list(res.columns) == list(DETECTOR_COLUMNS)
    assert res["kept"].dtype == bool


def test_per_frame_keep_mask_marks_the_highest_scores():
    """_per_frame_keep_mask marks the top-N correlation rows of each frame."""
    locs = np.zeros((6, len(DETECTOR_COLUMNS)), dtype=np.float64)
    locs[:, 4] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # frame
    locs[:, 3] = [0.1, 0.9, 0.2, 0.3, 0.4, 0.8]  # correlation

    keep = _per_frame_keep_mask(locs, 2)

    assert keep.tolist() == [False, True, True, False, True, True]


def test_per_frame_keep_mask_breaks_ties_by_detection_order():
    """Equal correlations keep the earlier detections, so results are stable."""
    locs = np.zeros((3, len(DETECTOR_COLUMNS)), dtype=np.float64)
    locs[:, 4] = 0.0

    keep = _per_frame_keep_mask(locs, 2)

    assert keep.tolist() == [True, True, False]


def _narrow_and_wide_frame() -> tuple[np.ndarray, np.ndarray]:
    """One frame holding a blob matching the reference and a much wider one."""
    narrow = np.zeros((64, 64))
    narrow[32, 20] = 1.0
    narrow = gaussian_filter(narrow, sigma=1.0)
    wide = np.zeros((64, 64))
    wide[32, 44] = 1.0
    wide = gaussian_filter(wide, sigma=3.0)
    return (narrow + wide)[np.newaxis], narrow[26:38, 14:26].copy()


def test_max_particles_per_frame_marks_the_best_match():
    """Per-frame limiting flags the best correlation but drops no rows."""
    img, ref = _narrow_and_wide_frame()

    unlimited = _run_ccr(img, ref, min_thresh=0.1)
    limited = _run_ccr(img, ref, min_thresh=0.1, max_particles_per_frame=1)

    assert len(unlimited) == 2  # the matching blob and the wider one
    assert len(limited) == 2  # the weaker match stays in the table ...
    assert limited["kept"].tolist() == [True, False]  # ... flagged as not kept
    assert limited["id"].tolist() == [0.0, 1.0]  # every row keeps its id
    best = limited.loc[limited["correlation"].idxmax()]
    assert limited.loc[limited["kept"], "id"].tolist() == [best["id"]]
    assert limited.loc[limited["kept"], "x"].iloc[0] == pytest.approx(best["x"])


def test_max_particles_per_frame_applies_to_each_frame():
    """Each frame is ranked on its own, so every frame keeps its best row."""
    frame, ref = _narrow_and_wide_frame()
    img = np.repeat(frame, 2, axis=0)

    res = _run_ccr(img, ref, min_thresh=0.1, max_particles_per_frame=1)

    assert res["frame"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert res["kept"].tolist() == [True, False, True, False]
    assert res["id"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_max_particles_per_frame_rejects_peak_picker():
    """The limit ranks by correlation, which 'Peak picker' does not produce."""
    with pytest.raises(ValueError, match="only supported for method='ccr'"):
        _run_peak_picker(_blob_stack(n_frames=1), max_particles_per_frame=1)


def test_per_frame_limit_is_optional():
    """The limit is off unless asked for; True uses the default count of 1."""
    img, ref = _narrow_and_wide_frame()
    kw = {"min_thresh": 0.1}

    assert _run_ccr(img, ref, **kw)["kept"].sum() == 2  # off by default
    assert _run_ccr(img, ref, max_particles_per_frame=None, **kw)["kept"].sum() == 2
    assert _run_ccr(img, ref, max_particles_per_frame=False, **kw)["kept"].sum() == 2
    assert _run_ccr(img, ref, max_particles_per_frame=True, **kw)["kept"].sum() == 1
    assert _run_ccr(img, ref, max_particles_per_frame=2, **kw)["kept"].sum() == 2
    # both rows stay in the table whichever limit is used
    assert len(_run_ccr(img, ref, max_particles_per_frame=True, **kw)) == 2


def test_max_particles_per_frame_must_be_positive():
    """A non-positive per-frame limit raises ValueError."""
    img, ref = _narrow_and_wide_frame()

    with pytest.raises(ValueError, match="max_particles_per_frame"):
        _run_ccr(img, ref, max_particles_per_frame=0)


# -------------------------------
# detector - helpers
# -------------------------------


def test_lookup_heights_uses_the_nearest_pixel_centre():
    """Half-pixel positions pick the nearest pixel centre, not the lower one."""
    stack = np.zeros((1, 8, 8))
    stack[0] = np.arange(1, 9, dtype=np.float64)[None, :]  # value = column + 1
    locs = np.zeros((5, len(DETECTOR_COLUMNS)), dtype=np.float64)
    locs[:, 0] = [3.5, 0.4, 7.4, -0.5, 7.5]  # x
    locs[:, 1] = 2.0  # y

    heights = _lookup_heights(stack, 0, locs)

    # 3.5 -> column 4; 0.4 -> column 0; 7.4 -> column 7.  A centre at -0.5 or
    # 7.5 is half a pixel outside the frame, so no pixel is read and z is 0.
    assert heights.tolist() == [5.0, 1.0, 8.0, 0.0, 0.0]


@pytest.mark.parametrize("position", [-1.5, -0.6, -0.5, -0.4, 3.5, 7.4, 7.5, 8.2])
def test_lookup_heights_matches_round_then_bounds(position):
    """Same pixel as MATLAB: round ties away from zero, then a bounds check."""
    stack = np.zeros((1, 8, 8))
    stack[0] = np.arange(1, 9, dtype=np.float64)[None, :]
    locs = np.zeros((1, len(DETECTOR_COLUMNS)), dtype=np.float64)
    locs[0, 0] = position
    locs[0, 1] = 2.0

    index = int(np.sign(position) * np.floor(abs(position) + 0.5))  # ties away
    expected = float(stack[0, 2, index]) if 0 <= index < 8 else 0.0

    assert _lookup_heights(stack, 0, locs).tolist() == [expected]


@pytest.mark.parametrize(
    ("rotated", "rows", "cols"),
    [
        (True, slice(12, 40 - 12 - 1), slice(10, 36 - 10 - 1)),
        (False, slice(12 - 1, 40 - 12), slice(10 - 1, 36 - 10)),
    ],
)
def test_crop_corr_edges_matches_matlab_index_ranges(rotated, rows, cols):
    """Crop bounds equal MATLAB's round(R)+1:end-R-1 / round(R):end-R ranges."""
    marker = np.arange(40 * 36, dtype=np.float64).reshape(40, 36)

    cropped = _crop_corr_edges(marker, (12, 10), rotated=rotated)

    np.testing.assert_array_equal(cropped, marker[rows, cols])


def test_crop_corr_edges_keeps_the_angle_axis():
    """A 3-D correlation stack is cropped spatially and keeps every angle."""
    marker = np.arange(2 * 40 * 36, dtype=np.float64).reshape(40, 36, 2)

    cropped = _crop_corr_edges(marker, (12, 10), rotated=True)

    assert cropped.shape == (15, 15, 2)
    np.testing.assert_array_equal(cropped, marker[12:27, 10:25, :])


def test_lookup_heights_returns_zero_outside_frame():
    """Heights come from the raw stack, and off-frame positions become 0."""
    stack = np.arange(4 * 8 * 8, dtype=np.float64).reshape(4, 8, 8)
    locs = np.zeros((3, len(DETECTOR_COLUMNS)), dtype=np.float64)
    locs[:, 0] = [3.0, -3.0, 3.0]  # x: inside, outside (left), inside
    locs[:, 1] = [2.0, 2.0, 9.0]  # y: inside, inside, outside (below)

    heights = _lookup_heights(stack, 1, locs)

    assert heights.tolist() == [float(stack[1, 2, 3]), 0.0, 0.0]


# -------------------------------
# peaks
# -------------------------------


def test_sample_line_follows_x_as_column_y_as_row():
    """Horizontal and vertical lines return the matching row and column."""
    rows, cols = np.mgrid[0:8, 0:10]
    ramp = (10.0 * rows + cols).astype(np.float64)

    np.testing.assert_allclose(_sample_line(ramp, 2, 3, 7, 3), ramp[3, 2:8])
    np.testing.assert_allclose(_sample_line(ramp, 4, 1, 4, 6), ramp[1:7, 4])


@pytest.mark.parametrize(
    ("x0", "y0", "x1", "y1", "n_expected"),
    [(0, 0, 5, 3, 6), (2, 2, 2, 2, 2)],
)
def test_sample_line_count(x0, y0, x1, y1, n_expected):
    """One sample per pixel along the longer axis, and never fewer than two."""
    assert _sample_line(np.zeros((8, 10)), x0, y0, x1, y1).size == n_expected


def test_sample_line_uses_nearest_neighbour_interpolation():
    """Samples are read like MATLAB improfile, whose default is 'nearest'."""
    img = np.arange(36, dtype=np.float64).reshape(6, 6)

    profile = _sample_line(img, 0.0, 0.0, 5.0, 3.0)

    # Bilinear interpolation would return 8.6, 15.4 and 16.4 here.
    np.testing.assert_allclose(profile, [0.0, 7.0, 8.0, 15.0, 16.0, 23.0])


def test_fast_peaks2d_coordinates_and_height():
    """Peaks carry 0-based coordinates, heights and zero prominence."""
    img = np.zeros((50, 50))
    img[10, 20] = 1.0
    img[30, 40] = 1.0
    img = gaussian_filter(img, sigma=1.0)

    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3)

    assert peaks.shape == (2, 4)
    peaks = peaks[peaks[:, 0].argsort()]
    assert peaks[:, 0].tolist() == pytest.approx([20.0, 40.0])  # x
    assert peaks[:, 1].tolist() == pytest.approx([10.0, 30.0])  # y
    assert peaks[:, 2].tolist() == pytest.approx([img[10, 20], img[30, 40]])
    assert peaks[:, 3].tolist() == [0.0, 0.0]  # prominence off by default


def test_fast_peaks2d_single_peak_prominence_is_its_height():
    """The highest peak reports its own height as prominence, as in MATLAB."""
    img = np.zeros((32, 32))
    img[16, 16] = 1.0
    img = gaussian_filter(img, sigma=1.0)

    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3, min_prom=1e-12)

    assert peaks.shape == (1, 4)
    assert peaks[0, 0] == pytest.approx(16.0)
    assert peaks[0, 1] == pytest.approx(16.0)
    assert peaks[0, 2] == pytest.approx(float(img[16, 16]))
    assert peaks[0, 3] == pytest.approx(float(img[16, 16]))


def test_fast_peaks2d_min_prom_filters_weak_peaks():
    """Peaks that do not clear min_prom are dropped, and prom is reported."""
    img = np.zeros((50, 50))
    img[10, 10] = 1.0  # tall peak
    img[30, 40] = 0.6  # shorter peak
    img = gaussian_filter(img, sigma=1.0)

    all_peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3, min_prom=1e-12)
    assert all_peaks.shape == (2, 4)
    tall, short = all_peaks[np.argsort(all_peaks[:, 2])[::-1]]
    assert tall[3] > short[3]

    kept = fast_peaks2d(img, thresh=0.05, kernel_size=3, min_prom=float(short[3]))

    assert kept.shape == (1, 4)
    assert kept[0, 0] == pytest.approx(tall[0])
    assert kept[0, 1] == pytest.approx(tall[1])
    assert kept[0, 3] == pytest.approx(tall[3])


def test_fast_peaks2d_rejects_non_2d():
    """A stack or higher-dimensional input raises ValueError."""
    with pytest.raises(ValueError, match="2-D image"):
        fast_peaks2d(np.zeros((4, 4, 4)), thresh=0.1, kernel_size=3)


# -------------------------------
# correlation
# -------------------------------


def test_normxcorr2_self_correlation_peak():
    """Self-correlation peaks at (H-1, W-1) with value 1."""
    img = gaussian_filter(_blob_stack(n_frames=1)[0], sigma=2.0)

    ccr = normxcorr2(img, img)

    assert ccr.shape == (127, 127)  # full: 2*64 - 1
    assert float(np.nanmax(ccr)) == pytest.approx(1.0, abs=1e-6)
    peak = np.unravel_index(np.nanargmax(ccr), ccr.shape)
    assert peak == (63, 63)  # (H-1, W-1)


def _triangle_subpixel_pair(
    size: int = 128, shift: tuple[float, float] = (2.3, -1.7)
) -> tuple[np.ndarray, np.ndarray]:
    """Frame pair replicating the synthetic sub-pixel alignment data.

    Binary triangle + Gaussian edge smoothing + cubic-spline sub-pixel shift;
    this combination makes the window variance of flat regions cancel to
    floating-point noise, which used to produce spurious correlation peaks far
    above 1.
    """
    from scipy.ndimage import shift as spline_shift

    w = max(5, int(size * 0.20))
    h = int(np.ceil(w * np.sqrt(3) / 2))
    mask = np.zeros((h, w), dtype=bool)
    for row in range(h):
        half = int((1.0 - (row + 0.5) / h) * w / 2)
        if half > 0:
            mask[row, half : w - half] = True
        elif w % 2 == 1:
            mask[row, w // 2] = True

    img = np.zeros((size, size))
    y0 = int(round(size / 2 - h / 2))
    x0 = int(round(size / 2 - w / 2))
    img[y0 : y0 + h, x0 : x0 + w] = mask.astype(np.float64)
    img = gaussian_filter(img, sigma=1.5, mode="constant", cval=0.0)
    img = img / img.max()

    ref = spline_shift(img, (0.0, 0.0), order=3, prefilter=True)
    mov = spline_shift(img, shift, order=3, prefilter=True)
    return ref, mov


def test_normxcorr2_flat_window_no_spurious_peak():
    """Flat windows with tiny ringing must not give spurious peaks above 1."""
    ref, mov = _triangle_subpixel_pair()

    ccr = normxcorr2(ref, mov)  # full-frame template, large near-flat regions

    assert float(np.nanmax(ccr)) <= 1.0 + 1e-9


def test_normxcorr2_constant_template_returns_zeros():
    """A constant (zero-energy) template gives an all-zero map."""
    img = np.ones((16, 16))

    ccr = normxcorr2(img, img)  # zero template energy

    assert ccr.shape == (31, 31)
    np.testing.assert_array_equal(ccr, np.zeros((31, 31)))


def _brute_force_normxcorr2(template: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Sliding-window definition of normalized cross-correlation.

    Independent of the FFT/running-sum implementation, with the same zero
    padding MATLAB uses at the edges.
    """
    t_rows, t_cols = template.shape
    template_zm = template - float(np.mean(template))
    padded = np.pad(image, ((t_rows - 1, t_rows - 1), (t_cols - 1, t_cols - 1)))
    out = np.zeros((padded.shape[0] - t_rows + 1, padded.shape[1] - t_cols + 1))
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            window = padded[row : row + t_rows, col : col + t_cols]
            window_zm = window - float(np.mean(window))
            denom = float(np.sqrt(np.sum(window_zm**2) * np.sum(template_zm**2)))
            out[row, col] = (
                0.0 if denom == 0.0 else float(np.sum(window_zm * template_zm) / denom)
            )
    return out


def test_normxcorr2_matches_brute_force_definition():
    """The fast implementation equals the sliding-window definition."""
    rng = np.random.default_rng(1234)
    image = rng.normal(size=(20, 24)) + 5.0
    template = rng.normal(size=(5, 7))

    got = normxcorr2(template, image)
    expected = _brute_force_normxcorr2(template, image)

    np.testing.assert_allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1e3, 1e9])
def test_normxcorr2_is_scale_free(scale):
    """Unit changes must not change the correlation (issue: absolute floor)."""
    ref, mov = _triangle_subpixel_pair()
    base = normxcorr2(ref, mov)
    peak = np.unravel_index(int(np.nanargmax(base)), base.shape)

    scaled = normxcorr2(ref * scale, mov * scale)

    # The match itself, and the peak location, are unchanged ...
    assert float(scaled[peak]) == pytest.approx(float(base[peak]), abs=1e-9)
    assert np.unravel_index(int(np.nanargmax(scaled)), scaled.shape) == peak
    assert float(np.nanmax(scaled)) > 0.9
    # ... while numerically flat windows, whose variance is at the round-off
    # level, can flip between 0 and a few thousandths.
    np.testing.assert_allclose(scaled, base, atol=1e-2)


def test_normxcorr2_rejects_template_larger_than_image():
    """A template bigger than the image raises ValueError, as in MATLAB."""
    with pytest.raises(ValueError, match="Template shape must be"):
        normxcorr2(np.zeros((9, 3)), np.zeros((8, 3)))


def test_normxcorr2_rejects_non_2d():
    """Only 2-D inputs are accepted."""
    with pytest.raises(ValueError, match="2-D template"):
        normxcorr2(np.zeros((4, 4, 4)), np.zeros((8, 8)))


# -------------------------------
# public API
# -------------------------------


def test_public_functions_expose_versions():
    """The ported entry points carry a version, like the other modules."""
    assert detector.__version__ == "0.1.0"
    assert fast_peaks2d.__version__ == "0.1.0"
    assert normxcorr2.__version__ == "0.1.0"


# -------------------------------
# Real AFM fixture (structural)
# -------------------------------


def test_real_afm_fixture_peak_picker(load_npz):
    """Run the detector on the shared real AFM fixture and check structure."""
    raw = load_npz("afm_0_00003_raw.npz")["data"]
    rng = float(np.nanmax(raw) - np.nanmin(raw))
    thresh = max(rng * 0.02, 1e-6)

    res = _run_peak_picker(raw, ref=3, filt_img=1.0, min_thresh=thresh)

    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0
    assert res["x"].between(0, raw.shape[1] - 1).all()
    assert res["y"].between(0, raw.shape[0] - 1).all()
    assert np.isfinite(res["z"]).all()
    assert res["frame"].eq(0.0).all()
