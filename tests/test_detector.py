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

import re

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter, rotate, zoom

from pnanolocz.correlation import normxcorr2
from pnanolocz.detector import (
    DETECTOR_COLUMNS,
    Detections,
    _crop_corr_edges,
    _detect_locs,
    _lookup_heights,
    detector,
    limit_per_frame,
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

    ``savedata.m`` exports ``x, y, Height, ccr, Frame, Angle``;
    """
    assert DETECTOR_COLUMNS == (
        "x",
        "y",
        "z",
        "correlation",
        "frame",
        "angle",
    )


def test_peak_picker_returns_named_dataframe():
    """Test detector with Peak picker returns a DataFrame with correct column names."""
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


def test_peak_picker_tracks_jittered_blob():
    """Test that each frame is detected at its own blob position."""
    res = _run_peak_picker(_blob_stack(n_frames=3, jitter=4))

    assert res["x"].tolist() == pytest.approx([32.0, 36.0, 40.0])
    assert res["y"].tolist() == pytest.approx([32.0, 36.0, 40.0])
    assert res["frame"].tolist() == [0.0, 1.0, 2.0]


def test_peak_picker_height_from_unfiltered_image():
    """Test that Z is looked up from the unfiltered stack even when filt_img > 0."""
    img = _blob_stack()
    res = _run_peak_picker(img, filt_img=0.5)

    raw_center = float(img[0, 32, 32])
    assert res["z"].tolist() == pytest.approx([raw_center] * 3)


def test_peak_picker_skips_frames_without_detections():
    """Test that frames with no peak contribute no rows and frames keep their index."""
    img = np.zeros((3, 64, 64), dtype=np.float64)
    img[1] = _blob_stack(n_frames=1)[0]

    res = _run_peak_picker(img)

    assert len(res) == 1
    assert res["frame"].tolist() == [1.0]


def test_peak_picker_fastdetect_locates_blob():
    """Test fast-detect finds the blob and rescales the position back to full size."""
    res = _run_peak_picker(_blob_stack(n_frames=1), fastdetect=True)

    assert len(res) == 1
    assert res["x"].iloc[0] == pytest.approx(32.0, abs=1.0)
    assert res["y"].iloc[0] == pytest.approx(32.0, abs=1.0)
    assert np.isfinite(res["z"]).all()


def test_empty_detection_returns_empty_table():
    """Check that no detections yield a 0-row table, not a phantom NaN localisation."""
    res = _run_peak_picker(np.zeros((2, 64, 64)), min_thresh=0.1)

    assert isinstance(res, pd.DataFrame)
    assert res.empty
    assert len(res) == 0
    assert list(res.columns) == list(DETECTOR_COLUMNS)


def test_2d_input_is_promoted_to_one_frame():
    """Test that a 2D input is treated as a single frame of a stack."""
    res = _run_peak_picker(_blob_stack(n_frames=1)[0])

    assert len(res) == 1
    assert res["frame"].tolist() == [0.0]


def test_input_array_not_mutated():
    """Test that the detector never modifies the caller's input or reference arrays."""
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
    """Inputs that are neither 2D nor 3D raise ValueError."""
    msg = "img must be 2D or 3D, got shape (2, 2, 2, 2)"
    with pytest.raises(ValueError, match=re.escape(msg)):
        _run_peak_picker(np.zeros((2, 2, 2, 2)))


def test_ccr_scalar_ref_raises():
    """CCR with a non-image reference raises ValueError."""
    with pytest.raises(ValueError, match="2D reference image"):
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
# detector - helpers
# -------------------------------


def test_lookup_heights_uses_the_nearest_pixel_centre():
    """Test that half-pixel positions pick the nearest pixel centre, not the lower."""
    stack = np.zeros((1, 8, 8))
    stack[0] = np.arange(1, 9, dtype=np.float64)[None, :]  # value = column + 1
    detections = Detections(
        x=np.array([3.5, 0.4, 7.4, -0.5, 7.5]),
        y=np.full(5, 2.0),
        z=np.zeros(5),
        correlation=np.zeros(5),
        frame=np.zeros(5, dtype=np.intp),
        angle=np.zeros(5),
    )

    heights = _lookup_heights(stack[0], detections.x, detections.y)
    # 3.5 -> column 4; 0.4 -> column 0; 7.4 -> column 7.  A centre at -0.5 or
    # 7.5 is half a pixel outside the frame, so no pixel is read and z is 0.
    assert heights.tolist() == [5.0, 1.0, 8.0, 0.0, 0.0]


@pytest.mark.parametrize("position", [-1.5, -0.6, -0.5, -0.4, 3.5, 7.4, 7.5, 8.2])
def test_lookup_heights_matches_round_then_bounds(position):
    """Check pixel choice same as MATLAB: round away from zero, then a bounds check."""
    frame = np.arange(1, 9, dtype=np.float64)[None, :].repeat(8, axis=0)

    x = np.array([position], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)

    index = int(np.sign(position) * np.floor(abs(position) + 0.5))  # ties away
    expected = float(frame[2, index]) if 0 <= index < 8 else 0.0

    assert _lookup_heights(frame, x, y).tolist() == [expected]


@pytest.mark.parametrize(
    ("rotated", "rows", "cols"),
    [
        (True, slice(12, 40 - 12 - 1), slice(10, 36 - 10 - 1)),
        (False, slice(12 - 1, 40 - 12), slice(10 - 1, 36 - 10)),
    ],
)
def test_crop_corr_edges_matches_matlab_index_ranges(rotated, rows, cols):
    """Check crop bounds equal MATLAB's round(R)+1:end-R-1 / round(R):end-R ranges."""
    marker = np.arange(40 * 36, dtype=np.float64).reshape(40, 36)

    cropped = _crop_corr_edges(marker, (12, 10), rotated=rotated)

    np.testing.assert_array_equal(cropped, marker[rows, cols])


def test_crop_corr_edges_keeps_the_angle_axis():
    """Test that 3D correlation stack is cropped spatially and keeps every angle."""
    marker = np.arange(2 * 40 * 36, dtype=np.float64).reshape(40, 36, 2)

    cropped = _crop_corr_edges(marker, (12, 10), rotated=True)

    assert cropped.shape == (15, 15, 2)
    np.testing.assert_array_equal(cropped, marker[12:27, 10:25, :])


def test_lookup_heights_returns_zero_outside_frame():
    """Test that heights come from the raw frame, and off-frame positions become 0."""
    stack = np.arange(4 * 8 * 8, dtype=np.float64).reshape(4, 8, 8)

    x = np.array([3.0, -3.0, 3.0], dtype=np.float64)  # inside, left of frame, inside
    y = np.array([2.0, 2.0, 9.0], dtype=np.float64)  # inside, inside, below frame

    heights = _lookup_heights(stack[1], x, y)

    assert heights.tolist() == [float(stack[1, 2, 3]), 0.0, 0.0]


# -------------------------------
# peaks
# -------------------------------


def test_sample_line_follows_x_as_column_y_as_row():
    """Test that horizontal and vertical lines return the matching row and column."""
    rows, cols = np.mgrid[0:8, 0:10]
    ramp = (10.0 * rows + cols).astype(np.float64)

    np.testing.assert_allclose(_sample_line(ramp, 2, 3, 7, 3), ramp[3, 2:8])
    np.testing.assert_allclose(_sample_line(ramp, 4, 1, 4, 6), ramp[1:7, 4])


@pytest.mark.parametrize(
    ("x0", "y0", "x1", "y1", "n_expected"),
    [(0, 0, 5, 3, 6), (2, 2, 2, 2, 2)],
)
def test_sample_line_count(x0, y0, x1, y1, n_expected):
    """Test that one sample per pixel along the longer axis, never fewer than two."""
    assert _sample_line(np.zeros((8, 10)), x0, y0, x1, y1).size == n_expected


def test_sample_line_uses_nearest_neighbour_interpolation():
    """Test that samples are read like MATLAB improfile, whose default is 'nearest'."""
    img = np.arange(36, dtype=np.float64).reshape(6, 6)

    profile = _sample_line(img, 0.0, 0.0, 5.0, 3.0)

    # Bilinear interpolation would return 8.6, 15.4 and 16.4 here.
    np.testing.assert_allclose(profile, [0.0, 7.0, 8.0, 15.0, 16.0, 23.0])


def test_fast_peaks2d_coordinates_and_height():
    """Test that peaks carry 0-based coordinates, heights and zero prominence."""
    img = np.zeros((50, 50))
    img[10, 20] = 1.0
    img[30, 40] = 1.0
    img = gaussian_filter(img, sigma=1.0)

    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3)

    assert len(peaks) == 2

    order = np.argsort(peaks.x)

    assert peaks.x[order].tolist() == [20, 40]
    assert peaks.y[order].tolist() == [10, 30]
    assert peaks.height[order].tolist() == pytest.approx([img[10, 20], img[30, 40]])
    assert peaks.prominence[order].tolist() == [0.0, 0.0]


def test_fast_peaks2d_single_peak_prominence_is_its_height():
    """Test that the highest peak reports its height as prominence, as in MATLAB."""
    img = np.zeros((32, 32))
    img[16, 16] = 1.0
    img = gaussian_filter(img, sigma=1.0)

    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3, min_prom=1e-12)

    assert len(peaks) == 1
    assert peaks.x[0] == pytest.approx(16.0)
    assert peaks.y[0] == pytest.approx(16.0)
    assert peaks.height[0] == pytest.approx(float(img[16, 16]))
    assert peaks.prominence[0] == pytest.approx(float(img[16, 16]))


def test_fast_peaks2d_min_prom_filters_weak_peaks():
    """Test that peaks that do not clear min_prom are dropped, and prom is reported."""
    img = np.zeros((50, 50))
    img[10, 10] = 1.0  # tall peak
    img[30, 40] = 0.6  # shorter peak
    img = gaussian_filter(img, sigma=1.0)

    all_peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3, min_prom=1e-12)
    order = np.argsort(all_peaks.height)

    tall_idx = order[1]
    short_idx = order[0]

    assert all_peaks.prominence[tall_idx] > all_peaks.prominence[short_idx]

    kept = fast_peaks2d(
        img,
        thresh=0.05,
        kernel_size=3,
        min_prom=float(all_peaks.prominence[short_idx]),
    )

    assert len(kept) == 1
    assert kept.x[0] == all_peaks.x[tall_idx]
    assert kept.y[0] == all_peaks.y[tall_idx]
    assert kept.prominence[0] == pytest.approx(all_peaks.prominence[tall_idx])


def test_fast_peaks2d_rejects_non_2d():
    """Test that a stack or higher-dimensional input raises ValueError."""
    with pytest.raises(ValueError, match="fast_peaks2d expects a 2D image"):
        fast_peaks2d(np.zeros((4, 4, 4)), thresh=0.1, kernel_size=3)


# -------------------------------
# correlation
# -------------------------------


def test_normxcorr2_self_correlation_peak():
    """Test a self-correlation peaks at (H-1, W-1) with value 1."""
    img = gaussian_filter(_blob_stack(n_frames=1)[0], sigma=2.0)

    ccr = normxcorr2(img, img)

    assert ccr.shape == (127, 127)  # full: 2*64 - 1
    assert float(np.nanmax(ccr)) == pytest.approx(1.0, abs=1e-6)
    peak = np.unravel_index(np.nanargmax(ccr), ccr.shape)
    assert peak == (63, 63)  # (H-1, W-1)


def _triangle_subpixel_pair(
    size: int = 128, shift: tuple[float, float] = (2.3, -1.7)
) -> tuple[np.ndarray, np.ndarray]:
    """Test frame pair replicating the synthetic sub-pixel alignment data.

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
    """Test that flat windows with tiny ringing must not give spurious peaks above 1."""
    ref, mov = _triangle_subpixel_pair()

    ccr = normxcorr2(ref, mov)  # full-frame template, large near-flat regions

    assert float(np.nanmax(ccr)) <= 1.0 + 1e-9


def test_normxcorr2_constant_template_returns_zeros():
    """Test that a constant (zero-energy) template gives an all-zero map."""
    img = np.ones((16, 16))

    ccr = normxcorr2(img, img)  # zero template energy

    assert ccr.shape == (31, 31)
    np.testing.assert_array_equal(ccr, np.zeros((31, 31)))


def _brute_force_normxcorr2(template: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Test a sliding-window definition of normalized cross-correlation.

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
    """Test that the fast implementation equals the sliding-window definition."""
    rng = np.random.default_rng(1234)
    image = rng.normal(size=(20, 24)) + 5.0
    template = rng.normal(size=(5, 7))

    got = normxcorr2(template, image)
    expected = _brute_force_normxcorr2(template, image)

    np.testing.assert_allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1e3, 1e9])
def test_normxcorr2_is_scale_free(scale):
    """Test that unit changes must not change the correlation."""
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
    """Test that a template bigger than the image raises ValueError, as in MATLAB."""
    with pytest.raises(ValueError, match="Template shape must be"):
        normxcorr2(np.zeros((9, 3)), np.zeros((8, 3)))


def test_normxcorr2_rejects_non_2d():
    """Test that only 2D inputs are accepted."""
    with pytest.raises(ValueError, match="2D template"):
        normxcorr2(np.zeros((4, 4, 4)), np.zeros((8, 8)))


# -------------------------------
# detector - _detect_locs
# -------------------------------


def _detect(stack: np.ndarray, **kwargs: object) -> Detections:
    """Call _detect_locs with peak-picker defaults on an already clean stack."""
    params: dict[str, object] = {
        "method": "peak picker",
        "ref_img": None,
        "peak_kernel_size": 3,
        "filt_img": 0.0,
        "filt_ccr": 0.0,
        "min_thresh": 0.05,
        "ex_edge": False,
        "fastdetect": False,
        "angle_array": np.array([0.0]),
    }
    params.update(kwargs)
    return _detect_locs(stack, **params)  # type: ignore[arg-type]


def test_detect_locs_returns_typed_detections():
    """Check that every field is a 1D array of the same length, with integer frames."""
    locs = _detect(_blob_stack(n_frames=3, jitter=4))

    assert isinstance(locs, Detections)
    assert len(locs) == 3
    for name, values in locs.as_dict().items():
        assert values.shape == (3,), name
    assert locs.frame.dtype == np.intp
    assert locs.x.dtype == locs.z.dtype == np.float64
    assert locs.frame.tolist() == [0, 1, 2]
    assert locs.x.tolist() == [32.0, 36.0, 40.0]


def test_detect_locs_empty_result_keeps_dtypes():
    """Check that no detections gives empty arrays of the usual dtypes not an error."""
    locs = _detect(np.zeros((2, 64, 64)), min_thresh=0.1)

    assert len(locs) == 0
    assert locs.frame.dtype == np.intp
    assert locs.x.dtype == np.float64
    assert list(locs.as_dict()) == list(DETECTOR_COLUMNS)


def test_detector_is_a_thin_wrapper_around_detect_locs():
    """Test that the public detector only validates inputs and wraps _detect_locs."""
    img = _blob_stack(n_frames=2, jitter=3)
    ref = img[0, 26:39, 26:39].copy()

    locs = _detect(img, method="ccr", ref_img=ref, peak_kernel_size=6, min_thresh=0.5)
    table = _run_ccr(img, ref)

    pd.testing.assert_frame_equal(table, pd.DataFrame(locs.as_dict()))


def test_detect_locs_fastdetect_reads_z_before_rescaling():
    """Heights come from the half-size stack; positions are doubled afterwards."""
    stack = _blob_stack(n_frames=1)
    half = zoom(stack, zoom=(1.0, 0.5, 0.5), order=1)

    locs = _detect(stack, fastdetect=True)

    assert len(locs) == 1
    row, col = int(locs.y[0]) // 2, int(locs.x[0]) // 2
    assert (locs.y[0], locs.x[0]) == (2.0 * row, 2.0 * col)
    assert locs.z[0] == pytest.approx(float(half[0, row, col]))


# -------------------------------
# public API
# -------------------------------


def test_public_functions_expose_versions():
    """Test the ported entry points carry a version, like the other modules."""
    assert detector.__version__ == "0.1.0"
    assert fast_peaks2d.__version__ == "0.1.0"
    assert normxcorr2.__version__ == "0.1.0"


# -------------------------------
# Real AFM fixture (structural)
# -------------------------------


def test_real_afm_fixture_peak_picker(load_npz):
    """Test that the detector runs on the shared real AFM fixture."""
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


# -------------------------------
# Test limit_per-frame
# --------------------------------


def test_limit_per_frame_keeps_top_n_in_original_order():
    """Check the best n rows of each frame survive in their original order."""
    detections = pd.DataFrame(
        {
            "frame": [0, 0, 0, 1, 1],
            "correlation": [0.5, 0.9, 0.9, 0.2, 0.8],
            "z": [3.0, 1.0, 2.0, 5.0, 4.0],
        }
    )

    assert limit_per_frame(detections, 2).index.tolist() == [1, 2, 3, 4]
    assert limit_per_frame(detections, 1, by="z").index.tolist() == [0, 3]
    assert len(detections) == 5  # input untouched


@pytest.mark.parametrize("n", [0, -1, 2.5, True])
def test_limit_per_frame_rejects_bad_n(n):
    """Test that N must be a positive integer."""
    with pytest.raises(ValueError, match="positive integer"):
        limit_per_frame(pd.DataFrame({"frame": [0], "correlation": [1.0]}), n)
