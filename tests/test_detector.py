"""Tests for pnanolocz.detector, fast_peaks2d and normxcorr2.

These modules port MATLAB ``Detector.m``, ``Fast_peaks2D.m`` and the MATLAB
built-in ``normxcorr2`` respectively.

Parity note
-----------
A committed MATLAB reference for ``Detector.m`` on the shared fixture image
(``detector_nanolocz_*.npz``) is not available yet, so the parity claim is
currently backed by hand-computable synthetic cases plus structural checks on
the real AFM fixture.  The golden parity fixture should be added once the
MATLAB export step is run (see CONTRIBUTING.md "Golden fixtures").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from pnanolocz.detector import DETECTOR_COLUMNS, Detector, detector
from pnanolocz.fast_peaks2d import fast_peaks2d
from pnanolocz.normxcorr2 import normxcorr2

# -------------------------------
# Helpers
# -------------------------------


def _blob_stack(
    n_frames: int = 3,
    shape: tuple[int, int] = (64, 64),
    center: tuple[int, int] = (32, 32),
    sigma: float = 1.0,
) -> np.ndarray:
    """Return a frame-first stack with one Gaussian blob per frame."""
    img = np.zeros((n_frames, *shape), dtype=np.float64)
    for f in range(n_frames):
        img[f, center[0], center[1]] = 1.0
        img[f] = gaussian_filter(img[f], sigma=sigma)
    return img


def _run_peak_picker(img: np.ndarray, **kwargs) -> pd.DataFrame:
    params = {
        "method": "Peak picker",
        "ref": 3,
        "filt_img": 0.0,
        "filt_ccr": 0.0,
        "min_thresh": 0.05,
        "ex_edge": False,
        "fastdetect": False,
    }
    params.update(kwargs)
    return detector(img, **params)


# -------------------------------
# detector - Peak picker
# -------------------------------


def test_peak_picker_returns_named_dataframe():
    """Peak picker returns a DataFrame with the documented column names."""
    img = _blob_stack()
    res = _run_peak_picker(img)

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == list(DETECTOR_COLUMNS)
    assert len(res) == 3
    # MATLAB 1-based coordinates and frames by default.
    assert res["x"].tolist() == pytest.approx([33.0, 33.0, 33.0])
    assert res["y"].tolist() == pytest.approx([33.0, 33.0, 33.0])
    assert res["frame"].tolist() == [1.0, 2.0, 3.0]


def test_peak_picker_height_from_unfiltered_image():
    """Z must be looked up from the unfiltered stack even when filt_img > 0."""
    img = _blob_stack()
    res = _run_peak_picker(img, filt_img=0.5)

    raw_center = img[0, 32, 32]
    assert res["z"].tolist() == pytest.approx([raw_center, raw_center, raw_center])


def test_peak_picker_zero_based_frames():
    """``matlab_frame_numbers=False`` gives 0-based frame numbers."""
    img = _blob_stack()
    res = _run_peak_picker(img, matlab_frame_numbers=False)
    assert res["frame"].tolist() == [0.0, 1.0, 2.0]


def test_peak_picker_frame_axis_last_matches_frame_first():
    """``frame_axis=-1`` accepts the MATLAB (H, W, N) layout."""
    first = _run_peak_picker(_blob_stack())
    last_layout = np.moveaxis(_blob_stack(), 0, -1)  # (H, W, N), MATLAB layout
    res = _run_peak_picker(last_layout, frame_axis=-1)
    cols = ["x", "y", "z", "frame"]
    np.testing.assert_allclose(res[cols].values, first[cols].values, rtol=1e-12)


def test_empty_detection_returns_single_nan_row():
    """No detections yield a single all-NaN row, matching MATLAB."""
    res = _run_peak_picker(np.zeros((2, 64, 64)), min_thresh=0.1)
    assert res.shape == (1, 8)
    assert res.isna().all().all()


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
    detector(
        img,
        method="ccr",
        ref=ref,
        filt_img=0.0,
        filt_ccr=0.0,
        min_thresh=0.5,
        ex_edge=False,
        fastdetect=False,
    )
    np.testing.assert_array_equal(ref, ref_before)


# -------------------------------
# detector - CCR
# -------------------------------


def test_ccr_self_reference_correlation_one():
    """Self-reference CCR gives unit correlation and angle 0."""
    img = _blob_stack(n_frames=1)
    ref = img[0].copy()
    res = detector(
        img,
        method="ccr",
        ref=ref,
        filt_img=0.0,
        filt_ccr=0.0,
        min_thresh=0.5,
        ex_edge=False,
        fastdetect=False,
    )
    assert len(res) == 1
    assert res["correlation"].tolist() == pytest.approx([1.0], abs=1e-6)
    assert res["angle"].tolist() == [0.0]


def test_ccr_rotation_picks_matching_angle():
    """A vertical bar template rotated 90 deg matches a horizontal bar image."""
    template = np.zeros((40, 40))
    template[10:30, 18:22] = 1.0  # vertical bar
    img = np.zeros((1, 40, 40))
    img[0, 18:22, 10:30] = 1.0  # horizontal bar

    res = detector(
        img,
        method="ccr",
        ref=template,
        filt_img=0.0,
        filt_ccr=0.0,
        min_thresh=0.5,
        ex_edge=False,
        fastdetect=False,
        angles=[0.0, 90.0],
    )
    assert len(res) == 1
    assert res["correlation"].tolist()[0] > 0.9
    assert res["angle"].tolist() == pytest.approx([-90.0])


def test_ccr_ex_edge_offset_matches_documented_approximation():
    """ex_edge=True/False conventions differ by exactly 0.5 px (documented)."""
    img = _blob_stack(n_frames=1)
    ref = img[0, 26:38, 26:38].copy()  # 12x12 reference around the blob

    no_edge = detector(
        img,
        method="ccr",
        ref=ref,
        filt_img=0.0,
        filt_ccr=0.0,
        min_thresh=0.5,
        ex_edge=False,
        fastdetect=False,
    )
    with_edge = detector(
        img,
        method="ccr",
        ref=ref,
        filt_img=0.0,
        filt_ccr=0.0,
        min_thresh=0.5,
        ex_edge=True,
        fastdetect=False,
    )

    assert len(no_edge) == len(with_edge) == 1
    assert with_edge["x"].iloc[0] - no_edge["x"].iloc[0] == pytest.approx(0.5, abs=0.01)
    assert with_edge["y"].iloc[0] - no_edge["y"].iloc[0] == pytest.approx(0.5, abs=0.01)


def test_ccr_fastdetect_matches_full_resolution_location():
    """Fast-detect locates the same particle as the full-resolution path."""
    img = _blob_stack(n_frames=1)
    ref = img[0, 26:38, 26:38].copy()

    full = detector(
        img,
        method="ccr",
        ref=ref,
        filt_img=0.0,
        filt_ccr=0.0,
        min_thresh=0.5,
        ex_edge=False,
        fastdetect=False,
    )
    fast = detector(
        img,
        method="ccr",
        ref=ref,
        filt_img=0.0,
        filt_ccr=0.0,
        min_thresh=0.5,
        ex_edge=False,
        fastdetect=True,
    )

    assert len(full) == len(fast) == 1
    assert fast["x"].iloc[0] == pytest.approx(full["x"].iloc[0], abs=1.0)
    assert fast["y"].iloc[0] == pytest.approx(full["y"].iloc[0], abs=1.0)


# -------------------------------
# detector - validation
# -------------------------------


def test_invalid_method_raises():
    """Unknown method names raise ValueError."""
    with pytest.raises(ValueError):
        _run_peak_picker(_blob_stack(n_frames=1), method="bogus")


def test_ccr_scalar_ref_raises():
    """CCR with a non-image reference raises ValueError."""
    with pytest.raises(ValueError):
        detector(
            _blob_stack(n_frames=1),
            method="ccr",
            ref=5.0,
            filt_img=0.0,
            filt_ccr=0.0,
            min_thresh=0.5,
            ex_edge=False,
            fastdetect=False,
        )


def test_2d_input_is_promoted_to_one_frame():
    """A 2-D input is promoted to a one-frame stack."""
    res = _run_peak_picker(_blob_stack(n_frames=1)[0])
    assert len(res) == 1
    assert res["frame"].tolist() == [1.0]


def test_detector_alias():
    """The MATLAB-style alias ``Detector`` points at ``detector``."""
    assert Detector is detector


# -------------------------------
# fast_peaks2d
# -------------------------------


def test_fast_peaks2d_coordinates_and_height():
    """Peaks carry 1-based coordinates, heights, and zero prominence."""
    img = np.zeros((50, 50))
    img[10, 20] = 1.0
    img[30, 40] = 1.0
    img = gaussian_filter(img, sigma=1.0)

    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3, matlab_indexing=True)
    assert peaks.shape == (2, 4)
    peaks = peaks[peaks[:, 0].argsort()]
    assert peaks[:, 0].tolist() == pytest.approx([21.0, 41.0])  # x (1-based)
    assert peaks[:, 1].tolist() == pytest.approx([11.0, 31.0])  # y (1-based)
    assert peaks[:, 2].tolist() == pytest.approx([img[10, 20], img[30, 40]])  # height
    assert peaks[:, 3].tolist() == [0.0, 0.0]  # prominence off by default


def test_fast_peaks2d_zero_based_coordinates():
    """``matlab_indexing=False`` gives 0-based coordinates."""
    img = np.zeros((50, 50))
    img[10, 20] = 1.0
    img = gaussian_filter(img, sigma=1.0)

    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=3, matlab_indexing=False)
    assert peaks.shape == (1, 4)
    assert peaks[0, 0] == pytest.approx(20.0)
    assert peaks[0, 1] == pytest.approx(10.0)


# -------------------------------
# normxcorr2
# -------------------------------


def test_normxcorr2_self_correlation_peak():
    """Self-correlation peaks at (H-1, W-1) with value 1."""
    img = gaussian_filter(_blob_stack(n_frames=1)[0], sigma=2.0)
    ccr = normxcorr2(img, img)
    assert ccr.shape == (127, 127)  # full: 2*64 - 1
    assert float(np.nanmax(ccr)) == pytest.approx(1.0, abs=1e-6)
    peak = np.unravel_index(np.nanargmax(ccr), ccr.shape)
    assert peak == (63, 63)  # (H-1, W-1)


def test_normxcorr2_constant_template_returns_zeros():
    """A constant (zero-energy) template gives an all-zero map."""
    img = np.ones((16, 16))
    ccr = normxcorr2(img, img)  # zero template energy
    assert ccr.shape == (31, 31)
    np.testing.assert_array_equal(ccr, np.zeros((31, 31)))


# -------------------------------
# Real AFM fixture (structural)
# -------------------------------


def test_real_afm_fixture_peak_picker(load_npz):
    """Run the detector on the shared real AFM fixture and check structure."""
    raw = load_npz("afm_0_00003_raw.npz")["data"]
    rng = float(np.nanmax(raw) - np.nanmin(raw))
    thresh = max(rng * 0.02, 1e-6)

    res = detector(
        raw,
        method="Peak picker",
        ref=3,
        filt_img=1.0,
        filt_ccr=0.0,
        min_thresh=thresh,
        ex_edge=False,
        fastdetect=False,
    )

    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0
    assert res["x"].between(1, raw.shape[1]).all()
    assert res["y"].between(1, raw.shape[0]).all()
    assert np.isfinite(res["z"]).all()
    assert res["frame"].eq(1.0).all()
