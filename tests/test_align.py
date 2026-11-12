"""Tests for the consolidated :mod:`pnanolocz.align` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter, rotate, shift

from pnanolocz.align import (
    _parabolic_offset,
    align_movie,
    align_rot,
    align_trans,
    im_to_polar,
)


def _blob(
    shape: tuple[int, int] = (64, 64), center: tuple[int, int] = (32, 32)
) -> np.ndarray:
    """Return a smooth Gaussian blob, used as a simple particle image."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.exp(
        -((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2.0 * 1.5**2)
    )


def _asymmetric(shape: tuple[int, int] = (96, 96)) -> np.ndarray:
    """Return an asymmetric shape so rotation angles are well defined."""
    img = np.zeros(shape, dtype=np.float64)
    img[30:52, 34:41] = 1.0  # vertical stem
    img[46:52, 34:62] = 0.8  # horizontal foot
    img[20:26, 52:58] = 0.6  # marker that breaks symmetry
    return gaussian_filter(img, sigma=1.0)


def _shifted_pair(dy: float = 2.0, dx: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a reference frame and a copy shifted by ``dy``/``dx`` pixels."""
    ref = _blob()
    moved = shift(ref, shift=(dy, dx), order=3, mode="constant", cval=0.0)
    return ref, moved


def test_align_trans_schema_and_identity() -> None:
    """A reference correlated with itself gives zero shifts and the right columns."""
    ref = _blob()
    res = align_trans(np.stack([ref, ref]), ref)

    assert list(res.columns) == ["frame", "dx", "dy"]
    assert len(res) == 2
    assert res["frame"].tolist() == [0, 1]
    assert np.issubdtype(res["frame"].dtype, np.integer)
    assert res["dx"].tolist() == pytest.approx([0.0, 0.0], abs=0.02)
    assert res["dy"].tolist() == pytest.approx([0.0, 0.0], abs=0.02)


def test_align_trans_recovers_known_shift() -> None:
    """Both correlation methods should recover a known integer translation."""
    ref, moved = _shifted_pair(dy=2.0, dx=3.0)
    stack = np.stack([ref, moved])

    fft = align_trans(stack, ref, method="FFT cross")
    cross = align_trans(stack, ref, method="Cross corr")

    # The returned values are displacements of the moved frame.
    for result in (fft, cross):
        assert result.loc[1, "dx"] == pytest.approx(3.0, abs=0.35)
        assert result.loc[1, "dy"] == pytest.approx(2.0, abs=0.35)


def test_align_trans_accepts_2d_image() -> None:
    """A single 2D image is treated as one frame."""
    ref = _blob()
    res = align_trans(ref, ref)

    assert len(res) == 1
    assert res.loc[0, "frame"] == 0
    assert res.loc[0, "dx"] == pytest.approx(0.0, abs=0.02)
    assert res.loc[0, "dy"] == pytest.approx(0.0, abs=0.02)


def test_align_rot_recovers_known_angle() -> None:
    """A target rotated by +20 degrees should need about -20 degrees."""
    ref = _asymmetric()
    target = rotate(ref, angle=20.0, reshape=False, order=3, mode="constant", cval=0.0)

    res = align_rot(ref, target, angle_range=(-60.0, 60.0))

    assert list(res.columns) == ["frame", "angle"]
    assert len(res) == 1
    assert res.loc[0, "angle"] == pytest.approx(-20.0, abs=1.0)


def test_align_rot_accepts_stack_and_keeps_ids() -> None:
    """A 3D target gives one row per frame, with 0-based frame numbers."""
    ref = _asymmetric()
    target = rotate(ref, angle=15.0, reshape=False, order=3, mode="constant", cval=0.0)

    res = align_rot(ref, np.stack([ref, target]), angle_range=(-45.0, 45.0))

    assert len(res) == 2
    assert res["frame"].tolist() == [0, 1]
    assert res.loc[0, "angle"] == pytest.approx(0.0, abs=0.01)
    assert res.loc[1, "angle"] == pytest.approx(-15.0, abs=1.0)


def test_align_movie_recovers_drift() -> None:
    """Movie drift uses the same displacement convention as align_trans."""
    ref, moved = _shifted_pair(dy=1.0, dx=-2.0)
    stack = np.stack([ref, moved])

    res = align_movie(stack, ref, method="FFT cross")

    assert list(res.columns) == ["frame", "dx", "dy"]
    assert len(res) == 2
    assert res.loc[0, "dx"] == pytest.approx(0.0, abs=0.02)
    assert res.loc[0, "dy"] == pytest.approx(0.0, abs=0.02)
    assert res.loc[1, "dx"] == pytest.approx(-2.0, abs=0.35)
    assert res.loc[1, "dy"] == pytest.approx(1.0, abs=0.35)


def test_align_movie_accepts_explicit_reference() -> None:
    """With ``full_image=False``, an explicit crop reference is used."""
    ref, moved = _shifted_pair(dy=1.0, dx=1.0)
    stack = np.stack([ref, moved])

    res = align_movie(
        stack,
        ref,
        ref_position=(0.0, 0.0, 64.0, 64.0),
        full_image=False,
        method="FFT cross",
    )

    assert len(res) == 2
    assert res.loc[1, "dx"] == pytest.approx(1.0, abs=0.35)
    assert res.loc[1, "dy"] == pytest.approx(1.0, abs=0.35)


def test_empty_stack_returns_empty_table() -> None:
    """An empty stack keeps the schema instead of raising."""
    ref = _blob()
    empty = np.zeros((0, 64, 64), dtype=np.float64)

    trans = align_trans(empty, ref)
    rot = align_rot(ref, empty)
    movie = align_movie(empty, ref)

    for table, columns in (
        (trans, ["frame", "dx", "dy"]),
        (rot, ["frame", "angle"]),
        (movie, ["frame", "dx", "dy"]),
    ):
        assert table.empty
        assert list(table.columns) == columns


def test_im_to_polar_shape() -> None:
    """The polar transform keeps the requested sampling shape."""
    polar = im_to_polar(_blob(), 0.0, 1.0, 12, 36)

    assert polar.shape == (12, 36)
    assert np.isfinite(polar).all()


def test_im_to_polar_rejects_nonfinite_radius() -> None:
    """The radius range must be finite."""
    with pytest.raises(ValueError, match="r_min and r_max must be finite"):
        im_to_polar(_blob(), 0.0, np.inf, 12, 36)


def test_parabolic_offset_handles_border_peak_and_asymmetry() -> None:
    """Border peaks are not refined; asymmetric peaks get a signed offset."""
    corr = np.zeros((8, 8), dtype=np.float64)
    corr[0, 0] = 1.0
    xo, yo = _parabolic_offset(corr, 0, 0)
    assert xo == pytest.approx(0.0, abs=0.02)
    assert yo == pytest.approx(0.0, abs=0.02)

    corr = np.zeros((8, 8), dtype=np.float64)
    corr[4, 4] = 1.0
    xo, yo = _parabolic_offset(corr, 4, 4)
    assert xo == pytest.approx(0.0, abs=0.02)
    assert yo == pytest.approx(0.0, abs=0.02)

    corr = np.zeros((8, 8), dtype=np.float64)
    corr[4, 4] = 1.0
    corr[4, 5] = 0.8
    corr[4, 3] = 0.2
    xo, _ = _parabolic_offset(corr, 4, 4)
    assert xo > 0.0


def test_align_movie_requires_reference_for_full_image() -> None:
    """``full_image=True`` needs an explicit reference."""
    with pytest.raises(ValueError, match="ref is required"):
        align_movie(_blob(), None)


def test_pixel_shift_must_be_finite_and_non_negative() -> None:
    """Invalid local-search limits are rejected consistently."""
    ref = _blob()

    with pytest.raises(ValueError, match="pixel_shift"):
        align_trans(ref, ref, pixel_shift=-1.0)

    with pytest.raises(ValueError, match="pixel_shift"):
        align_movie(ref, ref, pixel_shift=np.nan)


def test_invalid_methods_and_shapes_raise() -> None:
    """Bad method names, angle ranges and non-2D references are rejected."""
    ref = _blob()

    with pytest.raises(ValueError, match="method must be"):
        align_trans(ref, ref, method="bogus")

    with pytest.raises(ValueError, match="method must be"):
        align_rot(ref, ref, method="bogus")

    with pytest.raises(ValueError, match="min < max"):
        align_rot(ref, ref, angle_range=(10.0, 0.0))

    with pytest.raises(ValueError, match="ref must be 2D"):
        align_movie(ref, np.zeros((2, 64, 64)))

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        align_trans(np.zeros((2, 2, 2, 2)), ref)


def test_public_functions_expose_versions() -> None:
    """The public entry points carry a version, like the other modules."""
    assert align_trans.__version__ == "0.1.0"
    assert align_rot.__version__ == "0.1.0"
    assert align_movie.__version__ == "0.1.0"
    assert im_to_polar.__version__ == "0.1.0"


def test_result_tables_are_not_inputs() -> None:
    """Alignment tables are plain DataFrames with only computed columns."""
    ref = _blob()
    res = align_trans(np.stack([ref, ref]), ref)

    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"frame", "dx", "dy"}
    assert "angle" not in res.columns
    assert "id" not in res.columns
    assert "kept" not in res.columns
