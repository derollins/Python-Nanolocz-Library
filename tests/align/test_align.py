"""Tests for :mod:`pnanolocz.align.align`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import rotate

from pnanolocz.align import (
    ROTATION_COLUMNS,
    TRANSLATION_COLUMNS,
    Rotation,
    Translation,
    align_movie,
    align_rot,
    align_trans,
    im_to_polar,
)
from pnanolocz.align.align import (
    _align_movie,
    _align_rotation,
    _align_translation,
    _as_image,
    _as_stack,
    _method_key,
    _parabolic_offset,
    _validate_angle_range,
    _validate_pixel_shift,
    _validate_ref_position,
)

# --- records -------------------------------------------------------------


def test_translation_record_contract() -> None:
    """The translation record exposes its fields and mirrors them as a dict."""
    record = Translation(
        frame=np.array([0, 1], dtype=np.intp),
        dx=np.array([0.5, 1.5]),
        dy=np.array([-0.5, 2.0]),
    )

    assert len(record) == 2
    assert list(record.as_dict()) == ["frame", "dx", "dy"]
    assert record.as_dict()["dx"] is record.dx
    assert not hasattr(record, "__dict__")  # slots=True


def test_translation_record_is_frozen() -> None:
    """Fields cannot be rebound after construction."""
    record = Translation(
        frame=np.array([0], dtype=np.intp),
        dx=np.array([0.0]),
        dy=np.array([0.0]),
    )

    with pytest.raises(FrozenInstanceError):
        record.dx = np.array([1.0])  # type: ignore[misc]


def test_rotation_record_contract() -> None:
    """The rotation record exposes its fields and is frozen."""
    record = Rotation(
        frame=np.array([0, 1], dtype=np.intp),
        angle=np.array([-20.0, 0.0]),
    )

    assert len(record) == 2
    assert list(record.as_dict()) == ["frame", "angle"]

    with pytest.raises(FrozenInstanceError):
        record.angle = np.array([0.0, 0.0])  # type: ignore[misc]


def test_column_constants_match_records() -> None:
    """The exported column tuples follow the frozen record fields."""
    assert TRANSLATION_COLUMNS == ("frame", "dx", "dy")
    assert ROTATION_COLUMNS == ("frame", "angle")


# --- shared helpers ------------------------------------------------------


def test_as_stack_promotes_2d_and_zeroes_nonfinite() -> None:
    """A 2D image becomes a single frame and NaN/Inf values become zero."""
    stack = _as_stack(np.array([[np.nan, 1.0], [np.inf, -np.inf]]))

    assert stack.shape == (1, 2, 2)
    assert stack.dtype == np.float64
    assert np.isfinite(stack).all()
    assert stack[0, 0, 0] == 0.0
    assert stack[0, 1, 0] == 0.0
    assert stack[0, 1, 1] == 0.0


def test_as_stack_rejects_wrong_rank() -> None:
    """Only 2D and 3D input is accepted."""
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        _as_stack(np.zeros((2, 2, 2, 2)))


def test_as_image_requires_2d() -> None:
    """The reference must be a single 2D image."""
    with pytest.raises(ValueError, match="ref must be 2D"):
        _as_image(np.zeros((2, 64, 64)))


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


def test_validate_pixel_shift_rejects_negative_and_nonfinite() -> None:
    """The local-search limit must be finite and non-negative."""
    for value in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="pixel_shift"):
            _validate_pixel_shift(value)


def test_validate_angle_range_rejects_bad_ranges() -> None:
    """The angle range needs two finite values with min < max."""
    with pytest.raises(ValueError, match="two values"):
        _validate_angle_range((1.0,))

    with pytest.raises(ValueError, match="must be finite"):
        _validate_angle_range((0.0, np.inf))

    with pytest.raises(ValueError, match="min < max"):
        _validate_angle_range((1.0, 1.0))


def test_validate_ref_position_normalises_and_checks_values() -> None:
    """The optional crop rectangle is normalised to floats and validated."""
    assert _validate_ref_position(None) is None
    assert _validate_ref_position((0, 1, 2, 3)) == (0.0, 1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="four values"):
        _validate_ref_position((0.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="must be finite"):
        _validate_ref_position((0.0, np.nan, 2.0, 3.0))


def test_method_key_is_case_insensitive() -> None:
    """Method names are stripped, lower-cased and checked against the options."""
    assert _method_key(" Cross Corr ", {"cross corr"}) == "cross corr"

    with pytest.raises(ValueError, match="method must be"):
        _method_key("bogus", {"cross corr", "fft cross"})


# --- translation ---------------------------------------------------------


def test_align_trans_schema_and_identity(blob: np.ndarray) -> None:
    """A reference correlated with itself gives zero shifts and the right columns."""
    res = align_trans(np.stack([blob, blob]), blob)

    assert list(res.columns) == ["frame", "dx", "dy"]
    assert len(res) == 2
    assert res["frame"].tolist() == [0, 1]
    assert np.issubdtype(res["frame"].dtype, np.integer)
    assert res["dx"].tolist() == pytest.approx([0.0, 0.0], abs=0.02)
    assert res["dy"].tolist() == pytest.approx([0.0, 0.0], abs=0.02)


def test_align_trans_recovers_known_shift(
    blob: np.ndarray,
    shift_blob: Callable[[float, float], np.ndarray],
) -> None:
    """Both correlation methods should recover a known integer translation."""
    stack = np.stack([blob, shift_blob(2.0, 3.0)])

    fft = align_trans(stack, blob, method="FFT cross")
    cross = align_trans(stack, blob, method="Cross corr")

    # The returned values are displacements of the moved frame.
    for result in (fft, cross):
        assert result.loc[1, "dx"] == pytest.approx(3.0, abs=0.35)
        assert result.loc[1, "dy"] == pytest.approx(2.0, abs=0.35)


def test_align_trans_accepts_2d_image(blob: np.ndarray) -> None:
    """A single 2D image is treated as one frame."""
    res = align_trans(blob, blob)

    assert len(res) == 1
    assert res.loc[0, "frame"] == 0
    assert res.loc[0, "dx"] == pytest.approx(0.0, abs=0.02)
    assert res.loc[0, "dy"] == pytest.approx(0.0, abs=0.02)


def test_align_trans_empty_stack_returns_empty_table(
    blob: np.ndarray,
    empty_stack: np.ndarray,
) -> None:
    """An empty stack keeps the schema instead of raising."""
    table = align_trans(empty_stack, blob)

    assert table.empty
    assert list(table.columns) == ["frame", "dx", "dy"]


def test_align_trans_rejects_bad_pixel_shift(blob: np.ndarray) -> None:
    """Invalid local-search limits are rejected."""
    with pytest.raises(ValueError, match="pixel_shift"):
        align_trans(blob, blob, pixel_shift=-1.0)


def test_align_trans_rejects_bad_method_and_shape(blob: np.ndarray) -> None:
    """Bad method names and non-image stacks are rejected."""
    with pytest.raises(ValueError, match="method must be"):
        align_trans(blob, blob, method="bogus")

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        align_trans(np.zeros((2, 2, 2, 2)), blob)


def test_align_trans_exposes_version() -> None:
    """The public entry point carries a version, like the other modules."""
    assert align_trans.__version__ == "0.1.0"


def test_align_trans_table_has_only_computed_columns(blob: np.ndarray) -> None:
    """The table is a plain DataFrame with only the computed columns."""
    res = align_trans(np.stack([blob, blob]), blob)

    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"frame", "dx", "dy"}
    assert "angle" not in res.columns
    assert "id" not in res.columns


def test_align_translation_returns_record(blob: np.ndarray) -> None:
    """The private stage returns the record the public table is built from."""
    record = _align_translation(
        np.stack([blob, blob]),
        blob,
        pixel_shift=0.0,
        sub_pix=True,
        method="cross corr",
    )

    assert isinstance(record, Translation)
    assert list(record.as_dict()) == ["frame", "dx", "dy"]
    assert len(record) == 2


# --- rotation ------------------------------------------------------------


def test_align_rot_recovers_known_angle(asymmetric: np.ndarray) -> None:
    """A target rotated by +20 degrees should need about -20 degrees."""
    target = rotate(
        asymmetric,
        angle=20.0,
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
    )

    res = align_rot(asymmetric, target, angle_range=(-60.0, 60.0))

    assert list(res.columns) == ["frame", "angle"]
    assert len(res) == 1
    assert res.loc[0, "angle"] == pytest.approx(-20.0, abs=1.0)


def test_align_rot_accepts_stack_and_keeps_ids(asymmetric: np.ndarray) -> None:
    """A 3D target gives one row per frame, with 0-based frame numbers."""
    target = rotate(
        asymmetric,
        angle=15.0,
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
    )

    res = align_rot(
        asymmetric,
        np.stack([asymmetric, target]),
        angle_range=(-45.0, 45.0),
    )

    assert len(res) == 2
    assert res["frame"].tolist() == [0, 1]
    assert res.loc[0, "angle"] == pytest.approx(0.0, abs=0.01)
    assert res.loc[1, "angle"] == pytest.approx(-15.0, abs=1.0)


def test_align_rot_empty_stack_returns_empty_table(
    asymmetric: np.ndarray,
) -> None:
    """An empty target keeps the schema instead of raising."""
    table = align_rot(asymmetric, np.zeros((0, 96, 96)), angle_range=(-10.0, 10.0))

    assert table.empty
    assert list(table.columns) == ["frame", "angle"]


def test_align_rot_rejects_bad_method_and_range(asymmetric: np.ndarray) -> None:
    """Bad method names and reversed angle ranges are rejected."""
    with pytest.raises(ValueError, match="method must be"):
        align_rot(asymmetric, asymmetric, method="bogus")

    with pytest.raises(ValueError, match="min < max"):
        align_rot(asymmetric, asymmetric, angle_range=(10.0, 0.0))


def test_im_to_polar_shape(blob: np.ndarray) -> None:
    """The polar transform returns the requested radial/angular sampling."""
    polar = im_to_polar(blob, 0.0, 1.0, 12, 36)

    assert isinstance(polar, np.ndarray)
    assert polar.shape == (12, 36)
    assert np.isfinite(polar).all()


def test_im_to_polar_rejects_nonfinite_radius(blob: np.ndarray) -> None:
    """The radius range must be finite."""
    with pytest.raises(ValueError, match="r_min and r_max must be finite"):
        im_to_polar(blob, 0.0, np.inf, 12, 36)


def test_im_to_polar_rejects_bad_sampling(blob: np.ndarray) -> None:
    """The sample counts must be positive."""
    with pytest.raises(
        ValueError,
        match="radial_samples and angular_samples must be positive",
    ):
        im_to_polar(blob, 0.0, 1.0, 0, 36)


def test_align_rot_and_polar_expose_versions() -> None:
    """The public entry points carry a version, like the other modules."""
    assert align_rot.__version__ == "0.1.0"
    assert im_to_polar.__version__ == "0.1.0"


def test_align_rotation_returns_record(asymmetric: np.ndarray) -> None:
    """The private stage returns the record the public table is built from."""
    record = _align_rotation(
        asymmetric,
        np.stack([asymmetric]),
        angle_range=(-10.0, 10.0),
        method="rotation corr",
    )

    assert isinstance(record, Rotation)
    assert list(record.as_dict()) == ["frame", "angle"]
    assert len(record) == 1


# --- movie ---------------------------------------------------------------


def test_align_movie_recovers_drift(
    blob: np.ndarray,
    shift_blob: Callable[[float, float], np.ndarray],
) -> None:
    """Movie drift uses the same displacement convention as align_trans."""
    stack = np.stack([blob, shift_blob(1.0, -2.0)])

    res = align_movie(stack, blob, method="FFT cross")

    assert list(res.columns) == ["frame", "dx", "dy"]
    assert len(res) == 2
    assert res.loc[0, "dx"] == pytest.approx(0.0, abs=0.02)
    assert res.loc[0, "dy"] == pytest.approx(0.0, abs=0.02)
    assert res.loc[1, "dx"] == pytest.approx(-2.0, abs=0.35)
    assert res.loc[1, "dy"] == pytest.approx(1.0, abs=0.35)


def test_align_movie_accepts_explicit_reference(
    blob: np.ndarray,
    shift_blob: Callable[[float, float], np.ndarray],
) -> None:
    """With ``full_image=False``, an explicit crop reference is used."""
    stack = np.stack([blob, shift_blob(1.0, 1.0)])

    res = align_movie(
        stack,
        blob,
        ref_position=(0.0, 0.0, 64.0, 64.0),
        full_image=False,
        method="FFT cross",
    )

    assert len(res) == 2
    assert res.loc[1, "dx"] == pytest.approx(1.0, abs=0.35)
    assert res.loc[1, "dy"] == pytest.approx(1.0, abs=0.35)


def test_align_movie_empty_stack_returns_empty_table(
    blob: np.ndarray,
    empty_stack: np.ndarray,
) -> None:
    """An empty stack keeps the schema instead of raising."""
    table = align_movie(empty_stack, blob)

    assert table.empty
    assert list(table.columns) == ["frame", "dx", "dy"]


def test_align_movie_requires_reference_for_full_image(blob: np.ndarray) -> None:
    """``full_image=True`` needs an explicit reference."""
    with pytest.raises(ValueError, match="ref is required"):
        align_movie(blob, None)


def test_align_movie_rejects_non_2d_reference(blob: np.ndarray) -> None:
    """The reference must be a 2D image."""
    with pytest.raises(ValueError, match="ref must be 2D"):
        align_movie(blob, np.zeros((2, 64, 64)))


def test_align_movie_rejects_bad_pixel_shift(blob: np.ndarray) -> None:
    """NaN local-search limits are rejected."""
    with pytest.raises(ValueError, match="pixel_shift"):
        align_movie(blob, blob, pixel_shift=np.nan)


def test_align_movie_exposes_version() -> None:
    """The public entry point carries a version, like the other modules."""
    assert align_movie.__version__ == "0.1.0"


def test_align_movie_returns_translation_record(blob: np.ndarray) -> None:
    """The private stage returns the record the public table is built from."""
    record = _align_movie(
        np.stack([blob, blob]),
        blob,
        pixel_shift=0.0,
        sub_pix=True,
        filt_cr=0.0,
        method="cross corr",
        effective_shape=(64.0, 64.0),
    )

    assert isinstance(record, Translation)
    assert list(record.as_dict()) == ["frame", "dx", "dy"]
    assert len(record) == 2
