"""Contract and architecture tests for the alignment helper modules."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import pnanolocz.align_ptcloud as align_ptcloud_module
import pnanolocz.construct_particle_stack as construct_module
import pnanolocz.find_center as find_center_module
from pnanolocz.align import (
    AlignmentInput,
    MovieInput,
    PolarInput,
    Rotation,
    RotationInput,
    Translation,
    _compute_movie,
    _compute_polar,
    _compute_rotation,
    _compute_translation,
)
from pnanolocz.align_iterate import (
    IterateInput,
    IterateResult,
    _align_iterate,
    align_iterate,
)
from pnanolocz.align_ptcloud import (
    PointCloudInput,
    PointCloudResult,
    _align_ptcloud,
    align_ptcloud,
)
from pnanolocz.construct_particle_stack import (
    Particles,
    _construct_particle_stack,
    construct_particle_stack,
)
from pnanolocz.find_center import (
    CenterInput,
    _find_center_positions,
    _find_center_ptcloud,
    find_center_positions,
    find_center_ptcloud,
)


def _blob(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.exp(-((yy - 32.0) ** 2 + (xx - 32.0) ** 2) / (2.0 * 1.5**2))


def _locs_two_particles() -> np.ndarray:
    return np.array(
        [
            [32.0, 32.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [32.0, 32.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )


def test_particles_is_frozen_and_shared() -> None:
    """The particle record is frozen and reused by both modules."""
    part = Particles(image=np.zeros((1, 16, 16)), locs=np.zeros((1, 8)))

    assert construct_module.Particles is Particles
    with pytest.raises(FrozenInstanceError):
        part.image = np.ones((1, 16, 16))


def test_construct_particle_stack_returns_3d_for_2d_input() -> None:
    """A 2-D particle image is one particle, but the result stays 3-D."""
    locs = np.array([[32.0, 32.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    result = construct_particle_stack(_blob(), _blob(), locs)

    assert result.ndim == 3
    assert result.shape == (1, 64, 64)


def test_construct_particle_stack_rejects_locs_mismatch() -> None:
    """The number of localisations must match the particle images."""
    with pytest.raises(ValueError, match="rows must match"):
        construct_particle_stack(_blob(), np.zeros((2, 32, 32)), np.zeros((1, 8)))


def test_construct_module_has_no_matlab_aliases() -> None:
    """The public surface no longer exposes MATLAB wrappers or aliases."""
    assert not hasattr(construct_module, "construct_particle_stack_matlab")
    assert not hasattr(construct_module, "ConstructParticleStack")
    assert not hasattr(construct_module, "ParticleSet")


def test_align_iterate_returns_named_alignment_table() -> None:
    """The public adapter returns stack, per-particle table and reference."""
    stack, table, ref = align_iterate(
        np.stack([_blob(), _blob()]),
        _blob(),
        np.stack([_blob(), _blob()]),
        _locs_two_particles(),
        0,
        "FFT cross",
        0.0,
        0,
        "Rotation corr",
        0.0,
        0.0,
        False,
    )

    assert stack.ndim == 3
    assert stack.shape[0] == 2
    assert list(table.columns) == ["particle_id", "frame", "x", "y", "angle"]
    assert table["particle_id"].tolist() == [0, 1]
    assert ref.ndim == 2


def test_align_iterate_rejects_locs_mismatch() -> None:
    """Localisation rows and particle images must have the same length."""
    with pytest.raises(ValueError, match="same length"):
        align_iterate(
            np.stack([_blob(), _blob()]),
            _blob(),
            np.stack([_blob(), _blob()]),
            np.zeros((1, 8)),
            0,
            "FFT cross",
            0.0,
            0,
            "Rotation corr",
            0.0,
            0.0,
            False,
        )


def _reference_callback(lafm: np.ndarray, *args: object) -> np.ndarray:
    """Return three fixed reference detections for point-cloud tests."""
    del lafm, args
    return np.array(
        [
            [10.0, 10.0, 1.0, 0.0, 0.0],
            [20.0, 20.0, 1.0, 0.0, 0.0],
            [30.0, 30.0, 1.0, 0.0, 0.0],
        ]
    )


def _ptcloud_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.array(
        [
            [10.1, 10.2, 1.0],
            [20.1, 20.2, 1.0],
            [30.1, 30.2, 1.0],
            [10.2, 10.1, 1.0],
            [20.2, 20.1, 1.0],
            [30.2, 30.1, 1.0],
        ]
    )
    frame = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    particle_id = np.array([10, 11, 12, 20, 21, 22], dtype=np.intp)
    return xyz, frame, particle_id


def test_align_ptcloud_returns_named_tables() -> None:
    """Point-cloud alignment returns coordinate and quality DataFrames."""
    xyz, frame, particle_id = _ptcloud_arrays()

    aligned, quality = align_ptcloud(
        xyz,
        frame,
        particle_id,
        np.zeros((32, 32)),
        1.0,
        fast_peaks2d_fn=_reference_callback,
        localize_fn=_reference_callback,
        rmse_acceptance=100.0,
    )

    assert list(aligned.columns) == ["frame", "particle_id", "x", "y"]
    assert list(quality.columns) == ["frame", "rmse"]
    assert aligned["particle_id"].tolist() == particle_id.tolist()
    assert aligned["frame"].tolist() == frame.tolist()
    assert len(quality) == 2


def test_align_ptcloud_has_no_matlab_alias() -> None:
    """The MATLAB-cased alias is gone."""
    assert not hasattr(align_ptcloud_module, "align_PtCloud")


def test_align_ptcloud_rejects_invalid_xyz() -> None:
    """The NumPy point cloud must be (N, 3) or wider."""
    with pytest.raises(ValueError, match="xyz must be"):
        align_ptcloud(
            np.zeros((3, 2)),
            np.zeros(3, dtype=np.intp),
            np.zeros(3, dtype=np.intp),
            np.zeros((8, 8)),
            1.0,
            fast_peaks2d_fn=_reference_callback,
            localize_fn=_reference_callback,
        )


def test_find_center_rejects_invalid_fold_and_has_no_aliases() -> None:
    """Both center functions validate fold, and no MATLAB aliases remain."""
    img = _blob((32, 32))
    points = np.zeros((3, 3))

    with pytest.raises(ValueError, match="fold must be >= 1"):
        find_center_positions(0, img)

    with pytest.raises(ValueError, match="fold must be >= 1"):
        find_center_ptcloud(0, img, points)

    assert not hasattr(find_center_module, "FindCenterPositions")
    assert not hasattr(find_center_module, "FindCenter_ptCloud")


def test_find_center_positions_uniform_image_is_zero() -> None:
    """A uniform image has its centroid at the 0-based geometric centre."""
    centre = find_center_positions(1, np.ones((64, 64), dtype=np.float64))

    assert centre.tolist() == pytest.approx([0.0, 0.0], abs=1e-12)


def test_find_center_ptcloud_rejects_nonfinite_points() -> None:
    """Point clouds with NaN coordinates are rejected."""
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [np.nan, 0.0, 1.0]])

    with pytest.raises(ValueError, match="finite x, y and z"):
        find_center_ptcloud(2, _blob((16, 16)), points)


def test_construct_rejects_nonfinite_frame() -> None:
    """A missing frame index is invalid, not silently frame zero."""
    locs = np.array([[16.0, 16.0, 1.0, 0.0, np.nan, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="frame index must be finite"):
        construct_particle_stack(_blob((32, 32)), _blob((32, 32)), locs)


def test_construct_rejects_noninteger_frame() -> None:
    """Frame indices must be integers."""
    locs = np.array([[16.0, 16.0, 1.0, 0.0, 1.6, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="frame index must be an integer"):
        construct_particle_stack(_blob((32, 32)), _blob((32, 32)), locs)


def test_align_iterate_initializes_missing_angles() -> None:
    """Missing angle columns start at zero so rotation can accumulate."""
    locs = np.array(
        [
            [32.0, 32.0, 1.0, 0.0, 0.0],
            [32.0, 32.0, 1.0, 0.0, 1.0],
        ]
    )

    _, table, _ = align_iterate(
        np.stack([_blob(), _blob()]),
        _blob(),
        np.stack([_blob(), _blob()]),
        locs,
        0,
        "FFT cross",
        0.0,
        1,
        "Rotation corr",
        5.0,
        0.0,
        False,
    )

    assert np.isfinite(table["angle"].to_numpy()).all()


def test_align_iterate_rejects_noninteger_iterations() -> None:
    """Iteration counts are not silently truncated."""
    with pytest.raises(ValueError, match="tran_iterations must be an integer"):
        align_iterate(
            np.stack([_blob(), _blob()]),
            _blob(),
            np.stack([_blob(), _blob()]),
            np.zeros((2, 8)),
            1.5,
            "FFT cross",
            0.0,
            0,
            "Rotation corr",
            0.0,
            0.0,
            False,
        )


def test_align_ptcloud_empty_input_returns_empty_schema() -> None:
    """An empty cloud returns empty tables without calling callbacks."""

    def _unreachable(*args: object) -> np.ndarray:
        raise AssertionError("callbacks must not run for an empty cloud")

    aligned, quality = align_ptcloud(
        np.empty((0, 3)),
        np.empty(0, dtype=np.intp),
        np.empty(0, dtype=np.intp),
        np.zeros((8, 8)),
        1.0,
        fast_peaks2d_fn=_unreachable,
        localize_fn=_unreachable,
    )

    assert aligned.empty
    assert list(aligned.columns) == ["frame", "particle_id", "x", "y"]
    assert quality.empty
    assert list(quality.columns) == ["frame", "rmse"]


def test_align_ptcloud_rejects_noninteger_frame() -> None:
    """Frame values must be exact integers."""
    xyz, _, particle_id = _ptcloud_arrays()

    with pytest.raises(ValueError, match="frame values must be integers"):
        align_ptcloud(
            xyz,
            np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5]),
            particle_id,
            np.zeros((32, 32)),
            1.0,
            fast_peaks2d_fn=_reference_callback,
            localize_fn=_reference_callback,
        )


def test_align_trans_uses_frozen_input_and_result_records() -> None:
    """The translation public adapter has a typed frozen private boundary."""
    data = AlignmentInput(
        stack=np.stack([_blob(), _blob()]),
        reference=_blob(),
    )
    result = _compute_translation(
        data, pixel_shift=0.0, sub_pix=True, method="fft cross"
    )

    assert isinstance(data, AlignmentInput)
    assert isinstance(result, Translation)


def test_align_rot_uses_frozen_input_and_result_records() -> None:
    """The rotation public adapter has a typed frozen private boundary."""
    data = RotationInput(reference=_blob(), target=np.stack([_blob(), _blob()]))
    result = _compute_rotation(
        data, angle_range=(-30.0, 30.0), method="rotation corr"
    )

    assert isinstance(result, Rotation)


def test_align_movie_uses_frozen_input_and_result_records() -> None:
    """The movie public adapter has a typed frozen private boundary."""
    data = MovieInput(
        image=np.stack([_blob(), _blob()]),
        reference=_blob(),
        full_image=True,
        ref_position=None,
    )
    result = _compute_movie(
        data,
        pixel_shift=0.0,
        sub_pix=True,
        filt_cr=0.0,
        method="fft cross",
    )

    assert isinstance(result, Translation)


def test_im_to_polar_uses_frozen_input_record() -> None:
    """The polar public adapter builds a frozen ``PolarInput``."""
    data = PolarInput(
        image=_blob(),
        r_min=0.0,
        r_max=1.0,
        n_angles=12,
        n_radii=8,
    )
    result = _compute_polar(data)

    assert result.image.shape == (8, 12)


def test_find_center_uses_frozen_input_records() -> None:
    """Both center functions have typed frozen private boundaries."""
    image = _blob((32, 32))
    positions = _find_center_positions(CenterInput(image=image, fold=1))
    points = np.array([[0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [0.0, 8.0, 1.0]])
    cloud = _find_center_ptcloud(
        CenterInput(image=image, fold=2, points=points)
    )

    assert np.isfinite([positions.x, positions.y]).all()
    assert np.isfinite([cloud.x, cloud.y]).all()


def test_construct_uses_frozen_particles_stage() -> None:
    """The constructor private stage receives a frozen ``Particles``."""
    part = Particles(image=_blob(), locs=np.zeros((1, 8)))
    part.locs[0, :5] = [32.0, 32.0, 1.0, 0.0, 0.0]
    result = _construct_particle_stack(_blob(), part)

    assert result.image.ndim == 3


def test_align_iterate_uses_frozen_input_and_result_records() -> None:
    """The iterative stage receives an ``IterateInput`` and returns a record."""
    data = IterateInput(
        full_image=np.stack([_blob(), _blob()]),
        reference=_blob(),
        particles=Particles(
            image=np.stack([_blob(), _blob()]),
            locs=_locs_two_particles(),
        ),
        tran_iterations=0,
        translat_method="FFT cross",
        maxdrift=0.0,
        rot_iterations=0,
        rota_method="Rotation corr",
        maxang=0.0,
        thresh_min=0.0,
        autoupdateref=False,
    )
    result = _align_iterate(data, construct_particle_stack)

    assert isinstance(result, IterateResult)
    assert result.image.ndim == 3


def test_align_ptcloud_uses_frozen_input_and_result_records() -> None:
    """The ICP stage receives a ``PointCloudInput`` and returns a record."""
    xyz, frame, particle_id = _ptcloud_arrays()
    data = PointCloudInput(xyz=xyz, frame=frame, particle_id=particle_id)
    result = _align_ptcloud(
        data,
        np.zeros((32, 32)),
        1.0,
        fast_peaks2d_fn=_reference_callback,
        localize_fn=_reference_callback,
        rmse_acceptance=100.0,
    )

    assert isinstance(result, PointCloudResult)
    assert result.particle_id.tolist() == particle_id.tolist()
