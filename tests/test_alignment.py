"""Tests for the pnanolocz alignment family (NanoLocz alignment ports).

Covers align_trans (incl. sub-pixel parabolic refinement), align_rot,
align_iterate, align_movie, construct_particle_stack and find_center.
Sub-pixel accuracy is measured against continuous ground truth generated in
memory (triangle shape, Gaussian edge smoothing, cubic-spline sub-pixel
translation) - the same recipe as the Synthetic_Subpixel data used for the
MATLAB parity runs.  align_ptcloud is ported but not covered here; its smoke
tests arrive with the localization stage.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, rotate, shift

from pnanolocz.align_iterate import Particles, align_iterate
from pnanolocz.align_movie import align_movie
from pnanolocz.align_rot import align_rot
from pnanolocz.align_trans import (
    _subpixel_offset,
    _subpixel_offset_from_zoom,
    align_trans,
)
from pnanolocz.construct_particle_stack import ParticleSet, construct_particle_stack
from pnanolocz.find_center import find_center_positions


# -------------------------------
# align_trans
# -------------------------------
def make_subpixel_stack(
    n_frames: int = 8,
    size: int = 128,
    seed: int = 42,
    maxdrift: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-memory sub-pixel translated triangle stack with continuous GT drift."""
    rng = np.random.default_rng(seed)
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

    dx = np.zeros(n_frames)
    dy = np.zeros(n_frames)
    for f in range(1, n_frames):
        dx[f] = float(np.clip(dx[f - 1] + rng.uniform(-2.0, 2.0), -maxdrift, maxdrift))
        dy[f] = float(np.clip(dy[f - 1] + rng.uniform(-2.0, 2.0), -maxdrift, maxdrift))

    stack = np.stack(
        [shift(img, (dy[f], dx[f]), order=3, prefilter=True) for f in range(n_frames)]
    )
    return stack, dx, dy


def _errors(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    euc = np.mean(np.hypot(x - dx, y - dy))
    return euc


def test_fft_cross_subpixel_accuracy():
    """FFT phase correlation recovers continuous drift to ~0.02 px."""
    stack, dx, dy = make_subpixel_stack()
    x, y = align_trans(
        stack, stack[0], pixel_shift=0, sub_pix=False, method="FFT cross"
    )
    assert _errors(x, y, dx, dy) < 0.02


def test_cross_corr_parabola_subpixel_accuracy():
    """Cross corr + parabolic refinement recovers drift to ~0.05 px."""
    stack, dx, dy = make_subpixel_stack()
    x, y = align_trans(
        stack, stack[0], pixel_shift=0, sub_pix=True, method="Cross corr"
    )
    assert _errors(x, y, dx, dy) < 0.05


def test_cross_corr_integer_within_quantization():
    """Integer NCC peak stays within one pixel of the true drift."""
    stack, dx, dy = make_subpixel_stack()
    x, y = align_trans(
        stack, stack[0], pixel_shift=0, sub_pix=False, method="Cross corr"
    )
    euc = _errors(x, y, dx, dy)
    assert 0.0 < euc < 1.0


def test_input_not_mutated():
    """align_trans never modifies the caller's stack or reference."""
    stack, _, _ = make_subpixel_stack(n_frames=4)
    stack[0, 0, 0] = np.nan
    before = stack.copy()
    align_trans(stack, stack[0], pixel_shift=0, sub_pix=False, method="FFT cross")
    np.testing.assert_array_equal(stack, before)


def test_subpixel_offset_backward_alias():
    """Legacy zoom-name alias delegates to the parabola implementation."""
    corr = np.zeros((21, 21))
    corr[9:12, 9:12] = np.array([[0.7, 0.9, 0.75], [0.8, 1.0, 0.85], [0.6, 0.8, 0.7]])
    a = _subpixel_offset(corr, 10, 10)
    b = _subpixel_offset_from_zoom(corr, 10, 10)
    assert a == b


def test_align_trans_2d_input_promoted():
    """A 2-D image returns single-element shift arrays."""
    stack, _, _ = make_subpixel_stack(n_frames=2)
    x, y = align_trans(
        stack[0], stack[0], pixel_shift=0, sub_pix=False, method="FFT cross"
    )
    assert x.shape == (1,) and y.shape == (1,)


def test_align_trans_invalid_method_raises():
    """Unknown method names raise ValueError."""
    stack, _, _ = make_subpixel_stack(n_frames=2)
    with pytest.raises(ValueError):
        align_trans(stack, stack[0], pixel_shift=0, sub_pix=False, method="bogus")


# -------------------------------
# align_rot
# -------------------------------
def _vertical_bar(size: int = 64) -> np.ndarray:
    """Asymmetric template (vertical bar) so rotation is well-posed."""
    template = np.zeros((size, size))
    template[16:48, 30:34] = 1.0
    return template


@pytest.mark.parametrize("angle", [5.0, 12.0, -20.0, 25.0])
def test_rotation_corr_recovers_known_angles(angle: float):
    """Rotation corr recovers applied rotations within 1 degree."""
    template = _vertical_bar()
    rotated = rotate(template, angle=-angle, reshape=False, order=3, mode="constant")
    est = align_rot(template, rotated, (-30, 30), "Rotation corr")
    assert abs(abs(est) - abs(angle)) < 1.0


def test_polar_corr_recovers_angle():
    """Polar corr recovers a 12 degree rotation within 3 degrees."""
    template = _vertical_bar()
    rotated = rotate(template, angle=-12.0, reshape=False, order=3, mode="constant")
    est = align_rot(template, rotated, (-30, 30), "Polar Corr")
    assert abs(abs(est) - 12.0) < 3.0


def test_align_rot_invalid_method_raises():
    """Unknown method names raise ValueError."""
    template = _vertical_bar()
    with pytest.raises(ValueError):
        align_rot(template, template, (-30, 30), "bogus")


def test_bad_angle_range_raises():
    """A non-2-element angle range raises ValueError."""
    template = _vertical_bar()
    with pytest.raises(ValueError):
        align_rot(template, template, (0.0,), "Rotation corr")


# -------------------------------
# align_iterate
# -------------------------------
def _locs_from_drift(
    dx: np.ndarray, dy: np.ndarray, size: int, *, x_err: float = 0.0, y_err: float = 0.0
) -> np.ndarray:
    """MATLAB-style locs table: 1-based x/y, 0-based frames, empty angle col."""
    n = len(dx)
    locs = np.zeros((n, 8), dtype=np.float64)
    locs[:, 0] = size / 2 + dx + x_err + 1.0
    locs[:, 1] = size / 2 + dy + y_err + 1.0
    locs[:, 4] = np.arange(n)
    return locs


def _crops_around(stack: np.ndarray, locs: np.ndarray, radius: int = 20) -> np.ndarray:
    """Crop each frame around its loc, matching the particle-stack builder."""
    out = []
    for f in range(locs.shape[0]):
        cx = int(round(locs[f, 0] - 1))
        cy = int(round(locs[f, 1] - 1))
        y0 = max(0, cy - radius)
        x0 = max(0, cx - radius)
        out.append(stack[f, y0 : y0 + 2 * radius + 1, x0 : x0 + 2 * radius + 1])
    return np.stack(out)


def test_translation_only_recovers_gt_trajectory():
    """Translation iterations pull noisy initial locs onto the GT track."""
    size = 128
    stack, dx, dy = make_subpixel_stack(n_frames=8, size=size)

    # The reference is a crop of frame 0 with the SAME size as the particle
    # crops (align_trans's peak formula assumes template == frame size,
    # matching NanoLocz where the reference is the average particle image).
    centre = int(round(size / 2))
    ref = stack[0, centre - 20 : centre + 21, centre - 20 : centre + 21]

    locs = _locs_from_drift(dx, dy, size, x_err=2.0, y_err=-1.5)
    crops = _crops_around(stack, locs)

    part_out, _ = align_iterate(
        stack,
        ref,
        Particles(image=crops, locs=locs),
        tran_iterations=2,
        translat_method="Cross corr",
        maxdrift=8.0,
        rot_iterations=0,
        rota_method="Rotation corr",
        maxang=0.0,
        thresh_min=0.0,
        autoupdateref=True,
    )

    gt_x = size / 2 + dx + 1.0
    gt_y = size / 2 + dy + 1.0
    err = float(
        np.mean(np.hypot(part_out.locs[:, 0] - gt_x, part_out.locs[:, 1] - gt_y))
    )
    init_err = float(np.mean(np.hypot(locs[:, 0] - gt_x, locs[:, 1] - gt_y)))
    assert err < 0.5
    assert err < init_err


def test_rotation_iterations_accumulate_angles():
    """Rotation iterations accumulate per-frame angles into locs column 7."""
    size = 64
    template = np.zeros((size, size))
    template[16:48, 30:34] = 1.0
    angle_step = 8.0
    stack = np.stack(
        [
            rotate(
                template, angle=-angle_step * f, reshape=False, order=3, mode="constant"
            )
            for f in range(3)
        ]
    )

    locs = np.zeros((3, 8))
    locs[:, 0] = size / 2 + 1.0
    locs[:, 1] = size / 2 + 1.0
    locs[:, 4] = np.arange(3)

    part_out, _ = align_iterate(
        stack,
        stack[0],
        Particles(image=stack.copy(), locs=locs),
        tran_iterations=0,
        translat_method="Cross corr",
        maxdrift=0.0,
        rot_iterations=1,
        rota_method="Rotation corr",
        maxang=15.0,
        thresh_min=0.0,
        autoupdateref=False,
    )

    for f in range(3):
        assert abs(part_out.locs[f, 7] - angle_step * f) < 3.0


# -------------------------------
# align_movie
# -------------------------------
def test_cross_corr_movie_drift_recovers_gt():
    """Movie drift via Cross corr + sub-pixel lands within 0.5 px of GT."""
    stack, dx, dy = make_subpixel_stack(n_frames=6)
    ref_obj = {"image": stack[0]}
    x, y = align_movie(
        stack,
        ref_obj,
        pixel_shift=10,
        full_image=False,
        sub_pix=True,
        filt_cr=0,
    )
    euc = float(np.mean(np.hypot(x - dx, y - dy)))
    assert euc < 0.5


def test_fft_cross_movie_drift_recovers_gt():
    """Movie drift via FFT phase correlation lands within 0.05 px of GT."""
    stack, dx, dy = make_subpixel_stack(n_frames=6)
    ref_obj = {"image": stack[0]}
    x, y = align_movie(
        stack,
        ref_obj,
        pixel_shift=0,
        full_image=False,
        sub_pix=False,
        filt_cr=0,
        method="FFT cross",
    )
    euc = float(np.mean(np.hypot(x - dx, y - dy)))
    assert euc < 0.05


def test_align_movie_2d_input_promoted():
    """A 2-D image returns single-element drift arrays."""
    stack, _, _ = make_subpixel_stack(n_frames=2)
    x, y = align_movie(
        stack[0],
        {"image": stack[0]},
        pixel_shift=0,
        full_image=False,
        sub_pix=False,
        filt_cr=0,
        method="FFT cross",
    )
    assert x.shape == (1,) and y.shape == (1,)


# -------------------------------
# construct_particle_stack
# -------------------------------
def _two_particle_movie(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Movie with two Gaussian blobs at fixed known positions."""
    from scipy.ndimage import gaussian_filter

    movie = np.zeros((4, size, size))
    for f in range(4):
        movie[f, 20, 30] = 1.0
        movie[f, 44, 40] = 1.0
        movie[f] = gaussian_filter(movie[f], sigma=1.0)
    # MATLAB-style 1-based locs: x, y, z, corr, frame(1-based)
    locs = np.array(
        [
            [31.0, 21.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [41.0, 45.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    return movie, locs


def test_crops_shape_and_count():
    """Each loc produces one crop; size = Part.Image spatial size (MATLAB)."""
    movie, locs = _two_particle_movie()
    part = ParticleSet(image=np.zeros((1, 13, 13)), locs=locs)
    crops = construct_particle_stack(
        movie, part, quick=True, frame_axis=0, part_frame_axis=0, matlab_indexing=True
    )
    assert crops.shape == (2, 13, 13)


def test_crop_content_centred_on_particle():
    """The blob appears at the crop centre (within one pixel)."""
    movie, locs = _two_particle_movie()
    part = ParticleSet(image=np.zeros((1, 13, 13)), locs=locs)
    crops = construct_particle_stack(
        movie, part, quick=True, frame_axis=0, part_frame_axis=0, matlab_indexing=True
    )
    for k in range(2):
        peak = np.unravel_index(np.nanargmax(crops[k]), crops[k].shape)
        assert abs(peak[0] - 6) <= 1 and abs(peak[1] - 6) <= 1


def test_zero_based_frames_flag():
    """matlab_indexing=False interprets the frame column as 0-based."""
    movie, locs = _two_particle_movie()
    locs[:, 4] = 0.0  # 0-based frame labels
    part = ParticleSet(image=np.zeros((1, 13, 13)), locs=locs)
    crops = construct_particle_stack(
        movie, part, quick=True, frame_axis=0, part_frame_axis=0, matlab_indexing=False
    )
    assert crops.shape == (2, 13, 13)


# -------------------------------
# find_center
# -------------------------------
def _centered_shape(size: int = 64) -> np.ndarray:
    """A rotationally unique shape centred on the image centre."""
    from scipy.ndimage import gaussian_filter

    img = np.zeros((size, size))
    img[28:36, 30:34] = 1.0  # vertical bar at centre
    return gaussian_filter(img, sigma=1.0)


def test_rotational_center_near_image_center():
    """A centred shape rotates about the image centre; estimate ~ (0, 0)."""
    img = _centered_shape()
    center = find_center_positions(fold=4, input_img=img, align_exp=1.0)
    assert center.shape == (2,)
    assert np.hypot(center[0], center[1]) < 2.0


def test_fold_below_one_raises():
    """Fold must be >= 1."""
    img = _centered_shape()
    with pytest.raises(ValueError):
        find_center_positions(fold=0, input_img=img)
