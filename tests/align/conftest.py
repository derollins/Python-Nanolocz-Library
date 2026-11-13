"""Shared fixtures for the alignment tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift


@pytest.fixture
def blob() -> np.ndarray:
    """Return a smooth Gaussian blob, used as a simple particle image."""
    yy, xx = np.mgrid[:64, :64]
    return np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2.0 * 1.5**2))


@pytest.fixture
def asymmetric() -> np.ndarray:
    """Return an asymmetric shape so rotation angles are well defined."""
    img = np.zeros((96, 96), dtype=np.float64)
    img[30:52, 34:41] = 1.0  # vertical stem
    img[46:52, 34:62] = 0.8  # horizontal foot
    img[20:26, 52:58] = 0.6  # marker that breaks symmetry
    return gaussian_filter(img, sigma=1.0)


@pytest.fixture
def shift_blob(blob: np.ndarray) -> Callable[[float, float], np.ndarray]:
    """Return a function that shifts ``blob`` by ``dy``/``dx`` pixels."""

    def _shift(dy: float, dx: float) -> np.ndarray:
        return shift(blob, shift=(dy, dx), order=3, mode="constant", cval=0.0)

    return _shift


@pytest.fixture
def empty_stack() -> np.ndarray:
    """Return an empty frame-first stack with the default blob shape."""
    return np.zeros((0, 64, 64), dtype=np.float64)
