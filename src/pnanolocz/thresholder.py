"""
Thresholding and edge-detection utilities for NanoLocz-compatible AFM workflows.

This module ports the NanoLocz MATLAB ``thresholder.m`` routines to Python.
Every public threshold method returns a boolean *exclusion mask*:

    True  = excluded / masked pixel
    False = valid / included pixel

This differs from the original MATLAB numeric output, where valid pixels are
stored as ``1`` and excluded pixels are converted to ``NaN``.  The boolean
convention is easier to use in NumPy, and downstream modules convert it back to
NaN-outside semantics during fitting.

MATLAB alignment notes
----------------------
- ``bwareaopen(BW, N)`` is implemented with
  ``skimage.morphology.remove_small_objects(BW, max_size=N)``.
- ``~bwareaopen(~BW, N)`` is implemented as
  ``~remove_small_objects(~BW, max_size=N)``.
- Morphological operators are close but not bitwise-identical to MATLAB's image
  processing toolbox.
- ``line_step`` is a best-effort Python approximation because MATLAB
  ``findchangepts(..., 'Statistic', 'linear')`` has no exact NumPy equivalent.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, sobel
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import (
    closing,
    diamond,
    dilation,
    disk,
    erosion,
    remove_small_objects,
    skeletonize,
    thin,
)

try:  # Optional dependency used only by line_step.
    import ruptures as rpt
except Exception:  # pragma: no cover - optional dependency guard
    rpt = None  # type: ignore[assignment]

try:  # Optional dependency used only for skeleton graph pruning.
    import sknw
except Exception:  # pragma: no cover - optional dependency guard
    sknw = None  # type: ignore[assignment]


FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

_METHOD_MAP: dict[str, Callable[..., BoolArray]] = {}
F = TypeVar("F", bound=Callable[..., BoolArray])


def _register(name: str) -> Callable[[F], F]:
    """Register a thresholding method under a MATLAB-compatible name."""

    def decorator(func: F) -> F:
        _METHOD_MAP[name.lower()] = func
        return func

    return decorator


def _as_float_image(img: np.ndarray) -> FloatArray:
    """Return a float image while preserving shape."""
    return np.asarray(img, dtype=np.float64)


def _finite_values(arr: np.ndarray) -> np.ndarray:
    """Return finite values from an array as a 1-D vector."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return flat[np.isfinite(flat)]


def _replace_nonfinite_for_filter(arr: np.ndarray) -> FloatArray:
    """Replace NaN/Inf values before Gaussian or Sobel filtering.

    MATLAB image filtering functions often tolerate the NanoLocz workflow's
    numeric masks differently from SciPy.  For filtering, replacing non-finite
    values with the finite median prevents NaN propagation across the image.
    """
    out = np.asarray(arr, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if finite.any():
        fill = float(np.nanmedian(out[finite]))
    else:
        fill = 0.0
    out[~finite] = fill
    return np.asarray(out, dtype=np.float64)


def _binary_remove(binary: BoolArray) -> BoolArray:
    """Approximate MATLAB ``bwmorph(binary, 'remove')`` perimeter extraction."""
    fp = np.ones((3, 3), dtype=np.bool_)
    eroded = erosion(binary.astype(bool), footprint=fp)
    return np.asarray(binary & ~eroded, dtype=np.bool_)


@_register("selection")
def selection(
    img: np.ndarray[Any, np.dtype[Any]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Convert a user-supplied selection mask into an exclusion mask.

    MATLAB ``selection`` treats non-zero values as valid and zero values as
    excluded.  Therefore, a boolean input is interpreted as ``True = valid`` and
    is inverted before returning the Python convention ``True = excluded``.
    """
    del limits

    arr = np.asarray(img)
    if arr.dtype == np.bool_:
        valid = arr
    else:
        valid = (arr != 0) & np.isfinite(arr)

    return np.asarray(~valid, dtype=np.bool_)


@_register("histogram")
def histogram(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Mask pixels outside an inclusive intensity interval."""
    if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
        raise ValueError("histogram requires limits=(low, high)")

    low, high = float(limits[0]), float(limits[1])
    arr = _as_float_image(img)

    valid = np.isfinite(arr) & (arr >= low) & (arr <= high)
    return np.asarray(~valid, dtype=np.bool_)


@_register("otsu")
def otsu(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Single-level Otsu thresholding.

    MATLAB keeps pixels ``img <= threshold`` valid and masks pixels above the
    threshold.  Non-finite pixels are excluded.
    """
    del limits

    arr = _as_float_image(img)
    finite = np.isfinite(arr)

    if not finite.any():
        return np.ones_like(arr, dtype=np.bool_)

    thresh = float(threshold_otsu(arr[finite]))
    valid = finite & (arr <= thresh)
    return np.asarray(~valid, dtype=np.bool_)


@_register("2 level otsu")
def two_level_otsu(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Two-level Otsu thresholding matching MATLAB ``multithresh(img, 2)``.

    The middle intensity class is kept valid:
    ``threshold_low <= img <= threshold_high``.
    """
    del limits

    arr = _as_float_image(img)
    finite = np.isfinite(arr)

    if not finite.any():
        return np.ones_like(arr, dtype=np.bool_)

    values = arr[finite]
    try:
        low, high = threshold_multiotsu(values, classes=3)
    except Exception:
        # Stable fallback for constant or nearly constant data.
        low, high = np.nanpercentile(values, [33.333, 66.667])

    valid = finite & (arr >= float(low)) & (arr <= float(high))
    return np.asarray(~valid, dtype=np.bool_)


@_register("auto edges")
def auto_edges(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Sobel-gradient automatic edge masking.

    MATLAB logic:
    - Smooth with ``imgaussfilt(img, 2)``.
    - Compute Sobel gradients.
    - Use ``Grad = 2*xgrad.^2 + ygrad.^2``.
    - Threshold by ``min(Grad) + (mean(Grad) - min(Grad)) * factor``.
    - Clean with ``bwareaopen``, closing, dilation, and bridge.

    ``limits`` is optional.  If supplied, ``limits[0]`` is the diamond
    thickness and ``limits[1]`` is the threshold factor.  Invalid limits fall
    back to MATLAB defaults: thickness=5, factor=1.5.
    """
    arr = _replace_nonfinite_for_filter(img)

    if isinstance(limits, (tuple, list)) and len(limits) >= 2:
        try:
            thickness = int(round(float(limits[0])))
            factor = float(limits[1])
        except Exception:
            thickness, factor = 5, 1.5
        if thickness <= 0 or not np.isfinite(thickness):
            thickness = 5
        if factor <= 0 or not np.isfinite(factor):
            factor = 1.5
    else:
        thickness, factor = 5, 1.5

    sm = gaussian_filter(arr, sigma=2, mode="nearest")

    xgrad = sobel(sm, axis=1, mode="nearest")
    ygrad = sobel(sm, axis=0, mode="nearest")
    ygrad = ygrad - 0.5 * np.median(ygrad, axis=1, keepdims=True)

    grad = (2.0 * xgrad**2) + ygrad**2
    thresh = float(np.min(grad) + (np.mean(grad) - np.min(grad)) * factor)

    bw = grad > thresh
    bw = remove_small_objects(bw, max_size=100, connectivity=2)
    bw = ~remove_small_objects(~bw, max_size=50, connectivity=2)

    se = diamond(thickness)
    bw = closing(bw, footprint=se)
    bw = dilation(bw, footprint=se)
    bw = closing(bw, footprint=se)

    # Approximate bwmorph(BW, 'bridge') using a small diagonal closing footprint.
    bridge_fp = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.bool_)
    bw = closing(bw, footprint=bridge_fp)

    # MATLAB returns valid outside BW and NaN on BW.  Python returns excluded=True on BW.
    return np.asarray(bw, dtype=np.bool_)


@_register("hist edges")
def hist_edges(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | None = None,
) -> BoolArray:
    """Histogram-gated edge extraction.

    Pixels outside the intensity interval are converted into a perimeter mask
    and dilated with a disk of radius 3.
    """
    if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
        raise ValueError("hist edges requires limits=(low, high)")

    low, high = float(limits[0]), float(limits[1])
    arr = _replace_nonfinite_for_filter(img)

    sm = gaussian_filter(arr, sigma=2, mode="nearest")
    inside = (sm >= low) & (sm <= high)
    outside = ~inside

    perimeter = _binary_remove(np.asarray(outside, dtype=np.bool_))
    bw = dilation(perimeter, footprint=disk(3))

    return np.asarray(bw, dtype=np.bool_)


@_register("otsu edges")
def otsu_edges(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Otsu-gated edge extraction with MATLAB-like morphology."""
    del limits

    arr = _replace_nonfinite_for_filter(img)
    sm = gaussian_filter(arr, sigma=2, mode="nearest")

    values = _finite_values(sm)
    if values.size == 0:
        return np.zeros_like(sm, dtype=np.bool_)

    thresh = float(threshold_otsu(values))
    outside = ~(sm <= thresh)

    bw = _binary_remove(np.asarray(outside, dtype=np.bool_))
    bw = remove_small_objects(bw, max_size=100, connectivity=2)
    bw = ~remove_small_objects(~bw, max_size=50, connectivity=2)
    bw = dilation(bw, footprint=disk(3))
    bw = remove_small_objects(bw, max_size=100, connectivity=2)
    bw = ~remove_small_objects(~bw, max_size=50, connectivity=2)

    return np.asarray(bw, dtype=np.bool_)


def prune_skeleton_min_branch_length(
    skel: BoolArray,
    min_branch_length: int = 10,
) -> BoolArray:
    """Prune skeleton branches shorter than ``min_branch_length``.

    If the optional ``sknw`` package is unavailable, this function returns the
    skeleton unchanged.  This keeps the module importable in minimal
    environments while preserving the main NanoLocz workflow.
    """
    skel = np.asarray(skel, dtype=np.bool_)

    if sknw is None:
        return skel

    graph = sknw.build_sknw(skel.astype(np.uint8), multi=True)

    for start, end, key in list(graph.edges(keys=True)):
        if graph[start][end][key].get("weight", 0) < min_branch_length:
            graph.remove_edge(start, end, key)

    isolated = [node for node, degree in dict(graph.degree()).items() if degree == 0]
    graph.remove_nodes_from(isolated)

    out = np.zeros_like(skel, dtype=np.bool_)
    for _, _, _, data in graph.edges(keys=True, data=True):
        pts = data["pts"]
        out[pts[:, 0], pts[:, 1]] = True

    return np.asarray(out, dtype=np.bool_)


def _skeletonize_frame(binary_mask: BoolArray, min_branch_length: int = 10) -> BoolArray:
    """Thin, skeletonize, prune, and lightly reconnect an edge mask."""
    thinned = thin(np.asarray(binary_mask, dtype=np.bool_))
    skel = skeletonize(thinned)
    pruned = prune_skeleton_min_branch_length(skel, min_branch_length)
    return np.asarray(dilation(pruned), dtype=np.bool_)


@_register("otsu skel")
def otsu_skel(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Otsu thresholding followed by skeletonization."""
    del limits

    arr = _replace_nonfinite_for_filter(img)
    sm = gaussian_filter(arr, sigma=2, mode="nearest")

    values = _finite_values(sm)
    if values.size == 0:
        return np.zeros_like(sm, dtype=np.bool_)

    thresh = float(threshold_otsu(values))
    edges = ~(sm <= thresh)

    return _skeletonize_frame(np.asarray(edges, dtype=np.bool_), min_branch_length=10)


@_register("hist skel")
def hist_skel(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Histogram thresholding followed by skeletonization."""
    if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
        raise ValueError("hist skel requires limits=(low, high)")

    low, high = float(limits[0]), float(limits[1])
    arr = _replace_nonfinite_for_filter(img)
    sm = gaussian_filter(arr, sigma=2, mode="nearest")

    edges = ~((sm >= low) & (sm <= high))
    return _skeletonize_frame(np.asarray(edges, dtype=np.bool_), min_branch_length=10)


@_register("line_step")
def line_step(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Detect row-wise step changes and build an exclusion mask.

    This is a best-effort approximation of MATLAB ``findchangepts``.  It uses
    ``ruptures.Pelt`` when available.  If ``ruptures`` is unavailable or a row
    fails to fit, the row remains fully valid.
    """
    arr = _as_float_image(img)
    if arr.ndim != 2:
        raise ValueError("line_step expects a 2D image")
    if not (isinstance(limits, (tuple, list)) and len(limits) >= 2):
        print("line_step limits must be a sequence with at least two values")
        return np.zeros_like(arr, dtype=np.bool_)

    rows, cols = arr.shape
    penalty = float(limits[1])
    mask = np.zeros((rows, cols), dtype=np.bool_)

    if rpt is None:
        return mask

    for rr in range(rows):
        x = np.asarray(arr[rr, :], dtype=np.float64)
        finite = np.isfinite(x)
        if not finite.all():
            # Fill non-finite row values for the detector, but keep the output
            # mask convention simple.
            fill = float(np.nanmedian(x[finite])) if finite.any() else 0.0
            x = np.where(finite, x, fill)

        try:
            cps = rpt.Pelt(model="l2").fit(x).predict(pen=penalty)
        except Exception:
            cps = []

        cps = [int(cp) for cp in cps if 0 < int(cp) < cols]
        cps = [cp for cp in cps if 4 <= cp <= cols - 4]
        cps = sorted(set(cps))

        if not cps:
            continue

        for idx in range(len(cps) + 1):
            if idx == 0:
                cp = cps[0]
                left = x[max(cp - 3, 0) : cp + 1]
                right = x[cp : min(cp + 4, cols)]
                rising = left.size > 0 and right.size > 0 and float(np.mean(left)) < float(np.mean(right))
                mask[rr, 0 : cp + 1] = not rising

            elif idx == len(cps):
                cp = cps[-1]
                left = x[max(cp - 3, 0) : cp + 1]
                right = x[cp : min(cp + 4, cols)]
                falling_valid = left.size > 0 and right.size > 0 and float(np.mean(left)) > float(np.mean(right))
                mask[rr, cp:cols] = not falling_valid

            else:
                cp_prev = cps[idx - 1]
                cp_curr = cps[idx]
                left = x[max(cp_curr - 3, 0) : cp_curr + 1]
                right = x[cp_curr : min(cp_curr + 4, cols)]
                rising = left.size > 0 and right.size > 0 and float(np.mean(left)) < float(np.mean(right))
                mask[rr, cp_prev : cp_curr + 1] = not rising

    return np.asarray(mask, dtype=np.bool_)


@_register("adaptive")
def adaptive(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | None = None,
) -> BoolArray:
    """Adaptive Sobel/morphology-based masking.

    MATLAB keeps pixels that are outside the filled edge band and inside the
    intensity range.  Python returns the complement as an exclusion mask.
    """
    if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
        raise ValueError("adaptive requires limits=(low, high)")

    low, high = float(limits[0]), float(limits[1])
    arr = _replace_nonfinite_for_filter(img)

    sm = gaussian_filter(arr, sigma=0.1, mode="nearest")
    sob_mag = np.hypot(sobel(sm, axis=0, mode="nearest"), sobel(sm, axis=1, mode="nearest"))

    try:
        thresh = float(threshold_otsu(sob_mag))
        edge_band = sob_mag > thresh
    except Exception:
        edge_band = sob_mag > 0

    edge_band = closing(edge_band, footprint=disk(10))
    edge_band = remove_small_objects(edge_band, max_size=10, connectivity=2)

    se_line_vert = np.ones((10, 1), dtype=np.bool_)
    se_line_horz = np.ones((1, 10), dtype=np.bool_)

    dilated = dilation(edge_band, footprint=se_line_vert)
    dilated = dilation(dilated, footprint=se_line_horz)

    padded = np.pad(dilated, ((0, 0), (1, 1)), mode="constant", constant_values=True)
    filled = binary_fill_holes(padded)

    eroded = erosion(filled, footprint=se_line_horz)
    eroded = erosion(eroded, footprint=se_line_vert)

    final_band = eroded[:, 1:-1]

    finite = np.isfinite(img)
    valid = (~final_band) & finite & (np.asarray(img) >= low) & (np.asarray(img) <= high)

    return np.asarray(~valid, dtype=np.bool_)


@_register("parachute")
def parachute(
    img: FloatArray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> BoolArray:
    """Detect steep negative gradient features.

    This ports the MATLAB ``case 'parachute'`` branch.  ``limits[0]`` controls
    the threshold multiplier ``F`` and ``limits[1]`` controls the gradient
    direction multiplier.  The output marks detected features as excluded.
    """
    if isinstance(limits, (tuple, list)) and len(limits) >= 2:
        F = float(limits[0])
        direction = float(limits[1])
    else:
        F = 1.0
        direction = 1.0

    arr = _replace_nonfinite_for_filter(img)

    # MATLAB ``gradient(img)`` with one output is approximated here with the
    # X-gradient, which is the most useful direction for scan-line artifacts.
    _, gx = np.gradient(arr)
    h = gaussian_filter(direction * gx, sigma=1, mode="nearest")

    sigma = float(np.std(h))
    thresh = -F * 1.2 * sigma
    thresh2 = -5.0 * sigma

    bw = (h <= thresh) & (h >= thresh2)
    bw = remove_small_objects(bw, max_size=10, connectivity=2)

    return np.asarray(bw, dtype=np.bool_)


def apply_thresholder(
    img: np.ndarray,
    method: str,
    limits: tuple[float, float] | list[float] | str | None = None,
    invert: bool = False,
) -> BoolArray:
    """Apply a NanoLocz thresholding method to an image or frame-first stack.

    Parameters
    ----------
    img:
        2-D image ``(H, W)`` or frame-first stack ``(N, H, W)``.
    method:
        MATLAB-compatible method name.
    limits:
        Method-specific threshold parameters.
    invert:
        If ``True``, return the logical complement of the exclusion mask.
        This is the boolean equivalent of MATLAB's optional inverse masking.

    Returns
    -------
    ndarray of bool
        Exclusion mask with the same shape as ``img``.
    """
    method_lc = method.lower()
    if method_lc not in _METHOD_MAP:
        raise ValueError(f"Unknown thresholding method: {method!r}")

    if method_lc in {"histogram", "hist edges", "hist skel", "adaptive"}:
        if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
            raise ValueError(f"Method {method!r} requires limits=(low, high)")
        limits_safe = (float(limits[0]), float(limits[1]))
    elif method_lc == "line_step":
        limits_safe = limits
    elif method_lc in {"otsu", "2 level otsu", "otsu edges", "otsu skel", "selection"}:
        limits_safe = None
    else:
        # ``auto edges`` and ``parachute`` have optional limits.
        limits_safe = limits

    func = _METHOD_MAP[method_lc]
    arr = np.asarray(img)

    if arr.ndim == 2:
        result = func(arr, limits_safe)
    elif arr.ndim == 3:
        result = np.stack([func(frame, limits_safe) for frame in arr], axis=0)
    else:
        raise ValueError("img must be 2D or frame-first 3D with shape (N, H, W)")

    if invert:
        result = np.logical_not(result)

    return np.asarray(result, dtype=np.bool_)


apply_thresholder.__version__ = "0.2.0"  # type: ignore[attr-defined]


__all__ = [
    "apply_thresholder",
    "selection",
    "histogram",
    "otsu",
    "two_level_otsu",
    "auto_edges",
    "hist_edges",
    "otsu_edges",
    "otsu_skel",
    "hist_skel",
    "line_step",
    "adaptive",
    "parachute",
]
