"""
Automated NanoLocz AFM stack leveling routines.

This module ports MATLAB ``level_auto.m`` to Python and coordinates the three
lower-level modules:

- ``pnanolocz.level``
- ``pnanolocz.level_weighted``
- ``pnanolocz.thresholder``

Public Python conventions
-------------------------
- Image stacks are frame-first by default: ``(frames, rows, cols)``.
- MATLAB-style stacks ``(rows, cols, frames)`` are supported with
  ``frame_axis=-1``.
- Masks returned by ``thresholder`` use ``True = excluded`` and are passed
  directly into the leveling functions.

Important MATLAB alignment details
----------------------------------
The Gaussian-fit routines are intentionally implemented as multi-pass routines
rather than ordinary step lists.  MATLAB fits the histogram on the whole current
``result`` stack, not independently per frame.  This file preserves that
behavior.

Supported routines
------------------
- ``plane-line``
- ``iterative 1nm high``
- ``iterative -1nm low``
- ``iterative high low``
- ``Line1 + Otsu Line2``
- ``high-low x2 (fit)``
- ``iterative fit holes``
- ``iterative fit peaks``
- ``multi-plane-edges``
- ``multi-plane-otsu``
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from pnanolocz.level import apply_level
from pnanolocz.level_weighted import apply_level_weighted
from pnanolocz.thresholder import apply_thresholder

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]


def _step_level(polyx: float, polyy: float, method: str) -> dict[str, Any]:
    """Build a regular leveling step."""
    return {"kind": "level", "polyx": polyx, "polyy": polyy, "method": method}


def _step_weighted(polyx: float, polyy: float, method: str) -> dict[str, Any]:
    """Build a weighted-region leveling step."""
    return {"kind": "weighted", "polyx": polyx, "polyy": polyy, "method": method}


def _step_threshold(
    method: str, limits: Any = None, invert: bool = False
) -> dict[str, Any]:
    """Build a thresholding step."""
    return {"kind": "threshold", "method": method, "limits": limits, "invert": invert}


# Stepwise routines are the routines that can be expressed as a simple ordered
# sequence of thresholding/leveling calls.  Gaussian-fit routines are handled
# separately below because MATLAB fits their histograms on the full current stack.
STEPWISE_ROUTINES: dict[str, list[dict[str, Any]]] = {
    "plane-line": [
        _step_level(1, 1, "plane"),
        _step_level(1, 0, "med_line"),
    ],
    "iterative 1nm high": [
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-np.inf, 1.0)),
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-np.inf, 1.0)),
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-np.inf, 1.0)),
        _step_level(0, 0, "med_line"),
        _step_level(1, 0, "plane"),
        _step_threshold("histogram", (-np.inf, 1.0)),
        _step_level(0, 0, "med_line"),
        _step_level(2, 0, "plane"),
    ],
    "iterative -1nm low": [
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-1.0, np.inf)),
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-1.0, np.inf)),
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-1.0, np.inf)),
        _step_level(0, 0, "med_line"),
        _step_level(1, 0, "plane"),
        _step_threshold("histogram", (-1.0, np.inf)),
        _step_level(0, 0, "med_line"),
        _step_level(2, 0, "plane"),
    ],
    "iterative high low": [
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-1.0, 1.0)),
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-1.0, 1.0)),
        _step_level(1, 1, "plane"),
        _step_threshold("histogram", (-1.0, 1.0)),
        _step_level(0, 0, "med_line"),
        _step_level(1, 0, "plane"),
        _step_threshold("histogram", (-1.0, 1.0)),
        _step_level(0, 0, "med_line"),
        _step_level(2, 0, "plane"),
    ],
    "Line1 + Otsu Line2": [
        _step_level(1, 0, "line"),
        _step_threshold("otsu", None),
        _step_level(2, 0, "line"),
    ],
    "multi-plane-edges": [
        _step_level(1, 1, "plane"),
        _step_threshold("auto edges", (0, 0)),
        _step_weighted(2, 2, "plane"),
        # MATLAB contains a char-array-looking ['-inf', 'inf'] call here.
        # Passing infinities falls back to auto_edges defaults in thresholder.py.
        _step_threshold("auto edges", (-np.inf, np.inf)),
        _step_weighted(2, 2, "plane"),
        _step_weighted(0, 0, "med_line"),
        _step_weighted(2, 2, "plane"),
        _step_weighted(0, 0, "med_line"),
        _step_threshold("otsu", (0, 0)),
        _step_level(0, 0, "mean_plane"),
    ],
    "multi-plane-otsu": [
        _step_level(1, 1, "plane"),
        _step_threshold("otsu edges", (0, 0)),
        _step_weighted(2, 2, "plane"),
        _step_threshold("otsu edges", (0, 0)),
        _step_weighted(2, 2, "plane"),
        _step_threshold("otsu edges", (0, 0)),
        _step_weighted(2, 2, "plane"),
        _step_weighted(0, 0, "med_line"),
        _step_threshold("otsu edges", (0, 0)),
        _step_weighted(2, 2, "plane"),
        _step_weighted(0, 0, "med_line"),
        _step_threshold("otsu", (0, 0)),
        _step_level(0, 0, "mean_plane"),
    ],
}

SPECIAL_GAUSSIAN_ROUTINES = {
    "high-low x2 (fit)",
    "iterative fit holes",
    "iterative fit peaks",
}

# Public routine registry kept for backwards compatibility with tests or user code.
ROUTINES: dict[str, Any] = {
    **STEPWISE_ROUTINES,
    "high-low x2 (fit)": "special-gaussian",
    "iterative fit holes": "special-gaussian",
    "iterative fit peaks": "special-gaussian",
}


# MATLAB anisotropy preconditioning:
# std_x = std(mean(prev, 1)); std_y = std(mean(prev, 2));
# if std_y > factor * std_x: apply med_line(polyx=<strength>)
PRECOND_POLICIES: dict[str, dict[str, Any]] = {
    "iterative 1nm high": {
        "trigger": {"kind": "level", "method": "plane", "polyx": 1, "polyy": 1},
        "gates": [(7.0, 1.0), (5.0, 0.6)],
    },
    "iterative -1nm low": {
        "trigger": {"kind": "level", "method": "plane", "polyx": 1, "polyy": 1},
        "gates": [(7.0, 1.0), (5.0, 0.6)],
    },
    "iterative high low": {
        "trigger": {"kind": "level", "method": "plane", "polyx": 1, "polyy": 1},
        "gates": [(7.0, 1.0), (5.0, 0.6)],
    },
    "multi-plane-edges": {
        "trigger": {"kind": "level", "method": "plane", "polyx": 1, "polyy": 1},
        "gates": [(7.0, 1.0), (5.0, 0.6)],
    },
    "multi-plane-otsu": {
        "trigger": {"kind": "level", "method": "plane", "polyx": 1, "polyy": 1},
        "gates": [(5.7, 1.0)],
    },
}


def _as_frame_first(img: np.ndarray, frame_axis: int) -> tuple[FloatArray, bool, int]:
    """Convert an image or stack to frame-first order.

    Returns
    -------
    stack:
        Frame-first stack with shape ``(N, H, W)``.
    was_2d:
        Whether the input was a single 2-D image.
    normalized_axis:
        Normalized original frame axis, used for restoring output order.
    """
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True, 0

    if arr.ndim != 3:
        raise ValueError("img_stack must be 2D or 3D")

    normalized_axis = int(frame_axis) % arr.ndim
    stack = np.moveaxis(arr, normalized_axis, 0)
    return np.asarray(stack, dtype=np.float64), False, normalized_axis


def _restore_frame_axis(stack: FloatArray, was_2d: bool, frame_axis: int) -> np.ndarray:
    """Restore output dimensionality and frame-axis order."""
    if was_2d:
        return np.asarray(stack[0], dtype=np.float64)

    return np.asarray(np.moveaxis(stack, 0, frame_axis), dtype=np.float64)


def _normalize_filter_frames(
    filter_frames: Sequence[int] | None,
    n_frames: int,
    *,
    matlab_indexing: bool,
) -> list[int]:
    """Normalize requested frame indices.

    Python indexing is 0-based by default.  If ``matlab_indexing=True``, incoming
    frame indices are interpreted as MATLAB 1-based indices and shifted by -1.
    """
    if filter_frames is None:
        return list(range(n_frames))

    frames = [int(i) - 1 if matlab_indexing else int(i) for i in filter_frames]

    bad = [i for i in frames if i < 0 or i >= n_frames]
    if bad:
        raise IndexError(f"filter_frames contains out-of-range frame indices: {bad}")

    return frames


def _nanstd_1d(values: np.ndarray) -> float:
    """MATLAB-like sample standard deviation with NaN omission."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _compute_anisotropy_ratio(img: FloatArray) -> tuple[float, float, float]:
    """Compute MATLAB ``std_y / std_x`` from row and column mean profiles."""
    with np.errstate(invalid="ignore", divide="ignore"):
        col_means = np.nanmean(img, axis=0)
        row_means = np.nanmean(img, axis=1)

    std_x = _nanstd_1d(col_means)
    std_y = _nanstd_1d(row_means)

    if std_x == 0.0:
        ratio = float("inf") if std_y > 0.0 else 0.0
    else:
        ratio = std_y / std_x

    return std_x, std_y, float(ratio)


def _step_matches_trigger(step: dict[str, Any], trigger: dict[str, Any]) -> bool:
    """Return True if a step matches a preconditioning trigger."""
    for key, value in trigger.items():
        if step.get(key) != value:
            return False
    return True


def _maybe_apply_precondition(
    img: FloatArray,
    routine: str,
    step: dict[str, Any],
    already_applied: bool,
) -> tuple[FloatArray, bool]:
    """Apply MATLAB anisotropy-gated ``med_line`` preconditioning once.

    A previous Python implementation returned after the first gate regardless of
    whether the gate passed.  This corrected version checks all gates in order,
    exactly like MATLAB's ``if`` / ``elseif`` chain.
    """
    if already_applied:
        return np.asarray(img, dtype=np.float64), True

    policy = PRECOND_POLICIES.get(routine)
    if policy is None:
        return np.asarray(img, dtype=np.float64), False

    if not _step_matches_trigger(step, policy["trigger"]):
        return np.asarray(img, dtype=np.float64), False

    _, _, ratio = _compute_anisotropy_ratio(img)

    for factor, med_line_strength in policy["gates"]:
        if ratio > factor:
            out = apply_level(
                img,
                polyx=float(med_line_strength),
                polyy=0,
                method="med_line",
                mask=None,
            )
            return np.asarray(out, dtype=np.float64), True

    return np.asarray(img, dtype=np.float64), False


def _gauss1_model(x: np.ndarray, a1: float, b1: float, c1: float) -> np.ndarray:
    """MATLAB ``gauss1`` model: ``a1 * exp(-((x-b1)^2) / c1^2)``."""
    c = c1 if c1 != 0 else 1.0
    return np.asarray(a1 * np.exp(-((x - b1) ** 2) / (c**2)), dtype=np.float64)


def _compute_gauss_limits_from_stack(
    stack: FloatArray, kind: str
) -> tuple[float, float]:
    """Fit MATLAB-style ``gauss1`` limits from the whole current result stack.

    MATLAB uses:
    ``[hy, x] = hist(double(t(:)), 100); gfit = fit(x', hy', 'gauss1')``.

    This Python version uses ``np.histogram(..., bins=100)`` and ``curve_fit``.
    The fitted width is used as ``abs(c1)`` because the model is symmetric in
    the sign of ``c1`` but threshold limits must not be reversed.
    """
    data = np.asarray(stack, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]

    if data.size == 0:
        return -np.inf, np.inf

    hy, edges = np.histogram(data, bins=100)
    x = 0.5 * (edges[:-1] + edges[1:])

    mean_data = float(np.mean(data))
    std_data = float(np.std(data, ddof=1)) if data.size > 1 else 1.0
    if not np.isfinite(std_data) or std_data == 0.0:
        std_data = 1.0

    try:
        p0 = [float(np.max(hy)) if hy.size else 1.0, mean_data, std_data * np.sqrt(2.0)]
        popt, _ = curve_fit(
            _gauss1_model,
            x,
            hy.astype(np.float64),
            p0=p0,
            maxfev=10000,
        )
        _, b1, c1 = popt
        center = float(b1)
        width = abs(float(c1))
        if not np.isfinite(width) or width == 0.0:
            width = std_data * np.sqrt(2.0)
    except Exception:
        center = mean_data
        width = std_data * np.sqrt(2.0)

    delta = 1.5 * abs(float(width))

    if kind == "gauss_fit":
        return center - delta, center + delta
    if kind == "gauss_holes":
        return center - delta, np.inf
    if kind == "gauss_peaks":
        return -np.inf, center + delta

    raise ValueError(f"Unknown Gaussian limit kind: {kind!r}")


def _compute_gauss_limits(image: np.ndarray, kind: str) -> tuple[float, float]:
    """Compatibility entry point for the original per-image Gaussian fit."""
    return _compute_gauss_limits_from_stack(np.asarray(image, dtype=np.float64), kind)


def _matches_trigger(
    func_obj: Callable[..., Any],
    params: Mapping[str, Any],
    trigger_spec: Mapping[str, Any],
) -> bool:
    """Match an original callable-based step against a trigger specification."""
    expected = trigger_spec.get("after_step", trigger_spec)
    expected_func = expected.get("func")
    if expected_func is None and expected.get("kind") == "level":
        expected_func = "apply_level"
    if expected_func is not None and func_obj.__name__ != expected_func:
        return False
    return all(
        key in ("func", "kind") or params.get(key) == value
        for key, value in expected.items()
    )


def _maybe_inject_precond(
    img: FloatArray,
    routine: str,
    func_obj: Callable[..., Any],
    params: Mapping[str, Any],
    injected: bool,
    *,
    apply_level_fn: Callable[..., FloatArray] | None = None,
    debug: bool = False,
) -> tuple[FloatArray, bool]:
    """Compatibility wrapper for the original callable-based preconditioner."""
    if injected:
        return np.asarray(img, dtype=np.float64), True

    policy = PRECOND_POLICIES.get(routine)
    if policy is None or not _matches_trigger(func_obj, params, policy["trigger"]):
        return np.asarray(img, dtype=np.float64), False

    _, _, ratio = _compute_anisotropy_ratio(img)
    for factor, strength in policy["gates"]:
        if ratio > factor:
            fn = apply_level if apply_level_fn is None else apply_level_fn
            out = fn(img, polyx=strength, polyy=0, method="med_line", mask=None)
            if debug:
                print(f"[auto] precondition applied for ratio>{factor}")
            return np.asarray(out, dtype=np.float64), True

    return np.asarray(img, dtype=np.float64), False


def _apply_legacy_routine(
    stack: FloatArray,
    frames: Sequence[int],
    routine: str,
    steps: Sequence[dict[str, Any]],
) -> FloatArray:
    """Execute the original callable-based ROUTINES representation."""
    result = np.asarray(stack.copy(), dtype=np.float64)
    for frame_idx in frames:
        img = result[frame_idx]
        mask: BoolArray | None = None
        injected = False
        for step in steps:
            func = step["func"]
            params = {key: value for key, value in step.items() if key != "func"}
            if func is apply_thresholder:
                args = params.get("args")
                if (
                    isinstance(args, (list, tuple))
                    and len(args) == 1
                    and isinstance(args[0], str)
                    and args[0].startswith("gauss_")
                ):
                    args = _compute_gauss_limits(img, args[0])
                mask = np.asarray(
                    apply_thresholder(
                        img,
                        params["method"],
                        args,
                        invert=bool(params.get("invert", False)),
                    ),
                    dtype=np.bool_,
                )
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                continue

            img = func(
                img,
                mask=mask,
                **{
                    key: value
                    for key, value in params.items()
                    if key not in ("args", "invert")
                },
            )
            img, injected = _maybe_inject_precond(
                np.asarray(img, dtype=np.float64),
                routine=routine,
                func_obj=func,
                params=params,
                injected=injected,
            )
            result[frame_idx] = img
    return result


def _apply_threshold_step(img: FloatArray, step: dict[str, Any]) -> BoolArray:
    """Run a threshold step and normalize its output to a 2-D boolean mask."""
    mask = apply_thresholder(
        img,
        method=step["method"],
        limits=step.get("limits", None),
        invert=bool(step.get("invert", False)),
    )

    if mask.ndim != 2:
        raise RuntimeError(
            "Internal error: thresholding a single frame did not return a 2D mask"
        )

    return np.asarray(mask, dtype=np.bool_)


def _apply_stepwise_routine(
    stack: FloatArray,
    frames: Sequence[int],
    routine: str,
) -> FloatArray:
    """Apply a routine that is represented by a simple list of steps."""
    if routine not in STEPWISE_ROUTINES:
        raise ValueError(f"Unknown stepwise routine: {routine!r}")

    result = np.asarray(stack.copy(), dtype=np.float64)
    steps = STEPWISE_ROUTINES[routine]

    for frame_idx in frames:
        img = result[frame_idx]
        mask: BoolArray | None = None
        precondition_applied = False

        for step in steps:
            kind = step["kind"]

            if kind == "threshold":
                mask = _apply_threshold_step(img, step)
                continue

            if kind == "level":
                img = apply_level(
                    img,
                    polyx=step["polyx"],
                    polyy=step["polyy"],
                    method=step["method"],
                    mask=mask,
                )
            elif kind == "weighted":
                img = apply_level_weighted(
                    img,
                    polyx=int(step["polyx"]),
                    polyy=int(step["polyy"]),
                    method=step["method"],
                    mask=mask,
                )
            else:
                raise RuntimeError(f"Unknown internal step kind: {kind!r}")

            img, precondition_applied = _maybe_apply_precondition(
                np.asarray(img, dtype=np.float64),
                routine,
                step,
                precondition_applied,
            )

            result[frame_idx] = img

    return np.asarray(result, dtype=np.float64)


def _routine_high_low_x2_fit(stack: FloatArray, frames: Sequence[int]) -> FloatArray:
    """Implement MATLAB routine ``high-low x2 (fit)`` with stack-global fitting."""
    result = np.asarray(stack.copy(), dtype=np.float64)

    # First pass: flatten all requested frames before deriving the global histogram.
    for frame_idx in frames:
        prev = result[frame_idx]
        prev = apply_level(prev, 1, 1, "plane")
        prev = apply_level(prev, 0, 0, "med_line")
        result[frame_idx] = prev

    low, high = _compute_gauss_limits_from_stack(result, "gauss_fit")

    # Second pass: mask using the stack-global Gaussian bounds, then level again.
    for frame_idx in frames:
        prev = result[frame_idx]
        mask = apply_thresholder(prev, "histogram", (low, high), invert=False)
        prev = apply_level(prev, 1, 1, "plane", mask=mask)
        prev = apply_level(prev, 0, 0, "med_line", mask=mask)
        result[frame_idx] = prev

    return np.asarray(result, dtype=np.float64)


def _routine_iterative_fit_holes_or_peaks(
    stack: FloatArray,
    frames: Sequence[int],
    *,
    mode: str,
) -> FloatArray:
    """Implement MATLAB ``iterative fit holes`` and ``iterative fit peaks``.

    MATLAB performs two separate stack-global Gaussian fits:

    1. Fit after the first plane/median-line pass.
    2. Fit again after a masked plane/median-line refinement pass.
    """
    if mode == "holes":
        kind = "gauss_holes"
    elif mode == "peaks":
        kind = "gauss_peaks"
    else:
        raise ValueError("mode must be 'holes' or 'peaks'")

    result = np.asarray(stack.copy(), dtype=np.float64)

    # First pass before Gaussian fit #1.
    for frame_idx in frames:
        prev = result[frame_idx]
        prev = apply_level(prev, 2, 2, "plane")
        prev = apply_level(prev, 0, 0, "med_line")
        result[frame_idx] = prev

    low, high = _compute_gauss_limits_from_stack(result, kind)

    # Second pass before Gaussian fit #2.
    for frame_idx in frames:
        prev = result[frame_idx]
        mask = apply_thresholder(prev, "histogram", (low, high), invert=False)
        prev = apply_level(prev, 2, 2, "plane", mask=mask)
        prev = apply_level(prev, 0, 0, "med_line", mask=mask)
        result[frame_idx] = prev

    low, high = _compute_gauss_limits_from_stack(result, kind)

    # Third pass: final masked plane plus line correction.
    for frame_idx in frames:
        prev = result[frame_idx]
        mask = apply_thresholder(prev, "histogram", (low, high), invert=False)
        prev = apply_level(prev, 2, 2, "plane", mask=mask)
        prev = apply_level(prev, 1, 0, "line", mask=mask)
        result[frame_idx] = prev

    return np.asarray(result, dtype=np.float64)


def apply_level_auto(
    img_stack: np.ndarray,
    routine: str,
    filter_frames: Sequence[int] | None = None,
    *,
    frame_axis: int = 0,
    matlab_indexing: bool = False,
) -> np.ndarray:
    """Apply a NanoLocz automated leveling routine.

    Parameters
    ----------
    img_stack:
        2-D image or 3-D stack.  By default 3-D stacks are interpreted as
        ``(frames, rows, cols)``.
    routine:
        Name of one of the supported NanoLocz routines.
    filter_frames:
        Optional sequence of frame indices to process.  If omitted, all frames
        are processed.
    frame_axis:
        Axis containing frames.  Use ``frame_axis=-1`` for MATLAB-style
        ``(rows, cols, frames)`` input.
    matlab_indexing:
        If ``True``, ``filter_frames`` are interpreted as MATLAB 1-based frame
        indices.  If ``False``, Python 0-based indices are used.

    Returns
    -------
    ndarray
        Leveled image or stack with the same shape/order as the input.
    """
    if routine not in ROUTINES:
        valid = ", ".join(sorted(ROUTINES))
        raise ValueError(f"Unknown routine {routine!r}. Available routines: {valid}")

    stack, was_2d, original_frame_axis = _as_frame_first(img_stack, frame_axis)
    frames = _normalize_filter_frames(
        filter_frames,
        n_frames=stack.shape[0],
        matlab_indexing=matlab_indexing,
    )

    routine_steps = ROUTINES[routine]
    if (
        isinstance(routine_steps, list)
        and routine_steps
        and "func" in routine_steps[0]
    ):
        result = _apply_legacy_routine(stack, frames, routine, routine_steps)
        return _restore_frame_axis(result, was_2d, original_frame_axis)

    if routine == "high-low x2 (fit)":
        result = _routine_high_low_x2_fit(stack, frames)
    elif routine == "iterative fit holes":
        result = _routine_iterative_fit_holes_or_peaks(stack, frames, mode="holes")
    elif routine == "iterative fit peaks":
        result = _routine_iterative_fit_holes_or_peaks(stack, frames, mode="peaks")
    else:
        result = _apply_stepwise_routine(stack, frames, routine)

    return _restore_frame_axis(result, was_2d, original_frame_axis)


def level_auto(
    img: np.ndarray,
    filter_frames: Sequence[int] | None,
    routine: str,
) -> np.ndarray:
    """MATLAB-style compatibility wrapper.

    This wrapper assumes MATLAB input layout ``(rows, cols, frames)`` and
    MATLAB 1-based frame indices.  New Python code should usually call
    :func:`apply_level_auto` directly.
    """
    return apply_level_auto(
        img,
        routine=routine,
        filter_frames=filter_frames,
        frame_axis=-1,
        matlab_indexing=True,
    )


apply_level_auto.__version__ = "0.2.0"  # type: ignore[attr-defined]


__all__ = [
    "apply_level_auto",
    "level_auto",
    "ROUTINES",
    "STEPWISE_ROUTINES",
    "PRECOND_POLICIES",
    "_compute_gauss_limits",
    "_matches_trigger",
    "_maybe_inject_precond",
]
