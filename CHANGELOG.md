<!-- markdownlint-disable MD033 MD024-->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- 'CONTRIBUTING.md' file with contributor guidelines and development workflow.
- Python-native particle detection (`detector`, port of `Detector.m`) returning a
  0-based `pandas.DataFrame` in the NanoLocz `Part Locs` column layout
  (`x, y, z, correlation, frame, id, track_id, angle`) with a boolean `kept`
  column for interactive curation, and an opt-in `max_particles_per_frame` limit
  for CCR detection that flags all but the N best-correlation detections of each
  frame as `kept=False` without dropping rows (off by default; `True` uses the
  default count of 1); with `peaks.fast_peaks2d` (port of `Fast_peaks2D.m`) and
  `correlation.normxcorr2` (unit-independent, MATLAB-compatible normalized
  cross-correlation).
- `pandas` added to the core dependencies for detection-table outputs.

---

## [0.1.0] - 2026-03-02

### Changed

- **Major project restructure**:

  - Renamed package from `pnanolocz_lib` → `pnanolocz`.
  - Updated source directory structure to `src/pnanolocz/`.
  - Updated all import references and entry points.

### Added

- Initial public packaging setup for PyPI using `setuptools` + `setuptools_scm`.
- Comprehensive project metadata, classifiers, dependencies, and dev tools.
- Version writing via `src/pnanolocz/_version.py`.
- Entry point integration for `playnano` filters.

---

## [0.0.4] - 2026-03-02

### Added

- (none)

### Changed

- Updated project metadata: corrected author name to "Daniel E. Rollins".
- Standardised suppression of polynomial fitting warnings using
  `RuntimeWarning` and rank-related filters, replacing direct `RankWarning`
  references across leveling modules.
- Improved consistency in author attribution within `thresholder.py`.

### Fixed

- Eliminated dependency on `numpy.polynomial.polyutils.RankWarning` to avoid
  type-checking issues and NumPy internals.

### Infrastructure

- `.markdownlint.yaml` now ignores `LICENSE` and `LICENSE.md`.
- `.pre-commit-config.yaml` now excludes the `build/` directory from Black.

## [0.0.3] - 2026-01-14

### Summary

This release moves the Python NanoLocz implementation from being functionally similar to the MATLAB reference to being
**algorithmically aligned**. It standardizes mask semantics, improves numerical parity, completes several automated
routines, and clarifies public API behavior. These changes significantly improve reproducibility, validation, and ease
of extension.

### Breaking Changes

- **Minimum Python version** is now **3.11** (3.10 support removed).
- **Thresholding & mask polarity**:
  - New public API: `apply_thresholder(img, method, limits=None, invert=False)`.
  - All thresholders now return **boolean exclusion masks** (`True = excluded`).
  - Removes prior NaN-mask and mixed-polarity behavior.
- **Leveling functions now expect exclusion masks**:
  - Leveling internally converts exclusion ↔ validity masks.
  - Fitting uses **NaN-outside** semantics.
- **MATLAB behavioral alignment changes**:
  - `level_line` only runs the *column* stage when `polyy > 0`.
  - `med_line` interprets `polyx` as a **gain** when `polyx > 0`, matching MATLAB.

### Added / Improved

#### Mask Semantics (All Modules)

- Unified boolean exclusion masks across thresholding and leveling.
- Introduced `_validity_mask` to convert public exclusion masks to internal validity masks.
- All fits and summaries use NaN-outside semantics while preserving excluded pixels.

#### Core Leveling (`level.py`)

- Polynomial fits now use **1‑based indices** and **population standard deviation (`ddof=0`)**,
  matching MATLAB `polyfit(..., mu)`.
- Reworked methods (`plane`, `line`, `med_line`, `med_line_y`, `smed_line`, `mean_plane`, `log_y`) for:
  - alignment with MATLAB gating rules,
  - explicit low-sample fallbacks,
  - validity-mask‑only fitting.
- Guarantees: `background = input - leveled` under consistent semantics.

#### Automated Routines (`level_auto.py`)

- Implemented:
  - `multi-plane-edges`
  - `multi-plane-otsu`
- Restored missing histogram passes after plane leveling.
- Added anisotropy‑gated one‑off `med_line` preconditioning.
- MATLAB‑style Gaussian histogram (`gauss1`) fitting:
  - `gauss_fit`, `gauss_peaks`, `gauss_holes`
  - Adaptive bounds computed **per frame** (documented MATLAB deviation).

#### Thresholding (`thresholder.py`)

- New dispatcher: `apply_thresholder` (returns exclusion masks).
- Expanded & clarified methods (e.g., histogram, selection, otsu, edge masks, skel masks).
- Added advanced methods:
  - `line_step` (PELT change‑point)
  - `adaptive` (Sobel + morphology + inclusive gating)

#### Region‑Weighted Leveling (`level_weighted.py`)

- Region detection uses **8‑connectivity** and **min area = floor(1% of H×W)** (MATLAB rule).
- Weights ≤2% are zeroed (not renormalized), matching MATLAB.
- Evaluations use **1‑based coordinate grids**.
- Expanded documentation and edge‑case behavior notes.

#### Documentation

- Added `CITATION.cff` for formal citation.
- Updated README with:
  - Python 3.11+ requirement,
  - new APIs,
  - improved examples.
- Updated example notebook accordingly.

#### Tooling & CI

- CI now tests Python **3.11, 3.12, 3.13**.
- Pinned `scikit-image >= 0.26, < 0.27` for stability.
- Improved pre‑commit configuration and ignore rules.
- Standardized tooling (Black, Ruff, Mypy) to **py311**.

#### Tests & Resources

- Added `tests/conftest.py` with NPZ loader fixture.
- Added AFM resource data under `tests/resources/`.

### Migration Notes

#### Thresholding

Before:

```python
mask = thresholder(img, method="otsu")  # mixed polarity, NaN masks
```

After:

```python
mask_excl = apply_thresholder(img, method="otsu")  # True = excluded
valid = ~mask_excl
```

#### Leveling

Before:

```python
leveled = apply_level_weighted(img, 1, 1, method="line", mask=valid_mask)
```

After:

```python
leveled = apply_level_weighted(img, 1, 1, method="line", mask=mask_excl)
```

#### Automated Routines

```python
out = apply_level_auto(stack, routine="multi-plane-otsu")
```

### CI Note

To ensure setuptools-scm sees tags:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
```

---

[Unreleased]: https://github.com/derollins/Python-Nanolocz-Library/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/derollins/Python-Nanolocz-Library/releases/tag/v0.1.0
[0.0.4]: https://github.com/derollins/Python-Nanolocz-Library/releases/tag/v0.0.4
[0.0.3]: https://github.com/derollins/Python-Nanolocz-Library/releases/tag/v0.0.3
