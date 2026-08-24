# pnanolocz

<div align="center">

[![GitHub release](https://img.shields.io/github/v/release/derollins/Python-Nanolocz-Library?color=green)](https://github.com/derollins/Python-Nanolocz-Library/releases)
[![GitHub release downloads](https://img.shields.io/github/downloads/derollins/Python-Nanolocz-Library/latest/total?color=green)](https://github.com/derollins/Python-Nanolocz-Library/releases)
[![PyPI version](https://img.shields.io/pypi/v/pnanolocz?color=blue)](https://pypi.org/project/pnanolocz/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pnanolocz)
[![PyPI downloads](https://img.shields.io/pypi/dm/pnanolocz?color=blue)](https://pypi.org/project/pnanolocz/)
[![pre-commit](https://github.com/derollins/Python-Nanolocz-Library/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/derollins/Python-Nanolocz-Library/actions/workflows/pre-commit.yaml)
[![Tests](https://github.com/derollins/Python-Nanolocz-Library/actions/workflows/tests.yaml/badge.svg)](https://github.com/derollins/Python-Nanolocz-Library/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/derollins/Python-Nanolocz-Library/graph/badge.svg?token=W5NkAxjlmX)](https://codecov.io/gh/derollins/Python-Nanolocz-Library)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

*A Python library for AFM image flattening, background leveling, and edge detection, based on the MATLAB NanoLocz platform.*
Original NanoLocz MATLAB library is available here: <https://github.com/George-R-Heath/NanoLocz-Matlab-Library>

---

## 🔍 Key Features

- **Versatile Leveling**: Polynomial plane/line subtraction, median- and log‑based
flattening.
- **Automated Routines**: Multi‑frame “routines” (plane‑line, iterative high/low, Otsu pipelines, etc.).
- **Threshold & Edge Masks**: Histogram, Otsu, Sobel‑based edges, skeletonization, change‑point line steps.
- **2D & 3D Support**: Works on single images `(H,W)` or stacks `(N,H,W)`.
- **Batch‑Ready**: Scriptable for high‑speed AFM (HS‑AFM) and localization AFM (LAFM) workflows.

---

## 📦 Installation

```bash
pip install pnanolocz
```

Or clone & install locally:

```bash
git clone https://github.com/derollins/Python-Nanolocz-Library.git
cd Python-Nanolocz-Library
pip install -e .
```

Install optional features only when needed:

```bash
pip install -e ".[notebook]"  # Jupyter widgets and plotting
pip install -e ".[io]"        # HDF5 readers and writers
pip install -e ".[analysis]"  # tables, plots, and advanced segmentation
pip install -e ".[all,dev]"   # complete development environment
```

---

### Requirements

This library requires **Python 3.11 or newer** and uses modern scientific Python packages to replace MATLAB functionality
from the original NanoLocz platform:

- **NumPy** – Core numerical operations and array handling (replaces MATLAB’s matrix operations).
- **SciPy** – Polynomial fitting, signal processing, and optimization routines (similar to MATLAB’s polyfit, filter, etc.).
- **scikit-image** – Image processing tools for thresholding, edge detection, and morphological operations (analogous to
  MATLAB’s Image Processing Toolbox).
- **Matplotlib** – Visualization of AFM frames and masks (replaces MATLAB’s plotting functions).
- **ruptures** – Change-point detection for line-step analysis (provides advanced segmentation beyond MATLAB’s built-ins).
- **sknw** – Skeletonization and graph-based analysis for edge and structure detection.

These libraries allow the Python implementation to match or exceed MATLAB’s capabilities while remaining open-source and
easily extensible for AFM workflows.

### Stable API

The package root intentionally exposes a small, stable LAFM interface:

```python
from pnanolocz import (
    LAFMWorkflow,
    detector,
    lafm_movie_renderer,
    lafm_renderer,
    localize,
    localize_matlab,
)
```

`LAFMWorkflow.find_all_peaks()` uses OpenCV cubic localization (`cvcubic`) by
default. The previous SciPy `bicubic` method remains available explicitly.
Specialized functionality is imported from its defining submodule, for example
`from pnanolocz.align_trans import align_trans`.

Large validation images in `Software_testing_images` are maintained locally and
are not part of the repository. Tests requiring those files skip clearly when
the directory is unavailable.

---

## 🚀 Quickstart

```python
import numpy as np
from pnanolocz.level import apply_level
from pnanolocz.level_auto import apply_level_auto
from pnanolocz.thresholder import apply_thresholder
from pnanolocz.level_weighted import apply_weighted_level

# 1) Polynomial plane leveling
img = np.load("frame.npy")        # (H,W)
flat = apply_level(img, 2, 2, method="plane")

# 2) Region-weighted line leveling
frames = np.load("frames.npy")      # (N,H,W)
mask = thresholder(frames, method="threshold", limits=(1.5, 50))
levelled = apply_weighted_level(frames, 1, 0, method="line", mask=mask)

# 3) Automated multi‑frame pipeline
stack = np.load("stack.npy")      # (N,H,W)
out = apply_level_auto(stack, routine="multi-plane-otsu")

# 4) Otsu mask
mask = apply_thresholder(img, method="otsu", limits=None)
```

---

## 📖 Modules

- **`pnanolocz.level`**
  Core flattening / leveling (plane, line, median, smoothed, mean, log).

  Typical usage involves calling the `apply_level()` function with an image (2D)
  or image stack (3D) and specifying the desired method and polynomial orders.
  (see Quickstart above for an example)

    Available methods:

| Method       | Description                                                |
|--------------|------------------------------------------------------------|
| `plane`      | Polynomial line + plane subtraction in X and Y (centered fitting). |
| `line`       | Row‑wise and column‑wise polynomial leveling (each line individually). |
| `med_line`   | Row‑wise median line flattening.                           |
| `med_line_y` | Column‑wise median flattening.                             |
| `smed_line`  | Smoothed median line subtraction.                          |
| `mean_plane` | Global mean subtraction.                                   |
| `log_y`      | Logarithmic curve subtraction along the Y‑axis.            |

- **`pnanolocz.level_weighted`**
  Weighted-region flattening / leveling (plane, line, median and smoothed).

  Typical usage involves calling the `apply_weighted_level()` function with an
  image (2D) or image stack (3D) and specifying the desired method and polynomial
  orders. (see Quickstart above for an example)

    Available methods:

| Method       | Description                                                |
|--------------|------------------------------------------------------------|
| `plane`      | Region-weighted polynomial plane subtraction in X and Y.   |
| `line`       | Region-weighted row/column polynomial leveling.            |
| `med_line`   | Region-weighted row-wise median line flattening.           |
| `med_line_y` | Region-weighted column-wise median line flattening.        |
| `smed_line`  | Region-weighted smoothed median line subtraction.          |

- **`pnanolocz.thresholder`**
  Intensity / edge detection: histogram, Otsu, auto edges, skeleton, step detection.

  Typical usage involves calling the `apply_thresholder()` function with an image (2D)
  or image stack (3D) and specifying the desired method and polynomial orders.
  (see Quickstart above for an example)

    Available thresholder functions:

| Method       | Description |
|--------------|-------------|
| `selection` | Use user-supplied hand-drawn or binary mask. |
| `histogram` | Threshold by intensity limits. |
| `otsu` | Otsu's global thresholding method. |
| `auto_edges` | Detect edges using Sobel gradient and morphological filtering. |
| `hist_edges` | Detect edges by thresholding with histogram limits and morphological operations. |
| `otsu_edges` | Detect edges after Otsu thresholding using morphological operations. |
| `otsu_skel` | Skeletonize regions selected by Otsu thresholding. |
| `hist_skel` | Skeletonize regions selected by histogram thresholding. |
| `line_step` | Detect step changes along each row using PELT change point detection. |

- **`pnanolocz.level_auto`**
  Pre‑defined multi‑frame routines built from `level` + `thresholder`.

| Routine                | Description                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------|
| `plane-line`           | `plane` leveling followed by `med_line`.                                                      |
| `iterative 1nm high`   | Plane leveling + high‑side histogram threshold (1 nm) with iterative refinements.             |
| `iterative -1nm low`   | Plane leveling + low‑side histogram threshold (−1 nm) with iterative refinements.             |
| `iterative high low`   | Plane leveling + symmetric ±1 nm histogram threshold with iterative refinements.              |
| `Line1 + Otsu Line2`   | Single‑pass line leveling, Otsu‑mask, then a second line leveling.                            |
| `high-low x2 (fit)`    | Two‑stage plane + median line leveling, with Gaussian‑fit histogram threshold in between.     |
| `iterative fit holes`  | Iterative plane + median line leveling, masking “holes” via Gaussian‑fit low‑side threshold.  |
| `iterative fit peaks`  | Iterative plane + median line leveling, masking “peaks” via Gaussian‑fit high‑side threshold. |
| `multi-plane-edges`    | Iterative plane leveling combined with edge-based masks and region-weighted fits.             |
| `multi-plane-otsu`     | Iterative plane leveling guided by Otsu-edge masks for segmentation and background refinement.|

---

## 🔗 Links

- **Homepage**: [GitHub Repository](https://github.com/derollins/Python-Nanolocz-Library>)
  Explore the source code, documentation, and examples.
- **Issue Tracker**: <https://github.com/derollins/Python-Nanolocz-Library/issues>
  Use this to submit bug reports, feature requests, or ask questions.

---

## 📝 Citation

If you use this library, please cite:
> Heath, G.R. et al. *NanoLocz: Image analysis platform for AFM, high‑speed AFM and localization AFM.*
> Small Methods 2024, 2301766. <https://doi.org/10.1002/smtd.202301766>

and

> Rollins, D. E., & Heath, G. R. (2025). *Python-NanoLocz-Library: A Python implementation of the NanoLocz AFM leveling and
> analysis tools*. University of Leeds. <https://github.com/derollins/Python-Nanolocz-Library>

---

## ⚖️ License

Distributed under the terms of the [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.txt).
