# Contributing to the pnanolocz project

Thank you for your interest in contributing to **pnanolocz**! Contributions of all kinds
are welcome: bug reports, documentation improvements and new features.

Please read this guide before submitting a pull request.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Key design concepts](#key-design-concepts)
- [Running the Tests](#running-the-tests)
- [Code Style and Pre-commit](#code-style-and-pre-commit)
- [Type Hints](#type-hints)
- [Docstrings](#docstrings)
- [Branching and Pull Requests](#branching-and-pull-requests)
- [Changelog](#changelog)
- [Versioning](#versioning)
- [Writing Processing Tools](#writing-processing-tools)
- [Writing Analysis Tools](#writing-analysis-tools)
- [Notebooks](#notebooks)
- [Reporting Bugs and Requesting Features](#reporting-bugs-and-requesting-features)
- [AI Transparency](#ai-transparency)

---

## Getting Started

### Prerequisites

- [Miniforge](https://conda-forge.org/download/), alternatively your own preferred environment/package manager.
- Python 3.11 or newer.

### Setting up a development environment

```bash
git clone https://github.com/derollins/Python-Nanolocz-Library
cd Python-Nanolocz-Library
conda create -n pnanolocz-dev python=3.11
conda activate pnanolocz-dev
pip install -e ".[dev]"
```

This installs pnanolocz in editable mode along with all development dependencies including
testing and pre-commit tools.

---

## Project Structure

```txt
Python-Nanolocz-Library/
├── src/pnanolocz/
│   ├── level.py           # Background leveling and flattening for images and stacks.
│   ├── level_weighted.py  # Region-wise weighted polynomial / median background estimation.
│   ├── level_auto.py      # Automated, data-driven background correction workflows.
│   └── thresholder.py     # Thresholding and edge-detection routines returning boolean masks.
├── tests/
│   ├── test_level.py …    # One test module per source module.
│   └── resources/         # Golden fixtures: raw input arrays + reference outputs.
├── notebooks/             # Demo notebooks.
├── CHANGELOG.md
├── README.md
├── CITATION.cff
└── pyproject.toml
```

---

## Key design concepts

**pnanolocz aims to make NanoLocz's analysis tools available in Python.** It is a Python port of the
[MATLAB NanoLocz Library](https://github.com/George-R-Heath/NanoLocz-Matlab-Library). It can be called
from a script or interactive notebook, integrated into other applications or used as a plugin. The core
processing and analysis functions are the priority for porting since there are already a number of data
loading, saving and display tools available within the Python ecosystem. For this reason the functions within
the pnanolocz library should take a NumPy array and return an array or simple Python datatypes and structures
such as numbers, lists, dictionaries and frozen dataclasses of arrays/floats.

### The contract: arrays and parameters in, simple datatypes out

- **Inputs are NumPy arrays and plain parameters.** A function receives the data it needs as an
  argument. To deliver AFM data as a NumPy array you can use [AFMReader](https://github.com/AFM-SPM/AFMReader)
  or [playNano](https://github.com/derollins/playNano) to load instrument files.
- **Outputs are NumPy arrays and simple Python types.** Outputs should be arrays, plain numbers, and small
  transparent records (a frozen dataclass of arrays/floats is fine). A function returns its result
  for the caller to display, save, or pass on; the output should make sense with no display and no
  filesystem.
- **Functions compose.** Prefer plain functions that take their inputs and return their outputs, so
  callers can chain them and hold any intermediate values themselves, which keeps each step
  independently testable and reusable. Where several steps run in sequence, express them as functions
  that pass their results along, and add a thin `run_*` wrapper if a single-call convenience is useful.

### MATLAB alignment and reproducibility

**pnanolocz** ports aim for **numerical alignment with the MATLAB NanoLocz reference**. Because
NumPy/SciPy and MATLAB differ in numerical libraries, floating-point behaviour, and edge-case
handling, results may not be bit-for-bit identical. Where a deviation is intentional, document
it in the function or module docstring. New ports should be validated against reference output
using golden fixtures (see [Running the Tests](#running-the-tests)); a "parity" claim should be
backed by a committed reference from the MATLAB code if possible, not by hand-written expected
values.

### Dependencies

Keep the base install lean. General-purpose tools should rely only on the core dependencies.
Anything with a heavy or specialised dependency (large binary wheels, deep-learning stacks)
belongs behind an **optional extra**, and its imports should be deferred (imported inside the
function that needs them) so that `import pnanolocz` never requires an optional dependency and
a missing extra can never break an unrelated part of the package.

---

## Running the Tests

Tests are written with [pytest](https://pytest.org). To run the full test suite from the
project root:

```bash
pytest .
```

To run a specific test file:

```bash
pytest tests/test_level.py
```

All tests must pass before submitting a pull request. If you are adding new functionality,
please include tests covering the new behaviour. Tests live in the `tests/` folder and are
organised **one test module per source module** (`test_level.py`, `test_thresholder.py`, …);
larger subpackages may have their own test subdirectory.

### Golden fixtures

**pnanolocz** validates ports against reference output using fixtures committed under
`tests/resources/`. The established pattern is a raw input array (`*_raw.npz`) paired with the
expected **NanoLocz**/MATLAB output for a given method (`*_nanolocz_*.npz`); a test loads the raw
input, runs the Python implementation, and asserts agreement with the reference within a stated
tolerance. New ports should follow this pattern.

---

## Code Style and Pre-commit

**pnanolocz** uses [pre-commit](https://pre-commit.com) to enforce consistent code style and catch
common issues. The hooks run automatically on commit once installed:

```bash
pre-commit install
```

To run the hooks manually against all files:

```bash
pre-commit run --all-files
```

The pre-commit configuration pins Python to 3.11. If you are on a different version, the hooks
will still run but this is the version used in CI.

You do not need to manually configure formatting — the hooks handle this. If a hook modifies a
file, stage the changes and commit again.

Beyond what pre-commit enforces, please follow these general conventions:

- Prefer explicit over implicit; avoid ambiguous variable names and hidden state changes.
- Keep functions focused; extract helpers rather than nesting deeply.
- Prefer pure functions over stateful objects; keep transient pipeline state with the caller.
- Log significant internal decisions using the module-level `logger`
  (`logging.getLogger(__name__)`) rather than printing to stdout.
- Use `# noqa` comments sparingly and always with a specific rule code (e.g. `# noqa: E501`).

---

## Type Hints

All new code should include type hints. **pnanolocz** targets Python 3.11+, so built-in generics
(`list[int]`, `dict[str, Any]`) can be used directly. Import `Optional`, `Sequence`,
`Callable`, and similar from `typing` where needed. Type checking runs under `mypy --strict`
in CI, so annotations must be complete — including return types.

```python
# Good — arrays annotated, return type explicit
def process_frame(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    ...

# Good — Optional for arguments that can be None
def apply_thresholder(
    img: np.ndarray[Any, np.dtype[np.float64]],
    method: str,
    limits: tuple[float, float] | list[float] | str | None = None,
    invert: bool = False,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    ...
```

Return types should always be annotated. For functions that return nothing, annotate
`-> None` explicitly. Type hints on private helpers (`_` prefix) are encouraged but slightly
less strictly required than on public API.

---

## Docstrings

**pnanolocz** uses **NumPy-style docstrings** for all publicly exposed functions, methods, and
classes. **Every function, method, and class must have at least one line of documentation**,
undocumented code will not be accepted.

### One-line docstrings

Single line docstrings can be for private helpers and simple functions where the signature alone is not
self-explanatory.

### NumPy-style docstrings

Used for all public API — any function, method, or class that is part of the user-facing or
developer-facing interface:

```python
def apply_level(
    img: np.ndarray[Any, np.dtype[np.float64]],
    polyx: Optional[int] = None,
    polyy: Optional[int] = None,
    method: str = "plane",
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]] = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Apply a leveling method to an AFM image or stack.

    Public dispatcher for the leveling routines in this module. Normalizes
    inputs to a frame-first representation ``(N, H, W)``, applies the requested
    method frame-by-frame, and returns an array with the same shape as the input.

    Parameters
    ----------
    img : ndarray
        Input AFM image ``(H, W)`` or stack ``(N, H, W)``.
    polyx, polyy : int, optional
        Polynomial degrees (interpretation depends on ``method``).
    method : str
        Leveling method to apply.
    mask : ndarray of bool, optional
        Exclusion mask; ``True`` marks excluded pixels, ``False`` marks valid.

    Returns
    -------
    ndarray
        Leveled array with the same shape as ``img``.

    Notes
    -----
    Aims for numerical alignment with the MATLAB NanoLocz reference; intentional
    deviations are documented where they occur.
    """
```

### Sections to include

Include the following sections as applicable:

| Section | When to include |
| --- | --- |
| Summary (first line) | Always — one line, imperative mood |
| Extended description | When the summary alone is insufficient |
| `Parameters` | Any function that takes arguments |
| `Returns` | Any function that returns a value |
| `Raises` | When specific exceptions are raised intentionally |
| `See Also` | When related functions or classes exist |
| `Notes` | For algorithmic detail, MATLAB-alignment caveats, or references |
| `Examples` | For public API; optional for simple cases |

For **module-level docstrings**, follow the pattern established in `level.py` and
`thresholder.py`: a one-line summary, an extended description, a note on the MATLAB source and
any intentional deviations, a list of the methods the module provides, and an `Authors`
section.

---

## Branching and Pull Requests

- **`main` is reserved for releases.** Base pull requests on the current development branch
  where one is in use; open a discussion or issue first if you are unsure which branch to
  target.
- Use a descriptive branch name, e.g. `feature/lafm-render` or `fix/thresholder-none-index`.
- Keep pull requests focused, one feature or fix per PR where possible. In particular, do not
  bundle unrelated refactors of existing modules into a feature PR; propose those separately so
  each change can be reviewed on its own merits.
- CI (tests and pre-commit) runs automatically on pull requests.

### Pull request checklist

- [ ] The change respects the [Key design concepts](#key-design-concepts) (arrays in, simple
      types out; no file IO, display, GUI, or session state in the library)
- [ ] Tests pass locally (`pytest .`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Type checking passes (`mypy --strict`)
- [ ] New functionality is covered by tests, with golden fixtures for any MATLAB-ported behaviour
- [ ] Docstrings are present and follow NumPy style for all public API
- [ ] At least a one-line docstring is present on every function and method
- [ ] Type hints are included on all new functions and methods
- [ ] Any new heavy dependency is behind an optional extra and lazily imported
- [ ] A changelog entry has been added (see below)
- [ ] AI tool usage is disclosed if applicable (see [AI Transparency](#ai-transparency))

---

## Changelog

**pnanolocz** maintains a [CHANGELOG.md](CHANGELOG.md) in the repo root following the
[Keep a Changelog](https://keepachangelog.com) format.

When submitting a pull request, please add a brief entry under the relevant section
(`Added`, `Changed`, `Fixed`, `Documentation`) in the `Unreleased` section of `CHANGELOG.md`.
If your change is a **breaking change** — including any change to the name, signature, or
output of an existing public function — note it clearly under `Changed` with a migration note.

---

## Versioning

**pnanolocz** uses **VCS-based versioning** via `setuptools_scm` — the package version is derived
automatically from Git tags. **Do not manually edit version numbers** in any file. Releases are
made by tagging with a version string (e.g. `v0.2.0`).

Individual public functions carry their own version via a `__version__` attribute set on the
function (e.g. ``apply_level.__version__ = "0.1.0"``). Bump this when the behaviour of that
specific function changes, so that downstream callers and provenance records can track it.

---

## Writing Processing Tools

Processing tools operate on image data and return image data or masks. For example leveling
functions (returning a corrected array) and thresholding functions (returning a boolean mask).

Guidelines:

- Accept a NumPy array (a single frame ``(H, W)`` or a frame-first stack ``(N, H, W)``) and
  return a NumPy array of the same convention. Do not load or save files.
- Where a family of related methods exists, expose a single `apply_*` dispatcher that selects
  the method via a `method` argument, alongside the individual named functions (follow the
  structure of `level.py` and `thresholder.py`). Give the dispatcher a `__version__` attribute.
- Follow the mask convention used across the package: ``True`` = excluded/selected as
  appropriate to the routine, documented explicitly in the docstring.

### Exposing a filter to playNano

**pnanolocz**'s frame-based processing functions can be exposed to
[playNano](https://github.com/derollins/playNano) as filters via the `playnano.filters` entry
point group (2D frame → frame). Register the `apply_*` function in `pyproject.toml` under
`[project.entry-points."playnano.filters"]`, following the existing leveling entries. Only
frame-in/frame-out operations fit this model — multi-stage analyses (see below) do not, and
should not be forced into it.

---

## Writing Analysis Tools

Some **NanoLocz** capabilities are larger and multi-stage: they take a movie or stack and run several steps
to produce tables and/or image arrays, and so don't fit the single-function processing model
above. Such tools should be organised as **subpackages** under `src/pnanolocz/` (e.g.
`pnanolocz/<toolname>/`, with an `__init__` exposing its public functions) and follow the shape below.

Guidelines:

- **Express the pipeline as pure functions**, each taking its inputs and returning its output
  (a table, an image array, a small result record). The caller holds intermediate values between
  stages; the correct order of operations is enforced by the function signatures, not by a
  stateful object and runtime guards.
- Provide a single thin convenience function (e.g. `run_<tool>(data, params) -> result`) for the
  common "run it end to end" case. It should be a few lines of orchestration over the stage
  functions and hold no state itself.
- Keep the contract: **arrays in, simple types out.** Anything that would otherwise be a display
  or IO concern crosses the boundary as data — a region of interest is passed in as coordinates
  or a mask, a colormap is passed in as an array, an image result is returned as an array. The
  selection, raw-data loading, display, and saving are demonstrated in a notebook, not built into
  the subpackage.
- Put heavy dependencies behind an optional extra and import them lazily.
- Validate against the reference implementation with golden fixtures.

A new multi-stage tool should be accompanied by a demo notebook that runs it end to end on real
data (see [Notebooks](#notebooks)).

---

## Notebooks

The `notebooks/` directory holds demo notebooks that show how to use the library end to end.
Because loading, display, and saving live outside the package (see
[Key design concepts](#key-design-concepts)), the notebook is where those pieces come together:
it loads the raw data (for example with a data loader such as
[AFMReader](https://github.com/AFM-SPM/AFMReader) or [playNano](https://github.com/derollins/playNano)),
prepares any inputs such as a region of interest, calls the library functions, and displays and
saves the results.

Keep all of the loading, input selection, display, and saving in the notebook itself, or in a
small notebook-support module beside it, never in the package. A notebook accompanying a new
multi-stage tool should walk through its whole workflow this way, so the tool can be seen working
on real data without any of that scaffolding leaking into the library.

---

## Reporting Bugs and Requesting Features

Please use [GitHub Issues](https://github.com/derollins/Python-Nanolocz-Library/issues) to
report bugs or request features.

When reporting a bug, please include:

- Your operating system and Python version.
- The **pnanolocz** version (`pip show pnanolocz`).
- A minimal reproducible example if possible.
- The full traceback if an exception was raised.

For feature requests, a brief description of the use case is helpful alongside the proposed
behaviour.

---

## AI Transparency

AI-based tools may be used during development for tasks such as typing assistance, formatting,
debugging and refactoring suggestions, and documentation drafting.

Any use of AI tools in contributions to **pnanolocz** should be disclosed in one of the following
ways:

- A note in the module-level docstring of any file where AI tools contributed substantially to
  the logic or structure.
- A note in the pull request description.

AI-generated code must be reviewed, tested, and validated by the contributor before submission.
The contributor is responsible for the correctness of all submitted code regardless of how it
was produced.
