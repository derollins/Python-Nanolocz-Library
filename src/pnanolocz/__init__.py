"""Stable package interface for NanoLocz AFM and LAFM analysis."""

from importlib.metadata import PackageNotFoundError, version

from . import level, level_auto, level_weighted, thresholder
from .detector import detector
from .lafm_renderer import lafm_movie_renderer, lafm_renderer
from .lafm_workflow import LAFMWorkflow
from .localize import localize, localize_matlab

try:
    from ._version import version as __version__  # type: ignore[import-not-found]
except Exception:
    try:
        __version__ = version("pnanolocz")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "__version__",
    "detector",
    "localize",
    "localize_matlab",
    "LAFMWorkflow",
    "lafm_renderer",
    "lafm_movie_renderer",
    "level",
    "level_auto",
    "level_weighted",
    "thresholder",
]
