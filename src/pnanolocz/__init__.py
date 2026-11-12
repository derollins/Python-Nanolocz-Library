"""pnanolocz: AFM and HS-AFM data analysis tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__  # type: ignore[import-not-found, unused-ignore]
except Exception:
    try:
        __version__ = version("pnanolocz")
    except PackageNotFoundError:
        __version__ = "0.0.0"
