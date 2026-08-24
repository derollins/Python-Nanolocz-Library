"""Compatibility wrapper for ``pnanolocz.open_jpk.open_jpk_info``."""

from __future__ import annotations

try:
    from .open_jpk import open_JPK_info, open_jpk_info
except Exception:  # pragma: no cover
    from open_jpk import open_JPK_info, open_jpk_info  # type: ignore

__all__ = ["open_jpk_info", "open_JPK_info"]
