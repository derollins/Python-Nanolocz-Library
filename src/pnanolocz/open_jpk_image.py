"""Compatibility wrapper for ``pnanolocz.open_jpk.open_jpk_image``."""

from __future__ import annotations

try:
    from .open_jpk import open_JPK_image, open_jpk_image
except Exception:  # pragma: no cover
    from open_jpk import open_JPK_image, open_jpk_image  # type: ignore

__all__ = ["open_jpk_image", "open_JPK_image"]
