"""
Rendering parameter conversion for NanoLocz.

This module ports MATLAB ``Res_to_render.m``.
"""

from __future__ import annotations

import numpy as np


def _round_significant(value: float, digits: int = 2) -> float:
    """Round to significant figures, similar to MATLAB round(...,'significant')."""
    val = float(value)
    if val == 0 or not np.isfinite(val):
        return val
    return float(np.round(val, digits - int(np.floor(np.log10(abs(val)))) - 1))


def res_to_render(pixpernm: float, res: float) -> tuple[float, float]:
    """Calculate rendering marker size and expansion factor."""
    expand = _round_significant(10.0 / (float(pixpernm) * float(res)), 2)
    expand = min(max(expand, 1.0), 10.0)

    render_point = _round_significant(float(res) * float(pixpernm) / 3.0, 2)
    render_point = min(max(render_point, 0.2), 10.0)

    return float(render_point), float(expand)


# MATLAB-style alias.
Res_to_render = res_to_render

__all__ = ["res_to_render", "Res_to_render"]
