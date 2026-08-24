"""
Timestamp parsing for NanoLocz.

This module ports MATLAB ``time_elapsed.m``.  It extracts times in
``HH:MM:SS AM/PM`` form and returns elapsed seconds relative to the first parsed
timestamp.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np


_TIME_PATTERN = re.compile(r"(\d{2}:\d{2}:\d{2} [APap][Mm])")


def time_elapsed(strings: Iterable[str]) -> np.ndarray:
    """Return elapsed seconds from the first timestamp in a string sequence."""
    values = list(strings)
    seconds = np.zeros(len(values), dtype=np.float64)

    for i, item in enumerate(values):
        match = _TIME_PATTERN.search(str(item))
        if match:
            dt = datetime.strptime(match.group(1).upper(), "%I:%M:%S %p")
            seconds[i] = dt.hour * 3600 + dt.minute * 60 + dt.second

    if seconds.size == 0:
        return seconds

    elapsed = seconds - seconds[0]

    # Handle midnight wraparound.
    for i in range(1, elapsed.size):
        if elapsed[i] < elapsed[i - 1]:
            elapsed[i:] += 24 * 3600

    return elapsed


__all__ = ["time_elapsed"]
