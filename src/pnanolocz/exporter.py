"""
General data exporter for NanoLocz-compatible Python workflows.

This module ports NanoLocz ``exporter.m``.  It exports numerical arrays or
tabular data to common formats:

- ``.mat``
- ``.h5`` / ``.hdf5``
- ``.tif`` / ``.tiff``
- ``.txt``
- ``.csv``
- ``.xlsx``

Unlike MATLAB, this Python function does not open a save-file dialog.  A
``filepath`` must be provided, or a default filename is used in the current
working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _normalize_format(fmt: str | None, filepath: Path | None) -> str:
    """Normalize export format string to a lowercase extension."""
    if fmt is None or fmt == "":
        if filepath is not None and filepath.suffix:
            return filepath.suffix.lower()
        return ".mat"

    fmt = str(fmt).lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


def _normalize_filepath(filepath: str | Path | list[str] | tuple[str, ...] | None, fmt: str) -> Path:
    """Normalize filepath input and ensure it has the requested extension."""
    if filepath is None:
        path = Path("Data" + fmt)
    elif isinstance(filepath, (list, tuple)):
        if not filepath:
            path = Path("Data" + fmt)
        else:
            path = Path(filepath[0])
    else:
        path = Path(filepath)

    if path.suffix == "":
        path = path.with_suffix(fmt)

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _as_3d_slices(data: Any) -> list[np.ndarray]:
    """Split numerical data into 2-D slices for text/Excel export."""
    arr = np.asarray(data)

    if arr.ndim == 0:
        return [arr.reshape(1, 1)]
    if arr.ndim == 1:
        return [arr.reshape(-1, 1)]
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[:, :, i] for i in range(arr.shape[2])]

    raise ValueError("only up to 3-D arrays can be split into 2-D export slices")


def _export_mat(data: Any, filepath: Path) -> None:
    """Export MATLAB .mat file with variable name ``Data``."""
    try:
        from scipy.io import savemat
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("scipy is required for .mat export") from exc

    savemat(filepath, {"Data": data})


def _export_h5(data: Any, filepath: Path) -> None:
    """Export HDF5 file with dataset ``/Data``."""
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("h5py is required for .h5 export") from exc

    with h5py.File(filepath, "w") as h5:
        h5.create_dataset("Data", data=np.asarray(data))


def _export_tiff(data: Any, filepath: Path) -> None:
    """Export image or stack as TIFF."""
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("tifffile is required for TIFF export") from exc

    tifffile.imwrite(filepath, np.asarray(data))


def _export_table(data: Any, filepath: Path, fmt: str) -> None:
    """Export pandas-style tabular data."""
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("pandas is required for table export") from exc

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    if fmt == ".xlsx":
        df.to_excel(filepath, index=False)
    elif fmt == ".csv":
        df.to_csv(filepath, index=False)
    elif fmt == ".txt":
        df.to_csv(filepath, index=False, sep="\t")
    else:
        raise ValueError(f"unsupported table export format: {fmt}")


def _export_matrix_text(data: Any, filepath: Path, fmt: str) -> None:
    """Export numeric matrix/stack to text, CSV, or Excel."""
    slices = _as_3d_slices(data)

    if fmt == ".xlsx":
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError("pandas/openpyxl are required for .xlsx export") from exc

        with pd.ExcelWriter(filepath) as writer:
            for idx, matrix in enumerate(slices, start=1):
                pd.DataFrame(matrix).to_excel(writer, sheet_name=str(idx), index=False, header=False)
        return

    delimiter = "," if fmt == ".csv" else "\t"

    with open(filepath, "w", encoding="utf-8") as f:
        for idx, matrix in enumerate(slices):
            if idx > 0:
                f.write("\n")
            np.savetxt(f, np.asarray(matrix), delimiter=delimiter)


def exporter(
    data: Any,
    format: str | None = None,
    filepath: str | Path | list[str] | tuple[str, ...] | None = None,
    table: bool = False,
) -> Path:
    """Export data to disk.

    Parameters
    ----------
    data:
        Numerical matrix/stack or tabular data.
    format:
        File extension such as ``'.mat'``, ``'.h5'``, ``'.tiff'``,
        ``'.txt'``, ``'.xlsx'`` or ``'.csv'``.
    filepath:
        Output file path.  If omitted, ``Data.<format>`` is used.
    table:
        Treat ``data`` as tabular for text/Excel/CSV exports.

    Returns
    -------
    Path
        Saved file path.
    """
    path_hint = None
    if filepath is not None and not isinstance(filepath, (list, tuple)):
        path_hint = Path(filepath)

    fmt = _normalize_format(format, path_hint)
    path = _normalize_filepath(filepath, fmt)

    if fmt == ".mat":
        _export_mat(data, path)
    elif fmt in {".h5", ".hdf5"}:
        _export_h5(data, path)
    elif fmt in {".tif", ".tiff"}:
        _export_tiff(data, path)
    elif fmt in {".txt", ".xlsx", ".csv"}:
        if table:
            _export_table(data, path, fmt)
        else:
            _export_matrix_text(data, path, fmt)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    return path


__all__ = ["exporter"]
