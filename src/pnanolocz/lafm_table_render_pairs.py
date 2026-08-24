"""Render MATLAB/Python LAFM table pairs on a shared coordinate grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

from pnanolocz.lafm_renderer import _render_subset, _resolve_colormap


def _xyz(table: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(table, pd.DataFrame):
        arr = table.loc[:, ["x", "y", "z"]].to_numpy(float)
    else:
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("table must contain x, y, and z columns")
        arr = arr[:, :3]
    return arr[np.all(np.isfinite(arr), axis=1)]


def render_table_pair(
    matlab: np.ndarray | pd.DataFrame,
    python: np.ndarray | pd.DataFrame,
    *,
    img_gus: float = 1.0,
    expand: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render two tables with common x/y bounds, z mapping, and canvas."""
    mat = _xyz(matlab)
    py = _xyz(python)
    if not len(mat) or not len(py):
        raise ValueError("both localization tables must be non-empty")
    combined = np.vstack([mat, py])
    xmin, xmax = float(combined[:, 0].min()), float(combined[:, 0].max())
    ymin, ymax = float(combined[:, 1].min()), float(combined[:, 1].max())
    zmin, zmax = float(combined[:, 2].min()), float(combined[:, 2].max())
    mapping_zmax = zmax if zmax != zmin else zmin + 1.0
    cmap = _resolve_colormap("LAFM color")

    def prepare(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prepared = table.copy()
        prepared[:, 0] = np.round((table[:, 0] - xmin + 1.0) * expand)
        prepared[:, 1] = np.round((table[:, 1] - ymin + 1.0) * expand)
        xp = np.linspace(zmin, mapping_zmax, cmap.shape[0])
        indices = np.round(
            np.interp(
                prepared[:, 2],
                xp,
                np.arange(1, cmap.shape[0] + 1),
                left=1,
                right=cmap.shape[0],
            )
        )
        return prepared, indices

    mat_prepared, mat_indices = prepare(mat)
    py_prepared, py_indices = prepare(py)
    image_size = (
        int(max(mat_prepared[:, 1].max(), py_prepared[:, 1].max())) + 5,
        int(max(mat_prepared[:, 0].max(), py_prepared[:, 0].max())) + 5,
    )

    def render(prepared: np.ndarray, indices: np.ndarray) -> np.ndarray:
        raw = _render_subset(
            prepared,
            indices,
            cmap,
            image_size,
            img_gus=float(img_gus),
            expand=float(expand),
            prob=False,
        )
        scale = max(float(np.nanmax(raw)), 1.0)
        return np.asarray(np.clip(raw / scale, 0.0, 1.0), dtype=np.float64)

    metadata = {
        "z_range": [zmin, zmax],
        "xy_bounds": [xmin, xmax, ymin, ymax],
        "shape": list(image_size) + [3],
        "img_gus": float(img_gus),
        "expand": float(expand),
        "colormap": "LAFM color",
    }
    return render(mat_prepared, mat_indices), render(py_prepared, py_indices), metadata


def _new_render_directory(run_dir: Path) -> Path:
    candidate = run_dir / "renders"
    suffix = 2
    while candidate.exists():
        candidate = run_dir / f"renders_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _table_directories(run_path: Path) -> tuple[Path, Path]:
    """Resolve legacy or method-specific paired table directories."""
    for matlab_name, python_name in (
        ("matlab_tables", "python_tables"),
        ("matlab", "python"),
    ):
        matlab_dir = run_path / matlab_name
        python_dir = run_path / python_name
        if matlab_dir.is_dir() and python_dir.is_dir():
            return matlab_dir, python_dir
    raise RuntimeError(f"No paired MATLAB/Python table directories in {run_path}")


def render_all_pairs(run_dir: str | Path) -> Path:
    """Render all successful table pairs from a completed parity run."""
    run_path = Path(run_dir)
    matlab_dir, python_dir = _table_directories(run_path)
    matlab_manifest = json.loads((matlab_dir / "manifest.json").read_text())
    python_manifest = json.loads((python_dir / "manifest.json").read_text())
    mat_entries = {entry["source"]: entry for entry in matlab_manifest["files"]}
    py_entries = {entry["source"]: entry for entry in python_manifest["files"]}
    if set(mat_entries) != set(py_entries) or len(mat_entries) != 13:
        raise RuntimeError("Expected the same 13 successful inputs in both manifests")

    output = _new_render_directory(run_path)
    mat_out, py_out, compare_out = (
        output / "matlab",
        output / "python",
        output / "comparisons",
    )
    for directory in (mat_out, py_out, compare_out):
        directory.mkdir()

    cmap = ListedColormap(_resolve_colormap("LAFM color"), name="LAFM color")
    manifest: list[dict[str, Any]] = []
    for source in sorted(mat_entries):
        mat_entry, py_entry = mat_entries[source], py_entries[source]
        if mat_entry["status"] != "ok" or py_entry["status"] != "ok":
            raise RuntimeError(f"Unsuccessful table pair: {source}")
        mat_table = pd.read_csv(matlab_dir / mat_entry["table"])
        py_table = pd.read_csv(python_dir / py_entry["table"])
        mat_rgb, py_rgb, metadata = render_table_pair(mat_table, py_table)
        stem = Path(source).stem
        mat_tiff = mat_out / f"{stem}_lafm_color.tiff"
        py_tiff = py_out / f"{stem}_lafm_color.tiff"
        comparison = compare_out / f"{stem}_comparison.png"
        tifffile.imwrite(mat_tiff, mat_rgb.astype(np.float32), photometric="rgb")
        tifffile.imwrite(py_tiff, py_rgb.astype(np.float32), photometric="rgb")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].imshow(mat_rgb, origin="upper")
        axes[1].imshow(py_rgb, origin="upper")
        axes[0].set_title(f"MATLAB table — {source}")
        axes[1].set_title(f"Python table — {source}")
        for axis in axes:
            axis.set_axis_off()
        zmin, zmax = metadata["z_range"]
        if zmax == zmin:
            zmax = zmin + 1.0
        fig.colorbar(
            ScalarMappable(norm=Normalize(zmin, zmax), cmap=cmap),
            ax=axes,
            label="Height (z)",
            fraction=0.035,
        )
        fig.savefig(comparison, dpi=180)
        plt.close(fig)
        manifest.append(
            {
                "source": source,
                "status": "ok",
                "matlab_rows": int(len(mat_table)),
                "python_rows": int(len(py_table)),
                **metadata,
                "matlab_tiff": str(mat_tiff.relative_to(output)),
                "python_tiff": str(py_tiff.relative_to(output)),
                "comparison_png": str(comparison.relative_to(output)),
            }
        )

    payload = {"pairs": manifest, "pair_count": len(manifest)}
    (output / "render_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    """Render every MATLAB/Python localization-table pair in a parity run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render_all_pairs(args.run_dir))


if __name__ == "__main__":
    main()
