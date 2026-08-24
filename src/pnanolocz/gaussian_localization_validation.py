"""Gaussian-only MATLAB/Python localization validation."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from pnanolocz.fast_peaks2d import fast_peaks2d
from pnanolocz.lafm_table_parity import (
    TABLE_COLUMNS,
    create_run_directory,
    match_localization_tables,
    summarize_matches,
)
from pnanolocz.localize import localize


def _input_tiffs(input_dir: Path) -> list[Path]:
    files = sorted({*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")})
    stems = [path.stem.casefold() for path in files]
    if len(stems) != len(set(stems)):
        raise ValueError("TIFF inputs contain colliding stems")
    return files


def _direct_gaussian_stack(movie: np.ndarray, pixperfeat: float = 1.0) -> np.ndarray:
    stack = np.asarray(movie, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    if stack.ndim != 3:
        raise ValueError("movie must be a 2-D image or frame-first 3-D stack")
    rows: list[np.ndarray] = []
    for frame_index, frame in enumerate(stack):
        peaks = fast_peaks2d(frame, 0.0, 1, 0.0, matlab_indexing=True)
        if len(peaks):
            table = np.zeros((len(peaks), len(TABLE_COLUMNS)), dtype=np.float64)
            table[:, :4] = peaks
            table[:, 4:8] = frame_index + 1
            rows.append(table)
    if not rows:
        return np.empty((0, len(TABLE_COLUMNS)), dtype=np.float64)
    result = localize(stack, np.vstack(rows), "gaussian", pixperfeat)
    finite = np.all(np.isfinite(result[:, :2]), axis=1)
    result = result[finite]
    if len(result):
        x = np.floor(np.abs(result[:, 0]) + 0.5).astype(int) - 1
        y = np.floor(np.abs(result[:, 1]) + 0.5).astype(int) - 1
        frame = np.floor(np.abs(result[:, 4]) + 0.5).astype(int) - 1
        inside = (
            (x >= 0)
            & (x < stack.shape[2])
            & (y >= 0)
            & (y < stack.shape[1])
            & (frame >= 0)
            & (frame < stack.shape[0])
        )
        result = result[inside]
        result[:, 2] = stack[frame[inside], y[inside], x[inside]]
    return result


def export_python_gaussian_tables(
    input_dir: str | Path, output_dir: str | Path, *, pixperfeat: float = 1.0
) -> list[dict[str, Any]]:
    """Export Gaussian localization tables and a per-input manifest."""
    input_path, output_path = Path(input_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for path in _input_tiffs(input_path):
        entry: dict[str, Any] = {
            "source": path.name,
            "source_bytes": path.stat().st_size,
            "source_mtime_ns": path.stat().st_mtime_ns,
            "method": "gaussian",
            "status": "ok",
        }
        try:
            target = output_path / f"{path.stem}.csv"
            if target.exists():
                table = pd.read_csv(target)
                if tuple(table.columns) != TABLE_COLUMNS:
                    raise ValueError(
                        f"Existing table has incompatible schema: {target}"
                    )
                entry.update(table=target.name, rows=int(len(table)), resumed=True)
                manifest.append(entry)
                continue
            stack = tifffile.imread(path)
            table = _direct_gaussian_stack(stack, pixperfeat)
            pd.DataFrame(table, columns=TABLE_COLUMNS).to_csv(target, index=False)
            entry.update(table=target.name, rows=int(len(table)))
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
        manifest.append(entry)
    (output_path / "manifest.json").write_text(
        json.dumps(
            {"method": "gaussian", "pixperfeat": pixperfeat, "files": manifest},
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def _add_fit_columns(
    matched: pd.DataFrame, mat: pd.DataFrame, py: pd.DataFrame
) -> pd.DataFrame:
    """Attach Gaussian fit values using the exact matched source rows."""
    result = matched.copy()
    mi = result["matlab_index"].to_numpy(int)
    pi = result["python_index"].to_numpy(int)
    for column, label in (("extra10", "width"), ("extra11", "amplitude")):
        result[f"matlab_{label}"] = mat.iloc[mi][column].to_numpy(float)
        result[f"python_{label}"] = py.iloc[pi][column].to_numpy(float)
        result[f"delta_{label}"] = result[f"python_{label}"] - result[f"matlab_{label}"]
    return result


def _fit_metrics(matched: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for label in ("width", "amplitude"):
        values = matched[f"delta_{label}"].to_numpy(float)
        metrics[f"{label}_bias"] = float(np.mean(values)) if len(values) else np.nan
        metrics[f"{label}_mae"] = (
            float(np.mean(np.abs(values))) if len(values) else np.nan
        )
        metrics[f"{label}_rmse"] = (
            float(np.sqrt(np.mean(values**2))) if len(values) else np.nan
        )
    return metrics


def _json_safe(value: Any) -> Any:
    """Replace non-finite numbers recursively with JSON null values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _plots(
    stem: str,
    matched: pd.DataFrame,
    mat: pd.DataFrame,
    py: pd.DataFrame,
    source: Path,
    plots_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    image = tifffile.imread(source)
    if image.ndim == 3:
        image = image[0]
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray", origin="upper")
    mat_plot, py_plot = mat[mat["frame"] == 1], py[py["frame"] == 1]
    ax.scatter(mat_plot["x"] - 1, mat_plot["y"] - 1, marker="+", label="MATLAB")
    ax.scatter(
        py_plot["x"] - 1,
        py_plot["y"] - 1,
        facecolors="none",
        edgecolors="tab:orange",
        label="Python",
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.savefig(plots_dir / f"{stem}_overlay.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    if len(matched):
        ax.scatter(matched["delta_x"], matched["delta_y"])
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Python - MATLAB x (px)")
    ax.set_ylabel("Python - MATLAB y (px)")
    fig.savefig(plots_dir / f"{stem}_residuals.png", dpi=150)
    plt.close(fig)


def _validated_manifest_tables(
    matlab_path: Path, python_path: Path, method: str
) -> list[tuple[str, str]]:
    """Return source/table pairs after validating both manifest contracts."""
    manifests = []
    for label, path in (("MATLAB", matlab_path), ("Python", python_path)):
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing {label} manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("method") not in (None, method):
            raise RuntimeError(f"{label} manifest method is not {method!r}")
        failed = [
            entry for entry in payload.get("files", []) if entry.get("status") != "ok"
        ]
        if failed:
            raise RuntimeError(f"{label} manifest contains failed inputs: {failed}")
        manifests.append(payload)
    mat_files, py_files = manifests[0]["files"], manifests[1]["files"]
    mat_entries = {entry["source"]: entry for entry in mat_files}
    py_entries = {entry["source"]: entry for entry in py_files}
    if len(mat_entries) != len(mat_files) or len(py_entries) != len(py_files):
        raise RuntimeError("Manifest contains duplicate source entries")
    if not mat_entries or set(mat_entries) != set(py_entries):
        raise RuntimeError("MATLAB and Python manifests contain different inputs")
    pairs: list[tuple[str, str]] = []
    for source in sorted(mat_entries):
        mat_table = mat_entries[source].get("table")
        py_table = py_entries[source].get("table")
        if not mat_table or mat_table != py_table:
            raise RuntimeError(f"Table mapping mismatch for {source}")
        if (
            not (matlab_path / mat_table).exists()
            or not (python_path / py_table).exists()
        ):
            raise RuntimeError(f"Missing required table for {source}")
        pairs.append((source, mat_table))
    return pairs


def compare_gaussian_directories(
    matlab_dir: str | Path,
    python_dir: str | Path,
    output_root: str | Path,
    *,
    input_dir: str | Path,
    timestamp: str | None = None,
    enforce_thresholds: bool = True,
) -> Path:
    """Compare Gaussian tables and write a timestamped report directory."""
    matlab_path, python_path, input_path = (
        Path(matlab_dir),
        Path(python_dir),
        Path(input_dir),
    )
    pairs = _validated_manifest_tables(matlab_path, python_path, "gaussian")
    expected_sources = {path.name for path in _input_tiffs(input_path)}
    actual_sources = {source for source, _ in pairs}
    if not expected_sources or actual_sources != expected_sources:
        raise RuntimeError(
            f"Manifest/input TIFF mismatch: expected {sorted(expected_sources)}, "
            f"got {sorted(actual_sources)}"
        )
    run_dir = create_run_directory(output_root, timestamp=timestamp)
    matches_dir = run_dir / "matches"
    matches_dir.mkdir()
    archived = {"matlab": run_dir / "matlab", "python": run_dir / "python"}
    for label, source_dir in (("matlab", matlab_path), ("python", python_path)):
        archived[label].mkdir()
        shutil.copy2(source_dir / "manifest.json", archived[label] / "manifest.json")
        for _, table_name in pairs:
            shutil.copy2(source_dir / table_name, archived[label] / table_name)
    summaries: list[dict[str, Any]] = []
    all_matches: list[pd.DataFrame] = []
    for source_name, table_name in pairs:
        mat_file = matlab_path / table_name
        py_file = python_path / table_name
        mat, py = pd.read_csv(mat_file), pd.read_csv(py_file)
        matched, mat_only, py_only = match_localization_tables(mat, py)
        matched = _add_fit_columns(matched, mat, py)
        stem = mat_file.stem
        matched.to_csv(matches_dir / f"{stem}_matched.csv", index=False)
        mat_only.to_csv(matches_dir / f"{stem}_matlab_only.csv", index=False)
        py_only.to_csv(matches_dir / f"{stem}_python_only.csv", index=False)
        summary: dict[str, Any] = summarize_matches(matched, len(mat), len(py))
        summary.update(_fit_metrics(matched), source=stem)
        summaries.append(summary)
        all_matches.append(matched)
        _plots(stem, matched, mat, py, input_path / source_name, run_dir / "plots")
    files = pd.DataFrame(summaries)
    files.to_csv(run_dir / "per_file_metrics.csv", index=False)
    combined = (
        pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    )
    overall = summarize_matches(
        combined,
        int(files["matlab_count"].sum()) if len(files) else 0,
        int(files["python_count"].sum()) if len(files) else 0,
    )
    overall.update(_fit_metrics(combined))
    thresholds = {
        "match_rate_matlab_min": 0.99,
        "x_rmse_max": 0.1,
        "y_rmse_max": 0.1,
        "width_rmse_max": 0.6,
        "amplitude_rmse_max": 0.12,
    }

    def passes(metrics: dict[str, Any]) -> bool:
        return bool(
            metrics["match_rate_matlab"] >= thresholds["match_rate_matlab_min"]
            and metrics["x_rmse"] <= thresholds["x_rmse_max"]
            and metrics["y_rmse"] <= thresholds["y_rmse_max"]
            and metrics["width_rmse"] <= thresholds["width_rmse_max"]
            and metrics["amplitude_rmse"] <= thresholds["amplitude_rmse_max"]
        )

    files["passed"] = [passes(row) for row in files.to_dict("records")]
    files.to_csv(run_dir / "per_file_metrics.csv", index=False)
    file_failures = files.loc[~files["passed"], "source"].tolist()
    passed = passes(overall) and not file_failures
    pd.DataFrame([overall]).to_csv(run_dir / "overall_metrics.csv", index=False)
    metadata = {
        "method": "gaussian",
        "status": "pass" if passed else "fail",
        "files_compared": len(pairs),
        "match_max_distance_px": 0.75,
        "thresholds": thresholds,
        "file_failures": file_failures,
        "overall": overall,
        "configuration": {
            "pixperfeat": 1.0,
            "detector": {
                "height_threshold": 0.0,
                "kernel_size": 1,
                "prominence_threshold": 0.0,
            },
            "match_max_distance_px": 0.75,
            "overlay_frame": 1,
        },
        "provenance": {
            "input_dir": str(input_path.resolve()),
            "matlab_manifest": "matlab/manifest.json",
            "python_manifest": "python/manifest.json",
        },
        "environment": {
            "matlab": json.loads(
                (matlab_path / "manifest.json").read_text(encoding="utf-8")
            ).get("matlab_version", "not recorded by legacy export"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tifffile": tifffile.__version__,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False), encoding="utf-8"
    )
    (run_dir / "comparison_report.md").write_text(
        "# Gaussian MATLAB/Python Localization Comparison\n\n"
        + files.to_csv(index=False),
        encoding="utf-8",
    )
    if enforce_thresholds and not passed:
        raise RuntimeError(f"Gaussian parity thresholds failed; report: {run_dir}")
    return run_dir


__all__ = ["compare_gaussian_directories", "export_python_gaussian_tables"]


def main() -> None:
    """Run the Gaussian exporter or directory comparison from the command line."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--matlab", type=Path, required=True)
    compare.add_argument("--python", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        entries = export_python_gaussian_tables(args.input, args.output)
        if any(entry["status"] != "ok" for entry in entries):
            raise SystemExit(1)
    else:
        compare_gaussian_directories(
            args.matlab, args.python, args.output, input_dir=args.input
        )


if __name__ == "__main__":
    main()
