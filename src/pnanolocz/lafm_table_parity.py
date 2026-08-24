"""Direct MATLAB/Python LAFM localization-table parity utilities."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment

from pnanolocz.fast_peaks2d import fast_peaks2d
from pnanolocz.localize import localize

TABLE_COLUMNS = (
    "x",
    "y",
    "z",
    "prominence",
    "frame",
    "source_frame",
    "time",
    "correlation",
    "extra9",
    "extra10",
    "extra11",
    "extra12",
)


def _matlab_round(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def _gated_assignment(
    distances: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize valid matches, then minimize their total distance."""
    n_mat, n_py = distances.shape
    size = n_mat + n_py
    penalty = float(max_distance) + 1.0
    invalid = penalty * (size + 2)
    cost = np.full((size, size), invalid, dtype=np.float64)
    cost[:n_mat, :n_py] = np.where(distances <= float(max_distance), distances, invalid)
    cost[:n_mat, n_py:] = penalty
    cost[n_mat:, :n_py] = penalty
    cost[n_mat:, n_py:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    valid = (
        (rows < n_mat)
        & (cols < n_py)
        & (
            distances[rows.clip(max=n_mat - 1), cols.clip(max=n_py - 1)]
            <= float(max_distance)
        )
    )
    return rows[valid], cols[valid]


def create_run_directory(root: str | Path, *, timestamp: str | None = None) -> Path:
    """Create and return a uniquely timestamped report directory."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    stem = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root_path / stem
    suffix = 2
    while candidate.exists():
        candidate = root_path / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def direct_localize_stack(movie: np.ndarray) -> np.ndarray:
    """Run the fixed direct LAFM pipeline on a frame-first aligned stack."""
    stack = np.asarray(movie, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError("movie must be a 2-D image or frame-first 3-D stack")
    rows: list[np.ndarray] = []
    for frame_index, frame in enumerate(stack):
        peaks = fast_peaks2d(frame, 0.0, 1, 0.0, matlab_indexing=True)
        if len(peaks):
            table = np.zeros((len(peaks), len(TABLE_COLUMNS)), dtype=np.float64)
            table[:, :4] = peaks
            table[:, 4] = frame_index + 1
            table[:, 5] = frame_index + 1
            table[:, 6] = frame_index + 1
            table[:, 7] = frame_index + 1
            rows.append(table)
    if not rows:
        return np.empty((0, len(TABLE_COLUMNS)), dtype=np.float64)
    localized = localize(
        stack,
        np.vstack(rows),
        "bicubic",
        1.0,
        frame_axis=0,
        matlab_indexing=True,
    )
    finite = np.all(np.isfinite(localized[:, :5]), axis=1)
    localized = localized[finite]
    xs = _matlab_round(localized[:, 0]).astype(int)
    ys = _matlab_round(localized[:, 1]).astype(int)
    frames = _matlab_round(localized[:, 4]).astype(int)
    inside = (
        (xs > 0)
        & (xs < stack.shape[2])
        & (ys > 0)
        & (ys < stack.shape[1])
        & (frames > 0)
        & (frames <= stack.shape[0])
    )
    localized = localized[inside]
    xs, ys, frames = xs[inside] - 1, ys[inside] - 1, frames[inside] - 1
    localized[:, 2] = stack[frames, ys, xs]
    return np.asarray(localized, dtype=np.float64)


def _refresh_step3_z(table: pd.DataFrame, movie: np.ndarray) -> pd.DataFrame:
    """Reapply Step3 with MATLAB rounding to a previously localized table."""
    stack = np.asarray(movie, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    xs = _matlab_round(table["x"].to_numpy(float)).astype(int)
    ys = _matlab_round(table["y"].to_numpy(float)).astype(int)
    frames = _matlab_round(table["frame"].to_numpy(float)).astype(int)
    inside = (
        (xs > 0)
        & (xs < stack.shape[2])
        & (ys > 0)
        & (ys < stack.shape[1])
        & (frames > 0)
        & (frames <= stack.shape[0])
    )
    refreshed = table.loc[inside].copy()
    refreshed.loc[:, "z"] = stack[frames[inside] - 1, ys[inside] - 1, xs[inside] - 1]
    return refreshed


def _xyzf(table: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(table, pd.DataFrame):
        return table.loc[:, ["x", "y", "z", "frame"]].to_numpy(float)
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("table must provide x, y, z, and frame")
    return arr[:, :4]


def match_localization_tables(
    matlab: np.ndarray | pd.DataFrame,
    python: np.ndarray | pd.DataFrame,
    *,
    max_distance: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match rows one-to-one within each frame using gated Hungarian assignment."""
    mat = _xyzf(matlab)
    py = _xyzf(python)
    matched_rows: list[dict[str, float]] = []
    mat_only: list[np.ndarray] = []
    py_only: list[np.ndarray] = []
    frames = np.union1d(mat[:, 3] if len(mat) else [], py[:, 3] if len(py) else [])
    for frame in frames:
        mi = np.flatnonzero(mat[:, 3] == frame)
        pi = np.flatnonzero(py[:, 3] == frame)
        if not len(mi):
            py_only.extend(py[pi])
            continue
        if not len(pi):
            mat_only.extend(mat[mi])
            continue
        distances = np.linalg.norm(mat[mi, None, :2] - py[None, pi, :2], axis=2)
        mr, pr = _gated_assignment(distances, float(max_distance))
        used_m: set[int] = set()
        used_p: set[int] = set()
        for m_local, p_local in zip(mr, pr, strict=True):
            distance = float(distances[m_local, p_local])
            m_idx, p_idx = int(mi[m_local]), int(pi[p_local])
            used_m.add(m_idx)
            used_p.add(p_idx)
            row: dict[str, float] = {
                "frame": float(frame),
                "distance_xy": distance,
                "matlab_index": m_idx,
                "python_index": p_idx,
            }
            for col, idx in (("x", 0), ("y", 1), ("z", 2)):
                row[f"matlab_{col}"] = float(mat[m_idx, idx])
                row[f"python_{col}"] = float(py[p_idx, idx])
                row[f"delta_{col}"] = float(py[p_idx, idx] - mat[m_idx, idx])
            matched_rows.append(row)
        mat_only.extend(mat[idx] for idx in mi if int(idx) not in used_m)
        py_only.extend(py[idx] for idx in pi if int(idx) not in used_p)
    matched = pd.DataFrame(matched_rows)
    only_columns = ["x", "y", "z", "frame"]
    return (
        matched,
        pd.DataFrame(mat_only, columns=only_columns),
        pd.DataFrame(py_only, columns=only_columns),
    )


def summarize_matches(
    matched: pd.DataFrame, matlab_count: int, python_count: int
) -> dict[str, float | int]:
    """Summarize localization match counts and coordinate residuals."""
    metrics: dict[str, float | int] = {
        "matlab_count": int(matlab_count),
        "python_count": int(python_count),
        "matched_count": int(len(matched)),
        "matlab_only_count": int(matlab_count - len(matched)),
        "python_only_count": int(python_count - len(matched)),
        "match_rate_matlab": (
            float(len(matched) / matlab_count) if matlab_count else np.nan
        ),
    }
    for col in ("x", "y", "z"):
        delta = (
            matched[f"delta_{col}"].to_numpy(float) if len(matched) else np.array([])
        )
        metrics[f"{col}_bias"] = float(np.mean(delta)) if len(delta) else np.nan
        metrics[f"{col}_mae"] = (
            round(float(np.mean(np.abs(delta))), 12) if len(delta) else np.nan
        )
        metrics[f"{col}_rmse"] = (
            round(float(np.sqrt(np.mean(delta**2))), 12) if len(delta) else np.nan
        )
        metrics[f"{col}_median_ae"] = (
            float(np.median(np.abs(delta))) if len(delta) else np.nan
        )
        metrics[f"{col}_max_ae"] = (
            float(np.max(np.abs(delta))) if len(delta) else np.nan
        )
    return metrics


def export_python_tables(input_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Export interpolation localization tables for all input TIFF files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.tiff")):
        stat = path.stat()
        entry: dict[str, Any] = {
            "source": path.name,
            "source_bytes": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "status": "ok",
        }
        try:
            target = output_dir / f"{path.stem}.csv"
            if target.exists():
                existing_table = pd.read_csv(target)
                if tuple(existing_table.columns) != TABLE_COLUMNS:
                    raise ValueError(
                        f"Existing table has incompatible schema: {target}"
                    )
                stack = tifffile.imread(path)
                existing_table = _refresh_step3_z(existing_table, stack)
                existing_table.to_csv(target, index=False)
                existing = existing_table.loc[:, ["frame"]]
                stack_shape = stack.shape
                entry.update(
                    rows=int(len(existing)),
                    frames=int(stack_shape[0] if len(stack_shape) == 3 else 1),
                    shape=list(stack_shape),
                    table=target.name,
                    resumed=True,
                )
                manifest.append(entry)
                continue
            stack = tifffile.imread(path)
            table = direct_localize_stack(stack)
            pd.DataFrame(table, columns=TABLE_COLUMNS).to_csv(target, index=False)
            entry.update(
                rows=int(len(table)),
                frames=int(stack.shape[0] if stack.ndim == 3 else 1),
                shape=list(stack.shape),
                table=target.name,
            )
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
        manifest.append(entry)
    payload = {
        "pipeline_version": 1,
        "parameters": {
            "low_pass_sigma": 0,
            "high_pass_sigma": 0,
            "min_separation": 1,
            "height_threshold": 0,
            "prominence_threshold": 0,
            "localization_method": "bicubic",
            "pixperfeat": 1,
        },
        "files": manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return manifest


def compare_directories(
    matlab_dir: Path, python_dir: Path, output_dir: Path
) -> pd.DataFrame:
    """Compare exported MATLAB and Python interpolation tables."""
    matlab_manifest = json.loads((matlab_dir / "manifest.json").read_text())
    python_manifest = json.loads((python_dir / "manifest.json").read_text())
    matlab_files = matlab_manifest["files"]
    python_files = python_manifest["files"]
    for label, entries in (("MATLAB", matlab_files), ("Python", python_files)):
        failed = [entry for entry in entries if entry.get("status") != "ok"]
        if failed:
            raise RuntimeError(f"{label} manifest contains failed inputs: {failed}")
    matlab_sources = {entry["source"]: entry for entry in matlab_files}
    python_sources = {entry["source"]: entry for entry in python_files}
    if set(matlab_sources) != set(python_sources):
        raise RuntimeError("MATLAB and Python manifests contain different inputs")
    if len(matlab_sources) != 13:
        raise RuntimeError(
            f"Expected 13 aligned TIFF inputs, found {len(matlab_sources)}"
        )
    for source, mat_entry in matlab_sources.items():
        py_entry = python_sources[source]
        if mat_entry["table"] != py_entry["table"]:
            raise RuntimeError(f"Table-name mismatch for {source}")
        if not (matlab_dir / mat_entry["table"]).exists():
            raise RuntimeError(f"Missing MATLAB table for {source}")
        if not (python_dir / py_entry["table"]).exists():
            raise RuntimeError(f"Missing Python table for {source}")

    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    file_metrics: list[dict[str, Any]] = []
    frame_metrics: list[dict[str, Any]] = []
    all_matched: list[pd.DataFrame] = []
    for matlab_path in sorted(matlab_dir.glob("*.csv")):
        python_path = python_dir / matlab_path.name
        if not python_path.exists():
            continue
        mat = pd.read_csv(matlab_path)
        py = pd.read_csv(python_path)
        matched, mat_only, py_only = match_localization_tables(mat, py)
        stem = matlab_path.stem
        matched.to_csv(matches_dir / f"{stem}_matched.csv", index=False)
        all_matched.append(matched)
        mat_only.to_csv(matches_dir / f"{stem}_matlab_only.csv", index=False)
        py_only.to_csv(matches_dir / f"{stem}_python_only.csv", index=False)
        summary = summarize_matches(matched, len(mat), len(py))
        summary["source"] = stem
        file_metrics.append(summary)
        source_entry = next(
            entry for entry in matlab_files if entry["table"] == matlab_path.name
        )
        all_frames = range(1, int(source_entry["frames"]) + 1)
        for frame in all_frames:
            mf = mat[mat["frame"] == frame]
            pf = py[py["frame"] == frame]
            mm = matched[matched["frame"] == frame] if len(matched) else matched
            row = summarize_matches(mm, len(mf), len(pf))
            row.update(source=stem, frame=int(frame))
            frame_metrics.append(row)
    files = pd.DataFrame(file_metrics)
    frames = pd.DataFrame(frame_metrics)
    files.to_csv(output_dir / "per_file_metrics.csv", index=False)
    frames.to_csv(output_dir / "per_frame_metrics.csv", index=False)
    if len(files):
        combined = pd.concat(all_matched, ignore_index=True)
        overall = summarize_matches(
            combined,
            int(files["matlab_count"].sum()),
            int(files["python_count"].sum()),
        )
        overall["files_compared"] = int(len(files))
        pd.DataFrame([overall]).to_csv(output_dir / "overall_metrics.csv", index=False)
        report = [
            "# Direct LAFM MATLAB/Python Table Comparison",
            "",
            f"- Files compared: {overall['files_compared']}",
            f"- MATLAB rows: {overall['matlab_count']}",
            f"- Python rows: {overall['python_count']}",
            f"- Matched rows (≤0.75 px): {overall['matched_count']}",
            f"- MATLAB match rate: {overall['match_rate_matlab']:.2%}",
            f"- x MAE / RMSE: {overall['x_mae']:.6g} / {overall['x_rmse']:.6g} px",
            f"- y MAE / RMSE: {overall['y_mae']:.6g} / {overall['y_rmse']:.6g} px",
            f"- z MAE / RMSE: {overall['z_mae']:.6g} / {overall['z_rmse']:.6g}",
            "",
            "## Per-file metrics",
            "",
            "```csv",
            files.to_csv(index=False).strip(),
            "```",
        ]
        (output_dir / "comparison_report.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
    return files


def main() -> None:
    """Run interpolation export or comparison from the command line."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    py_cmd = sub.add_parser("python")
    py_cmd.add_argument("--input", type=Path, required=True)
    py_cmd.add_argument("--output", type=Path, required=True)
    cmp_cmd = sub.add_parser("compare")
    cmp_cmd.add_argument("--matlab", type=Path, required=True)
    cmp_cmd.add_argument("--python", type=Path, required=True)
    cmp_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "python":
        export_python_tables(args.input, args.output)
    else:
        compare_directories(args.matlab, args.python, args.output)


if __name__ == "__main__":
    main()
