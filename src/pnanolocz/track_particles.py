"""
Particle tracking utilities for NanoLocz.

This module ports MATLAB ``track_particles.m`` and the embedded
``simpletracker`` implementation by Jean-Yves Tinevez.  The high-level
``track_particles`` function assigns a track ID to each particle observation
based on x/y coordinates and frame numbers.

The implementation follows the MATLAB workflow:

1. Split input detections into one coordinate array per frame.
2. Link detections in consecutive frames with a distance-limited assignment.
3. Optionally close gaps by linking unmatched sources to unmatched targets in
   later frames.
4. Rebuild tracks from the directed adjacency graph.
5. Return one track ID per original detection row.

Python notes
------------
- Coordinates are treated as plain numeric positions.  Frame numbers may be
  MATLAB-style 1-based integers; output IDs are 1-based to match MATLAB.
- The default assignment method is Hungarian assignment via
  ``scipy.optimize.linear_sum_assignment``.
- A nearest-neighbour method is also provided for parity with the original
  ``simpletracker`` options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

FloatArray = np.ndarray
IntArray = np.ndarray


@dataclass
class SimpleTrackerResult:
    """Container returned by :func:`simpletracker`.

    Attributes
    ----------
    tracks:
        List of tracks.  Each track is an array of global observation indices
        in the concatenated frame order.  Indices are 0-based Python indices.
    adjacency_tracks:
        List of adjacency edge chains.  Each array contains global node indices
        belonging to one directed track path.  This is retained for MATLAB API
        parity; for most NanoLocz use, ``tracks`` is sufficient.
    adjacency:
        Dense boolean adjacency matrix.  ``adjacency[i, j]`` is true when
        observation ``i`` links to observation ``j``.  A dense matrix is used to
        keep the module lightweight; for very large movies a sparse matrix can
        be substituted later.
    frame_offsets:
        Starting global index of every frame in the concatenated node list.
    """
    tracks: list[np.ndarray]
    adjacency_tracks: list[np.ndarray]
    adjacency: np.ndarray
    frame_offsets: np.ndarray


def _as_points_list(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Validate and normalize a frame-wise point list."""
    out: list[np.ndarray] = []
    for frame in points:
        arr = np.asarray(frame, dtype=np.float64)
        if arr.size == 0:
            arr = np.empty((0, 2), dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Each points frame must be a 2-D array")
        out.append(arr)
    return out


def hungarianlinker(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Link source points to target points with a distance-limited assignment.

    Parameters
    ----------
    source, target:
        Arrays with shapes ``(n_source, n_dim)`` and ``(n_target, n_dim)``.
    max_distance:
        Maximum allowed linking distance.  Pairs farther than this are rejected.

    Returns
    -------
    target_indices:
        Length ``n_source`` array.  Entry ``i`` is the matched target index, or
        ``-1`` when source ``i`` is unmatched.
    target_distances:
        Length ``n_source`` array with matched distances, or ``nan`` for
        unmatched sources.
    unassigned_targets:
        Target indices that were not assigned.
    total_cost:
        Sum of accepted assignment distances.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)

    n_source = src.shape[0]
    n_target = tgt.shape[0]

    target_indices = np.full(n_source, -1, dtype=np.int64)
    target_distances = np.full(n_source, np.nan, dtype=np.float64)

    if n_source == 0:
        return target_indices, target_distances, np.arange(n_target, dtype=np.int64), 0.0
    if n_target == 0:
        return target_indices, target_distances, np.empty(0, dtype=np.int64), 0.0

    distances = cdist(src, tgt)
    finite_max = np.isfinite(max_distance)
    gate = float(max_distance) if finite_max else np.inf

    # ``linear_sum_assignment`` needs finite costs.  Invalid links receive a
    # large penalty so they will be ignored after assignment.
    if finite_max:
        large = gate * 1_000_000 + 1_000_000
        cost = np.where(distances <= gate, distances, large)
    else:
        large = np.nanmax(distances) * 1_000_000 + 1_000_000 if distances.size else 1e12
        cost = distances.copy()

    rows, cols = linear_sum_assignment(cost)

    accepted_cost = 0.0
    assigned_targets: set[int] = set()
    for r, c in zip(rows, cols, strict=False):
        d = float(distances[r, c])
        if d <= gate:
            target_indices[r] = int(c)
            target_distances[r] = d
            accepted_cost += d
            assigned_targets.add(int(c))

    unassigned_targets = np.array(
        [j for j in range(n_target) if j not in assigned_targets],
        dtype=np.int64,
    )

    return target_indices, target_distances, unassigned_targets, float(accepted_cost)


def nearestneighborlinker(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour linker used for gap closing in MATLAB.

    The original simpletracker uses nearest-neighbour linking when closing gaps.
    This implementation greedily accepts candidate pairs in increasing distance
    order while enforcing one-to-one assignments.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)

    n_source = src.shape[0]
    n_target = tgt.shape[0]

    target_indices = np.full(n_source, -1, dtype=np.int64)
    target_distances = np.full(n_source, np.nan, dtype=np.float64)

    if n_source == 0:
        return target_indices, target_distances, np.arange(n_target, dtype=np.int64)
    if n_target == 0:
        return target_indices, target_distances, np.empty(0, dtype=np.int64)

    distances = cdist(src, tgt)
    candidates = np.argwhere(distances <= float(max_distance))

    # Sort all valid source-target candidates by distance.
    if candidates.size:
        order = np.argsort(distances[candidates[:, 0], candidates[:, 1]])
        candidates = candidates[order]

    used_sources: set[int] = set()
    used_targets: set[int] = set()

    for r, c in candidates:
        rr = int(r)
        cc = int(c)
        if rr in used_sources or cc in used_targets:
            continue
        target_indices[rr] = cc
        target_distances[rr] = float(distances[rr, cc])
        used_sources.add(rr)
        used_targets.add(cc)

    unassigned_targets = np.array(
        [j for j in range(n_target) if j not in used_targets],
        dtype=np.int64,
    )

    return target_indices, target_distances, unassigned_targets


def _frame_offsets(n_cells: np.ndarray) -> np.ndarray:
    """Return starting global index of each frame."""
    offsets = np.zeros(n_cells.size, dtype=np.int64)
    if n_cells.size > 1:
        offsets[1:] = np.cumsum(n_cells[:-1])
    return offsets


def _rebuild_tracks_from_adjacency(adjacency: np.ndarray) -> list[np.ndarray]:
    """Rebuild directed path tracks from an adjacency matrix."""
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return []

    incoming = adjacency.any(axis=0)
    outgoing = adjacency.any(axis=1)

    # Track starts are nodes with no predecessor.  Isolated detections are also
    # starts and will become one-node tracks, matching simpletracker behaviour.
    starts = np.flatnonzero(~incoming)

    tracks: list[np.ndarray] = []
    visited: set[int] = set()

    for start in starts:
        current = int(start)
        chain = [current]
        visited.add(current)

        while outgoing[current]:
            next_nodes = np.flatnonzero(adjacency[current])
            if next_nodes.size == 0:
                break

            # simpletracker creates one outgoing link per source.  If a malformed
            # matrix has several, choose the first deterministic edge.
            nxt = int(next_nodes[0])
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt

        tracks.append(np.asarray(chain, dtype=np.int64))

    # Safety net for cycles or any nodes not reached from starts.
    for node in range(n_nodes):
        if node not in visited:
            tracks.append(np.asarray([node], dtype=np.int64))

    return tracks


def simpletracker(
    points: Sequence[np.ndarray],
    *,
    max_linking_distance: float = np.inf,
    max_gap_closing: int = 3,
    method: Literal["Hungarian", "NearestNeighbor"] = "Hungarian",
    debug: bool = False,
) -> SimpleTrackerResult:
    """Track frame-wise point detections with optional gap closing.

    Parameters mirror the MATLAB ``simpletracker`` key-value arguments, but use
    Pythonic snake_case names.
    """
    pts = _as_points_list(points)
    n_slices = len(pts)
    n_cells = np.array([frame.shape[0] for frame in pts], dtype=np.int64)
    offsets = _frame_offsets(n_cells)
    n_total = int(n_cells.sum())

    adjacency = np.zeros((n_total, n_total), dtype=bool)

    unmatched_targets: list[np.ndarray] = [np.arange(n_cells[i], dtype=np.int64) for i in range(n_slices)]
    unmatched_sources: list[np.ndarray] = [np.arange(n_cells[i], dtype=np.int64) for i in range(n_slices)]

    # ------------------------------------------------------------------
    # 1) Consecutive frame-to-frame linking.
    # ------------------------------------------------------------------
    for frame_idx in range(n_slices - 1):
        source = pts[frame_idx]
        target = pts[frame_idx + 1]

        if method.lower() == "nearestneighbor":
            target_indices, _, unassigned = nearestneighborlinker(
                source,
                target,
                max_linking_distance,
            )
        else:
            target_indices, _, unassigned, _ = hungarianlinker(
                source,
                target,
                max_linking_distance,
            )

        unmatched_targets[frame_idx + 1] = unassigned
        unmatched_sources[frame_idx] = np.flatnonzero(target_indices == -1).astype(np.int64)

        for local_source, local_target in enumerate(target_indices):
            if local_target == -1:
                continue
            row = offsets[frame_idx] + local_source
            col = offsets[frame_idx + 1] + int(local_target)
            adjacency[row, col] = True

        if debug:
            linked = int(np.sum(target_indices != -1))
            print(f"[simpletracker] frame {frame_idx + 1}->{frame_idx + 2}: {linked} links")

    # Last frame has no outgoing links, so every unlinked target there remains
    # available only as a possible gap-closing target.
    if n_slices:
        unmatched_sources[-1] = np.arange(n_cells[-1], dtype=np.int64)

    # ------------------------------------------------------------------
    # 2) Gap closing.  This mirrors the MATLAB loop: unmatched sources from
    # frame i are matched to unmatched targets in frames i+2 ... i+gap.
    # ------------------------------------------------------------------
    max_gap = int(max_gap_closing)
    for frame_idx in range(max(0, n_slices - 2)):
        src_unmatched = unmatched_sources[frame_idx].copy()
        if src_unmatched.size == 0:
            continue

        for target_frame in range(frame_idx + 2, min(frame_idx + max_gap, n_slices - 1) + 1):
            tgt_unmatched = unmatched_targets[target_frame].copy()
            if src_unmatched.size == 0 or tgt_unmatched.size == 0:
                continue

            source = pts[frame_idx][src_unmatched, :]
            target = pts[target_frame][tgt_unmatched, :]

            target_indices, _, _ = nearestneighborlinker(
                source,
                target,
                max_linking_distance,
            )

            linked_mask = target_indices != -1
            for k, target_local_pos in enumerate(target_indices):
                if target_local_pos == -1:
                    continue
                local_source = int(src_unmatched[k])
                local_target = int(tgt_unmatched[int(target_local_pos)])
                row = offsets[frame_idx] + local_source
                col = offsets[target_frame] + local_target
                adjacency[row, col] = True

                if debug:
                    print(
                        "[simpletracker] gap link "
                        f"frame {frame_idx + 1} source {local_source} -> "
                        f"frame {target_frame + 1} target {local_target}"
                    )

            # Remove sources and targets that have just been linked from the
            # available unmatched pools, as in MATLAB.
            linked_source_positions = np.flatnonzero(linked_mask)
            linked_target_positions = target_indices[linked_mask].astype(np.int64)

            if linked_source_positions.size:
                src_unmatched = np.delete(src_unmatched, linked_source_positions)

            if linked_target_positions.size:
                unmatched_targets[target_frame] = np.delete(
                    unmatched_targets[target_frame],
                    linked_target_positions,
                )

            if src_unmatched.size == 0:
                break

        unmatched_sources[frame_idx] = src_unmatched

    tracks = _rebuild_tracks_from_adjacency(adjacency)
    adjacency_tracks = [track.copy() for track in tracks]

    return SimpleTrackerResult(
        tracks=tracks,
        adjacency_tracks=adjacency_tracks,
        adjacency=adjacency,
        frame_offsets=offsets,
    )


def track_particles(
    xy: np.ndarray,
    frames: np.ndarray,
    max_step: float,
    max_missing_frames: int,
    *,
    method: Literal["Hungarian", "NearestNeighbor"] = "Hungarian",
    matlab_frame_indexing: bool = True,
) -> np.ndarray:
    """Assign NanoLocz track IDs to particle observations.

    Parameters
    ----------
    xy:
        ``(N, 2)`` x/y coordinates.
    frames:
        Length ``N`` frame number for each observation.
    max_step:
        Maximum linking distance between two detections.
    max_missing_frames:
        Maximum number of frames over which a gap may be closed.
    method:
        Consecutive-frame linking method.  MATLAB defaults to ``Hungarian``.
    matlab_frame_indexing:
        If true, frame labels are interpreted as MATLAB-style 1-based labels.
        If false, zero-based frame labels are accepted and internally shifted.

    Returns
    -------
    ndarray
        Length ``N`` array of 1-based track IDs.  Zero is used only for rows
        that cannot be mapped, which should not normally happen.
    """
    coords = np.asarray(xy, dtype=np.float64)
    fr = np.asarray(frames).astype(int).ravel()

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("xy must be an Nx2 or wider coordinate array")
    if coords.shape[0] != fr.size:
        raise ValueError("xy and frames must have the same number of rows")
    if fr.size == 0:
        return np.empty(0, dtype=np.int64)

    frame_labels = fr.copy()
    if not matlab_frame_indexing:
        frame_labels = frame_labels + 1

    max_frame = int(np.nanmax(frame_labels))
    if max_frame < 1:
        raise ValueError("frame labels must be positive or set matlab_frame_indexing=False")

    # Build MATLAB-style ``coods`` cell array and a parallel mapping back to the
    # original row indices.  This mapping is essential because simpletracker
    # returns indices in concatenated frame order.
    points: list[np.ndarray] = []
    original_indices_by_frame: list[np.ndarray] = []

    for frame_number in range(1, max_frame + 1):
        pos = np.flatnonzero(frame_labels == frame_number)
        points.append(coords[pos, 0:2])
        original_indices_by_frame.append(pos.astype(np.int64))

    result = simpletracker(
        points,
        max_linking_distance=float(max_step),
        max_gap_closing=int(max_missing_frames),
        method=method,
    )

    concatenated_original_indices = (
        np.concatenate(original_indices_by_frame)
        if original_indices_by_frame
        else np.empty(0, dtype=np.int64)
    )

    track_ids = np.zeros(coords.shape[0], dtype=np.int64)

    for track_number, track in enumerate(result.tracks, start=1):
        # Convert global simpletracker node IDs back to original input rows.
        valid_nodes = track[(track >= 0) & (track < concatenated_original_indices.size)]
        original_rows = concatenated_original_indices[valid_nodes]
        track_ids[original_rows] = int(track_number)

    return track_ids


__all__ = [
    "SimpleTrackerResult",
    "hungarianlinker",
    "nearestneighborlinker",
    "simpletracker",
    "track_particles",
]
