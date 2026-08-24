"""Reusable orchestration for the interactive LAFM Notebook workbench."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

from pnanolocz.afm_colormap import afm_colormap
from pnanolocz.align_iterate import Particles, align_iterate
from pnanolocz.detector import detector
from pnanolocz.construct_particle_stack import (
    ParticleSet,
    construct_particle_stack,
)
from pnanolocz.filter_movie import filter_movie
from pnanolocz.fast_peaks2d import fast_peaks2d
from pnanolocz.lafm_renderer import lafm_renderer
from pnanolocz.level_auto import apply_level_auto
from pnanolocz.localize import localize
from pnanolocz.rem_outliers import rem_outliers


COLORMAP_NAMES = (
    "LAFM color",
    "magma",
    "plasma",
    "inferno",
    "viridis",
    "gray",
    "Rainbow",
    "hot",
    "jet",
    "AFM brown",
    "AFM dark gold",
    "AFM gold",
    "fire",
)
DEFAULT_COLORMAP = "LAFM color"


def resolve_lafm_colormap(name: str) -> np.ndarray:
    """Resolve a requested built-in or NanoLocz LUT to normalized RGB."""
    if name not in COLORMAP_NAMES:
        raise ValueError(f"Unknown colormap {name!r}")
    return np.asarray(afm_colormap(name), dtype=np.float64)[:, :3]


def _frame_first(movie: np.ndarray) -> np.ndarray:
    arr = np.asarray(movie, dtype=np.float64)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("movie must be a 2-D image or frame-first 3-D stack")
    return arr


@dataclass
class LAFMWorkflow:
    """Stateful, testable LAFM analysis workflow used by the Notebook UI."""

    raw_movie: np.ndarray
    source_name: str = "lafm_input.tiff"
    processed_movie: np.ndarray | None = None
    roi_bounds: tuple[int, int, int, int] | None = None
    roi_movie: np.ndarray | None = None
    roi_reference: np.ndarray | None = None
    roi_reference_frame: int | None = None
    particle_movie: np.ndarray | None = None
    initial_locs: np.ndarray | None = None
    reference: np.ndarray | None = None
    correlation_locs: np.ndarray | None = None
    aligned_locs: np.ndarray | None = None
    localized_locs: np.ndarray | None = None
    localization_include: np.ndarray | None = None
    lafm_z_filter_limits: tuple[float, float] | None = None
    rendered_locs: np.ndarray | None = None
    rendered_lafm_locs: np.ndarray | None = None
    rendered_probability_locs: np.ndarray | None = None
    rendered_rgb: np.ndarray | None = None
    rendered_probability: np.ndarray | None = None
    z_limits: np.ndarray | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_array(
        cls,
        movie: np.ndarray,
        *,
        source_name: str = "lafm_input.tiff",
    ) -> "LAFMWorkflow":
        return cls(raw_movie=_frame_first(movie).copy(), source_name=source_name)

    @classmethod
    def from_tiff(cls, path: str | Path) -> "LAFMWorkflow":
        source = Path(path)
        return cls.from_array(tifffile.imread(source), source_name=source.name)

    def _invalidate_from_roi(self) -> None:
        self.initial_locs = None
        self.reference = None
        self.correlation_locs = None
        self.aligned_locs = None
        self.localized_locs = None
        self.localization_include = None
        self.lafm_z_filter_limits = None
        self.rendered_locs = None
        self.rendered_lafm_locs = None
        self.rendered_probability_locs = None
        self.rendered_rgb = None
        self.rendered_probability = None
        self.z_limits = None
        self.particle_movie = None

    def preprocess(
        self,
        *,
        use_leveling: bool = False,
        level_routine: str = "plane-line",
        use_filtering: bool = False,
        filter_name: str = "Gaussian",
        filter_strength: float = 1.0,
    ) -> np.ndarray:
        """Optionally level/filter the movie; both are disabled by default."""
        result = self.raw_movie.copy()
        if use_leveling:
            result = apply_level_auto(
                result,
                routine=level_routine,
                frame_axis=0,
            )
        if use_filtering:
            result = filter_movie(
                result,
                filter_name,
                filter_strength,
                frame_axis=0,
            )
        self.processed_movie = np.asarray(result, dtype=np.float64)
        self.settings.update(
            {
                "use_leveling": bool(use_leveling),
                "level_routine": level_routine,
                "use_filtering": bool(use_filtering),
                "filter_name": filter_name,
                "filter_strength": float(filter_strength),
            }
        )
        self.roi_bounds = None
        self.roi_movie = None
        self.roi_reference = None
        self.roi_reference_frame = None
        self._invalidate_from_roi()
        return self.processed_movie

    def set_roi(self, bounds: tuple[int, int, int, int]) -> np.ndarray:
        """Select ``(x0, y0, x1, y1)`` and crop the same ROI from all frames."""
        movie = self.processed_movie
        if movie is None:
            movie = self.preprocess()
        x0, y0, x1, y1 = (int(value) for value in bounds)
        rows, cols = movie.shape[1:]
        x0, x1 = max(0, x0), min(cols, x1)
        y0, y1 = max(0, y0), min(rows, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("ROI must be non-empty and inside the image")
        self.roi_bounds = (x0, y0, x1, y1)
        self.roi_movie = movie[:, y0:y1, x0:x1].copy()
        self.roi_reference = None
        self.roi_reference_frame = None
        self._invalidate_from_roi()
        return self.roi_movie

    def set_reference_roi(
        self,
        bounds: tuple[int, int, int, int],
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Select a representative particle template without cropping the movie."""
        movie = self.processed_movie
        if movie is None:
            movie = self.preprocess()
        frame = int(frame_index)
        if not 0 <= frame < movie.shape[0]:
            raise ValueError("frame_index is outside the movie")
        x0, y0, x1, y1 = (int(value) for value in bounds)
        rows, cols = movie.shape[1:]
        x0, x1 = max(0, x0), min(cols, x1)
        y0, y1 = max(0, y0), min(rows, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Reference ROI must be non-empty and inside the image")
        self.roi_bounds = (x0, y0, x1, y1)
        self.roi_reference_frame = frame
        self.roi_reference = movie[frame, y0:y1, x0:x1].copy()
        self.roi_movie = movie
        self._invalidate_from_roi()
        self.settings.update(
            {
                "reference_roi_bounds": self.roi_bounds,
                "reference_roi_frame": frame,
            }
        )
        return self.roi_reference

    def _require_roi(self) -> np.ndarray:
        if self.roi_movie is None:
            raise RuntimeError("Select an ROI before running this stage")
        return self.roi_movie

    def detect_initial(
        self,
        *,
        method: str = "ROI",
        peak_size: int = 5,
        image_filter_sigma: float = 1.0,
        threshold: float = 0.1,
        correlation_filter_sigma: float = 1.0,
        fast_find: bool = True,
        exclude_edges: bool = True,
        correlation_min: float = 0.5,
        correlation_max: float = 1.0,
    ) -> np.ndarray:
        movie = self._require_roi()
        method_lc = method.strip().lower()
        if method_lc == "roi":
            if self.roi_reference is None:
                raise RuntimeError("Select an ROI Reference before ROI detection")
            locs = detector(
                movie,
                method="ccr",
                ref=self.roi_reference,
                filt_img=float(image_filter_sigma),
                filt_ccr=float(correlation_filter_sigma),
                min_thresh=float(correlation_min),
                ex_edge=bool(exclude_edges),
                fastdetect=bool(fast_find),
                angles=[0.0],
                frame_axis=0,
            )
            detection_method = "ROI reference cross-correlation"
            locs = locs[
                (locs[:, 3] >= float(correlation_min))
                & (locs[:, 3] <= float(correlation_max))
            ]
        elif method_lc == "peaks":
            locs = detector(
                movie,
                method="Peak picker",
                ref=int(peak_size),
                filt_img=float(image_filter_sigma),
                filt_ccr=0.0,
                min_thresh=float(threshold),
                ex_edge=bool(exclude_edges),
                fastdetect=bool(fast_find),
                frame_axis=0,
            )
            detection_method = "Peak picker"
        else:
            raise ValueError("method must be 'ROI' or 'Peaks'")
        if locs.shape[0] == 0:
            raise RuntimeError(
                "Initial per-frame particle detection found no particles"
            )
        self.initial_locs = np.asarray(locs, dtype=np.float64)
        if method_lc == "roi" and self.roi_reference is not None:
            self.particle_movie = construct_particle_stack(
                movie,
                ParticleSet(
                    image=self.roi_reference[np.newaxis, :, :],
                    locs=self.initial_locs,
                ),
                quick=True,
                frame_axis=0,
                part_frame_axis=0,
                matlab_indexing=True,
            )
        self.settings.update(
            {
                "peak_size": int(peak_size),
                "image_filter_sigma": float(image_filter_sigma),
                "initial_threshold": float(threshold),
                "initial_detection_method": detection_method,
                "correlation_filter_sigma": float(correlation_filter_sigma),
                "fast_find": bool(fast_find),
                "exclude_edges": bool(exclude_edges),
                "correlation_min": float(correlation_min),
                "correlation_max": float(correlation_max),
            }
        )
        self.reference = None
        self.correlation_locs = None
        return self.initial_locs

    def detected_particle_preview(
        self,
        *,
        crop_radius: int = 5,
        frame_number: int | None = None,
    ) -> np.ndarray:
        """Return one detected particle crop without calculating an average."""
        locs = self.correlation_locs
        if locs is None:
            locs = self.initial_locs
        if locs is None:
            raise RuntimeError("Detect particles before requesting a preview")
        if frame_number is not None:
            locs = locs[np.rint(locs[:, 4]).astype(int) == int(frame_number)]
            if len(locs) == 0:
                raise RuntimeError(f"No detected particle in frame {frame_number}")
        crops, _ = self._particle_crops(locs, crop_radius)
        return crops[0].copy()

    def _particle_crops(
        self,
        locs: np.ndarray,
        crop_radius: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        movie = self._require_roi()
        radius = int(crop_radius)
        if radius < 1:
            raise ValueError("crop_radius must be positive")
        size = radius * 2 + 1
        crops: list[np.ndarray] = []
        kept: list[np.ndarray] = []
        for row in np.asarray(locs, dtype=np.float64):
            x = int(np.sign(row[0]) * np.floor(abs(row[0]) + 0.5)) - 1
            y = int(np.sign(row[1]) * np.floor(abs(row[1]) + 0.5)) - 1
            frame = int(np.sign(row[4]) * np.floor(abs(row[4]) + 0.5)) - 1
            if not (0 <= frame < movie.shape[0]):
                continue
            y0, y1 = y - radius, y + radius + 1
            x0, x1 = x - radius, x + radius + 1
            if y0 < 0 or x0 < 0 or y1 > movie.shape[1] or x1 > movie.shape[2]:
                continue
            crop = movie[frame, y0:y1, x0:x1]
            if crop.shape == (size, size):
                crops.append(crop)
                kept.append(row.copy())
        if not crops:
            raise RuntimeError("No detected particles are far enough from ROI edges")
        return np.asarray(crops, dtype=np.float64), np.asarray(kept, dtype=np.float64)

    def calculate_average_reference(self, *, crop_radius: int = 5) -> np.ndarray:
        if self.initial_locs is None:
            raise RuntimeError("Run initial detection before calculating a reference")
        if self.particle_movie is not None:
            crops = self.particle_movie
        else:
            crops, kept = self._particle_crops(self.initial_locs, crop_radius)
            self.initial_locs = kept
        self.reference = np.nanmean(crops, axis=0)
        self.settings["crop_radius"] = int(crop_radius)
        return np.asarray(self.reference, dtype=np.float64)

    def detect_with_reference(
        self,
        *,
        correlation_filter_sigma: float = 1.0,
        threshold: float = 0.5,
        image_filter_sigma: float = 1.0,
        fast_find: bool = True,
        exclude_edges: bool = True,
        correlation_max: float = 1.0,
    ) -> np.ndarray:
        movie = self._require_roi()
        if self.reference is None:
            raise RuntimeError("Calculate an average reference before redetection")
        locs = detector(
            movie,
            method="ccr",
            ref=self.reference,
            filt_img=float(image_filter_sigma),
            filt_ccr=float(correlation_filter_sigma),
            min_thresh=float(threshold),
            ex_edge=bool(exclude_edges),
            fastdetect=bool(fast_find),
            angles=[0.0],
            frame_axis=0,
        )
        locs = locs[locs[:, 3] <= float(correlation_max)]
        if locs.shape[0] == 0:
            raise RuntimeError("Reference detection found no particles")
        self.correlation_locs = np.asarray(locs, dtype=np.float64)
        self.particle_movie = construct_particle_stack(
            movie,
            ParticleSet(
                image=self.reference[np.newaxis, :, :],
                locs=self.correlation_locs,
            ),
            quick=True,
            frame_axis=0,
            part_frame_axis=0,
            matlab_indexing=True,
        )
        self.settings.update(
            {
                "correlation_filter_sigma": float(correlation_filter_sigma),
                "correlation_threshold": float(threshold),
            }
        )
        return self.correlation_locs

    def align_translation(
        self,
        *,
        iterations: int = 2,
        method: str = "Cross corr",
        max_drift: float = 3.0,
        auto_update_reference: bool = True,
    ) -> np.ndarray:
        if self.correlation_locs is None or self.reference is None:
            raise RuntimeError("Run reference detection before alignment")
        if self.particle_movie is None:
            crops, kept = self._particle_crops(
                self.correlation_locs,
                int(self.settings.get("crop_radius", 5)),
            )
        else:
            crops = self.particle_movie
            kept = self.correlation_locs.copy()
        locs_zero_frame = kept.copy()
        locs_zero_frame[:, 4] -= 1.0
        part = Particles(image=crops, locs=locs_zero_frame)
        aligned_part, aligned_reference = align_iterate(
            self._require_roi(),
            self.reference,
            part,
            tran_iterations=int(iterations),
            translat_method=method,
            maxdrift=float(max_drift),
            rot_iterations=0,
            rota_method="Rotation corr",
            maxang=0.0,
            thresh_min=0.0,
            autoupdateref=bool(auto_update_reference),
            frame_axis=0,
        )
        aligned_locs = np.asarray(aligned_part.locs, dtype=np.float64)
        aligned_locs[:, 4] += 1.0
        self.aligned_locs = aligned_locs
        self.particle_movie = np.asarray(aligned_part.image, dtype=np.float64)
        self.reference = np.asarray(aligned_reference, dtype=np.float64)
        self.settings.update(
            {
                "translation_iterations": int(iterations),
                "translation_method": method,
                "max_drift": float(max_drift),
                "auto_update_reference": bool(auto_update_reference),
            }
        )
        return self.reference

    def recalculate_correlation(self, *, threshold: float | None = None) -> np.ndarray:
        if threshold is None:
            threshold = float(self.settings.get("correlation_threshold", 0.3))
        return self.detect_with_reference(
            correlation_filter_sigma=float(
                self.settings.get("correlation_filter_sigma", 0.5)
            ),
            threshold=float(threshold),
        )

    def find_all_peaks(
        self,
        *,
        localization_method: str = "cvcubic",
        pixperfeat: float = 1.0,
        low_pass_sigma: float = 0.0,
        high_pass_sigma: float = 0.0,
        min_separation: int = 1,
        height_threshold: float = 0.0,
        prominence_threshold: float = 0.0,
    ) -> np.ndarray:
        """Detect every image peak, then run sub-pixel localization."""
        movie = (
            self.particle_movie
            if self.particle_movie is not None
            else self._require_roi()
        )
        filtered = np.asarray(movie, dtype=np.float64).copy()
        if float(low_pass_sigma) > 0:
            filtered = np.asarray(
                [
                    gaussian_filter(frame, sigma=float(low_pass_sigma))
                    for frame in filtered
                ]
            )
        if float(high_pass_sigma) > 0:
            filtered = np.asarray(
                [
                    frame
                    - gaussian_filter(frame, sigma=float(high_pass_sigma))
                    for frame in filtered
                ]
            )
        rows: list[np.ndarray] = []
        source_locs = self.aligned_locs
        if source_locs is None:
            source_locs = self.correlation_locs
        if source_locs is None:
            source_locs = self.initial_locs
        for frame_index, frame in enumerate(filtered):
            peaks = fast_peaks2d(
                frame,
                thresh=float(height_threshold),
                kernel_size=int(min_separation),
                min_prom=float(prominence_threshold),
                matlab_indexing=True,
            )
            if len(peaks):
                table = np.zeros((len(peaks), 12), dtype=np.float64)
                table[:, :4] = peaks[:, :4]
                table[:, 4] = frame_index + 1
                if source_locs is not None and frame_index < len(source_locs):
                    table[:, 5] = source_locs[frame_index, 4]
                    table[:, 6] = source_locs[frame_index, 4]
                    table[:, 7] = source_locs[frame_index, 3]
                rows.append(table)
        if not rows:
            raise RuntimeError("Find all peaks detected no peaks in the movie")
        locs = np.vstack(rows)
        localized = localize(
            movie,
            locs,
            localization_method,
            float(pixperfeat),
            frame_axis=0,
            matlab_indexing=True,
        )
        valid = localized[np.all(np.isfinite(localized[:, :5]), axis=1)]
        if valid.shape[0] == 0:
            raise RuntimeError("Sub-pixel localization retained no particles")
        valid = self.postprocess_lafm_z(valid, movie)
        self.localized_locs = valid
        self.localization_include = np.ones(len(valid), dtype=bool)
        self.lafm_z_filter_limits = (
            float(np.min(valid[:, 2])),
            float(np.max(valid[:, 2])),
        )
        self.rendered_locs = None
        self.rendered_lafm_locs = None
        self.rendered_probability_locs = None
        self.rendered_rgb = None
        self.rendered_probability = None
        self.z_limits = None
        self.settings.update(
            {
                "localization_method": localization_method,
                "pixperfeat": float(pixperfeat),
                "peak_low_pass_sigma": float(low_pass_sigma),
                "peak_high_pass_sigma": float(high_pass_sigma),
                "peak_min_separation": int(min_separation),
                "peak_height_threshold": float(height_threshold),
                "peak_prominence_threshold": float(prominence_threshold),
                "particle_stack_shape": [int(value) for value in movie.shape],
                "localization_count": int(valid.shape[0]),
            }
        )
        return valid

    def postprocess_lafm_z(
        self,
        locs: np.ndarray,
        particle_movie: np.ndarray,
    ) -> np.ndarray:
        """Run MATLAB Step3: replace z from the unfiltered particle stack."""
        table = np.asarray(locs, dtype=np.float64).copy()
        movie = _frame_first(particle_movie)
        xs_matlab = np.rint(table[:, 0]).astype(int)
        ys_matlab = np.rint(table[:, 1]).astype(int)
        frames_matlab = np.rint(table[:, 4]).astype(int)
        inside = (
            (xs_matlab > 0)
            & (xs_matlab < movie.shape[2])
            & (ys_matlab > 0)
            & (ys_matlab < movie.shape[1])
            & (frames_matlab > 0)
            & (frames_matlab <= movie.shape[0])
        )
        table = table[inside]
        xs = xs_matlab[inside] - 1
        ys = ys_matlab[inside] - 1
        frames = frames_matlab[inside] - 1
        table[:, 2] = movie[frames, ys, xs]
        return table

    @property
    def included_localizations(self) -> np.ndarray:
        """Return the MATLAB ``LAFM.Locs(LAFM.IncludeLocs,:)`` view."""
        if self.localized_locs is None:
            raise RuntimeError("Run Find all peaks before filtering LAFM")
        if (
            self.localization_include is None
            or len(self.localization_include) != len(self.localized_locs)
        ):
            self.localization_include = np.ones(len(self.localized_locs), dtype=bool)
        return self.localized_locs[self.localization_include]

    def filter_lafm_localizations(
        self,
        z_min: float,
        z_max: float,
    ) -> np.ndarray:
        """Apply the inclusive MATLAB LAFM z threshold mask."""
        if self.localized_locs is None:
            raise RuntimeError("Run Find all peaks before filtering LAFM")
        lower, upper = float(z_min), float(z_max)
        if lower > upper:
            raise ValueError("LAFM z minimum must not exceed maximum")
        z = self.localized_locs[:, 2]
        self.localization_include = (z >= lower) & (z <= upper)
        self.lafm_z_filter_limits = (lower, upper)
        self.rendered_locs = None
        self.rendered_lafm_locs = None
        self.rendered_probability_locs = None
        self.rendered_rgb = None
        self.rendered_probability = None
        self.z_limits = None
        return self.included_localizations

    def reset_lafm_filter(self) -> np.ndarray:
        """Reset IncludeLocs to the complete Step3 localization table."""
        if self.localized_locs is None:
            raise RuntimeError("Run Find all peaks before filtering LAFM")
        self.localization_include = np.ones(len(self.localized_locs), dtype=bool)
        self.lafm_z_filter_limits = (
            float(np.min(self.localized_locs[:, 2])),
            float(np.max(self.localized_locs[:, 2])),
        )
        self.rendered_locs = None
        self.rendered_lafm_locs = None
        self.rendered_probability_locs = None
        self.rendered_rgb = None
        self.rendered_probability = None
        self.z_limits = None
        return self.included_localizations

    def render_lafm(
        self,
        *,
        colormap_name: str = DEFAULT_COLORMAP,
        img_gus: float = 1.0,
        expand: float = 1.0,
        delete_outliers: float = 4.0,
        colorlimits: tuple[float, float] = (0.0, 1.0),
        colorlimit_mode: str = "Max Min",
        lafm_z_range: tuple[float, float] | None = None,
        probability_z_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render an existing localization table without re-localizing."""
        if self.localized_locs is None:
            raise RuntimeError("Run Find all peaks before rendering LAFM")
        cmap = resolve_lafm_colormap(colormap_name)
        valid = rem_outliers(self.included_localizations, float(delete_outliers))
        if len(valid) == 0:
            raise RuntimeError("Outlier removal retained no localizations")
        finite_z = valid[:, 2][np.isfinite(valid[:, 2])]
        if finite_z.size == 0:
            raise RuntimeError("No finite localization heights are available")
        full_range = (float(np.min(finite_z)), float(np.max(finite_z)))

        def select_z(
            table: np.ndarray,
            limits: tuple[float, float] | None,
            label: str,
        ) -> tuple[np.ndarray, tuple[float, float]]:
            lower, upper = full_range if limits is None else map(float, limits)
            if lower > upper:
                raise ValueError(f"{label} z minimum must not exceed maximum")
            selected = table[(table[:, 2] >= lower) & (table[:, 2] <= upper)]
            return selected, (lower, upper)

        lafm_valid, lafm_range = select_z(valid, lafm_z_range, "LAFM")
        probability_valid, probability_range = select_z(
            valid, probability_z_range, "Probability"
        )
        rgb_source = lafm_valid if len(lafm_valid) else valid
        rgb, z_limits = lafm_renderer(
            rgb_source,
            float(img_gus),
            float(expand),
            cmap,
            False,
            list(lafm_range) if lafm_z_range is not None else list(colorlimits),
            "Manual" if lafm_z_range is not None else colorlimit_mode,
        )
        if len(lafm_valid) == 0:
            rgb = np.zeros_like(rgb)
            z_limits = np.asarray(lafm_range, dtype=np.float64)
        probability_source = (
            probability_valid if len(probability_valid) else valid
        )
        probability, _ = lafm_renderer(
            probability_source,
            float(img_gus),
            float(expand),
            cmap,
            True,
            (
                list(probability_range)
                if probability_z_range is not None
                else list(colorlimits)
            ),
            "Manual" if probability_z_range is not None else colorlimit_mode,
        )
        if len(probability_valid) == 0:
            probability = np.zeros_like(probability)
        self.rendered_rgb = rgb
        self.rendered_probability = probability
        self.rendered_locs = valid.copy()
        self.rendered_lafm_locs = lafm_valid.copy()
        self.rendered_probability_locs = probability_valid.copy()
        self.z_limits = z_limits
        self.settings.update(
            {
                "colormap": colormap_name,
                "img_gus": float(img_gus),
                "expand": float(expand),
                "delete_outliers": float(delete_outliers),
                "colorlimits": [float(colorlimits[0]), float(colorlimits[1])],
                "colorlimit_mode": colorlimit_mode,
                "lafm_render_z_range": list(lafm_range),
                "probability_render_z_range": list(probability_range),
            }
        )
        return rgb, probability, z_limits

    def localize_and_render(
        self,
        *,
        localization_method: str = "cvcubic",
        pixperfeat: float = 1.0,
        colormap_name: str = DEFAULT_COLORMAP,
        img_gus: float = 1.0,
        expand: float = 1.0,
        delete_outliers: float = 4.0,
        colorlimits: tuple[float, float] = (0.0, 1.0),
        colorlimit_mode: str = "Max Min",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compatibility wrapper for the now-separated localization/render stages."""
        localized = self.find_all_peaks(
            localization_method=localization_method,
            pixperfeat=pixperfeat,
        )
        rgb, probability, z_limits = self.render_lafm(
            colormap_name=colormap_name,
            img_gus=img_gus,
            expand=expand,
            delete_outliers=delete_outliers,
            colorlimits=colorlimits,
            colorlimit_mode=colorlimit_mode,
        )
        return localized, rgb, probability, z_limits

    def _new_output_dir(self, output_root: str | Path) -> Path:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(self.source_name).stem)
        candidate = root / f"{stem}_lafm"
        suffix = 2
        while candidate.exists():
            candidate = root / f"{stem}_lafm_{suffix}"
            suffix += 1
        candidate.mkdir()
        return candidate

    def save_results(self, output_root: str | Path) -> dict[str, Path]:
        """Save all reproducibility artifacts without overwriting prior runs."""
        if (
            self.localized_locs is None
            or self.rendered_rgb is None
            or self.rendered_probability is None
            or self.rendered_locs is None
            or self.reference is None
            or self.roi_movie is None
        ):
            raise RuntimeError("Complete localization and rendering before saving")
        import matplotlib.pyplot as plt

        directory = self._new_output_dir(output_root)
        paths = {
            "localizations_csv": directory / "localizations.csv",
            "localizations_all_csv": directory / "localizations_all.csv",
            "localizations_rendered_csv": directory / "localizations_rendered.csv",
            "rgb_png": directory / "lafm_rgb.png",
            "rgb_tiff": directory / "lafm_rgb.tiff",
            "probability_png": directory / "lafm_probability.png",
            "probability_tiff": directory / "lafm_probability.tiff",
            "roi_movie_tiff": directory / "roi_movie.tiff",
            "reference_tiff": directory / "average_reference.tiff",
            "settings_json": directory / "settings.json",
        }
        np.savetxt(
            paths["localizations_csv"],
            self.localized_locs,
            delimiter=",",
            header=",".join(f"column_{idx + 1}" for idx in range(self.localized_locs.shape[1])),
            comments="",
        )
        np.savetxt(
            paths["localizations_all_csv"],
            self.localized_locs,
            delimiter=",",
            header=",".join(
                f"column_{idx + 1}" for idx in range(self.localized_locs.shape[1])
            ),
            comments="",
        )
        np.savetxt(
            paths["localizations_rendered_csv"],
            self.rendered_locs,
            delimiter=",",
            header=",".join(
                f"column_{idx + 1}" for idx in range(self.rendered_locs.shape[1])
            ),
            comments="",
        )
        rgb_scale = max(float(np.nanmax(self.rendered_rgb)), 1.0)
        probability_scale = max(float(np.nanmax(self.rendered_probability)), 1.0)
        plt.imsave(paths["rgb_png"], np.clip(self.rendered_rgb / rgb_scale, 0, 1))
        plt.imsave(
            paths["probability_png"],
            self.rendered_probability,
            cmap="gray",
            vmin=0,
            vmax=probability_scale,
        )
        tifffile.imwrite(
            paths["rgb_tiff"],
            self.rendered_rgb.astype(np.float32),
            photometric="rgb",
        )
        tifffile.imwrite(
            paths["probability_tiff"],
            self.rendered_probability.astype(np.float32),
            photometric="minisblack",
        )
        tifffile.imwrite(
            paths["roi_movie_tiff"],
            self.roi_movie.astype(np.float32),
            photometric="minisblack",
        )
        tifffile.imwrite(
            paths["reference_tiff"],
            self.reference.astype(np.float32),
            photometric="minisblack",
        )
        payload = {
            "source_name": self.source_name,
            "roi_bounds": self.roi_bounds,
            "z_limits": None if self.z_limits is None else self.z_limits.tolist(),
            "lafm_z_filter_limits": self.lafm_z_filter_limits,
            "localizations_all_count": int(len(self.localized_locs)),
            "localizations_included_count": int(len(self.included_localizations)),
            "localizations_rendered_count": int(len(self.rendered_locs)),
            **self.settings,
        }
        paths["settings_json"].write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return paths


__all__ = [
    "COLORMAP_NAMES",
    "DEFAULT_COLORMAP",
    "LAFMWorkflow",
    "resolve_lafm_colormap",
]
