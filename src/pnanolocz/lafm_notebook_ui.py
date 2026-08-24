"""ipywidgets user interface for the LAFM Notebook workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.widgets import RectangleSelector

from pnanolocz.lafm_workflow import (
    COLORMAP_NAMES,
    DEFAULT_COLORMAP,
    LAFMWorkflow,
    resolve_lafm_colormap,
)
from pnanolocz.level_auto import ROUTINES

try:
    import ipywidgets as widgets
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The LAFM Notebook requires ipywidgets. Install it with "
        "`python -m pip install ipywidgets ipympl`."
    ) from exc


FILTER_NAMES = (
    "Gaussian",
    "-Gaussian",
    "Wiener",
    "Disk",
    "Sphere",
    "Laplacian",
    "Peak sharp",
    "High-pass",
)

LAFM_RENDER_COLORMAPS = COLORMAP_NAMES


class LAFMWorkbench:
    """Interactive Notebook controller around :class:`LAFMWorkflow`."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.image_dir = (
            self.project_root / "Software_testing_images" / "LAFM testing"
        )
        self.output_root = (
            self.project_root
            / "Software_testing_images"
            / "test_output"
            / "lafm_notebook"
        )
        self.image_paths = sorted(self.image_dir.glob("*.tiff"))
        if not self.image_paths:
            raise FileNotFoundError(f"No TIFF images found in {self.image_dir}")

        self.workflow: LAFMWorkflow | None = None
        self.roi_selector: RectangleSelector | None = None
        self._syncing_render_ranges = False
        self._build_widgets()
        self._build_figures()
        self._wire_events()
        self._set_stage("loaded")
        self._load_selected_file()

    def _build_widgets(self) -> None:
        auto = widgets.Layout(width="auto")
        self.file_dropdown = widgets.Dropdown(
            options=[(path.name, path) for path in self.image_paths],
            description="TIFF:",
            layout=widgets.Layout(width="520px"),
        )
        self.frame_slider = widgets.IntSlider(
            min=0,
            max=0,
            value=0,
            description="Frame:",
            continuous_update=False,
            layout=widgets.Layout(width="420px"),
        )
        self.colormap_dropdown = widgets.Dropdown(
            options=LAFM_RENDER_COLORMAPS,
            value=DEFAULT_COLORMAP,
            description="Colormap:",
            layout=auto,
        )
        self.use_leveling = widgets.Checkbox(
            value=False,
            description="Enable leveling",
        )
        self.level_routine = widgets.Dropdown(
            options=sorted(ROUTINES),
            value="plane-line",
            description="Routine:",
            layout=auto,
        )
        self.use_filtering = widgets.Checkbox(
            value=False,
            description="Enable filtering",
        )
        self.filter_name = widgets.Dropdown(
            options=FILTER_NAMES,
            value="Gaussian",
            description="Filter:",
            layout=auto,
        )
        self.filter_strength = widgets.FloatText(
            value=1.0,
            description="Strength:",
            layout=auto,
        )
        self.apply_preprocess = widgets.Button(
            description="Apply / reset preprocessing",
            button_style="info",
        )

        self.detection_method = widgets.ToggleButtons(
            options=("Peaks", "ROI"),
            value="ROI",
            description="Select Method:",
        )
        self.fast_find = widgets.Checkbox(value=True, description="Fast Find")
        self.exclude_edges = widgets.Checkbox(
            value=True,
            description="Exclude edges",
        )
        self.peak_size = widgets.IntText(value=5, description="Min separation:")
        self.initial_threshold = widgets.FloatText(
            value=0.1,
            description="Min height:",
        )
        self.image_filter_sigma = widgets.FloatText(
            value=1.0,
            description="Filter image σ:",
        )
        self.crop_radius = widgets.IntText(value=5, description="Crop radius:")
        self.correlation_sigma = widgets.FloatText(
            value=1.0,
            description="Filter xcor σ:",
        )
        self.correlation_threshold = widgets.FloatText(
            value=0.5,
            description="Correlation min:",
        )
        self.correlation_max = widgets.FloatText(
            value=1.0,
            description="Correlation max:",
        )
        self.tracking_max_step = widgets.FloatText(
            value=10.0,
            description="Max step:",
        )
        self.tracking_max_missing = widgets.IntText(
            value=3,
            description="Max missing:",
        )
        self.detect_initial_button = widgets.Button(
            description="1. Detect particles in every frame",
            button_style="primary",
        )
        self.reference_button = widgets.Button(
            description="2. Average detected particles",
        )
        self.detect_reference_button = widgets.Button(
            description="3. Re-detect particles with average reference",
        )

        self.translation_method = widgets.Dropdown(
            options=("Cross corr", "FFT cross"),
            value="Cross corr",
            description="Method:",
        )
        self.translation_iterations = widgets.ToggleButtons(
            options=(1, 2),
            value=2,
            description="Iterations:",
        )
        self.max_drift = widgets.FloatText(value=3.0, description="Max drift:")
        self.auto_reference = widgets.Checkbox(
            value=True,
            description="Auto update reference",
        )
        self.align_button = widgets.Button(
            description="Run translation alignment",
            button_style="warning",
        )
        self.recalculate_button = widgets.Button(
            description="Recalculate correlation",
        )

        self.localization_method = widgets.Dropdown(
            options=(
                "bicubic",
                "cvcubic",
                "bilinear",
                "lanczos3",
                "lanczos2",
                "gaussian",
                "sphere",
            ),
            value="cvcubic",
            description="Sub-pixel:",
        )
        self.pixperfeat = widgets.FloatText(value=1.0, description="Pix/feature:")
        self.peak_low_pass = widgets.FloatText(
            value=0.0, description="Low-pass Gaussian:"
        )
        self.peak_high_pass = widgets.FloatText(
            value=0.0, description="High-pass Off:"
        )
        self.peak_min_separation = widgets.IntText(
            value=1, description="Min separation:"
        )
        self.peak_height = widgets.FloatText(value=0.0, description="Height:")
        self.peak_prominence = widgets.FloatText(
            value=0.0, description="Prominence:"
        )
        self.preview_peaks = widgets.Checkbox(
            value=False, description="Preview peaks"
        )
        self.localization_view = widgets.ToggleButtons(
            options=("All particles (MATLAB)", "Current particle"),
            value="All particles (MATLAB)",
            description="Peak view:",
        )
        self.localization_scope = widgets.ToggleButtons(
            options=("Included peaks", "All peaks"),
            value="Included peaks",
            description="Show:",
        )
        self.lafm_z_min = widgets.FloatText(
            value=0.0,
            description="LAFM z min:",
        )
        self.lafm_z_max = widgets.FloatText(
            value=1.0,
            description="LAFM z max:",
        )
        self.apply_lafm_filter = widgets.Button(
            description="Apply LAFM z filter",
        )
        self.reset_lafm_filter = widgets.Button(
            description="Reset LAFM z filter",
        )
        self.lafm_filter_count = widgets.HTML(value="Included: 0 / 0")
        self.img_gus = widgets.FloatText(value=1.0, description="Render σ:")
        self.expand = widgets.FloatText(value=5.0, description="Pixel expansion:")
        self.delete_outliers = widgets.FloatText(
            value=4.0, description="Delete outliers:"
        )
        self.colorlimit_mode = widgets.Dropdown(
            options=("Max Min", "Exc outliers", "Manual"),
            value="Exc outliers",
            description="Z limits:",
        )
        self.z_min = widgets.FloatText(value=0.0, description="Z min:")
        self.z_max = widgets.FloatText(value=1.0, description="Z max:")
        self.lafm_render_z_range = widgets.FloatRangeSlider(
            value=(0.0, 1.0), min=0.0, max=1.0, step=0.01,
            description="LAFM z range:", continuous_update=False,
            readout_format=".3g", layout=widgets.Layout(width="520px"),
        )
        self.probability_render_z_range = widgets.FloatRangeSlider(
            value=(0.0, 1.0), min=0.0, max=1.0, step=0.01,
            description="Probability z range:", continuous_update=False,
            readout_format=".3g", layout=widgets.Layout(width="520px"),
        )
        self.localize_button = widgets.Button(
            description="1. Find all peaks",
            button_style="primary",
        )
        self.render_button = widgets.Button(
            description="2. Render LAFM",
            button_style="success",
        )
        self.save_button = widgets.Button(
            description="Save LAFM results",
            button_style="success",
        )

        self.status = widgets.HTML()
        self.log = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #bbb",
                max_height="180px",
                overflow_y="auto",
            )
        )

        preprocessing = widgets.VBox(
            [
                widgets.HTML(
                    "<b>1. Input and optional preprocessing</b> "
                    "(test images: both switches remain off)"
                ),
                self.file_dropdown,
                widgets.HBox([self.use_leveling, self.level_routine]),
                widgets.HBox(
                    [
                        self.use_filtering,
                        self.filter_name,
                        self.filter_strength,
                    ]
                ),
                self.apply_preprocess,
            ]
        )
        detection = widgets.VBox(
            [
                widgets.HTML(
                    "<b>2. ROI and two-stage particle detection</b><br>"
                    "Draw a tight ROI around one representative particle. This "
                    "ROI Reference is cross-correlated against every full movie "
                    "frame. Detected particle crops are averaged only when stage "
                    "2 is clicked; that average becomes the stage-3 reference."
                ),
                widgets.HBox(
                    [
                        self.detection_method,
                        self.fast_find,
                        self.exclude_edges,
                    ]
                ),
                widgets.HBox(
                    [
                        self.image_filter_sigma,
                        self.correlation_sigma,
                        self.correlation_threshold,
                        self.correlation_max,
                    ]
                ),
                widgets.HBox(
                    [
                        self.peak_size,
                        self.initial_threshold,
                        self.crop_radius,
                    ]
                ),
                widgets.HTML("<b>Delete detections / tracking settings</b>"),
                widgets.HBox(
                    [
                        self.tracking_max_step,
                        self.tracking_max_missing,
                    ]
                ),
                widgets.HBox(
                    [
                        self.detect_initial_button,
                        self.reference_button,
                        self.detect_reference_button,
                    ]
                ),
            ]
        )
        alignment = widgets.VBox(
            [
                widgets.HTML("<b>3. Iterative translation alignment</b>"),
                widgets.HBox(
                    [
                        self.translation_method,
                        self.translation_iterations,
                        self.max_drift,
                        self.auto_reference,
                    ]
                ),
                widgets.HBox([self.align_button, self.recalculate_button]),
            ]
        )
        localization = widgets.VBox(
            [
                widgets.HTML("<b>4. Localization, render and save</b>"),
                widgets.HBox(
                    [
                        self.localization_method,
                        self.pixperfeat,
                        self.peak_min_separation,
                    ]
                ),
                widgets.HBox(
                    [
                        self.peak_low_pass,
                        self.peak_high_pass,
                        self.peak_height,
                        self.peak_prominence,
                        self.preview_peaks,
                    ]
                ),
                widgets.HBox([self.localization_view, self.localization_scope]),
                widgets.HTML("<b>Step3 LAFM post-processing / background</b>"),
                widgets.HBox(
                    [
                        self.lafm_z_min,
                        self.lafm_z_max,
                        self.apply_lafm_filter,
                        self.reset_lafm_filter,
                        self.lafm_filter_count,
                    ]
                ),
                widgets.HTML("<b>LAFM render settings</b>"),
                widgets.HBox(
                    [
                        self.img_gus,
                        self.expand,
                        self.delete_outliers,
                    ]
                ),
                widgets.HBox(
                    [
                        self.colormap_dropdown,
                        self.colorlimit_mode,
                        self.z_min,
                        self.z_max,
                    ]
                ),
                self.lafm_render_z_range,
                self.probability_render_z_range,
                widgets.HBox(
                    [self.localize_button, self.render_button, self.save_button]
                ),
            ]
        )
        self.controls = widgets.Accordion(
            children=[preprocessing, detection, alignment, localization],
            titles=(
                "1 — Load / preprocess",
                "2 — ROI / detect / reference",
                "3 — Align / recalculate",
                "4 — Localize / render / save",
            ),
            selected_index=0,
        )

    def _build_figures(self) -> None:
        self.input_output = widgets.Output()
        self.roi_output = widgets.Output()
        self.detection_output = widgets.Output()
        self.localization_output = widgets.Output()
        self.render_output = widgets.Output()

        with self.input_output:
            self.input_fig, self.input_ax = plt.subplots(figsize=(12, 7))
            if hasattr(self.input_fig.canvas, "header_visible"):
                self.input_fig.canvas.header_visible = False
            self.input_artist = self.input_ax.imshow(
                np.zeros((2, 2)),
                cmap="afmhot",
                origin="upper",
                interpolation="nearest",
                aspect="equal",
            )
            self.input_ax.set_title("Input — Matplotlib afmhot")
            plt.show()

        with self.roi_output:
            self.roi_fig, self.roi_ax = plt.subplots(figsize=(12, 7))
            if hasattr(self.roi_fig.canvas, "header_visible"):
                self.roi_fig.canvas.header_visible = False
            self.roi_artist = self.roi_ax.imshow(
                np.zeros((2, 2)),
                cmap="afmhot",
                origin="upper",
                interpolation="nearest",
                aspect="equal",
            )
            self.roi_ax.set_title("ROI preview — Matplotlib afmhot")
            plt.show()

        with self.detection_output:
            self.detect_fig, (
                self.particles_ax,
                self.detected_particle_ax,
                self.reference_ax,
            ) = plt.subplots(
                1,
                3,
                figsize=(11, 3.5),
            )
            if hasattr(self.detect_fig.canvas, "header_visible"):
                self.detect_fig.canvas.header_visible = False
            self.particle_artist = self.particles_ax.imshow(
                np.zeros((2, 2)),
                cmap="afmhot",
                origin="upper",
                interpolation="nearest",
            )
            self.particle_scatter: Any = None
            self.detected_particle_artist = self.detected_particle_ax.imshow(
                np.zeros((2, 2)),
                cmap="afmhot",
                origin="upper",
                interpolation="nearest",
            )
            self.reference_artist = self.reference_ax.imshow(
                np.zeros((2, 2)),
                cmap="afmhot",
                origin="upper",
                interpolation="nearest",
            )
            self.particles_ax.set_title("Detected particles")
            self.detected_particle_ax.set_title("Detected particle — current frame")
            self.reference_ax.set_title("Average reference")
            plt.show()

        with self.localization_output:
            self.localization_fig, (
                self.localization_input_ax,
                self.localization_peak_ax,
            ) = plt.subplots(1, 2, figsize=(9, 4))
            if hasattr(self.localization_fig.canvas, "header_visible"):
                self.localization_fig.canvas.header_visible = False
            self.localization_input_artist = self.localization_input_ax.imshow(
                np.zeros((2, 2)), cmap="afmhot", origin="upper"
            )
            self.localization_peak_artist = self.localization_peak_ax.imshow(
                np.zeros((2, 2)), cmap="afmhot", origin="upper"
            )
            self.localization_scatter: Any = None
            self.localization_input_ax.set_title("Original frame")
            self.localization_peak_ax.set_title("Localized peaks")
            plt.show()

        with self.render_output:
            self.render_fig, (self.rgb_ax, self.probability_ax) = plt.subplots(
                1,
                2,
                figsize=(7, 3.5),
            )
            if hasattr(self.render_fig.canvas, "header_visible"):
                self.render_fig.canvas.header_visible = False
            self.rgb_artist = self.rgb_ax.imshow(np.zeros((2, 2, 3)), origin="upper")
            self.probability_artist = self.probability_ax.imshow(
                np.zeros((2, 2)),
                cmap="gray",
                origin="upper",
            )
            self.rgb_ax.set_title("LAFM colour render")
            self.probability_ax.set_title("Probability")
            placeholder = ScalarMappable(
                norm=Normalize(0.0, 1.0), cmap=self._mpl_colormap()
            )
            self.lafm_z_colorbar = self.render_fig.colorbar(
                placeholder, ax=self.rgb_ax, label="Height (z)"
            )
            self.probability_density_colorbar = self.render_fig.colorbar(
                self.probability_artist,
                ax=self.probability_ax,
                label="Localization density",
            )
            plt.show()

        self.figures = widgets.Tab(
            children=[
                self.input_output,
                self.roi_output,
                self.detection_output,
                self.localization_output,
                self.render_output,
            ],
            titles=(
                "Input (afmhot)",
                "ROI (afmhot)",
                "Detections / average",
                "Find peaks",
                "LAFM render",
            ),
        )
        for figure in (
            self.input_fig,
            self.roi_fig,
            self.detect_fig,
            self.localization_fig,
            self.render_fig,
        ):
            canvas = figure.canvas
            if hasattr(canvas, "layout"):
                canvas.layout.width = "auto"
                canvas.layout.height = "auto"
                canvas.layout.margin = "0 auto"
            if hasattr(canvas, "resizable"):
                canvas.resizable = False

    def _wire_events(self) -> None:
        self.file_dropdown.observe(self._load_selected_file, names="value")
        self.frame_slider.observe(self._show_current_frame, names="value")
        self.localization_view.observe(
            self._update_localization_frame,
            names="value",
        )
        self.localization_scope.observe(
            self._update_localization_frame,
            names="value",
        )
        self.colormap_dropdown.observe(self._update_display_colormap, names="value")
        self.lafm_render_z_range.observe(
            self._render_controls_changed, names="value"
        )
        self.probability_render_z_range.observe(
            self._render_controls_changed, names="value"
        )
        self.detection_method.observe(
            self._update_detection_method_controls,
            names="value",
        )
        self.apply_preprocess.on_click(self._run_preprocess)
        self.detect_initial_button.on_click(self._run_initial_detection)
        self.reference_button.on_click(self._run_reference)
        self.detect_reference_button.on_click(self._run_reference_detection)
        self.align_button.on_click(self._run_alignment)
        self.recalculate_button.on_click(self._run_recalculation)
        self.localize_button.on_click(self._run_localization)
        self.apply_lafm_filter.on_click(self._apply_lafm_z_filter)
        self.reset_lafm_filter.on_click(self._reset_lafm_z_filter)
        self.render_button.on_click(self._run_render)
        self.save_button.on_click(self._run_save)

        self.roi_selector = RectangleSelector(
            self.input_ax,
            self._select_roi,
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords="pixels",
            interactive=True,
        )
        self._update_detection_method_controls()

    def _update_detection_method_controls(self, *_: Any) -> None:
        roi_mode = self.detection_method.value == "ROI"
        for control in (
            self.correlation_sigma,
            self.correlation_threshold,
            self.correlation_max,
        ):
            control.disabled = not roi_mode
        self.peak_size.disabled = roi_mode
        self.initial_threshold.disabled = roi_mode

    def _record(self, message: str, *, error: bool = False) -> None:
        color = "#a00" if error else "#064"
        self.status.value = f"<span style='color:{color}'>{message}</span>"
        with self.log:
            print(message)

    def _set_stage(self, stage: str) -> None:
        """Enable actions only after their prerequisites are available."""
        current = {
            "loaded": 0,
            "roi": 1,
            "initial": 2,
            "reference": 3,
            "redetected": 4,
            "aligned": 5,
            "recalculated": 6,
            "localized": 7,
            "rendered": 8,
        }[stage]
        self.detect_initial_button.disabled = current < 1
        self.reference_button.disabled = current < 2
        self.detect_reference_button.disabled = current < 3
        self.align_button.disabled = current < 4
        self.recalculate_button.disabled = current < 5
        self.localize_button.disabled = current < 4
        self.render_button.disabled = current < 7
        self.save_button.disabled = current < 8

    def _resize_input_figure(
        self,
        image_shape: tuple[int, int],
        roi_shape: tuple[int, int] | None = None,
    ) -> None:
        """Fit complete images inside a bounded box without distortion."""
        shapes = (image_shape, roi_shape or image_shape)
        for figure, axis, (rows, cols) in zip(
            (self.input_fig, self.roi_fig),
            (self.input_ax, self.roi_ax),
            shapes,
        ):
            scale = min(9.0 / max(cols, 1), 5.5 / max(rows, 1))
            width = cols * scale
            height = rows * scale
            figure.set_size_inches(width, height, forward=True)
            axis.set_aspect("equal", adjustable="box")
            figure.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.92)

    def _guard(self, action: Any) -> None:
        try:
            action()
        except Exception as exc:
            self._record(f"{type(exc).__name__}: {exc}", error=True)

    def _load_selected_file(self, *_: Any) -> None:
        def action() -> None:
            self.workflow = LAFMWorkflow.from_tiff(self.file_dropdown.value)
            self.workflow.preprocess()
            self._set_stage("loaded")
            self.frame_slider.max = self.workflow.raw_movie.shape[0] - 1
            self.frame_slider.value = min(
                self.frame_slider.value,
                self.frame_slider.max,
            )
            self._show_current_frame()
            self._record(
                f"Loaded {self.file_dropdown.value.name}: "
                f"{self.workflow.raw_movie.shape}. Preprocessing is off."
            )

        self._guard(action)

    def _current_frame(self) -> np.ndarray:
        if self.workflow is None or self.workflow.processed_movie is None:
            raise RuntimeError("Load an image first")
        return self.workflow.processed_movie[self.frame_slider.value]

    def _mpl_colormap(self) -> ListedColormap:
        return ListedColormap(
            resolve_lafm_colormap(self.colormap_dropdown.value),
            name=self.colormap_dropdown.value,
        )

    @staticmethod
    def _set_image(
        artist: Any,
        ax: Any,
        image: np.ndarray,
        *,
        cmap: Any | None = None,
    ) -> None:
        artist.set_data(image)
        if cmap is not None and image.ndim == 2:
            artist.set_cmap(cmap)
        if image.ndim == 2 and np.isfinite(image).any():
            artist.set_clim(float(np.nanmin(image)), float(np.nanmax(image)))
        rows, cols = image.shape[:2]
        artist.set_extent((-0.5, cols - 0.5, rows - 0.5, -0.5))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

    def _show_current_frame(self, *_: Any) -> None:
        if self.workflow is None:
            return
        frame = self._current_frame()
        self._set_image(
            self.input_artist,
            self.input_ax,
            frame,
            cmap="afmhot",
        )
        self.input_ax.set_title(
            f"{self.file_dropdown.value.name} — frame {self.frame_slider.value}"
        )
        roi_shape = None
        if self.workflow.roi_movie is not None:
            roi_shape = self.workflow.roi_movie.shape[1:]
        self._resize_input_figure(frame.shape, roi_shape)
        self.input_fig.canvas.draw_idle()
        self._update_detection_frame()
        self._update_localization_frame()

    def _update_detection_frame(self) -> None:
        if self.workflow is None or self.workflow.roi_movie is None:
            return
        locs = self.workflow.correlation_locs
        if locs is None:
            locs = self.workflow.initial_locs
        if locs is not None:
            self._overlay_particles(locs, "Detected particles", select_tab=False)
            try:
                crop = self.workflow.detected_particle_preview(
                    crop_radius=self.crop_radius.value,
                    frame_number=self.frame_slider.value + 1,
                )
                self._set_image(
                    self.detected_particle_artist,
                    self.detected_particle_ax,
                    crop,
                    cmap="afmhot",
                )
                self.detected_particle_ax.set_title(
                    f"Detected particle — frame {self.frame_slider.value}"
                )
            except RuntimeError:
                self.detected_particle_ax.set_title(
                    f"No detected particle — frame {self.frame_slider.value}"
                )
            self.detect_fig.canvas.draw_idle()

    def _update_localization_frame(self, *_: Any) -> None:
        if self.workflow is None or self.workflow.localized_locs is None:
            return
        if self.workflow.particle_movie is not None:
            particle_index = min(
                self.frame_slider.value,
                self.workflow.particle_movie.shape[0] - 1,
            )
            frame_image = self.workflow.particle_movie[particle_index]
        else:
            frame_image = self._current_frame()
        frame_number = self.frame_slider.value + 1
        all_locs = self.workflow.localized_locs
        if self.localization_scope.value == "Included peaks":
            locs = self.workflow.included_localizations
        else:
            locs = all_locs
        selected = locs[np.rint(locs[:, 4]).astype(int) == frame_number]
        self._set_image(
            self.localization_input_artist,
            self.localization_input_ax,
            frame_image,
            cmap="afmhot",
        )
        if self.localization_scatter is not None:
            self.localization_scatter.remove()
        self.localization_input_ax.set_title(f"Original — frame {self.frame_slider.value}")
        if self.localization_view.value == "All particles (MATLAB)":
            self.localization_peak_artist.set_visible(False)
            self.localization_peak_ax.set_facecolor("#262626")
            centered_x = locs[:, 0] - np.median(locs[:, 0])
            centered_y = locs[:, 1] - np.median(locs[:, 1])
            self.localization_scatter = self.localization_peak_ax.scatter(
                centered_x,
                centered_y,
                c=locs[:, 2],
                cmap="afmhot",
                marker="s",
                s=4,
                linewidths=0,
            )
            x_margin = max(0.5, float(np.ptp(centered_x)) * 0.05)
            y_margin = max(0.5, float(np.ptp(centered_y)) * 0.05)
            self.localization_peak_ax.set_xlim(
                float(np.min(centered_x)) - x_margin,
                float(np.max(centered_x)) + x_margin,
            )
            self.localization_peak_ax.set_ylim(
                float(np.max(centered_y)) + y_margin,
                float(np.min(centered_y)) - y_margin,
            )
            self.localization_peak_ax.set_aspect("equal", adjustable="box")
            self.localization_peak_ax.set_title(
                f"{len(locs)} {self.localization_scope.value.lower()} "
                "— centered overlay"
            )
        else:
            self.localization_peak_artist.set_visible(True)
            self.localization_peak_ax.set_facecolor("white")
            self._set_image(
                self.localization_peak_artist,
                self.localization_peak_ax,
                frame_image,
                cmap="afmhot",
            )
            self.localization_scatter = self.localization_peak_ax.scatter(
                selected[:, 0] - 1,
                selected[:, 1] - 1,
                s=40,
                facecolors="none",
                edgecolors="cyan",
            )
            self.localization_peak_ax.set_title(
                f"Localized peaks — frame {self.frame_slider.value}: {len(selected)}"
            )
        self.localization_fig.canvas.draw_idle()

    def _update_lafm_filter_count(self) -> None:
        if self.workflow is None or self.workflow.localized_locs is None:
            self.lafm_filter_count.value = "Included: 0 / 0"
            return
        included = len(self.workflow.included_localizations)
        total = len(self.workflow.localized_locs)
        self.lafm_filter_count.value = f"Included: {included} / {total}"

    def _apply_lafm_z_filter(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            included = self.workflow.filter_lafm_localizations(
                self.lafm_z_min.value,
                self.lafm_z_max.value,
            )
            self._update_lafm_filter_count()
            self._sync_render_z_ranges()
            self._update_localization_frame()
            self._set_stage("localized")
            self._record(
                f"LAFM z filter retained {len(included)} / "
                f"{len(self.workflow.localized_locs)} localizations."
            )

        self._guard(action)

    def _reset_lafm_z_filter(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            included = self.workflow.reset_lafm_filter()
            assert self.workflow.lafm_z_filter_limits is not None
            self.lafm_z_min.value, self.lafm_z_max.value = (
                self.workflow.lafm_z_filter_limits
            )
            self._update_lafm_filter_count()
            self._sync_render_z_ranges()
            self._update_localization_frame()
            self._set_stage("localized")
            self._record(f"LAFM z filter reset; all {len(included)} peaks included.")

        self._guard(action)

    def _update_display_colormap(self, *_: Any) -> None:
        self.probability_artist.set_cmap(self._mpl_colormap())
        if self.workflow is not None and self.workflow.rendered_rgb is not None:
            self._run_render(None)
        elif self.workflow is not None and self.workflow.localized_locs is not None:
            self._record("Colormap changed. Click render again before saving.")

    def _render_controls_changed(self, *_: Any) -> None:
        if (
            not self._syncing_render_ranges
            and self.workflow is not None
            and self.workflow.rendered_rgb is not None
        ):
            self._run_render(None)

    def _sync_render_z_ranges(self) -> None:
        if self.workflow is None or self.workflow.localized_locs is None:
            return
        z = self.workflow.included_localizations[:, 2]
        z = z[np.isfinite(z)]
        if z.size == 0:
            return
        lower, upper = float(np.min(z)), float(np.max(z))
        if lower == upper:
            upper = lower + 1.0
        step = max((upper - lower) / 200.0, np.finfo(float).eps)
        self._syncing_render_ranges = True
        try:
            for slider in (
                self.lafm_render_z_range,
                self.probability_render_z_range,
            ):
                slider.min = min(float(slider.min), lower)
                slider.max = max(float(slider.max), upper)
                slider.value = (lower, upper)
                slider.min = lower
                slider.max = upper
                slider.step = step
        finally:
            self._syncing_render_ranges = False

    def _run_preprocess(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            self.workflow.preprocess(
                use_leveling=self.use_leveling.value,
                level_routine=self.level_routine.value,
                use_filtering=self.use_filtering.value,
                filter_name=self.filter_name.value,
                filter_strength=self.filter_strength.value,
            )
            self._set_stage("loaded")
            self._show_current_frame()
            self._record(
                "Preprocessing applied. Select a new ROI; downstream state reset."
            )

        self._guard(action)

    def _select_roi(self, click: Any, release: Any) -> None:
        def action() -> None:
            if None in (click.xdata, click.ydata, release.xdata, release.ydata):
                return
            assert self.workflow is not None
            rows, cols = self._current_frame().shape
            x0 = max(0, int(np.floor(min(click.xdata, release.xdata))))
            x1 = min(cols, int(np.ceil(max(click.xdata, release.xdata))))
            y0 = max(0, int(np.floor(min(click.ydata, release.ydata))))
            y1 = min(rows, int(np.ceil(max(click.ydata, release.ydata))))
            preview = self.workflow.set_reference_roi(
                (x0, y0, x1, y1),
                frame_index=self.frame_slider.value,
            )
            self.crop_radius.value = max(1, min(preview.shape) // 2)
            self._set_image(
                self.roi_artist,
                self.roi_ax,
                preview,
                cmap="afmhot",
            )
            self.roi_ax.set_title(f"ROI x=[{x0},{x1}), y=[{y0},{y1})")
            self._resize_input_figure(self._current_frame().shape, preview.shape)
            self.roi_fig.canvas.draw_idle()
            self._set_stage("roi")
            self.controls.selected_index = 1
            self.figures.selected_index = 1
            self._record(
                f"ROI Reference selected from frame {self.frame_slider.value}: "
                f"{self.workflow.roi_bounds}, template {preview.shape}. "
                f"Detection will search all {self.workflow.roi_movie.shape[0]} frames."
            )

        self._guard(action)

    def _overlay_particles(
        self,
        locs: np.ndarray,
        title: str,
        *,
        select_tab: bool = True,
    ) -> None:
        assert self.workflow is not None and self.workflow.roi_movie is not None
        frame = self.frame_slider.value + 1
        selected = locs[np.rint(locs[:, 4]).astype(int) == frame]
        image = self.workflow.roi_movie[self.frame_slider.value]
        self._set_image(
            self.particle_artist,
            self.particles_ax,
            image,
            cmap="afmhot",
        )
        if self.particle_scatter is not None:
            self.particle_scatter.remove()
        self.particle_scatter = self.particles_ax.scatter(
            selected[:, 0] - 1,
            selected[:, 1] - 1,
            s=35,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1,
        )
        self.particles_ax.set_title(f"{title}: {len(locs)} total")
        self.detect_fig.canvas.draw_idle()
        if select_tab:
            self.figures.selected_index = 2

    def _run_initial_detection(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            locs = self.workflow.detect_initial(
                method=self.detection_method.value,
                peak_size=self.peak_size.value,
                image_filter_sigma=self.image_filter_sigma.value,
                correlation_filter_sigma=self.correlation_sigma.value,
                threshold=self.initial_threshold.value,
                fast_find=self.fast_find.value,
                exclude_edges=self.exclude_edges.value,
                correlation_min=self.correlation_threshold.value,
                correlation_max=self.correlation_max.value,
            )
            self.workflow.settings.update(
                {
                    "tracking_max_step": self.tracking_max_step.value,
                    "tracking_max_missing": self.tracking_max_missing.value,
                }
            )
            self._overlay_particles(locs, "Initial particles")
            particle = self.workflow.detected_particle_preview(
                crop_radius=self.crop_radius.value,
                frame_number=self.frame_slider.value + 1,
            )
            self._set_image(
                self.detected_particle_artist,
                self.detected_particle_ax,
                particle,
                cmap="afmhot",
            )
            self.detected_particle_ax.set_title(
                f"Detected particle — frame {self.frame_slider.value}"
            )
            self.detect_fig.canvas.draw_idle()
            self._set_stage("initial")
            frames, counts = np.unique(
                np.rint(locs[:, 4]).astype(int),
                return_counts=True,
            )
            self._record(
                f"Initial particle detection found {len(locs)} particles across "
                f"{len(frames)}/{self.workflow.roi_movie.shape[0]} frames; "
                f"per-frame range {counts.min()}-{counts.max()}."
            )

        self._guard(action)

    def _run_reference(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            reference = self.workflow.calculate_average_reference(
                crop_radius=self.crop_radius.value
            )
            self._set_image(
                self.reference_artist,
                self.reference_ax,
                reference,
                cmap="afmhot",
            )
            self.reference_ax.set_title(
                f"Average of {len(self.workflow.initial_locs)} detected particle crops"
            )
            self.detect_fig.canvas.draw_idle()
            self._set_stage("reference")
            self._record(
                f"Average particle reference calculated from "
                f"{len(self.workflow.initial_locs)} frame-specific crops: "
                f"{reference.shape}."
            )

        self._guard(action)

    def _run_reference_detection(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            locs = self.workflow.detect_with_reference(
                correlation_filter_sigma=self.correlation_sigma.value,
                threshold=self.correlation_threshold.value,
                image_filter_sigma=self.image_filter_sigma.value,
                fast_find=self.fast_find.value,
                exclude_edges=self.exclude_edges.value,
                correlation_max=self.correlation_max.value,
            )
            self._overlay_particles(locs, "Particles from average reference")
            self._set_stage("redetected")
            self.controls.selected_index = 2
            frames = np.unique(np.rint(locs[:, 4]).astype(int))
            self._record(
                f"Average-reference detection found {len(locs)} particles "
                f"across {len(frames)}/{self.workflow.roi_movie.shape[0]} frames."
            )

        self._guard(action)

    def _run_alignment(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            reference = self.workflow.align_translation(
                iterations=self.translation_iterations.value,
                method=self.translation_method.value,
                max_drift=self.max_drift.value,
                auto_update_reference=self.auto_reference.value,
            )
            self._set_image(
                self.reference_artist,
                self.reference_ax,
                reference,
                cmap="afmhot",
            )
            self.reference_ax.set_title("Aligned auto-updated reference")
            self.detect_fig.canvas.draw_idle()
            self._set_stage("aligned")
            self._record(
                f"Translation alignment complete: "
                f"{self.translation_iterations.value} iteration(s), "
                f"{self.translation_method.value}."
            )

        self._guard(action)

    def _run_recalculation(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            locs = self.workflow.recalculate_correlation(
                threshold=self.correlation_threshold.value
            )
            self._overlay_particles(locs, "Recalculated correlation")
            self._set_stage("recalculated")
            self.controls.selected_index = 3
            self._record(
                f"Correlation recalculated; inspect overlay for {len(locs)} particles."
            )

        self._guard(action)

    def _run_localization(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            localized = self.workflow.find_all_peaks(
                localization_method=self.localization_method.value,
                pixperfeat=self.pixperfeat.value,
                low_pass_sigma=self.peak_low_pass.value,
                high_pass_sigma=self.peak_high_pass.value,
                min_separation=self.peak_min_separation.value,
                height_threshold=self.peak_height.value,
                prominence_threshold=self.peak_prominence.value,
            )
            assert self.workflow.lafm_z_filter_limits is not None
            self.lafm_z_min.value, self.lafm_z_max.value = (
                self.workflow.lafm_z_filter_limits
            )
            self._update_lafm_filter_count()
            self._sync_render_z_ranges()
            self._update_localization_frame()
            self.figures.selected_index = 3
            self._set_stage("localized")
            self._record(
                f"Find all peaks completed: {len(localized)} sub-pixel "
                f"localizations. Review this count, then click Render LAFM."
            )

        self._guard(action)

    def _run_render(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            rgb, probability, z_limits = self.workflow.render_lafm(
                colormap_name=self.colormap_dropdown.value,
                img_gus=self.img_gus.value,
                expand=self.expand.value,
                delete_outliers=self.delete_outliers.value,
                colorlimits=(self.z_min.value, self.z_max.value),
                colorlimit_mode=self.colorlimit_mode.value,
                lafm_z_range=tuple(self.lafm_render_z_range.value),
                probability_z_range=tuple(
                    self.probability_render_z_range.value
                ),
            )
            rgb_scale = max(float(np.nanmax(rgb)), 1.0)
            rgb_display = np.clip(rgb / rgb_scale, 0.0, 1.0)
            self._set_image(self.rgb_artist, self.rgb_ax, rgb_display)
            self._set_image(
                self.probability_artist,
                self.probability_ax,
                probability,
                cmap=self._mpl_colormap(),
            )
            self.lafm_z_colorbar.remove()
            self.probability_density_colorbar.remove()
            z_mappable = ScalarMappable(
                norm=Normalize(float(z_limits[0]), float(z_limits[1])),
                cmap=self._mpl_colormap(),
            )
            self.lafm_z_colorbar = self.render_fig.colorbar(
                z_mappable, ax=self.rgb_ax, label="Height (z)"
            )
            self.probability_density_colorbar = self.render_fig.colorbar(
                self.probability_artist,
                ax=self.probability_ax,
                label="Localization density",
            )
            self.rgb_ax.set_title(
                "LAFM render; "
                + (
                    f"z={z_limits.tolist()}"
                    if len(self.workflow.rendered_lafm_locs)
                    else "selected z range is empty"
                )
            )
            self.probability_ax.set_title(
                "Probability render"
                if len(self.workflow.rendered_probability_locs)
                else "Probability: selected z range is empty"
            )
            self.render_fig.canvas.draw_idle()
            self._set_stage("rendered")
            self.figures.selected_index = 4
            self._record(
                f"Rendered {len(self.workflow.rendered_locs)} localized "
                f"particles with {self.colormap_dropdown.value}."
            )

        self._guard(action)

    def _run_save(self, _: Any) -> None:
        def action() -> None:
            assert self.workflow is not None
            paths = self.workflow.save_results(self.output_root)
            directory = next(iter(paths.values())).parent
            self._record(f"Saved {len(paths)} outputs to {directory}.")

        self._guard(action)

    def widget(self) -> widgets.Widget:
        """Return the complete dashboard widget."""
        return widgets.VBox(
            [
                widgets.HTML(
                    "<h2>LAFM Notebook Workbench</h2>"
                    "<p>Run stages from 1 to 4. Test images default to no "
                    "leveling and no filtering.</p>"
                ),
                self.controls,
                self.status,
                self.figures,
                widgets.HBox(
                    [
                        widgets.HTML("<b>Frame viewer:</b>"),
                        self.frame_slider,
                    ]
                ),
                widgets.HTML("<b>Run log</b>"),
                self.log,
            ]
        )


def launch_lafm_workbench(
    project_root: str | Path | None = None,
    *,
    display_ui: bool = True,
) -> LAFMWorkbench:
    """Create the workbench and optionally display it in the current Notebook."""
    root = Path.cwd() if project_root is None else Path(project_root)
    if root.name == "notebooks":
        root = root.parent
    workbench = LAFMWorkbench(root)
    if display_ui:
        display(workbench.widget())
    return workbench


__all__ = ["FILTER_NAMES", "LAFMWorkbench", "launch_lafm_workbench"]
