"""Survey Observability Field map for the Rivelero Observability workflow.

ObservabilityMapWidget extends RasterMapWidget with SOF-specific layers:

- categorical analysis state (outside domain / invalid / blind / observable);
- observable space and blind spots within analysable space;
- exposure count and globally normalized exposure;
- visibility of one selected sampling unit;
- a single scatter overlay of canonical Viewpoints with click selection.

The widget displays arrays already owned by the canonical
SurveyObservabilityField and VisibilityStore. It performs no visibility or
exposure calculation, and holds a single raster artist, one colourbar and one
Viewpoint collection that are updated in place when the layer changes, so the
map extent and colour meaning stay stable while navigating.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

try:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QWidget,
    )

from rivelero.analysis.coverage import coverage_class_raster
from rivelero.gui.application_state import VisualizationLayer
from rivelero.gui.raster_map import RasterMapWidget, project_viewpoints
from rivelero.gui.theme import COLORS, SPACING
from rivelero.observability.masks import ObservabilityState
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.visualization.analysis import (
    coverage_class_colormap,
    coverage_class_legend_handles,
    EXPOSURE_DIFFERENCE_CMAP,
    comparison_change_legend_handles,
    scenario_change_colormap,
    scenario_change_legend_handles,
    unit_contribution_colormap,
    unit_contribution_legend_handles,
)
from rivelero.visualization.maps import raster_extent
from rivelero.visualization.observability import (
    CONTEXT_COLOR,
    EXPOSURE_CMAP,
    OBSERVABILITY_STATE_COLORS,
    OBSERVABILITY_STATE_LABELS,
    observability_state_colormap,
    observability_state_legend_handles,
)


class ObservabilityMapMode(str, Enum):
    """Layers the map can display. Values match VisualizationLayer."""

    OBSERVABILITY_STATE = VisualizationLayer.OBSERVABILITY_STATE.value
    OBSERVABLE_SPACE = VisualizationLayer.OBSERVABLE_SPACE.value
    EXPOSURE = VisualizationLayer.EXPOSURE.value
    NORMALIZED_EXPOSURE = VisualizationLayer.NORMALIZED_EXPOSURE.value
    BLIND_SPOTS = VisualizationLayer.BLIND_SPOTS.value
    INDIVIDUAL_VISIBILITY = VisualizationLayer.EFFECTIVE_VISIBILITY.value
    COVERAGE_CLASS = VisualizationLayer.COVERAGE_CLASS.value
    UNIT_CONTRIBUTION = VisualizationLayer.UNIT_CONTRIBUTION.value
    SCENARIO_CHANGE = VisualizationLayer.SCENARIO_CHANGE.value
    SCENARIO_EXPOSURE = VisualizationLayer.SCENARIO_EXPOSURE.value
    COMPARISON_CHANGE = VisualizationLayer.COMPARISON_CHANGE.value
    EXPOSURE_DIFFERENCE = VisualizationLayer.EXPOSURE_DIFFERENCE.value


MODE_LABELS: dict[ObservabilityMapMode, str] = {
    ObservabilityMapMode.OBSERVABILITY_STATE: "Analysis state",
    ObservabilityMapMode.OBSERVABLE_SPACE: "Observable space",
    ObservabilityMapMode.EXPOSURE: "Exposure",
    ObservabilityMapMode.NORMALIZED_EXPOSURE: "Normalized exposure",
    ObservabilityMapMode.BLIND_SPOTS: "Blind spots",
    ObservabilityMapMode.INDIVIDUAL_VISIBILITY: "Selected unit visibility",
    ObservabilityMapMode.COVERAGE_CLASS: "Unique / repeated coverage",
    ObservabilityMapMode.UNIT_CONTRIBUTION: "Selected unit contribution",
    ObservabilityMapMode.SCENARIO_CHANGE: "Baseline → scenario change",
    ObservabilityMapMode.SCENARIO_EXPOSURE: "Scenario exposure",
    ObservabilityMapMode.COMPARISON_CHANGE: "Coverage change",
    ObservabilityMapMode.EXPOSURE_DIFFERENCE: "Exposure difference",
}

# Layers offered by default (the Observability page). Other pages pass their
# own subset, e.g. Analysis & Design adds the coverage-class layer.
OBSERVABILITY_PAGE_MODES = (
    ObservabilityMapMode.OBSERVABILITY_STATE,
    ObservabilityMapMode.OBSERVABLE_SPACE,
    ObservabilityMapMode.EXPOSURE,
    ObservabilityMapMode.NORMALIZED_EXPOSURE,
    ObservabilityMapMode.BLIND_SPOTS,
    ObservabilityMapMode.INDIVIDUAL_VISIBILITY,
)

CONTINUOUS_MODES = frozenset(
    {
        ObservabilityMapMode.EXPOSURE,
        ObservabilityMapMode.NORMALIZED_EXPOSURE,
    }
)

_NOT_ANALYSABLE_HANDLE_LABEL = "Outside domain / invalid (not shown)"

# Surveys can hold tens of thousands of Viewpoints; shrink markers for them.
_DENSE_SURVEY_THRESHOLD = 2_000


class ObservabilityMapWidget(RasterMapWidget):
    """Map of SOF products with a Viewpoint overlay."""

    viewpoint_selected = Signal(str)

    # Emitted once with (x, y) in the SOF CRS after start_point_pick().
    point_picked = Signal(float, float)

    mode_changed = Signal(str)

    def __init__(
        self,
        *,
        modes: Iterable[ObservabilityMapMode] = OBSERVABILITY_PAGE_MODES,
        labels: dict[ObservabilityMapMode, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self._modes = tuple(ObservabilityMapMode(mode) for mode in modes)
        # Per-page wording, e.g. "Baseline exposure" next to scenario layers.
        self._labels = {**MODE_LABELS, **(labels or {})}
        if not self._modes:
            raise ValueError("At least one map mode is required.")
        self._sof: SurveyObservabilityField | None = None
        # Derived once per SOF: used by the state layer and hover readout.
        self._state_cache: np.ndarray | None = None
        self._coverage_cache: np.ndarray | None = None
        self._mode = self._modes[0]

        self._terrain_path: Path | None = None
        self._terrain_data: np.ndarray | None = None

        self._viewpoints: list[Any] = []
        self._viewpoint_ids: list[str] = []
        self._viewpoint_xy = np.empty((0, 2), dtype=float)
        self._selected_viewpoint_id: str | None = None

        self._individual_mask: np.ndarray | None = None
        self._individual_description: str | None = None

        # Derived UnitContributionClass raster of the selected unit.
        self._unit_classes: np.ndarray | None = None
        self._unit_description: str | None = None

        # Derived scenario layers (never the baseline arrays themselves).
        self._scenario_exposure: np.ndarray | None = None
        self._scenario_classes: np.ndarray | None = None

        self._picking_point = False

        # Derived left -> right comparison layers.
        self._comparison_classes: np.ndarray | None = None
        self._exposure_difference: np.ndarray | None = None

        self._layer_image = None

        # Temporary design candidates: (id, x, y) in the SOF CRS.
        self._candidate_points: list[tuple[str, float, float]] = []
        self._candidate_artist = None
        self._terrain_image = None
        self._viewpoint_artist = None
        self._selected_artist = None
        self._selected_label = None
        self._legend = None
        self._message_artist = None

        # True once an SOF has been drawn; afterwards the axes limits (which
        # the toolbar pans and zooms) are the source of the current extent.
        self._has_drawn = False
        self._extent_reset_pending = False

        super().__init__(parent=parent)

        self._build_controls()
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def sof(self) -> SurveyObservabilityField | None:
        return self._sof

    @property
    def mode(self) -> ObservabilityMapMode:
        return self._mode

    @property
    def modes(self) -> tuple[ObservabilityMapMode, ...]:
        return self._modes

    @property
    def legend_labels(self) -> list[str]:
        """Labels of the current categorical legend (empty if none)."""
        if self._legend is None:
            return []
        return [text.get_text() for text in self._legend.get_texts()]

    @property
    def layer_array(self) -> np.ndarray | None:
        """Array currently displayed by the raster layer, if any."""
        if self._layer_image is None or not self._layer_image.get_visible():
            return None
        return self._layer_image.get_array()

    @property
    def viewpoint_count(self) -> int:
        return len(self._viewpoint_ids)

    def set_field(self, sof: SurveyObservabilityField | None) -> None:
        """Display ``sof``; ``None`` shows the empty state."""

        if sof is not None and not isinstance(sof, SurveyObservabilityField):
            raise TypeError("sof must be a SurveyObservabilityField or None.")

        if sof is self._sof:
            return

        previous = self._sof
        self._sof = sof
        self._state_cache = None
        self._coverage_cache = None

        if sof is None:
            self._individual_mask = None
            self._unit_classes = None
            self._scenario_exposure = None
            self._scenario_classes = None
            self._comparison_classes = None
            self._exposure_difference = None
        else:
            extent = raster_extent(
                shape=sof.exposure_count.shape,
                transform=sof.transform,
            )
            same_grid = (
                previous is not None
                and previous.transform == sof.transform
                and previous.exposure_count.shape == sof.exposure_count.shape
            )
            self.full_extent = extent
            # Rebuilding over the same grid keeps the user's zoom.
            if not same_grid or self.current_extent is None:
                self.current_extent = extent
                self._extent_reset_pending = True
            if (
                self._individual_mask is not None
                and self._individual_mask.shape != sof.exposure_count.shape
            ):
                self._individual_mask = None
            # Contribution/scenario/comparison layers are relative to one SOF.
            self._unit_classes = None
            self._scenario_exposure = None
            self._scenario_classes = None
            self._comparison_classes = None
            self._exposure_difference = None

        self._project_viewpoints()
        self.display_combo.setEnabled(sof is not None)
        self._render()

    def set_terrain(self, path: str | Path | None) -> None:
        """Use the elevation raster at ``path`` as a grey context underlay."""

        source = None if path is None else Path(path).expanduser().resolve()

        if source == self._terrain_path:
            return

        self._terrain_path = source
        self._terrain_data = None

        if source is not None and source.is_file():
            with rasterio.open(source) as dataset:
                self._terrain_data = dataset.read(1, masked=True)

        self._render()

    def set_viewpoints(self, viewpoints: Iterable[Any]) -> None:
        """Set the canonical Viewpoints drawn as one scatter collection."""
        self._viewpoints = list(viewpoints)
        self._project_viewpoints()
        self._draw_viewpoints()
        self.canvas.draw_idle()

    def set_selected_viewpoint(self, viewpoint_id: str | None) -> None:
        """Highlight one canonical Viewpoint."""
        viewpoint_id = None if viewpoint_id is None else str(viewpoint_id)
        if viewpoint_id == self._selected_viewpoint_id:
            return
        self._selected_viewpoint_id = viewpoint_id
        self.zoom_selected_button.setEnabled(viewpoint_id is not None)
        self._draw_selected()
        self.canvas.draw_idle()

    def set_individual_visibility(
        self,
        mask: np.ndarray | None,
        description: str | None = None,
    ) -> None:
        """Set the selected unit's effective visibility mask for display."""

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if self._sof is not None and mask.shape != self._sof.exposure_count.shape:
                raise ValueError("Visibility mask shape does not match the SOF.")

        self._individual_mask = mask
        self._individual_description = description

        if self._mode == ObservabilityMapMode.INDIVIDUAL_VISIBILITY:
            self._render()

    def set_unit_contribution(
        self,
        classes: np.ndarray | None,
        description: str | None = None,
    ) -> None:
        """Set the selected unit's UnitContributionClass raster."""

        if classes is not None:
            classes = np.asarray(classes, dtype=np.uint8)
            if (
                self._sof is not None
                and classes.shape != self._sof.exposure_count.shape
            ):
                raise ValueError("Contribution raster shape does not match the SOF.")

        self._unit_classes = classes
        self._unit_description = description

        if self._mode == ObservabilityMapMode.UNIT_CONTRIBUTION:
            self._render()

    def set_scenario_layers(
        self,
        exposure: np.ndarray | None,
        change_classes: np.ndarray | None,
    ) -> None:
        """Set the scenario exposure and ScenarioChangeClass rasters."""

        for array in (exposure, change_classes):
            if (
                array is not None
                and self._sof is not None
                and array.shape != self._sof.exposure_count.shape
            ):
                raise ValueError("Scenario raster shape does not match the SOF.")

        self._scenario_exposure = exposure
        self._scenario_classes = change_classes

        if self._mode in (
            ObservabilityMapMode.SCENARIO_CHANGE,
            ObservabilityMapMode.SCENARIO_EXPOSURE,
        ):
            self._render()

    def set_candidate_points(self, points) -> None:
        """Show temporary candidates as a separate marker collection."""
        self._candidate_points = [(str(i), float(x), float(y)) for i, x, y in points]
        self._draw_candidates()
        self.canvas.draw_idle()

    def _draw_candidates(self) -> None:
        show = self._sof is not None and bool(self._candidate_points)
        if not show:
            if self._candidate_artist is not None:
                self._candidate_artist.set_visible(False)
            return
        xy = np.array([[x, y] for _i, x, y in self._candidate_points])
        if self._candidate_artist is None:
            self._candidate_artist = self.axes.scatter(
                xy[:, 0],
                xy[:, 1],
                s=70,
                marker="D",
                facecolors="white",
                edgecolors=COLORS.text_primary,
                linewidths=1.6,
                zorder=6,
            )
        else:
            self._candidate_artist.set_offsets(xy)
        self._candidate_artist.set_visible(True)

    def set_comparison_layers(
        self,
        change_classes: np.ndarray | None,
        exposure_difference: np.ndarray | None,
    ) -> None:
        """Set left -> right change classes and signed exposure difference."""
        self._comparison_classes = change_classes
        self._exposure_difference = exposure_difference
        if self._mode in (
            ObservabilityMapMode.COMPARISON_CHANGE,
            ObservabilityMapMode.EXPOSURE_DIFFERENCE,
        ):
            self._render()

    @property
    def picking_point(self) -> bool:
        return self._picking_point

    def start_point_pick(self) -> None:
        """Emit point_picked for the next click on the map."""
        self._picking_point = True
        self._status_label.setText("Click the map to place the candidate.")

    def cancel_point_pick(self) -> None:
        self._picking_point = False
        self._update_status()

    def set_mode(self, mode: ObservabilityMapMode | str) -> None:
        """Switch the displayed layer."""

        mode = ObservabilityMapMode(mode)

        if mode not in self._modes:
            raise ValueError(f"Map mode {mode.value!r} is not offered here.")

        if mode == self._mode:
            return

        self._mode = mode

        index = self.display_combo.findData(mode.value)
        if index >= 0 and self.display_combo.currentIndex() != index:
            self.display_combo.blockSignals(True)
            self.display_combo.setCurrentIndex(index)
            self.display_combo.blockSignals(False)

        self._render()
        self.mode_changed.emit(mode.value)

    def zoom_to_selected(self) -> None:
        """Zoom around the selected Viewpoint."""

        if self._selected_viewpoint_id is None or self.full_extent is None:
            return

        xy = self._selected_xy()
        if xy is None:
            return

        left, right, bottom, top = self.full_extent
        half_width = max((right - left) * 0.1, 1.0)
        half_height = max((top - bottom) * 0.1, 1.0)
        x, y = xy
        self.set_view_extent(
            (x - half_width, x + half_width, y - half_height, y + half_height)
        )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        controls = QWidget(self)
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs)
        layout.setSpacing(SPACING.sm)

        layout.addWidget(QLabel("Display"))

        self.display_combo = QComboBox()
        for mode in self._modes:
            self.display_combo.addItem(self._labels[mode], mode.value)
        self.display_combo.setEnabled(False)
        layout.addWidget(self.display_combo)

        self.terrain_checkbox = QCheckBox("Terrain")
        self.terrain_checkbox.setChecked(True)
        self.terrain_checkbox.setToolTip(
            "Show terrain beneath cells that have no value in this layer."
        )
        layout.addWidget(self.terrain_checkbox)

        self.viewpoints_checkbox = QCheckBox("Viewpoints")
        self.viewpoints_checkbox.setChecked(True)
        layout.addWidget(self.viewpoints_checkbox)

        layout.addStretch(1)

        self.zoom_selected_button = QPushButton("Zoom to selected")
        self.zoom_selected_button.setEnabled(False)
        layout.addWidget(self.zoom_selected_button)

        self.reset_view_button = QPushButton("Full extent")
        layout.addWidget(self.reset_view_button)

        # Insert below the navigation toolbar, as WorldMapWidget does.
        self.layout().insertWidget(1, controls)

        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        self.terrain_checkbox.toggled.connect(lambda _checked: self._render())
        self.viewpoints_checkbox.toggled.connect(self._on_viewpoints_toggled)
        self.zoom_selected_button.clicked.connect(self.zoom_to_selected)
        self.reset_view_button.clicked.connect(self.reset_view)

    def _on_display_changed(self, _index: int) -> None:
        self.set_mode(self.display_combo.currentData())

    def _on_viewpoints_toggled(self, _checked: bool) -> None:
        self._draw_viewpoints()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # RasterMapWidget hooks
    # ------------------------------------------------------------------

    def _draw_raster(self) -> None:
        # Arrays are already in memory; only the view limits change.
        self._apply_limits()
        self.canvas.draw_idle()

    def redraw_overlays(self) -> None:
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self._has_drawn and not self._extent_reset_pending:
            # Preserve interactive pan/zoom across layer changes.
            left, right = self.axes.get_xlim()
            bottom, top = self.axes.get_ylim()
            self.current_extent = (left, right, bottom, top)
        self._extent_reset_pending = False
        self._has_drawn = self._sof is not None

        self._draw_terrain()
        self._draw_layer()
        self._draw_viewpoints()
        self._draw_candidates()
        self._apply_limits()
        self._update_status()
        self.canvas.draw_idle()

    def _draw_terrain(self) -> None:
        show = (
            self._sof is not None
            and self._terrain_data is not None
            and self.terrain_checkbox.isChecked()
            and self._mode != ObservabilityMapMode.EXPOSURE_DIFFERENCE
            and self._terrain_data.shape == self._sof.exposure_count.shape
        )

        if not show:
            if self._terrain_image is not None:
                self._terrain_image.set_visible(False)
            return

        extent = self.full_extent

        if self._terrain_image is None:
            self._terrain_image = self.axes.imshow(
                self._terrain_data,
                extent=extent,
                origin="upper",
                cmap="Greys_r",
                alpha=0.55,
                interpolation="nearest",
                zorder=0,
            )
        else:
            self._terrain_image.set_data(self._terrain_data)
            self._terrain_image.set_extent(extent)
            self._terrain_image.set_clim(
                float(self._terrain_data.min()),
                float(self._terrain_data.max()),
            )

        self._terrain_image.set_visible(True)

    def _draw_layer(self) -> None:
        if self._legend is not None:
            self._legend.remove()
            self._legend = None

        spec = self._layer_spec()

        if spec is None:
            if self._layer_image is not None:
                self._layer_image.set_visible(False)
            self.hide_colorbar()
            self._show_message(self._empty_message())
            self.axes.set_title("")
            return

        data, cmap, norm, colorbar_label, handles = spec
        self._show_message(None)

        hatched = self._mode == ObservabilityMapMode.EXPOSURE_DIFFERENCE
        self.axes.patch.set_hatch("////" if hatched else None)
        self.axes.patch.set_edgecolor(COLORS.border_strong if hatched else "none")

        if self._layer_image is None:
            self._layer_image = self.axes.imshow(
                data,
                extent=self.full_extent,
                origin="upper",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                zorder=1,
            )
        else:
            self._layer_image.set_data(data)
            self._layer_image.set_cmap(cmap)
            self._layer_image.set_norm(norm)
            self._layer_image.set_extent(self.full_extent)

        self._layer_image.set_visible(True)

        if colorbar_label is not None:
            self.show_colorbar(self._layer_image, colorbar_label)
            if self._mode in (
                ObservabilityMapMode.EXPOSURE,
                ObservabilityMapMode.SCENARIO_EXPOSURE,
                ObservabilityMapMode.EXPOSURE_DIFFERENCE,
            ):
                self._colorbar.locator = MaxNLocator(integer=True)
                self._colorbar.update_ticks()
        else:
            self.hide_colorbar()

        if handles:
            self._legend = self.axes.legend(
                handles=handles,
                loc="upper right",
                fontsize=8,
                framealpha=0.92,
            )
            self._legend.set_zorder(10)

        title = self._labels[self._mode]
        if (
            self._mode == ObservabilityMapMode.INDIVIDUAL_VISIBILITY
            and self._individual_description
        ):
            title = f"{title} — {self._individual_description}"
        elif (
            self._mode == ObservabilityMapMode.UNIT_CONTRIBUTION
            and self._unit_description
        ):
            title = f"{title} — {self._unit_description}"
        self.axes.set_title(title, fontsize=10, color=COLORS.text_primary)

        unit = _linear_unit(self._sof.crs)
        self.axes.set_xlabel(f"X ({unit})" if unit else "X")
        self.axes.set_ylabel(f"Y ({unit})" if unit else "Y")
        self.axes.ticklabel_format(style="plain", useOffset=False, axis="both")
        self.axes.set_aspect("equal", adjustable="box")

    def _layer_spec(self):
        """Return (data, cmap, norm, colorbar label, legend handles)."""

        sof = self._sof

        if sof is None:
            return None

        analysable = sof.analysable_mask
        not_analysable = Patch(
            facecolor="none",
            edgecolor=COLORS.border_strong,
            label=_NOT_ANALYSABLE_HANDLE_LABEL,
        )

        if self._mode == ObservabilityMapMode.OBSERVABILITY_STATE:
            cmap, norm = observability_state_colormap()
            return (
                self._state(),
                cmap,
                norm,
                None,
                observability_state_legend_handles(),
            )

        if self._mode == ObservabilityMapMode.COMPARISON_CHANGE:
            if self._comparison_classes is None:
                return None
            cmap, norm = scenario_change_colormap()
            return (
                self._comparison_classes,
                cmap,
                norm,
                None,
                comparison_change_legend_handles(),
            )

        if self._mode == ObservabilityMapMode.EXPOSURE_DIFFERENCE:
            if self._exposure_difference is None:
                return None
            data = np.ma.masked_where(
                ~analysable | np.ma.getmaskarray(self._exposure_difference),
                np.ma.getdata(self._exposure_difference).astype(np.float32),
            )
            # Symmetric about zero so gains and losses of equal size have
            # equal visual weight.
            limit = float(max(1, int(np.abs(data).max()) if data.count() else 1))
            return (
                data,
                EXPOSURE_DIFFERENCE_CMAP,
                Normalize(vmin=-limit, vmax=limit),
                "Exposure difference (right − left, sampling units)",
                # Non-analysable cells are hatched so they cannot be read as
                # the neutral "no difference" colour.
                [
                    Patch(
                        facecolor="white",
                        edgecolor=COLORS.border_strong,
                        hatch="////",
                        label="Outside domain / invalid (not compared)",
                    )
                ],
            )

        if self._mode == ObservabilityMapMode.SCENARIO_CHANGE:
            if self._scenario_classes is None:
                return None
            cmap, norm = scenario_change_colormap()
            return (
                self._scenario_classes,
                cmap,
                norm,
                None,
                scenario_change_legend_handles(),
            )

        if self._mode == ObservabilityMapMode.SCENARIO_EXPOSURE:
            if self._scenario_exposure is None:
                return None
            data = np.ma.masked_where(
                ~analysable,
                self._scenario_exposure.astype(np.float32),
            )
            # Shared scale with the baseline so colours compare directly.
            maximum = max(
                1,
                sof.maximum_exposure,
                int(self._scenario_exposure[analysable].max(initial=0)),
            )
            return (
                data,
                EXPOSURE_CMAP,
                Normalize(vmin=0.0, vmax=float(maximum)),
                "Scenario exposure (number of sampling units)",
                [],
            )

        if self._mode == ObservabilityMapMode.UNIT_CONTRIBUTION:
            if self._unit_classes is None:
                return None
            cmap, norm = unit_contribution_colormap()
            return (
                self._unit_classes,
                cmap,
                norm,
                None,
                unit_contribution_legend_handles(),
            )

        if self._mode == ObservabilityMapMode.COVERAGE_CLASS:
            cmap, norm = coverage_class_colormap()
            return (
                self._coverage_classes(),
                cmap,
                norm,
                None,
                coverage_class_legend_handles(),
            )

        if self._mode == ObservabilityMapMode.OBSERVABLE_SPACE:
            observable_color = OBSERVABILITY_STATE_COLORS[
                ObservabilityState.OBSERVABLE
            ]
            data = np.ma.masked_where(
                ~analysable,
                sof.observable_mask.astype(np.uint8),
            )
            cmap, norm = _binary_colormap(CONTEXT_COLOR, observable_color)
            return (
                data,
                cmap,
                norm,
                None,
                [
                    Patch(
                        facecolor=observable_color,
                        label="Observable (exposure ≥ 1)",
                    ),
                    Patch(
                        facecolor=CONTEXT_COLOR,
                        label="Not observable",
                    ),
                    not_analysable,
                ],
            )

        if self._mode == ObservabilityMapMode.BLIND_SPOTS:
            blind_color = OBSERVABILITY_STATE_COLORS[
                ObservabilityState.BLIND_SPOT
            ]
            data = np.ma.masked_where(
                ~analysable,
                sof.blindspot_mask.astype(np.uint8),
            )
            cmap, norm = _binary_colormap(CONTEXT_COLOR, blind_color)
            return (
                data,
                cmap,
                norm,
                None,
                [
                    Patch(
                        facecolor=blind_color,
                        label=OBSERVABILITY_STATE_LABELS[
                            ObservabilityState.BLIND_SPOT
                        ],
                    ),
                    Patch(facecolor=CONTEXT_COLOR, label="Observable"),
                    not_analysable,
                ],
            )

        if self._mode == ObservabilityMapMode.EXPOSURE:
            data = np.ma.masked_where(
                ~analysable,
                sof.exposure_count.astype(np.float32),
            )
            # The scale spans the whole field, so zooming never changes
            # what a colour means.
            norm = Normalize(vmin=0.0, vmax=float(max(1, sof.maximum_exposure)))
            return (
                data,
                EXPOSURE_CMAP,
                norm,
                "Exposure (number of sampling units)",
                [],
            )

        if self._mode == ObservabilityMapMode.NORMALIZED_EXPOSURE:
            data = np.ma.masked_where(~analysable, sof.normalized_exposure)
            return (
                data,
                EXPOSURE_CMAP,
                Normalize(vmin=0.0, vmax=1.0),
                "Normalized exposure (fraction of active sampling units)",
                [],
            )

        if self._mode == ObservabilityMapMode.INDIVIDUAL_VISIBILITY:
            if self._individual_mask is None:
                return None
            visible_color = OBSERVABILITY_STATE_COLORS[
                ObservabilityState.OBSERVABLE
            ]
            data = np.ma.masked_where(
                ~analysable,
                self._individual_mask.astype(np.uint8),
            )
            cmap, norm = _binary_colormap(CONTEXT_COLOR, visible_color)
            return (
                data,
                cmap,
                norm,
                None,
                [
                    Patch(facecolor=visible_color, label="Visible from unit"),
                    Patch(facecolor=CONTEXT_COLOR, label="Not visible from unit"),
                    not_analysable,
                ],
            )

        return None

    def _empty_message(self) -> str:
        if self._sof is None:
            return "No current observability field.\nBuild one to see results."
        if self._mode == ObservabilityMapMode.UNIT_CONTRIBUTION:
            return (
                "Analyse contributions, then select a sampling unit\n"
                "in the table to map its contribution."
            )
        return (
            "Select a Viewpoint on the map or table, then show or compute\n"
            "its visibility in the panel below."
        )

    def _show_message(self, text: str | None) -> None:
        if text is None:
            if self._message_artist is not None:
                self._message_artist.set_visible(False)
            return

        if self._message_artist is None:
            self._message_artist = self.axes.text(
                0.5,
                0.5,
                text,
                transform=self.axes.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color=COLORS.text_secondary,
                zorder=20,
            )
        else:
            self._message_artist.set_text(text)

        self._message_artist.set_visible(True)

    # ------------------------------------------------------------------
    # Viewpoint overlay
    # ------------------------------------------------------------------

    def _project_viewpoints(self) -> None:
        if self._sof is None or not self._viewpoints:
            self._viewpoint_ids = []
            self._viewpoint_xy = np.empty((0, 2), dtype=float)
            return

        ids, xs, ys = project_viewpoints(self._viewpoints, self._sof.crs)
        self._viewpoint_ids = ids
        self._viewpoint_xy = np.column_stack([xs, ys]) if ids else np.empty((0, 2))

    def _draw_viewpoints(self) -> None:
        show = (
            self._sof is not None
            and self.viewpoints_checkbox.isChecked()
            and len(self._viewpoint_ids) > 0
        )

        if not show:
            if self._viewpoint_artist is not None:
                self._viewpoint_artist.set_visible(False)
            self._draw_selected()
            return

        dense = len(self._viewpoint_ids) > _DENSE_SURVEY_THRESHOLD

        if self._viewpoint_artist is None:
            # One PathCollection for the whole survey, never one artist per
            # Viewpoint.
            self._viewpoint_artist = self.axes.scatter(
                self._viewpoint_xy[:, 0],
                self._viewpoint_xy[:, 1],
                s=8 if dense else 22,
                marker="o",
                facecolors=COLORS.text_primary,
                edgecolors="white",
                linewidths=0.4 if dense else 0.7,
                picker=True,
                pickradius=6,
                zorder=5,
            )
        else:
            self._viewpoint_artist.set_offsets(self._viewpoint_xy)
            self._viewpoint_artist.set_sizes([8 if dense else 22])

        self._viewpoint_artist.set_visible(True)
        self._draw_selected()

    def _selected_xy(self) -> tuple[float, float] | None:
        if self._selected_viewpoint_id is None:
            return None
        try:
            index = self._viewpoint_ids.index(self._selected_viewpoint_id)
        except ValueError:
            return None
        x, y = self._viewpoint_xy[index]
        return float(x), float(y)

    def _draw_selected(self) -> None:
        xy = self._selected_xy() if self._sof is not None else None

        if xy is None:
            for artist in (self._selected_artist, self._selected_label):
                if artist is not None:
                    artist.set_visible(False)
            return

        if self._selected_artist is None:
            self._selected_artist = self.axes.scatter(
                [xy[0]],
                [xy[1]],
                s=150,
                marker="o",
                facecolors="none",
                edgecolors=COLORS.text_primary,
                linewidths=2.0,
                zorder=6,
            )
            # Only the selected Viewpoint is labelled; labelling every point
            # would make large surveys unreadable.
            self._selected_label = self.axes.annotate(
                self._selected_viewpoint_id,
                xy,
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
                color=COLORS.text_primary,
                zorder=7,
            )
        else:
            self._selected_artist.set_offsets([xy])
            self._selected_label.xy = xy
            self._selected_label.set_text(self._selected_viewpoint_id)

        self._selected_artist.set_visible(True)
        self._selected_label.set_visible(True)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_pick(self, event) -> None:
        if (
            self._viewpoint_artist is None
            or event.artist is not self._viewpoint_artist
            or not len(event.ind)
        ):
            return

        index = int(event.ind[0])

        if not 0 <= index < len(self._viewpoint_ids):
            return

        viewpoint_id = self._viewpoint_ids[index]
        self.set_selected_viewpoint(viewpoint_id)
        self.viewpoint_selected.emit(viewpoint_id)

    def _on_press(self, event) -> None:
        if (
            not self._picking_point
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.button != 1
            or self._toolbar.mode
        ):
            return
        self._picking_point = False
        self._update_status()
        self.point_picked.emit(float(event.xdata), float(event.ydata))

    def _on_motion(self, event) -> None:
        if (
            self._sof is None
            or event.inaxes is not self.axes
            or event.xdata is None
        ):
            return

        column, row = ~self._sof.transform * (event.xdata, event.ydata)
        row, column = int(np.floor(row)), int(np.floor(column))
        rows, columns = self._sof.exposure_count.shape

        if not (0 <= row < rows and 0 <= column < columns):
            return

        state = ObservabilityState(int(self._state()[row, column]))
        text = (
            f"x {event.xdata:,.1f}  y {event.ydata:,.1f}  ·  "
            f"{OBSERVABILITY_STATE_LABELS[state]}"
        )

        if state in (ObservabilityState.BLIND_SPOT, ObservabilityState.OBSERVABLE):
            text += f"  ·  exposure {int(self._sof.exposure_count[row, column])}"

        self._status_label.setText(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coverage_classes(self) -> np.ndarray:
        if self._coverage_cache is None:
            self._coverage_cache = coverage_class_raster(
                self._sof.exposure_count,
                analysis_mask=self._sof.analysis_mask,
                valid_mask=self._sof.valid_mask,
            )
        return self._coverage_cache

    def _state(self) -> np.ndarray:
        if self._state_cache is None:
            self._state_cache = self._sof.observability_state
        return self._state_cache

    def _apply_limits(self) -> None:
        if self.current_extent is None:
            return
        left, right, bottom, top = self.current_extent
        self.axes.set_xlim(left, right)
        self.axes.set_ylim(bottom, top)

    def _update_status(self) -> None:
        if self._sof is None:
            self._status_label.setText("")
            return
        rows, columns = self._sof.exposure_count.shape
        self._status_label.setText(
            f"CRS: {self._sof.crs.to_string()} | Grid: {columns:,} × {rows:,}"
        )


def _binary_colormap(off_color: str, on_color: str):
    cmap = ListedColormap([off_color, on_color])
    return cmap, BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)


def _linear_unit(crs) -> str | None:
    try:
        if crs is None or not crs.is_projected:
            return None
        unit = str(crs.linear_units or "").strip()
    except AttributeError:
        return None
    if unit.lower() in {"metre", "meter", "metres", "meters", "m"}:
        return "m"
    return unit or None
