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

from rivelero.gui.application_state import VisualizationLayer
from rivelero.gui.raster_map import RasterMapWidget, project_viewpoints
from rivelero.gui.theme import COLORS, SPACING
from rivelero.observability.masks import ObservabilityState
from rivelero.observability.survey_field import SurveyObservabilityField
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


MODE_LABELS: dict[ObservabilityMapMode, str] = {
    ObservabilityMapMode.OBSERVABILITY_STATE: "Analysis state",
    ObservabilityMapMode.OBSERVABLE_SPACE: "Observable space",
    ObservabilityMapMode.EXPOSURE: "Exposure",
    ObservabilityMapMode.NORMALIZED_EXPOSURE: "Normalized exposure",
    ObservabilityMapMode.BLIND_SPOTS: "Blind spots",
    ObservabilityMapMode.INDIVIDUAL_VISIBILITY: "Selected unit visibility",
}

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

    mode_changed = Signal(str)

    def __init__(self, *, parent: QWidget | None = None) -> None:
        self._sof: SurveyObservabilityField | None = None
        # Derived once per SOF: used by the state layer and hover readout.
        self._state_cache: np.ndarray | None = None
        self._mode = ObservabilityMapMode.OBSERVABILITY_STATE

        self._terrain_path: Path | None = None
        self._terrain_data: np.ndarray | None = None

        self._viewpoints: list[Any] = []
        self._viewpoint_ids: list[str] = []
        self._viewpoint_xy = np.empty((0, 2), dtype=float)
        self._selected_viewpoint_id: str | None = None

        self._individual_mask: np.ndarray | None = None
        self._individual_description: str | None = None

        self._layer_image = None
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

        if sof is None:
            self._individual_mask = None
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

    def set_mode(self, mode: ObservabilityMapMode | str) -> None:
        """Switch the displayed layer."""

        mode = ObservabilityMapMode(mode)

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
        for mode, label in MODE_LABELS.items():
            self.display_combo.addItem(label, mode.value)
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
        self._apply_limits()
        self._update_status()
        self.canvas.draw_idle()

    def _draw_terrain(self) -> None:
        show = (
            self._sof is not None
            and self._terrain_data is not None
            and self.terrain_checkbox.isChecked()
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
            if self._mode == ObservabilityMapMode.EXPOSURE:
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

        title = MODE_LABELS[self._mode]
        if (
            self._mode == ObservabilityMapMode.INDIVIDUAL_VISIBILITY
            and self._individual_description
        ):
            title = f"{title} — {self._individual_description}"
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
