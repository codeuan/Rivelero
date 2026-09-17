"""Interactive terrain/domain map for the Rivelero World workflow.

WorldMapWidget extends RasterMapWidget with Rivelero-specific spatial
semantics:

- canonical Viewpoint overlay;
- selected Viewpoint highlighting;
- AnalysisDomain overlay;
- user-drawn AnalysisDomain polygons;
- explicit domain-drawing lifecycle;
- survey/terrain CRS transformation for display only.

The widget does not construct or mutate AnalysisDomain itself. Drawn
vertices are emitted to WorldPage, which passes them through domain_service.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import PolygonSelector
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.crs import CRS

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QWidget,
    )

except ImportError:
    try:
        from PyQt6.QtCore import Qt, pyqtSignal as Signal
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QWidget,
        )

    except ImportError as exc:
        raise ImportError(
            "Rivelero GUI requires PySide6 or PyQt6."
        ) from exc


from rivelero.core.domain import AnalysisDomain
from rivelero.gui.raster_map import RasterMapWidget
from rivelero.gui.theme import SPACING


class WorldMapDisplayMode(str, Enum):
    """Primary semantic overlay displayed over terrain."""

    SURVEY = "survey"
    DOMAIN = "domain"
    SURVEY_AND_DOMAIN = "survey_and_domain"


class WorldMapWidget(RasterMapWidget):
    """Terrain map with Survey and AnalysisDomain overlays."""

    viewpoint_selected = Signal(
        str
    )

    domain_polygon_drawn = Signal(
        object
    )

    domain_drawing_started = Signal()

    domain_drawing_cancelled = Signal()

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
    ) -> None:

        self._viewpoints: list[
            Any
        ] = []

        self._resolved_viewpoints: list[
            tuple[
                str,
                float,
                float,
            ]
        ] = []

        self._selected_viewpoint_id: (
            str | None
        ) = None

        self._domain: (
            AnalysisDomain | None
        ) = None

        self._display_mode = (
            WorldMapDisplayMode
            .SURVEY_AND_DOMAIN
        )

        self._show_viewpoint_ids = False

        self._viewpoint_artist: (
            PathCollection | None
        ) = None

        self._selected_artist = None

        self._domain_artists: list[Any] = []

        self._id_artists = []

        self._polygon_selector: (
            PolygonSelector | None
        ) = None

        self._draft_polygon_artist = None

        super().__init__(
            parent=parent
        )

        self._build_world_controls()

        self.canvas.mpl_connect(
            "pick_event",
            self._on_pick
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_viewpoints(
        self,
        viewpoints: Iterable[Any],
    ) -> None:
        """Set canonical Viewpoints to display."""

        self._viewpoints = list(
            viewpoints
        )

        self._resolve_viewpoints()

        self.redraw_overlays()

    def set_domain(
        self,
        domain: AnalysisDomain | None,
    ) -> None:
        """Set canonical AnalysisDomain overlay."""

        if (
            domain is not None
            and not isinstance(
                domain,
                AnalysisDomain,
            )
        ):
            raise TypeError(
                "domain must be an AnalysisDomain or None."
            )

        if (
            domain is not None
            and self.metadata is not None
            and domain.crs != self.metadata.crs
        ):
            raise ValueError(
                "AnalysisDomain CRS must match the terrain CRS "
                "before it can be displayed."
            )

        self._domain = domain

        self.redraw_overlays()

    def set_selected_viewpoint(
        self,
        viewpoint_id: str | None,
    ) -> None:
        """Highlight one canonical Viewpoint."""

        self._selected_viewpoint_id = (
            None
            if viewpoint_id is None
            else str(
                viewpoint_id
            )
        )

        self.redraw_overlays()

    def set_show_viewpoint_ids(
        self,
        enabled: bool,
    ) -> None:

        self._show_viewpoint_ids = bool(
            enabled
        )

        self.redraw_overlays()

    def zoom_to_selected(
        self,
    ) -> None:
        """Zoom around selected Viewpoint."""

        if (
            self._selected_viewpoint_id
            is None
            or self.metadata is None
        ):
            return

        selected = next(
            (
                item
                for item
                in self._resolved_viewpoints
                if item[0]
                == self._selected_viewpoint_id
            ),
            None,
        )

        if selected is None:
            return

        _, x, y = selected

        if self.full_extent is None:
            return

        left, right, bottom, top = (
            self.full_extent
        )

        full_width = (
            right
            - left
        )

        full_height = (
            top
            - bottom
        )

        half_width = max(
            full_width * 0.05,
            self.metadata.resolution_x
            * 10.0,
        )

        half_height = max(
            full_height * 0.05,
            self.metadata.resolution_y
            * 10.0,
        )

        self.set_view_extent(
            (
                x - half_width,
                x + half_width,
                y - half_height,
                y + half_height,
            )
        )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_world_controls(
        self,
    ) -> None:

        controls = QWidget(
            self
        )

        layout = QHBoxLayout(
            controls
        )

        layout.setContentsMargins(
            SPACING.sm,
            SPACING.xs,
            SPACING.sm,
            SPACING.xs,
        )

        layout.setSpacing(
            SPACING.sm
        )

        layout.addWidget(
            QLabel(
                "Display"
            )
        )

        self.display_combo = QComboBox()

        self.display_combo.addItem(
            "Survey + domain",
            (
                WorldMapDisplayMode
                .SURVEY_AND_DOMAIN
                .value
            ),
        )

        self.display_combo.addItem(
            "Survey",
            WorldMapDisplayMode.SURVEY.value,
        )

        self.display_combo.addItem(
            "Analysis area",
            WorldMapDisplayMode.DOMAIN.value,
        )

        layout.addWidget(
            self.display_combo
        )

        self.ids_checkbox = QCheckBox(
            "Viewpoint IDs"
        )

        layout.addWidget(
            self.ids_checkbox
        )

        layout.addStretch(
            1
        )

        self.zoom_selected_button = QPushButton(
            "Zoom to selected"
        )

        self.zoom_selected_button.setEnabled(
            False
        )

        self.draw_domain_button = QPushButton(
            "Draw analysis area"
        )

        self.draw_domain_button.setCheckable(
            True
        )

        self.cancel_draw_button = QPushButton(
            "Cancel drawing"
        )

        self.cancel_draw_button.setVisible(
            False
        )

        layout.addWidget(
            self.zoom_selected_button
        )

        layout.addWidget(
            self.draw_domain_button
        )

        layout.addWidget(
            self.cancel_draw_button
        )

        # RasterMapWidget root layout already contains toolbar/canvas/status.
        # Insert semantic controls below the navigation toolbar.
        self.layout().insertWidget(
            1,
            controls,
        )

        self.display_combo.currentIndexChanged.connect(
            self._on_display_changed
        )

        self.ids_checkbox.toggled.connect(
            self.set_show_viewpoint_ids
        )

        self.zoom_selected_button.clicked.connect(
            self.zoom_to_selected
        )

        self.draw_domain_button.toggled.connect(
            self._on_draw_toggled
        )

        self.cancel_draw_button.clicked.connect(
            self.cancel_domain_drawing
        )

    # ------------------------------------------------------------------
    # Raster hook
    # ------------------------------------------------------------------

    def set_raster(
        self,
        path,
    ) -> None:

        super().set_raster(
            path
        )

        self._resolve_viewpoints()

        if (
            self._domain is not None
            and self._domain.crs
            != self.metadata.crs
        ):
            # Never silently transform a canonical AnalysisDomain.
            self._domain = None

        self.redraw_overlays()

    # ------------------------------------------------------------------
    # Viewpoint resolution
    # ------------------------------------------------------------------

    def _resolve_viewpoints(
        self,
    ) -> None:

        self._resolved_viewpoints = []

        if (
            self.metadata is None
            or not self._viewpoints
        ):
            return

        target_crs = (
            self.metadata.crs
        )

        transformer_cache: dict[
            str,
            Transformer,
        ] = {}

        for viewpoint in self._viewpoints:

            try:
                viewpoint_id = str(
                    viewpoint.viewpoint_id
                )

                x = float(
                    viewpoint.x
                )

                y = float(
                    viewpoint.y
                )

                source_crs = (
                    CRS.from_user_input(
                        viewpoint.crs
                    )
                )

            except (
                AttributeError,
                TypeError,
                ValueError,
            ):
                continue

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            if source_crs != target_crs:

                key = source_crs.to_string()

                transformer = (
                    transformer_cache.get(
                        key
                    )
                )

                if transformer is None:

                    transformer = (
                        Transformer.from_crs(
                            PyprojCRS.from_user_input(
                                source_crs.to_string()
                            ),
                            PyprojCRS.from_user_input(
                                target_crs.to_string()
                            ),
                            always_xy=True,
                        )
                    )

                    transformer_cache[
                        key
                    ] = transformer

                x, y = transformer.transform(
                    x,
                    y,
                )

            self._resolved_viewpoints.append(
                (
                    viewpoint_id,
                    float(x),
                    float(y),
                )
            )

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def _remove_semantic_overlays(
        self,
    ) -> None:

        for artist in (
            self._viewpoint_artist,
            self._selected_artist,
            self._draft_polygon_artist,
        ):

            if artist is not None:

                try:
                    artist.remove()

                except ValueError:
                    pass

        self._viewpoint_artist = None
        self._selected_artist = None

        # Draft artist is managed by drawing mode and should not be erased
        # merely because another semantic overlay changed.
        if self._polygon_selector is None:
            self._draft_polygon_artist = None

        for artist in self._id_artists:

            try:
                artist.remove()

            except ValueError:
                pass

        self._id_artists = []

    def _draw_semantic_overlays(
        self,
    ) -> None:

        if self.metadata is None:
            return

        show_survey = (
            self._display_mode
            in {
                WorldMapDisplayMode.SURVEY,
                WorldMapDisplayMode.SURVEY_AND_DOMAIN,
            }
        )

        show_domain = (
            self._display_mode
            in {
                WorldMapDisplayMode.DOMAIN,
                WorldMapDisplayMode.SURVEY_AND_DOMAIN,
            }
        )

        if (
            show_domain
            and self._domain is not None
        ):
            self._draw_domain()

        if (
            show_survey
            and self._resolved_viewpoints
        ):
            self._draw_viewpoints()

    def _draw_viewpoints(
        self,
    ) -> None:

        ids = [
            item[0]
            for item
            in self._resolved_viewpoints
        ]

        xs = np.asarray(
            [
                item[1]
                for item
                in self._resolved_viewpoints
            ],
            dtype=float,
        )

        ys = np.asarray(
            [
                item[2]
                for item
                in self._resolved_viewpoints
            ],
            dtype=float,
        )

        self._viewpoint_artist = (
            self.axes.scatter(
                xs,
                ys,
                s=28,
                marker="o",
                edgecolors="white",
                linewidths=0.6,
                picker=True,
                pickradius=7,
                zorder=5,
                label="Viewpoints",
            )
        )

        # Keep IDs on the artist so pick_event index remains deterministic.
        self._viewpoint_artist._rivelero_viewpoint_ids = ids

        if self._show_viewpoint_ids:

            # Avoid making large opportunistic surveys unreadable.
            if len(ids) <= 250:

                for viewpoint_id, x, y in (
                    self._resolved_viewpoints
                ):

                    artist = self.axes.annotate(
                        viewpoint_id,
                        (
                            x,
                            y,
                        ),
                        xytext=(
                            4,
                            4,
                        ),
                        textcoords="offset points",
                        fontsize=7,
                        zorder=7,
                    )

                    self._id_artists.append(
                        artist
                    )

        if self._selected_viewpoint_id is not None:

            selected = next(
                (
                    item
                    for item
                    in self._resolved_viewpoints
                    if item[0]
                    == self._selected_viewpoint_id
                ),
                None,
            )

            if selected is not None:

                _, x, y = selected

                self._selected_artist = (
                    self.axes.scatter(
                        [x],
                        [y],
                        s=115,
                        marker="o",
                        facecolors="none",
                        linewidths=2.0,
                        zorder=8,
                        label="Selected Viewpoint",
                    )
                )

    def _draw_domain(
        self,
    ) -> None:

        geometry = self._domain.geometry

        polygons = []

        if geometry.geom_type == "Polygon":
            polygons = [
                geometry
            ]

        elif geometry.geom_type == "MultiPolygon":
            polygons = list(
                geometry.geoms
            )

        else:
            return

        # Keep one artist reference for the first polygon and allow the
        # remaining patches to be cleaned by axes collections on the next
        # full semantic redraw.
        first_artist = None

        for polygon in polygons:

            coords = np.asarray(
                polygon.exterior.coords
            )

            artist = MplPolygon(
                coords,
                closed=True,
                fill=True,
                alpha=0.16,
                linewidth=2.0,
                zorder=4,
            )

            self.axes.add_patch(
                artist
            )

            if first_artist is None:
                first_artist = artist

        self._domain_artist = (
            first_artist
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_pick(
        self,
        event,
    ) -> None:

        if (
            self._viewpoint_artist is None
            or event.artist
            is not self._viewpoint_artist
            or not event.ind
        ):
            return

        index = int(
            event.ind[0]
        )

        if (
            index < 0
            or index
            >= len(
                self._resolved_viewpoints
            )
        ):
            return

        viewpoint_id = (
            self._resolved_viewpoints[
                index
            ][0]
        )

        self.set_selected_viewpoint(
            viewpoint_id
        )

        self.viewpoint_selected.emit(
            viewpoint_id
        )

    # ------------------------------------------------------------------
    # Display mode
    # ------------------------------------------------------------------

    def _on_display_changed(
        self,
        _index: int,
    ) -> None:

        value = (
            self.display_combo
            .currentData()
        )

        self._display_mode = (
            WorldMapDisplayMode(
                value
            )
        )

        self.redraw_overlays()

    # ------------------------------------------------------------------
    # Polygon drawing
    # ------------------------------------------------------------------

    def _on_draw_toggled(
        self,
        enabled: bool,
    ) -> None:

        if enabled:
            self.start_domain_drawing()

        else:
            self.cancel_domain_drawing(
                emit_signal=False
            )

    def start_domain_drawing(
        self,
    ) -> None:
        """Activate Matplotlib PolygonSelector."""

        if self.metadata is None:

            self.draw_domain_button.blockSignals(
                True
            )

            self.draw_domain_button.setChecked(
                False
            )

            self.draw_domain_button.blockSignals(
                False
            )

            return

        self.cancel_domain_drawing(
            emit_signal=False
        )

        # Disable toolbar navigation so clicks belong unambiguously to the
        # polygon selector.
        if self._toolbar.mode:

            if "pan" in str(
                self._toolbar.mode
            ).lower():
                self._toolbar.pan()

            elif "zoom" in str(
                self._toolbar.mode
            ).lower():
                self._toolbar.zoom()

        self._polygon_selector = (
            PolygonSelector(
                self.axes,
                self._complete_domain_polygon,
                useblit=True,
            )
        )

        self.draw_domain_button.blockSignals(
            True
        )

        self.draw_domain_button.setChecked(
            True
        )

        self.draw_domain_button.blockSignals(
            False
        )

        self.draw_domain_button.setText(
            "Drawing analysis area…"
        )

        self.cancel_draw_button.setVisible(
            True
        )

        self.domain_drawing_started.emit()

    def cancel_domain_drawing(
        self,
        *,
        emit_signal: bool = True,
    ) -> None:
        """Deactivate polygon drawing without changing current domain."""

        if self._polygon_selector is not None:

            self._polygon_selector.disconnect_events()

            self._polygon_selector = None

        self.draw_domain_button.blockSignals(
            True
        )

        self.draw_domain_button.setChecked(
            False
        )

        self.draw_domain_button.blockSignals(
            False
        )

        self.draw_domain_button.setText(
            "Draw analysis area"
        )

        self.cancel_draw_button.setVisible(
            False
        )

        if emit_signal:
            self.domain_drawing_cancelled.emit()

        self.canvas.draw_idle()

    def _complete_domain_polygon(
        self,
        vertices,
    ) -> None:

        clean_vertices = [
            (
                float(x),
                float(y),
            )
            for x, y
            in vertices
        ]

        if len(
            clean_vertices
        ) < 3:
            return

        self.cancel_domain_drawing(
            emit_signal=False
        )

        self.domain_polygon_drawn.emit(
            clean_vertices
        )