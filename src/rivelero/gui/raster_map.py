"""Reusable raster map infrastructure for Rivelero GUI maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.crs import CRS

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget

from rivelero.gui.environment_import import RasterMetadata, inspect_elevation_raster

try:
    from shiboken6 import isValid as _qt_object_alive
except ImportError:  # PyQt6
    try:
        from PyQt6 import sip as _sip

        def _qt_object_alive(obj) -> bool:
            return not _sip.isdeleted(obj)
    except ImportError:
        def _qt_object_alive(obj) -> bool:
            return True


class SafeFigureCanvas(FigureCanvasQTAgg):
    """FigureCanvasQTAgg whose deferred redraw tolerates widget deletion.

    Matplotlib queues ``draw_idle`` with a zero-delay timer. If the widget is
    destroyed first (a closed dialog or discarded page), the queued callback
    would touch a deleted Qt object and raise; the redraw is simply skipped.
    """

    def _draw_idle(self):
        if _qt_object_alive(self):
            super()._draw_idle()


def project_viewpoints(
    viewpoints: Iterable[Any],
    target_crs: CRS,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return Viewpoint IDs and coordinates transformed for display.

    Coordinates are transformed into ``target_crs`` for display only; the
    canonical Viewpoints are never modified. Viewpoints with missing or
    non-finite coordinates are skipped.
    """

    ids: list[str] = []
    xs: list[float] = []
    ys: list[float] = []

    transformers: dict[str, Transformer] = {}

    for viewpoint in viewpoints:
        try:
            viewpoint_id = str(viewpoint.viewpoint_id)
            x = float(viewpoint.x)
            y = float(viewpoint.y)
            source_crs = CRS.from_user_input(viewpoint.crs)
        except (AttributeError, TypeError, ValueError):
            continue

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        if source_crs != target_crs:
            key = source_crs.to_string()
            transformer = transformers.get(key)

            if transformer is None:
                transformer = Transformer.from_crs(
                    PyprojCRS.from_user_input(source_crs.to_string()),
                    PyprojCRS.from_user_input(target_crs.to_string()),
                    always_xy=True,
                )
                transformers[key] = transformer

            x, y = transformer.transform(x, y)

        ids.append(viewpoint_id)
        xs.append(float(x))
        ys.append(float(y))

    return ids, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


class RasterMapWidget(QWidget):
    """Display a georeferenced raster with navigation and extent control."""

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metadata: RasterMetadata | None = None
        self.raster_path: Path | None = None
        self.full_extent: tuple[float, float, float, float] | None = None
        self.current_extent: tuple[float, float, float, float] | None = None
        self._raster_artist = None

        # One colourbar axis is reused for the widget's lifetime so that
        # layer changes never stack additional colourbars.
        self._colorbar = None
        self._colorbar_axes = None

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = SafeFigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.ax = self.axes
        self._toolbar = NavigationToolbar2QT(self.canvas, self)
        self._status_label = QLabel()
        self._status_label.setProperty("secondaryText", True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self._status_label)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_raster(self, path: str | Path) -> None:
        """Load raster metadata and draw the complete raster."""
        source = Path(path).expanduser().resolve()
        self.metadata = inspect_elevation_raster(source)
        self.raster_path = source
        bounds = self.metadata.bounds
        self.full_extent = (
            float(bounds.left),
            float(bounds.right),
            float(bounds.bottom),
            float(bounds.top),
        )
        self.current_extent = self.full_extent
        self._draw_raster()
        self.redraw_overlays()

    def set_view_extent(
        self,
        extent: tuple[float, float, float, float],
        *,
        reload: bool = True,
    ) -> None:
        """Set the visible map extent, optionally reloading the raster window."""
        if len(extent) != 4:
            raise ValueError("extent must contain left, right, bottom, top.")
        left, right, bottom, top = (float(value) for value in extent)
        if not left < right or not bottom < top:
            raise ValueError("extent bounds must be increasing.")
        self.current_extent = (left, right, bottom, top)
        if reload:
            self._draw_raster()
            self.redraw_overlays()
        else:
            self.axes.set_xlim(left, right)
            self.axes.set_ylim(bottom, top)
            self.canvas.draw_idle()

    def reset_view(self) -> None:
        """Return to the complete raster extent."""
        if self.full_extent is not None:
            self.set_view_extent(self.full_extent)

    def redraw_overlays(self) -> None:
        """Hook for semantic subclasses such as WorldMapWidget."""
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Colourbar lifecycle
    # ------------------------------------------------------------------

    @property
    def colorbar_visible(self) -> bool:
        """Whether a continuous colourbar is currently shown."""
        return (
            self._colorbar is not None
            and self._colorbar_axes is not None
            and self._colorbar_axes.get_visible()
        )

    @property
    def colorbar_label(self) -> str | None:
        if self._colorbar is None:
            return None
        return self._colorbar.ax.get_ylabel()

    def show_colorbar(self, mappable, label: str) -> None:
        """Show the single colourbar for ``mappable`` with ``label``."""
        if self._colorbar_axes is None:
            divider = make_axes_locatable(self.axes)
            self._colorbar_axes = divider.append_axes(
                "right", size="3.5%", pad=0.12
            )
        if self._colorbar is None:
            self._colorbar = self.figure.colorbar(
                mappable, cax=self._colorbar_axes
            )
        else:
            self._colorbar.update_normal(mappable)
        self._colorbar.set_label(label)
        self._colorbar_axes.set_visible(True)

    def hide_colorbar(self) -> None:
        """Hide the colourbar, e.g. for categorical layers with a legend."""
        if self._colorbar_axes is not None:
            self._colorbar_axes.set_visible(False)

    def _draw_raster(self) -> None:
        if self.metadata is None or self.raster_path is None:
            return
        with rasterio.open(self.raster_path) as dataset:
            data = dataset.read(1, masked=True)
        self.axes.clear()
        self._raster_artist = self.axes.imshow(
            data,
            extent=self.full_extent,
            origin="upper",
            cmap="terrain",
            interpolation="nearest",
            zorder=0,
        )
        self.show_colorbar(self._raster_artist, "Elevation")
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_aspect("equal", adjustable="box")
        if self.current_extent is not None:
            self.axes.set_xlim(self.current_extent[0], self.current_extent[1])
            self.axes.set_ylim(self.current_extent[2], self.current_extent[3])
        self._status_label.setText(
            f"CRS: {self.metadata.crs} | Size: {self.metadata.width:,} x {self.metadata.height:,}"
        )
        self.canvas.draw_idle()
