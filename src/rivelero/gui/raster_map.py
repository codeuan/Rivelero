"""Reusable raster map infrastructure for Rivelero GUI maps."""

from __future__ import annotations

from pathlib import Path

import rasterio
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget

from rivelero.gui.environment_import import RasterMetadata, inspect_elevation_raster


class RasterMapWidget(QWidget):
    """Display a georeferenced raster with navigation and extent control."""

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metadata: RasterMetadata | None = None
        self.raster_path: Path | None = None
        self.full_extent: tuple[float, float, float, float] | None = None
        self.current_extent: tuple[float, float, float, float] | None = None
        self._raster_artist = None

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
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
