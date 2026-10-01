"""Optional OpenStreetMap background for Rivelero's Matplotlib maps.

BasemapLayer draws :func:`rivelero.visualization.basemap.render_basemap`
beneath every other artist of one Axes and keeps it matched to the visible
extent: after a pan or zoom the tiles for the new view are fetched on a
worker thread and swapped in, so the map never waits for the network.

The background is display-only context. It never changes axis limits or
data limits, so autoscaling and "Full extent" behave as without it.
"""

from __future__ import annotations

from typing import Any, Callable

from matplotlib.image import AxesImage

try:
    from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal
    from PySide6.QtWidgets import QCheckBox, QWidget
except ImportError:
    from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal as Signal
    from PyQt6.QtWidgets import QCheckBox, QWidget

from rasterio.crs import CRS

from rivelero.visualization.basemap import (
    OSM_ATTRIBUTION,
    BasemapImage,
    TileFetcher,
    render_basemap,
)

BASEMAP_ZORDER = -10
# Label of the background image artist (hidden from legends).
BASEMAP_LABEL = "_basemap"
# Opacity of the layers drawn over the background, so it shows through.
OVERLAY_ALPHA = 0.6
# Delay after the last pan/zoom step before new tiles are requested.
REFRESH_DELAY_MS = 350
# Output pixels are capped so a very large window stays quick to warp.
MAX_IMAGE_SIDE = 2048

TOOLTIP = (
    "Show OpenStreetMap beneath the map to check where it lies in the real "
    "world. Needs an internet connection; tiles are cached. Display only."
)


class _ResultSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class _RenderTask(QRunnable):
    def __init__(self, generation: int, render: Callable[[], BasemapImage]) -> None:
        super().__init__()
        self.generation = generation
        self.render = render
        self.signals = _ResultSignals()

    def run(self) -> None:
        try:
            result = self.render()
        except Exception as exc:  # noqa: BLE001 - reported on the map
            self.signals.failed.emit(self.generation, str(exc))
        else:
            self.signals.finished.emit(self.generation, result)


class BasemapLayer(QObject):
    """OpenStreetMap background for ``axes``, toggled by :attr:`toggle`."""

    # Emitted whenever the layer is switched on or off.
    toggled = Signal(bool)

    def __init__(
        self,
        axes,
        canvas,
        *,
        parent: QWidget | None = None,
        fetch_tile: TileFetcher | None = None,
        synchronous: bool = False,
    ) -> None:
        super().__init__(parent)
        self.axes = axes
        self.canvas = canvas
        self.fetch_tile = fetch_tile
        # Render on the calling thread instead of the thread pool (tests).
        self.synchronous = synchronous

        self._crs: CRS | None = None
        self._image: BasemapImage | None = None
        self._artist: AxesImage | None = None
        self._text_artist = None
        self._message: str | None = None
        self._requested_view: tuple | None = None
        self._generation = 0
        self._tasks: set[_RenderTask] = set()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFRESH_DELAY_MS)
        self._timer.timeout.connect(self.update_now)

        self.toggle = QCheckBox("OpenStreetMap", parent)
        self.toggle.setToolTip(TOOLTIP)
        self.toggle.toggled.connect(self._on_toggled)
        self._update_toggle()

        canvas.mpl_connect("draw_event", self._on_draw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.toggle.isChecked() and self._crs is not None

    @property
    def image(self) -> BasemapImage | None:
        """The background currently drawn, if any."""
        return self._image if self.enabled else None

    @property
    def message(self) -> str | None:
        """Loading or error message shown on the map, if any."""
        return self._message

    @property
    def overlay_alpha(self) -> float | None:
        """Opacity for raster layers above the background (None: opaque)."""
        return OVERLAY_ALPHA if self.enabled else None

    def set_enabled(self, enabled: bool) -> None:
        self.toggle.setChecked(bool(enabled))

    def set_crs(self, crs: Any | None) -> None:
        """Set the map CRS; ``None`` (no georeferencing) disables the layer."""

        try:
            resolved = None if crs is None else CRS.from_user_input(crs)
        except Exception:  # noqa: BLE001 - an unusable CRS simply disables it
            resolved = None
        if resolved is not None and not (resolved.is_projected or resolved.is_geographic):
            resolved = None
        if resolved == self._crs:
            return
        self._crs = resolved
        self._image = None
        self._requested_view = None
        self._update_toggle()
        self.refresh()

    def refresh(self) -> None:
        """Re-attach the background after the Axes was cleared or redrawn.

        Map widgets call this after ``axes.clear()``; the current background
        is re-added at once and new tiles follow if the view changed.
        """

        self._detach()
        if not self.enabled:
            self._message = None
            return
        self._attach()
        self._timer.start()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_toggle(self) -> None:
        available = self._crs is not None
        self.toggle.setEnabled(available)
        self.toggle.setToolTip(
            TOOLTIP if available
            else "OpenStreetMap needs map coordinates in a known CRS."
        )

    def _on_toggled(self, _checked: bool) -> None:
        self._requested_view = None
        self.refresh()
        self.toggled.emit(self.enabled)
        self.canvas.draw_idle()

    def _current_view(self) -> tuple | None:
        left, right = sorted(self.axes.get_xlim())
        bottom, top = sorted(self.axes.get_ylim())
        if not (left < right and bottom < top):
            return None
        box = self.axes.get_window_extent()
        scale = max(box.width, box.height, 1.0) / MAX_IMAGE_SIDE
        width = max(1, int(box.width / max(scale, 1.0)))
        height = max(1, int(box.height / max(scale, 1.0)))
        return (left, right, bottom, top), (width, height)

    def _on_draw(self, _event) -> None:
        if not self.enabled:
            return
        cleared = (
            self._text_artist is None
            or self._text_artist.axes is not self.axes
            or (self._artist is not None and self._artist.axes is not self.axes)
        )
        if cleared:
            # The Axes was cleared by its widget: put the background back.
            self._detach()
            self._attach()
            self.canvas.draw_idle()
        view = self._current_view()
        if view is not None and view != self._requested_view:
            self._timer.start()

    def update_now(self) -> None:
        """Request tiles for the current view now instead of after a delay."""
        if not self.enabled:
            return
        view = self._current_view()
        if view is None or view == self._requested_view:
            return
        self._requested_view = view
        self._generation += 1
        extent, size = view
        crs = self._crs
        fetch = self.fetch_tile

        def render() -> BasemapImage:
            return render_basemap(extent, crs, size, fetch_tile=fetch)

        if self._image is None:
            self._set_message("Loading OpenStreetMap…")

        task = _RenderTask(self._generation, render)
        task.signals.finished.connect(self._on_finished)
        task.signals.failed.connect(self._on_failed)
        if self.synchronous:
            task.run()
            return
        task.setAutoDelete(False)
        self._tasks.add(task)
        task.signals.finished.connect(lambda *_: self._tasks.discard(task))
        task.signals.failed.connect(lambda *_: self._tasks.discard(task))
        QThreadPool.globalInstance().start(task)

    def _on_finished(self, generation: int, image: BasemapImage) -> None:
        if generation != self._generation or not self.enabled or image.crs != self._crs:
            return
        self._image = image
        self._message = None
        self._detach()
        self._attach()
        self.canvas.draw_idle()

    def _on_failed(self, generation: int, reason: str) -> None:
        if generation != self._generation or not self.enabled:
            return
        # Retry on the next pan/zoom or toggle rather than looping.
        self._set_message(f"OpenStreetMap unavailable ({reason})")

    def _set_message(self, text: str | None) -> None:
        self._message = text
        self._detach()
        self._attach()
        self.canvas.draw_idle()

    def _attach(self) -> None:
        if self._image is not None and self._artist is None:
            self._artist = _add_image_without_limits(self.axes, self._image)
        if self._text_artist is None:
            self._text_artist = self.axes.text(
                0.995,
                0.005,
                self._message or OSM_ATTRIBUTION,
                transform=self.axes.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#333333",
                zorder=30,
                bbox={"boxstyle": "square,pad=0.2", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.75},
            )
            # Not part of the data: never counted by autoscaling or layout.
            self._text_artist.set_in_layout(False)

    def _detach(self) -> None:
        for artist in (self._artist, self._text_artist):
            if artist is not None and artist.axes is not None:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):
                    pass
        self._artist = None
        self._text_artist = None


def _add_image_without_limits(axes, image: BasemapImage) -> AxesImage:
    """Add ``image`` beneath all artists without touching limits or datalim."""

    artist = AxesImage(axes, origin="upper", interpolation="bilinear", zorder=BASEMAP_ZORDER)
    artist.set_data(image.rgba)

    autoscale = (axes.get_autoscalex_on(), axes.get_autoscaley_on())
    data_limits = axes.dataLim.frozen()
    ignore = axes.ignore_existing_data_limits
    axes.set_autoscale_on(False)
    artist.set_extent(image.extent)
    axes.set_autoscalex_on(autoscale[0])
    axes.set_autoscaley_on(autoscale[1])
    axes.dataLim.set(data_limits)
    axes.ignore_existing_data_limits = ignore
    artist.sticky_edges.x[:] = []
    artist.sticky_edges.y[:] = []

    axes.add_image(artist)
    artist.set_label(BASEMAP_LABEL)
    return artist
