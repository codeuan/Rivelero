from __future__ import annotations

from collections import Counter
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_hex
from matplotlib.patches import FancyArrow
from matplotlib.ticker import MaxNLocator

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QComboBox,
        QCheckBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QComboBox,
        QCheckBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.gui.survey_qc import SurveySpatialQC


DISPLAY_MODES = (
    "all",
    "heading",
    "fov",
    "observer_height",
    "sensor",
    "source",
    "platform",
    "observation_sequence",
)


class SurveyMapWidget(QWidget):
    """Reusable Matplotlib survey map that works with canonical Rivelero data."""

    viewpoint_selected = Signal(str)
    selection_changed = Signal(str)

    def __init__(
        self,
        *,
        configuration: ViewpointConfiguration | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.configuration: ViewpointConfiguration | None = None
        self.qc: SurveySpatialQC | None = None
        self.current_display_mode = "all"
        self.selected_viewpoint_id: str | None = None
        self._show_viewpoint_ids = False
        self._show_orientation = False
        self._point_viewpoint_ids: list[str] = []
        self._selection_blocked = False

        self._build_ui()
        self.set_configuration(configuration)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        self.mode_combo = QComboBox()
        for label, mode in (
            ("All Viewpoints", "all"),
            ("Heading availability", "heading"),
            ("FOV availability", "fov"),
            ("Observer-height availability", "observer_height"),
            ("Sensor", "sensor"),
            ("Source", "source"),
            ("Platform", "platform"),
            ("Observation sequence", "observation_sequence"),
        ):
            self.mode_combo.addItem(label, mode)
        self.mode_combo.setCurrentText("All Viewpoints")
        controls.addWidget(QLabel("Colour / display by:"))
        controls.addWidget(self.mode_combo, 1)

        self.ids_toggle = QCheckBox("Viewpoint IDs")
        self.orientation_toggle = QCheckBox("Show orientation")
        self.orientation_toggle.setChecked(True)
        controls.addWidget(self.ids_toggle)
        controls.addWidget(self.orientation_toggle)

        self.reset_button = QPushButton("Reset")
        controls.addWidget(self.reset_button)

        layout.addLayout(controls)

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(360)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.mpl_connect("pick_event", self._on_pick_event)
        layout.addWidget(self.canvas, 1)

        self.qc_label = QLabel()
        self.qc_label.setWordWrap(True)
        self.qc_label.setProperty("secondaryText", True)
        layout.addWidget(self.qc_label)

        self.mode_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        self.ids_toggle.toggled.connect(self._set_show_ids)
        self.orientation_toggle.toggled.connect(self._set_show_orientation)
        self.reset_button.clicked.connect(self._reset_view)

    def set_configuration(self, configuration: ViewpointConfiguration | None) -> None:
        self.configuration = configuration
        if configuration is None or not configuration.viewpoints:
            self.selected_viewpoint_id = None
            self.qc = SurveySpatialQC.from_configuration(configuration)
            self._redraw()
            return

        self.qc = SurveySpatialQC.from_configuration(configuration)
        if self.selected_viewpoint_id is not None:
            if self.selected_viewpoint_id not in self.qc.viewpoint_by_id:
                self.selected_viewpoint_id = None
        self._redraw()

    def set_display_mode(self, mode: str) -> None:
        if mode not in DISPLAY_MODES:
            raise ValueError(f"Unsupported display mode: {mode!r}")
        self.current_display_mode = mode
        self.mode_combo.setCurrentText(self._display_mode_label(mode))
        self._redraw()

    def set_selected_viewpoint_id(self, viewpoint_id: str | None, *, emit_signal: bool = True) -> None:
        if self.configuration is None:
            self.selected_viewpoint_id = None
            return

        if viewpoint_id is not None and viewpoint_id not in self.qc.viewpoint_by_id:
            raise ValueError(f"Unknown viewpoint_id: {viewpoint_id}")

        if self.selected_viewpoint_id == viewpoint_id:
            if emit_signal:
                self.selection_changed.emit(viewpoint_id or "")
            return

        self.selected_viewpoint_id = viewpoint_id
        self._redraw()
        if emit_signal:
            self.selection_changed.emit(viewpoint_id or "")

    @property
    def show_viewpoint_ids(self) -> bool:
        return self._show_viewpoint_ids

    @show_viewpoint_ids.setter
    def show_viewpoint_ids(self, enabled: bool) -> None:
        self._show_viewpoint_ids = bool(enabled)
        self._redraw()

    @property
    def show_orientation(self) -> bool:
        return self._show_orientation

    @show_orientation.setter
    def show_orientation(self, enabled: bool) -> None:
        self._show_orientation = bool(enabled)
        self._redraw()

    def _on_display_mode_changed(self, index: int) -> None:
        mode = self.mode_combo.itemData(index)
        if mode is None:
            return
        self.current_display_mode = str(mode)
        self._redraw()

    def _set_show_ids(self, enabled: bool) -> None:
        self.show_viewpoint_ids = enabled

    def _set_show_orientation(self, enabled: bool) -> None:
        self.show_orientation = enabled

    def _reset_view(self) -> None:
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _on_pick_event(self, event: MouseEvent) -> None:
        if not hasattr(event, "artist") or event.artist is None:
            return
        if len(getattr(event, "ind", ())) == 0:
            return
        index = int(event.ind[0])
        if index < 0 or index >= len(self._point_viewpoint_ids):
            return
        viewpoint_id = self._point_viewpoint_ids[index]
        self._selection_blocked = True
        try:
            self.set_selected_viewpoint_id(viewpoint_id, emit_signal=False)
        finally:
            self._selection_blocked = False
        self.viewpoint_selected.emit(viewpoint_id)

    def _redraw(self) -> None:
        self.ax.clear()

        if self.configuration is None or not self.configuration.viewpoints:
            self.ax.text(0.5, 0.5, "No Viewpoints", ha="center", va="center")
            self.ax.set_axis_off()
            self.qc_label.setText("This survey contains no Viewpoints.")
            self.canvas.draw_idle()
            return

        self.qc = SurveySpatialQC.from_configuration(self.configuration)

        if not self.qc.has_consistent_crs:
            self.ax.text(
                0.5,
                0.5,
                "Inconsistent Viewpoint CRS\nMap view rejected for safety.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self.ax.set_axis_off()
            self.qc_label.setText(
                "CRS mismatch: Viewpoints do not share a single coordinate reference system."
            )
            self.canvas.draw_idle()
            return

        viewpoints = list(self.configuration.viewpoints)
        xs = [float(vp.x) for vp in viewpoints]
        ys = [float(vp.y) for vp in viewpoints]
        self._point_viewpoint_ids = [vp.viewpoint_id for vp in viewpoints]

        colors = [self._color_for_viewpoint(vp) for vp in viewpoints]
        collection = self.ax.scatter(
            xs,
            ys,
            s=35,
            c=colors,
            edgecolors="white",
            linewidths=0.7,
            picker=True,
            zorder=5,
        )
        collection.set_label("Viewpoints")
        self._maybe_draw_orientation(viewpoints)
        self._maybe_draw_observation_sequence(viewpoints)

        if self.selected_viewpoint_id is not None:
            idx = self._index_for_selected()
            if idx is not None:
                self.ax.scatter(
                    [xs[idx]],
                    [ys[idx]],
                    s=120,
                    facecolors="none",
                    edgecolors="#1F2937",
                    linewidths=2.0,
                    zorder=10,
                )

        if self._show_viewpoint_ids and len(viewpoints) <= 500:
            for viewpoint in viewpoints:
                self.ax.annotate(
                    str(viewpoint.viewpoint_id),
                    (float(viewpoint.x), float(viewpoint.y)),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    zorder=7,
                    alpha=0.8,
                )

        if len(xs) > 1:
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.autoscale(True)
        self.ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
        self.ax.set_xlabel(self._axis_label("x"))
        self.ax.set_ylabel(self._axis_label("y"))
        self.ax.set_title("Survey map")

        self._update_qc_label()
        self.canvas.draw_idle()

    def _axis_label(self, axis_name: str) -> str:
        crs = self.qc.crs if self.qc is not None else None
        if crs is None:
            return axis_name.upper()
        if axis_name == "x":
            return "X / easting"
        return "Y / northing"

    def _index_for_selected(self) -> int | None:
        if self.selected_viewpoint_id is None:
            return None
        for index, viewpoint_id in enumerate(self._point_viewpoint_ids):
            if viewpoint_id == self.selected_viewpoint_id:
                return index
        return None

    def _maybe_draw_orientation(self, viewpoints: list[Any]) -> None:
        if not self._show_orientation:
            return
        if len(viewpoints) > 1500:
            return
        for viewpoint in viewpoints:
            if viewpoint.heading_deg is None:
                continue
            heading = float(viewpoint.heading_deg) % 360.0
            angle = (90.0 - heading) * (3.141592653589793 / 180.0)
            length = self._orientation_length(viewpoints)
            x0 = float(viewpoint.x)
            y0 = float(viewpoint.y)
            x1 = x0 + length * (2.0 if heading == 0 else 0.0)
            y1 = y0 + length * (2.0 if heading == 0 else 0.0)
            dx = length * __import__("math").sin(angle)
            dy = length * __import__("math").cos(angle)
            arrow = FancyArrow(
                x0,
                y0,
                dx,
                dy,
                width=0.2,
                head_width=length * 0.18,
                head_length=length * 0.24,
                length_includes_head=True,
                color="#3F6F78",
                alpha=0.8,
                zorder=6,
            )
            self.ax.add_patch(arrow)

    def _orientation_length(self, viewpoints: list[Any]) -> float:
        if not viewpoints:
            return 1.0
        xs = [float(vp.x) for vp in viewpoints]
        ys = [float(vp.y) for vp in viewpoints]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        return span * 0.05

    def _maybe_draw_observation_sequence(self, viewpoints: list[Any]) -> None:
        if self.current_display_mode != "observation_sequence":
            return
        if self.configuration is None:
            return
        events = list(self.configuration.observation_events)
        if not events:
            self.ax.text(
                0.5,
                0.5,
                "This survey contains no ObservationEvents.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return
        xs, ys = [], []
        for event in events:
            point = self.configuration.get_viewpoint(event.viewpoint_id)
            xs.append(float(point.x))
            ys.append(float(point.y))
        self.ax.plot(xs, ys, linestyle="--", linewidth=1.2, color="#7A65A8", alpha=0.7, zorder=4)

    def _color_for_viewpoint(self, viewpoint: Any) -> str:
        mode = self.current_display_mode
        if mode == "all":
            return to_hex("#3F6F78")
        if mode == "heading":
            return to_hex("#4E9B67") if viewpoint.heading_deg is not None else to_hex("#C58A2A")
        if mode == "fov":
            return to_hex("#4E9B67") if viewpoint.horizontal_fov_deg is not None else to_hex("#C58A2A")
        if mode == "observer_height":
            return to_hex("#4E9B67") if viewpoint.observer_height_m is not None else to_hex("#C58A2A")
        if mode == "sensor":
            key = viewpoint.sensor_id or "Unknown"
            return self._category_color(key)
        if mode == "source":
            key = viewpoint.source or "Unknown"
            return self._category_color(key)
        if mode == "platform":
            key = viewpoint.platform or "Unknown"
            return self._category_color(key)
        if mode == "observation_sequence":
            if self.configuration is None:
                return to_hex("#3F6F78")
            events = list(self.configuration.observation_events)
            if not events:
                return to_hex("#A7AFB3")
            index_by_id = {event.viewpoint_id: idx for idx, event in enumerate(events)}
            point_index = index_by_id.get(viewpoint.viewpoint_id, 0)
            palette = [
                "#3F6F78",
                "#557FA3",
                "#4E9B67",
                "#C58A2A",
                "#7A65A8",
                "#B85450",
            ]
            return palette[point_index % len(palette)]
        return to_hex("#3F6F78")

    def _category_color(self, category: str) -> str:
        palette = [
            "#3F6F78",
            "#557FA3",
            "#4E9B67",
            "#C58A2A",
            "#7A65A8",
            "#B85450",
            "#5B7382",
            "#9B7B5C",
        ]
        counts = Counter()
        if self.configuration is not None:
            for viewpoint in self.configuration.viewpoints:
                if self.current_display_mode == "sensor":
                    counts[viewpoint.sensor_id or "Unknown"] += 1
                elif self.current_display_mode == "source":
                    counts[viewpoint.source or "Unknown"] += 1
                else:
                    counts[viewpoint.platform or "Unknown"] += 1
        ordered = list(counts.keys())
        if category in ordered:
            idx = ordered.index(category)
        else:
            idx = 0
        return palette[min(idx, len(palette) - 1)]

    def _display_mode_label(self, mode: str) -> str:
        labels = {
            "all": "All Viewpoints",
            "heading": "Heading availability",
            "fov": "FOV availability",
            "observer_height": "Observer-height availability",
            "sensor": "Sensor",
            "source": "Source",
            "platform": "Platform",
            "observation_sequence": "Observation sequence",
        }
        return labels.get(mode, "All Viewpoints")

    def _update_qc_label(self) -> None:
        if self.qc is None:
            self.qc_label.setText("")
            return

        if self.configuration is None or not self.configuration.viewpoints:
            self.qc_label.setText("This survey contains no Viewpoints.")
            return

        if self.current_display_mode in {"heading", "fov", "observer_height"}:
            if self.current_display_mode == "heading":
                missing = self.qc.missing_heading_count
                available = self.qc.total_viewpoints - missing
                text = f"Heading available: {available:,} | Heading missing: {missing:,}"
            elif self.current_display_mode == "fov":
                missing = self.qc.missing_fov_count
                available = self.qc.total_viewpoints - missing
                text = f"FOV available: {available:,} | FOV missing: {missing:,}"
            else:
                missing = self.qc.missing_observer_height_count
                available = self.qc.total_viewpoints - missing
                text = f"Observer height available: {available:,} | Missing: {missing:,}"
        else:
            text = (
                f"CRS: {self.qc.crs.to_string() if self.qc.crs is not None else 'unavailable'} | "
                f"Viewpoints: {self.qc.total_viewpoints:,} | "
                f"Repeated locations: {self.qc.repeated_location_count:,}"
            )
        self.qc_label.setText(text)
