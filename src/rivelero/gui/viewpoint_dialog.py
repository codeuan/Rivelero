"""Canonical Viewpoint add/edit dialog."""

from __future__ import annotations

import json
import math
from typing import Any

try:
    from PySide6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGroupBox,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGroupBox,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )

from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.components import CollapsibleSection


class ViewpointDialog(QDialog):
    """Create or edit one canonical :class:`Viewpoint`."""

    def __init__(
        self,
        *,
        sensors: dict[str, Any] | None = None,
        viewpoint: Viewpoint | None = None,
        title: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            title or ("Edit Viewpoint" if viewpoint else "Add Viewpoint")
        )
        self.setMinimumWidth(560)
        self.result_viewpoint: Viewpoint | None = None
        self._original = viewpoint
        self._fields: dict[str, QLineEdit] = {}

        outer = QVBoxLayout(self)
        form = QFormLayout()
        for name, label in (
            ("viewpoint_id", "Viewpoint ID"),
            ("x", "X / Easting"),
            ("y", "Y / Northing"),
            ("crs", "CRS"),
            ("z", "Elevation / Z"),
            ("observer_height_m", "Observer height (m)"),
            ("heading_deg", "Heading (degrees)"),
            ("pitch_deg", "Pitch (degrees)"),
            ("roll_deg", "Roll (degrees)"),
            ("horizontal_fov_deg", "Horizontal field of view (degrees)"),
            ("vertical_fov_deg", "Vertical field of view (degrees)"),
            ("platform", "Platform"),
            ("source", "Source"),
            ("source_id", "Source ID"),
        ):
            self._add_field(form, name, label)

        self.sensor_combo = QComboBox()
        self.sensor_combo.addItem("No Sensor", None)
        for sensor_id in sorted((sensors or {}).keys()):
            self.sensor_combo.addItem(sensor_id, sensor_id)
        form.addRow("Sensor", self.sensor_combo)
        outer.addLayout(form)

        advanced = CollapsibleSection("Advanced metadata", expanded=False)
        advanced_form = QFormLayout()
        self._add_field(advanced_form, "position_uncertainty_m", "Position uncertainty (m)")
        self._add_field(advanced_form, "orientation_uncertainty_deg", "Orientation uncertainty (degrees)")
        self.provenance_edit = QPlainTextEdit()
        self.provenance_edit.setPlaceholderText("{}")
        self.provenance_edit.setMaximumHeight(90)
        advanced_form.addRow("Provenance JSON", self.provenance_edit)
        self.extra_edit = QPlainTextEdit()
        self.extra_edit.setPlaceholderText("{}")
        self.extra_edit.setMaximumHeight(90)
        advanced_form.addRow("Extra metadata JSON", self.extra_edit)
        advanced.content_layout.addLayout(advanced_form)
        outer.addWidget(advanced)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._populate(viewpoint)

    def _add_field(self, form: QFormLayout, name: str, label: str) -> None:
        field = QLineEdit()
        field.setPlaceholderText("Blank = missing")
        self._fields[name] = field
        form.addRow(label, field)

    def _populate(self, viewpoint: Viewpoint | None) -> None:
        if viewpoint is None:
            self._fields["crs"].setText("EPSG:4326")
            self.provenance_edit.setPlainText("{}")
            self.extra_edit.setPlainText("{}")
            return
        for name in self._fields:
            value = getattr(viewpoint, name)
            self._fields[name].setText("" if value is None else str(value))
        index = self.sensor_combo.findData(viewpoint.sensor_id)
        self.sensor_combo.setCurrentIndex(max(index, 0))
        self.provenance_edit.setPlainText(json.dumps(viewpoint.provenance, indent=2))
        self.extra_edit.setPlainText(json.dumps(viewpoint.extra_metadata, indent=2))

    @staticmethod
    def _optional_float(field: QLineEdit, label: str) -> float | None:
        text = field.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value

    @staticmethod
    def _json_object(editor: QPlainTextEdit, label: str) -> dict[str, Any]:
        text = editor.toPlainText().strip() or "{}"
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must contain valid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object.")
        return value

    def _accept(self) -> None:
        try:
            viewpoint = Viewpoint(
                viewpoint_id=self._fields["viewpoint_id"].text().strip(),
                x=float(self._fields["x"].text()),
                y=float(self._fields["y"].text()),
                crs=self._fields["crs"].text().strip(),
                z=self._optional_float(self._fields["z"], "Elevation"),
                observer_height_m=self._optional_float(self._fields["observer_height_m"], "Observer height"),
                heading_deg=self._optional_float(self._fields["heading_deg"], "Heading"),
                pitch_deg=self._optional_float(self._fields["pitch_deg"], "Pitch"),
                roll_deg=self._optional_float(self._fields["roll_deg"], "Roll"),
                horizontal_fov_deg=self._optional_float(self._fields["horizontal_fov_deg"], "Horizontal FOV"),
                vertical_fov_deg=self._optional_float(self._fields["vertical_fov_deg"], "Vertical FOV"),
                sensor_id=self.sensor_combo.currentData(),
                platform=self._fields["platform"].text().strip() or None,
                source=self._fields["source"].text().strip() or None,
                source_id=self._fields["source_id"].text().strip() or None,
                position_uncertainty_m=self._optional_float(self._fields["position_uncertainty_m"], "Position uncertainty"),
                orientation_uncertainty_deg=self._optional_float(self._fields["orientation_uncertainty_deg"], "Orientation uncertainty"),
                provenance=self._json_object(self.provenance_edit, "Provenance"),
                extra_metadata=self._json_object(self.extra_edit, "Extra metadata"),
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid Viewpoint", str(exc))
            return
        self.result_viewpoint = viewpoint
        self.accept()
