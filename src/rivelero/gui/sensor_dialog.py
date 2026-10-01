"""Canonical Sensor editor and Sensor manager dialogs."""

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
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QHBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QHBoxLayout,
        QWidget,
    )

from rivelero.core.sensor import Sensor, SensorModality
from rivelero.gui.components import CollapsibleSection


class SensorDialog(QDialog):
    """Create or edit one canonical Sensor."""

    def __init__(self, *, sensor: Sensor | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Sensor" if sensor else "Add Sensor")
        self.setMinimumWidth(560)
        self.result_sensor: Sensor | None = None
        self._fields: dict[str, QLineEdit] = {}
        outer = QVBoxLayout(self)
        form = QFormLayout()
        for name, label in (
            ("sensor_id", "Sensor ID"),
            ("manufacturer", "Manufacturer"),
            ("model", "Model"),
            ("horizontal_fov_deg", "Horizontal FOV (degrees)"),
            ("vertical_fov_deg", "Vertical FOV (degrees)"),
        ):
            self._add_field(form, name, label)
        self.modality_combo = QComboBox()
        for modality in SensorModality:
            self.modality_combo.addItem(modality.value.replace("_", " ").title(), modality.value)
        form.addRow("Modality", self.modality_combo)
        outer.addLayout(form)

        advanced = CollapsibleSection("Advanced sensor metadata", expanded=False)
        advanced_form = QFormLayout()
        for name, label in (
            ("image_width_px", "Image width (px)"),
            ("image_height_px", "Image height (px)"),
            ("focal_length_mm", "Focal length (mm)"),
            ("sensor_width_mm", "Sensor width (mm)"),
            ("sensor_height_mm", "Sensor height (mm)"),
            ("spectral_bands", "Spectral bands (comma separated)"),
            ("wavelength_range_nm", "Wavelength range (nm, min,max)"),
            ("spatial_resolution", "Spatial resolution"),
            ("source", "Source"),
        ):
            self._add_field(advanced_form, name, label)
        self.extra_edit = QPlainTextEdit()
        self.extra_edit.setPlaceholderText("{}")
        self.extra_edit.setMaximumHeight(90)
        advanced_form.addRow("Extra metadata JSON", self.extra_edit)
        advanced.content_layout.addLayout(advanced_form)
        outer.addWidget(advanced)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._populate(sensor)

    def _add_field(self, form: QFormLayout, name: str, label: str) -> None:
        field = QLineEdit()
        field.setPlaceholderText("Blank = missing")
        self._fields[name] = field
        form.addRow(label, field)

    def _populate(self, sensor: Sensor | None) -> None:
        if sensor is None:
            self.extra_edit.setPlainText("{}")
            return
        for name, field in self._fields.items():
            value = getattr(sensor, name)
            if name == "spectral_bands" and value:
                value = ", ".join(value)
            elif name == "wavelength_range_nm" and value is not None:
                value = ", ".join(str(item) for item in value)
            field.setText("" if value is None else str(value))
        index = self.modality_combo.findData(getattr(sensor.modality, "value", sensor.modality))
        self.modality_combo.setCurrentIndex(max(index, 0))
        self.extra_edit.setPlainText(json.dumps(sensor.extra_metadata, indent=2))

    @staticmethod
    def _optional_number(field: QLineEdit, label: str, *, integer: bool = False) -> float | int | None:
        text = field.text().strip()
        if not text:
            return None
        try:
            value = int(text) if integer else float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if not integer and not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")
        return value

    def _accept(self) -> None:
        try:
            bands = tuple(item.strip() for item in self._fields["spectral_bands"].text().split(",") if item.strip())
            wavelength_text = self._fields["wavelength_range_nm"].text().strip()
            wavelength = None if not wavelength_text else tuple(float(item.strip()) for item in wavelength_text.split(","))
            metadata = json.loads(self.extra_edit.toPlainText().strip() or "{}")
            if not isinstance(metadata, dict):
                raise ValueError("Extra metadata must be a JSON object.")
            self.result_sensor = Sensor(
                sensor_id=self._fields["sensor_id"].text().strip(),
                modality=self.modality_combo.currentData(),
                manufacturer=self._fields["manufacturer"].text().strip() or None,
                model=self._fields["model"].text().strip() or None,
                image_width_px=self._optional_number(self._fields["image_width_px"], "Image width", integer=True),
                image_height_px=self._optional_number(self._fields["image_height_px"], "Image height", integer=True),
                focal_length_mm=self._optional_number(self._fields["focal_length_mm"], "Focal length"),
                sensor_width_mm=self._optional_number(self._fields["sensor_width_mm"], "Sensor width"),
                sensor_height_mm=self._optional_number(self._fields["sensor_height_mm"], "Sensor height"),
                horizontal_fov_deg=self._optional_number(self._fields["horizontal_fov_deg"], "Horizontal FOV"),
                vertical_fov_deg=self._optional_number(self._fields["vertical_fov_deg"], "Vertical FOV"),
                spectral_bands=bands,
                wavelength_range_nm=wavelength,
                spatial_resolution=self._optional_number(self._fields["spatial_resolution"], "Spatial resolution"),
                source=self._fields["source"].text().strip() or None,
                extra_metadata=metadata,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Invalid Sensor", str(exc))
            return
        self.accept()


class SensorManagerDialog(QDialog):
    """Manage canonical Sensors without mutating ApplicationState directly."""

    def __init__(self, *, sensors: dict[str, Sensor], referenced_ids: set[str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Sensors")
        self.setMinimumSize(760, 440)
        self.sensors = dict(sensors)
        self.referenced_ids = set(referenced_ids or ())
        outer = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("Sensor ID", "Modality", "Manufacturer", "Model", "Horizontal FOV", "Vertical FOV"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        outer.addWidget(self.table)
        actions = QHBoxLayout()
        for label, handler in (("Add", self._add), ("Edit", self._edit), ("Delete", self._delete)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            actions.addWidget(button)
        actions.addStretch(1)
        outer.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._refresh()

    def _selected_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def _refresh(self) -> None:
        self.table.setRowCount(0)
        for row, sensor in enumerate(self.sensors.values()):
            self.table.insertRow(row)
            values = (sensor.sensor_id, getattr(sensor.modality, "value", sensor.modality), sensor.manufacturer or "", sensor.model or "", "" if sensor.horizontal_fov_deg is None else str(sensor.horizontal_fov_deg), "" if sensor.vertical_fov_deg is None else str(sensor.vertical_fov_deg))
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def _add(self) -> None:
        dialog = SensorDialog(parent=self)
        if dialog.exec() and dialog.result_sensor is not None:
            if dialog.result_sensor.sensor_id in self.sensors:
                QMessageBox.warning(self, "Duplicate Sensor", "That Sensor ID already exists.")
                return
            self.sensors[dialog.result_sensor.sensor_id] = dialog.result_sensor
            self._refresh()

    def _edit(self) -> None:
        sensor_id = self._selected_id()
        if sensor_id is None:
            return
        dialog = SensorDialog(sensor=self.sensors[sensor_id], parent=self)
        if dialog.exec() and dialog.result_sensor is not None:
            replacement = dialog.result_sensor
            if replacement.sensor_id != sensor_id and replacement.sensor_id in self.sensors:
                QMessageBox.warning(self, "Duplicate Sensor", "That Sensor ID already exists.")
                return
            if replacement.sensor_id != sensor_id and sensor_id in self.referenced_ids:
                QMessageBox.warning(self, "Sensor is in use", "A referenced Sensor cannot be renamed here.")
                return
            was_referenced = sensor_id in self.referenced_ids
            del self.sensors[sensor_id]
            self.sensors[replacement.sensor_id] = replacement
            self.referenced_ids.discard(sensor_id)
            if was_referenced:
                self.referenced_ids.add(replacement.sensor_id)
            self._refresh()

    def _delete(self) -> None:
        sensor_id = self._selected_id()
        if sensor_id is None:
            return
        if sensor_id in self.referenced_ids:
            QMessageBox.warning(self, "Sensor is in use", "This Sensor is referenced by a Viewpoint and cannot be deleted.")
            return
        if QMessageBox.question(self, "Delete Sensor", f"Delete Sensor {sensor_id!r}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            del self.sensors[sensor_id]
            self._refresh()
