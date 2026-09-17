"""Canonical ObservationEvent editor and manager dialogs."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

try:
    from PySide6.QtWidgets import (
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

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.observation import ObservationEvent
from rivelero.gui.components import CollapsibleSection


class ObservationEventDialog(QDialog):
    """Create or edit one event without resolving geometry defaults."""

    def __init__(self, *, configuration: ViewpointConfiguration, event: ObservationEvent | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Observation Event" if event else "Add Observation Event")
        self.setMinimumWidth(600)
        self.result_event: ObservationEvent | None = None
        self._fields: dict[str, QLineEdit] = {}
        outer = QVBoxLayout(self)
        form = QFormLayout()
        for name, label in (
            ("event_id", "Event ID"),
            ("viewpoint_id", "Viewpoint ID"),
            ("timestamp", "Timestamp (ISO 8601, optional)"),
            ("sequence_id", "Sequence ID"),
            ("sequence_index", "Sequence index"),
            ("image_id", "Image ID"),
            ("source", "Source"),
        ):
            field = QLineEdit()
            field.setPlaceholderText("Blank = missing / no override")
            self._fields[name] = field
            form.addRow(label, field)
        self._viewpoint_ids = tuple(viewpoint.viewpoint_id for viewpoint in configuration.viewpoints)
        self._viewpoint_field = self._fields["viewpoint_id"]
        self._viewpoint_field.setPlaceholderText("Existing Viewpoint ID")
        outer.addLayout(form)

        advanced = CollapsibleSection("Observation geometry overrides", expanded=False)
        override_form = QFormLayout()
        for name, label in (
            ("observer_height_m", "Observer height (m)"),
            ("heading_deg", "Heading (degrees)"),
            ("pitch_deg", "Pitch (degrees)"),
            ("roll_deg", "Roll (degrees)"),
            ("horizontal_fov_deg", "Horizontal FOV (degrees)"),
            ("vertical_fov_deg", "Vertical FOV (degrees)"),
        ):
            field = QLineEdit()
            field.setPlaceholderText("Blank = no event-level override")
            self._fields[name] = field
            override_form.addRow(label, field)
        self.acquisition_edit = QPlainTextEdit()
        self.acquisition_edit.setPlaceholderText("{}")
        self.acquisition_edit.setMaximumHeight(75)
        override_form.addRow("Acquisition conditions JSON", self.acquisition_edit)
        self.uncertainty_edit = QPlainTextEdit()
        self.uncertainty_edit.setPlaceholderText("{}")
        self.uncertainty_edit.setMaximumHeight(75)
        override_form.addRow("Metadata uncertainty JSON", self.uncertainty_edit)
        self.extra_edit = QPlainTextEdit()
        self.extra_edit.setPlaceholderText("{}")
        self.extra_edit.setMaximumHeight(75)
        override_form.addRow("Extra metadata JSON", self.extra_edit)
        advanced.content_layout.addLayout(override_form)
        outer.addWidget(advanced)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._populate(event)

    def _populate(self, event: ObservationEvent | None) -> None:
        if event is None:
            self.acquisition_edit.setPlainText("{}")
            self.uncertainty_edit.setPlainText("{}")
            self.extra_edit.setPlainText("{}")
            return
        for name, field in self._fields.items():
            value = getattr(event, name)
            field.setText("" if value is None else (value.isoformat() if isinstance(value, datetime) else str(value)))
        self.acquisition_edit.setPlainText(json.dumps(event.acquisition_conditions, indent=2))
        self.uncertainty_edit.setPlainText(json.dumps(event.metadata_uncertainty, indent=2))
        self.extra_edit.setPlainText(json.dumps(event.extra_metadata, indent=2))

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
        try:
            value = json.loads(editor.toPlainText().strip() or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must contain valid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object.")
        return value

    def _accept(self) -> None:
        try:
            event_id = self._fields["event_id"].text().strip()
            viewpoint_id = self._fields["viewpoint_id"].text().strip()
            if viewpoint_id not in self._viewpoint_ids:
                raise ValueError("Viewpoint ID must reference an existing Viewpoint.")
            timestamp_text = self._fields["timestamp"].text().strip()
            timestamp = None if not timestamp_text else datetime.fromisoformat(timestamp_text)
            sequence_text = self._fields["sequence_index"].text().strip()
            sequence_index = None if not sequence_text else int(sequence_text)
            self.result_event = ObservationEvent(
                event_id=event_id,
                viewpoint_id=viewpoint_id,
                timestamp=timestamp,
                sequence_id=self._fields["sequence_id"].text().strip() or None,
                sequence_index=sequence_index,
                image_id=self._fields["image_id"].text().strip() or None,
                source=self._fields["source"].text().strip() or None,
                observer_height_m=self._optional_float(self._fields["observer_height_m"], "Observer height"),
                heading_deg=self._optional_float(self._fields["heading_deg"], "Heading"),
                pitch_deg=self._optional_float(self._fields["pitch_deg"], "Pitch"),
                roll_deg=self._optional_float(self._fields["roll_deg"], "Roll"),
                horizontal_fov_deg=self._optional_float(self._fields["horizontal_fov_deg"], "Horizontal FOV"),
                vertical_fov_deg=self._optional_float(self._fields["vertical_fov_deg"], "Vertical FOV"),
                acquisition_conditions=self._json_object(self.acquisition_edit, "Acquisition conditions"),
                metadata_uncertainty=self._json_object(self.uncertainty_edit, "Metadata uncertainty"),
                extra_metadata=self._json_object(self.extra_edit, "Extra metadata"),
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid Observation Event", str(exc))
            return
        self.accept()


class ObservationEventManagerDialog(QDialog):
    """Manage ordered canonical ObservationEvents."""

    def __init__(self, *, configuration: ViewpointConfiguration, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Observation Events")
        self.setMinimumSize(900, 480)
        self.configuration = configuration
        self.events = list(configuration.observation_events)
        outer = QVBoxLayout(self)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(("Event ID", "Viewpoint ID", "Timestamp", "Sequence ID", "Index", "Image ID", "Source"))
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

    def _refresh(self) -> None:
        self.table.setRowCount(0)
        for row, event in enumerate(self.events):
            self.table.insertRow(row)
            values = (event.event_id, event.viewpoint_id, "" if event.timestamp is None else event.timestamp.isoformat(), event.sequence_id or "", "" if event.sequence_index is None else str(event.sequence_index), event.image_id or "", event.source or "")
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def _selected_index(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _add(self) -> None:
        dialog = ObservationEventDialog(configuration=self.configuration, parent=self)
        if dialog.exec() and dialog.result_event is not None:
            if any(event.event_id == dialog.result_event.event_id for event in self.events):
                QMessageBox.warning(self, "Duplicate Event", "That Event ID already exists.")
                return
            self.events.append(dialog.result_event)
            self._refresh()

    def _edit(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        dialog = ObservationEventDialog(configuration=self.configuration, event=self.events[index], parent=self)
        if dialog.exec() and dialog.result_event is not None:
            replacement = dialog.result_event
            if any(event.event_id == replacement.event_id and position != index for position, event in enumerate(self.events)):
                QMessageBox.warning(self, "Duplicate Event", "That Event ID already exists.")
                return
            self.events[index] = replacement
            self._refresh()

    def _delete(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        event = self.events[index]
        if QMessageBox.question(self, "Delete Observation Event", f"Delete event {event.event_id!r}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            del self.events[index]
            self._refresh()
