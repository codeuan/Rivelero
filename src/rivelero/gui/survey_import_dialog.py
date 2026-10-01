"""Interactive survey-import dialog for the Rivelero GUI.

This dialog converts external CSV survey tables into canonical Rivelero
objects through rivelero.gui.survey_import.

Responsibilities
----------------
The dialog:

- selects Viewpoint, Sensor and ObservationEvent CSV files;
- previews source columns and rows;
- allows explicit Viewpoint field mapping;
- defines source and target CRS;
- validates the import before accepting it;
- returns a SurveyImportResult.

The dialog does not modify ApplicationState directly. The caller decides
whether an accepted SurveyImportResult becomes the active survey.

Provider-specific acquisition (Google Street View, Mapillary, etc.) does not
belong here. Future provider adapters should ultimately produce the same
canonical Rivelero objects.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

except ImportError:
    try:
        from PyQt6.QtCore import Qt, pyqtSignal as Signal
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QStackedWidget,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )

    except ImportError as exc:
        raise ImportError(
            "Rivelero GUI requires PySide6 or PyQt6."
        ) from exc


from rasterio.crs import CRS

from rivelero.gui.components import (
    BadgeType,
    CollapsibleSection,
    ContentCard,
    LabeledValue,
    PageHeader,
    StatusBadge,
    make_primary_button,
    make_secondary_button,
)
from rivelero.gui.survey_import import (
    ObservationEventColumnMapping,
    SensorColumnMapping,
    SurveyImportError,
    SurveyImportOptions,
    SurveyImportResult,
    ViewpointColumnMapping,
    detect_crs_column,
    import_survey_csv,
)
from rivelero.gui.theme import (
    SPACING,
    refresh_style,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Used only when the Viewpoints file does not declare its CRS.
DEFAULT_SOURCE_CRS = "EPSG:4326"


NONE_COLUMN = "— Not mapped —"

PREVIEW_ROWS = 8

VIEWPOINT_REQUIRED_FIELDS = (
    "viewpoint_id",
)

VIEWPOINT_MAPPING_FIELDS = (
    (
        "viewpoint_id",
        "Viewpoint ID",
        True,
    ),
    (
        "x",
        "X / Easting",
        False,
    ),
    (
        "y",
        "Y / Northing",
        False,
    ),
    (
        "longitude",
        "Longitude",
        False,
    ),
    (
        "latitude",
        "Latitude",
        False,
    ),
    (
        "z",
        "Elevation / Z",
        False,
    ),
    (
        "observer_height_m",
        "Observer height",
        False,
    ),
    (
        "heading_deg",
        "Heading",
        False,
    ),
    (
        "pitch_deg",
        "Pitch",
        False,
    ),
    (
        "roll_deg",
        "Roll",
        False,
    ),
    (
        "horizontal_fov_deg",
        "Horizontal FOV",
        False,
    ),
    (
        "vertical_fov_deg",
        "Vertical FOV",
        False,
    ),
    (
        "sensor_id",
        "Sensor ID",
        False,
    ),
    (
        "platform",
        "Platform",
        False,
    ),
    (
        "source",
        "Source",
        False,
    ),
    (
        "source_id",
        "Source ID",
        False,
    ),
    (
        "position_uncertainty_m",
        "Position uncertainty",
        False,
    ),
    (
        "orientation_uncertainty_deg",
        "Orientation uncertainty",
        False,
    ),
)


# ---------------------------------------------------------------------------
# File selector
# ---------------------------------------------------------------------------


class CsvFileSelector(QWidget):
    """Reusable CSV path selector."""

    path_changed = Signal(
        object
    )

    def __init__(
        self,
        label: str,
        *,
        required: bool,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self.required = bool(
            required
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            SPACING.xs
        )

        title_row = QHBoxLayout()

        title = QLabel(
            label
        )

        title_font = title.font()
        title_font.setBold(
            True
        )
        title.setFont(
            title_font
        )

        title_row.addWidget(
            title
        )

        if required:
            required_label = QLabel(
                "Required"
            )
            required_label.setProperty(
                "badge",
                "advanced",
            )
            title_row.addWidget(
                required_label
            )

        title_row.addStretch(
            1
        )

        layout.addLayout(
            title_row
        )

        row = QHBoxLayout()

        row.setSpacing(
            SPACING.sm
        )

        self.path_edit = QLineEdit()

        self.path_edit.setPlaceholderText(
            (
                "Select CSV file…"
                if required
                else "Optional CSV file…"
            )
        )

        self.browse_button = QPushButton(
            "Browse"
        )

        self.clear_button = QPushButton(
            "Clear"
        )

        self.clear_button.setEnabled(
            False
        )

        row.addWidget(
            self.path_edit,
            1,
        )

        row.addWidget(
            self.browse_button
        )

        row.addWidget(
            self.clear_button
        )

        layout.addLayout(
            row
        )

        self.browse_button.clicked.connect(
            self._browse
        )

        self.clear_button.clicked.connect(
            self.clear
        )

        self.path_edit.textChanged.connect(
            self._on_text_changed
        )

    @property
    def path(
        self,
    ) -> Path | None:

        text = (
            self.path_edit
            .text()
            .strip()
        )

        if not text:
            return None

        return Path(
            text
        ).expanduser()

    def set_path(
        self,
        path: str | Path | None,
    ) -> None:

        self.path_edit.setText(
            ""
            if path is None
            else str(
                Path(
                    path
                ).expanduser()
            )
        )

    def clear(
        self,
    ) -> None:

        self.path_edit.clear()

    def _browse(
        self,
    ) -> None:

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if filename:
            self.set_path(
                filename
            )

    def _on_text_changed(
        self,
        _text: str,
    ) -> None:

        self.clear_button.setEnabled(
            self.path is not None
        )

        self.path_changed.emit(
            self.path
        )


# ---------------------------------------------------------------------------
# Field-mapping row
# ---------------------------------------------------------------------------


class MappingRow(QWidget):
    """One canonical-field to source-column mapping."""

    mapping_changed = Signal()

    def __init__(
        self,
        canonical_name: str,
        label: str,
        *,
        required: bool,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self.canonical_name = (
            canonical_name
        )

        self.required = bool(
            required
        )

        layout = QHBoxLayout(
            self
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            SPACING.md
        )

        label_text = (
            f"{label} *"
            if required
            else label
        )

        self.label = QLabel(
            label_text
        )

        self.label.setMinimumWidth(
            180
        )

        self.combo = QComboBox()

        self.combo.setMinimumWidth(
            220
        )

        layout.addWidget(
            self.label
        )

        layout.addWidget(
            self.combo,
            1,
        )

        self.combo.currentIndexChanged.connect(
            self.mapping_changed.emit
        )

    @property
    def source_column(
        self,
    ) -> str | None:

        value = (
            self.combo.currentData()
        )

        if value is None:
            return None

        return str(
            value
        )

    def set_columns(
        self,
        columns: list[str],
    ) -> None:
        """Populate source columns and automatically suggest a match."""

        current = (
            self.source_column
        )

        self.combo.blockSignals(
            True
        )

        try:
            self.combo.clear()

            self.combo.addItem(
                NONE_COLUMN,
                None,
            )

            for column in columns:
                self.combo.addItem(
                    column,
                    column,
                )

            suggestion = (
                _suggest_column(
                    self.canonical_name,
                    columns,
                )
            )

            target = (
                current
                if current in columns
                else suggestion
            )

            if target is not None:

                index = self.combo.findData(
                    target
                )

                if index >= 0:
                    self.combo.setCurrentIndex(
                        index
                    )

        finally:
            self.combo.blockSignals(
                False
            )


# ---------------------------------------------------------------------------
# Import dialog
# ---------------------------------------------------------------------------


class SurveyImportDialog(QDialog):
    """Interactive generic CSV survey importer."""

    import_completed = Signal(
        object
    )

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self._result: (
            SurveyImportResult | None
        ) = None

        self._viewpoint_columns: list[
            str
        ] = []

        self._viewpoint_preview: list[
            dict[str, str]
        ] = []

        self._validation_current = False

        # Source CRS taken from the Viewpoints file's 'crs' column (so a
        # later file can replace it), and the note explaining it.
        self._auto_source_crs: str | None = None
        self._crs_note: tuple[str, str] | None = None

        self._configure_dialog()

        self._build_interface()

        self._connect_signals()

        self._update_navigation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def import_result(
        self,
    ) -> SurveyImportResult | None:
        """Return accepted SurveyImportResult, if available."""

        return self._result

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _configure_dialog(
        self,
    ) -> None:

        self.setWindowTitle(
            "Import survey — Rivelero"
        )

        self.setMinimumSize(
            980,
            720,
        )

        self.resize(
            1120,
            820,
        )

        self.setModal(
            True
        )

    def _build_interface(
        self,
    ) -> None:

        root = QVBoxLayout(
            self
        )

        root.setContentsMargins(
            SPACING.xl,
            SPACING.xl,
            SPACING.xl,
            SPACING.xl,
        )

        root.setSpacing(
            SPACING.lg
        )

        self.header = PageHeader(
            "Import survey",
            (
                "Convert external observation tables into canonical "
                "Rivelero Viewpoints, Sensors and ObservationEvents."
            ),
        )

        root.addWidget(
            self.header
        )

        root.addWidget(
            self._build_step_indicator()
        )

        self.stack = QStackedWidget()

        self.files_page = (
            self._build_files_page()
        )

        self.coordinates_page = (
            self._build_coordinates_page()
        )

        self.mapping_page = (
            self._build_mapping_page()
        )

        self.validation_page = (
            self._build_validation_page()
        )

        for page in (
            self.files_page,
            self.coordinates_page,
            self.mapping_page,
            self.validation_page,
        ):
            self.stack.addWidget(
                page
            )

        root.addWidget(
            self.stack,
            1,
        )

        root.addWidget(
            self._build_navigation()
        )

    def _build_step_indicator(
        self,
    ) -> QWidget:

        widget = QFrame()

        widget.setProperty(
            "subtleCard",
            True,
        )

        layout = QHBoxLayout(
            widget
        )

        layout.setContentsMargins(
            SPACING.md,
            SPACING.sm,
            SPACING.md,
            SPACING.sm,
        )

        layout.setSpacing(
            SPACING.md
        )

        self.step_labels = []

        labels = (
            "1  Files",
            "2  Coordinates",
            "3  Field mapping",
            "4  Validate",
        )

        for index, text in enumerate(
            labels
        ):

            label = QLabel(
                text
            )

            label.setProperty(
                "stepActive",
                index == 0,
            )

            self.step_labels.append(
                label
            )

            layout.addWidget(
                label
            )

            if index < len(labels) - 1:

                separator = QLabel(
                    "›"
                )

                separator.setProperty(
                    "secondaryText",
                    True,
                )

                layout.addWidget(
                    separator
                )

        layout.addStretch(
            1
        )

        return widget

    # ------------------------------------------------------------------
    # Step 1 — files
    # ------------------------------------------------------------------

    def _build_files_page(
        self,
    ) -> QWidget:

        page = QWidget()

        layout = QVBoxLayout(
            page
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            SPACING.lg
        )

        card = ContentCard(
            title="Source files",
            description=(
                "A Viewpoint table is required. Sensor and "
                "ObservationEvent tables are optional."
            ),
        )

        self.viewpoints_selector = (
            CsvFileSelector(
                "Viewpoints",
                required=True,
            )
        )

        self.sensors_selector = (
            CsvFileSelector(
                "Sensors",
                required=False,
            )
        )

        self.events_selector = (
            CsvFileSelector(
                "Observation events",
                required=False,
            )
        )

        card.add_widget(
            self.viewpoints_selector
        )

        card.add_widget(
            self.sensors_selector
        )

        card.add_widget(
            self.events_selector
        )

        layout.addWidget(
            card
        )

        self.preview_card = ContentCard(
            title="Viewpoint preview",
            description=(
                "The preview shows source data exactly as supplied. "
                "No visibility defaults are applied during import."
            ),
        )

        self.preview_status = QLabel(
            "Select a Viewpoint CSV to preview its contents."
        )

        self.preview_status.setProperty(
            "secondaryText",
            True,
        )

        self.preview_table = QTableWidget()

        self.preview_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        self.preview_table.setAlternatingRowColors(
            True
        )

        self.preview_card.add_widget(
            self.preview_status
        )

        self.preview_card.add_widget(
            self.preview_table
        )

        layout.addWidget(
            self.preview_card,
            1,
        )

        return page

    # ------------------------------------------------------------------
    # Step 2 — coordinates
    # ------------------------------------------------------------------

    def _build_coordinates_page(
        self,
    ) -> QWidget:

        page = QWidget()

        layout = QVBoxLayout(
            page
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            SPACING.lg
        )

        card = ContentCard(
            title="Coordinate reference system",
            description=(
                "Tell Rivelero how source coordinates are defined. "
                "X/Y values are never silently assumed to be "
                "longitude/latitude."
            ),
        )

        form = QFormLayout()

        form.setHorizontalSpacing(
            SPACING.xl
        )

        form.setVerticalSpacing(
            SPACING.md
        )

        self.source_crs_edit = QLineEdit(
            DEFAULT_SOURCE_CRS
        )

        self.source_crs_edit.setPlaceholderText(
            "e.g. EPSG:4326"
        )

        form.addRow(
            "Source CRS",
            self.source_crs_edit,
        )

        self.transform_coordinates = (
            QCheckBox(
                "Transform imported coordinates"
            )
        )

        form.addRow(
            "",
            self.transform_coordinates,
        )

        self.target_crs_edit = QLineEdit()

        self.target_crs_edit.setPlaceholderText(
            "e.g. EPSG:32633"
        )

        self.target_crs_edit.setEnabled(
            False
        )

        form.addRow(
            "Target CRS",
            self.target_crs_edit,
        )

        card.add_layout(
            form
        )

        self.crs_status = QLabel()

        self.crs_status.setWordWrap(
            True
        )

        card.add_widget(
            self.crs_status
        )

        layout.addWidget(
            card
        )

        explanation = ContentCard(
            title="Why this matters",
            description=(
                "Visibility distance, observer height and terrain geometry "
                "depend on spatial units. For analysis, a suitable projected "
                "CRS is generally preferable to geographic degrees."
            ),
            subtle=True,
        )

        layout.addWidget(
            explanation
        )

        layout.addStretch(
            1
        )

        return page

    # ------------------------------------------------------------------
    # Step 3 — mapping
    # ------------------------------------------------------------------

    def _build_mapping_page(
        self,
    ) -> QWidget:

        page = QWidget()

        outer = QVBoxLayout(
            page
        )

        outer.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        outer.setSpacing(
            SPACING.lg
        )

        card = ContentCard(
            title="Viewpoint field mapping",
            description=(
                "Map source columns to canonical Rivelero fields. "
                "Obvious names are suggested automatically, but the "
                "mapping remains explicit and editable."
            ),
        )

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        mapping_content = QWidget()

        mapping_layout = QVBoxLayout(
            mapping_content
        )

        mapping_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        mapping_layout.setSpacing(
            SPACING.sm
        )

        self.mapping_rows: dict[
            str,
            MappingRow,
        ] = {}

        basic_fields = {
            "viewpoint_id",
            "x",
            "y",
            "longitude",
            "latitude",
            "observer_height_m",
            "heading_deg",
            "horizontal_fov_deg",
            "sensor_id",
            "platform",
            "source",
            "source_id",
        }

        self.advanced_mapping = (
            CollapsibleSection(
                "Advanced fields",
                description=(
                    "Optional orientation, elevation and uncertainty "
                    "metadata."
                ),
                badge=BadgeType.ADVANCED,
                expanded=False,
            )
        )

        for (
            canonical_name,
            label,
            required,
        ) in VIEWPOINT_MAPPING_FIELDS:

            row = MappingRow(
                canonical_name,
                label,
                required=required,
            )

            self.mapping_rows[
                canonical_name
            ] = row

            row.mapping_changed.connect(
                self._invalidate_validation
            )

            if canonical_name in basic_fields:
                mapping_layout.addWidget(
                    row
                )

            else:
                self.advanced_mapping.content_layout.addWidget(
                    row
                )

        mapping_layout.addWidget(
            self.advanced_mapping
        )

        mapping_layout.addStretch(
            1
        )

        scroll.setWidget(
            mapping_content
        )

        card.add_widget(
            scroll
        )

        self.mapping_status = QLabel()

        self.mapping_status.setWordWrap(
            True
        )

        card.add_widget(
            self.mapping_status
        )

        outer.addWidget(
            card,
            1,
        )

        return page

    # ------------------------------------------------------------------
    # Step 4 — validation
    # ------------------------------------------------------------------

    def _build_validation_page(
        self,
    ) -> QWidget:

        page = QWidget()

        layout = QVBoxLayout(
            page
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            SPACING.lg
        )

        project_card = ContentCard(
            title="Survey identity",
            description=(
                "These values identify the imported "
                "ViewpointConfiguration inside Rivelero."
            ),
        )

        form = QFormLayout()

        self.configuration_id_edit = QLineEdit(
            "imported_survey"
        )

        self.configuration_name_edit = QLineEdit(
            "Imported survey"
        )

        form.addRow(
            "Configuration ID",
            self.configuration_id_edit,
        )

        form.addRow(
            "Name",
            self.configuration_name_edit,
        )

        project_card.add_layout(
            form
        )

        layout.addWidget(
            project_card
        )

        self.validation_card = ContentCard(
            title="Validation",
            description=(
                "Rivelero will construct canonical objects without "
                "installing them into the current project."
            ),
        )

        self.validation_message = QLabel(
            "Run validation to inspect the import."
        )

        self.validation_message.setWordWrap(
            True
        )

        self.validation_card.add_widget(
            self.validation_message
        )

        self.validation_details = QWidget()

        details = QGridLayout(
            self.validation_details
        )

        details.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        details.setHorizontalSpacing(
            SPACING.xxl
        )

        self.valid_viewpoints = LabeledValue(
            "Viewpoints",
            "—",
            vertical=True,
        )

        self.valid_sensors = LabeledValue(
            "Sensors",
            "—",
            vertical=True,
        )

        self.valid_events = LabeledValue(
            "Observation events",
            "—",
            vertical=True,
        )

        self.valid_warnings = LabeledValue(
            "Warnings",
            "—",
            vertical=True,
        )

        self.valid_errors = LabeledValue(
            "Errors",
            "—",
            vertical=True,
        )

        details.addWidget(
            self.valid_viewpoints,
            0,
            0,
        )

        details.addWidget(
            self.valid_sensors,
            0,
            1,
        )

        details.addWidget(
            self.valid_events,
            0,
            2,
        )

        details.addWidget(
            self.valid_warnings,
            1,
            0,
        )

        details.addWidget(
            self.valid_errors,
            1,
            1,
        )

        self.validation_details.hide()

        self.validation_card.add_widget(
            self.validation_details
        )

        self.validate_button = (
            make_secondary_button(
                "Run validation"
            )
        )

        self.validation_card.add_widget(
            self.validate_button
        )

        layout.addWidget(
            self.validation_card
        )

        self.issue_card = ContentCard(
            title="Issues",
            description=(
                "Import errors indicate invalid rows. Missing optional "
                "metadata such as heading is not itself an import error."
            ),
            subtle=True,
        )

        self.issue_text = QLabel(
            "No validation results yet."
        )

        self.issue_text.setWordWrap(
            True
        )

        self.issue_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.issue_card.add_widget(
            self.issue_text
        )

        layout.addWidget(
            self.issue_card
        )

        layout.addStretch(
            1
        )

        return page

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _build_navigation(
        self,
    ) -> QWidget:

        frame = QFrame()

        frame.setProperty(
            "subtleCard",
            True,
        )

        layout = QHBoxLayout(
            frame
        )

        layout.setContentsMargins(
            SPACING.md,
            SPACING.sm,
            SPACING.md,
            SPACING.sm,
        )

        self.back_button = QPushButton(
            "Back"
        )

        self.cancel_button = QPushButton(
            "Cancel"
        )

        self.next_button = (
            make_primary_button(
                "Next"
            )
        )

        self.import_button = (
            make_primary_button(
                "Import survey"
            )
        )

        self.import_button.hide()

        layout.addWidget(
            self.cancel_button
        )

        layout.addWidget(
            self.back_button
        )

        layout.addStretch(
            1
        )

        layout.addWidget(
            self.next_button
        )

        layout.addWidget(
            self.import_button
        )

        return frame

    def _connect_signals(
        self,
    ) -> None:

        self.viewpoints_selector.path_changed.connect(
            self._on_viewpoints_path_changed
        )

        self.sensors_selector.path_changed.connect(
            self._invalidate_validation
        )

        self.events_selector.path_changed.connect(
            self._invalidate_validation
        )

        self.source_crs_edit.textChanged.connect(
            self._invalidate_validation
        )

        self.target_crs_edit.textChanged.connect(
            self._invalidate_validation
        )

        self.transform_coordinates.toggled.connect(
            self._on_transform_toggled
        )

        self.configuration_id_edit.textChanged.connect(
            self._invalidate_validation
        )

        self.configuration_name_edit.textChanged.connect(
            self._invalidate_validation
        )

        self.validate_button.clicked.connect(
            self.validate_import
        )

        self.back_button.clicked.connect(
            self.previous_step
        )

        self.next_button.clicked.connect(
            self.next_step
        )

        self.cancel_button.clicked.connect(
            self.reject
        )

        self.import_button.clicked.connect(
            self._accept_import
        )

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------

    @property
    def current_step(
        self,
    ) -> int:

        return self.stack.currentIndex()

    def next_step(
        self,
    ) -> None:

        if not self._validate_current_step():
            return

        if self.current_step < (
            self.stack.count() - 1
        ):
            self.stack.setCurrentIndex(
                self.current_step + 1
            )

        self._update_navigation()

    def previous_step(
        self,
    ) -> None:

        if self.current_step > 0:
            self.stack.setCurrentIndex(
                self.current_step - 1
            )

        self._update_navigation()

    def _update_navigation(
        self,
    ) -> None:

        step = self.current_step

        self.back_button.setEnabled(
            step > 0
        )

        final = (
            step
            == self.stack.count() - 1
        )

        self.next_button.setVisible(
            not final
        )

        self.import_button.setVisible(
            final
        )

        self.import_button.setEnabled(
            final
            and self._validation_current
            and self._result is not None
        )

        for index, label in enumerate(
            self.step_labels
        ):
            label.setProperty(
                "stepActive",
                index == step,
            )

            label.style().unpolish(
                label
            )

            label.style().polish(
                label
            )

    def _validate_current_step(
        self,
    ) -> bool:

        if self.current_step == 0:
            return self._validate_files_step()

        if self.current_step == 1:
            return self._validate_crs_step()

        if self.current_step == 2:
            return self._validate_mapping_step()

        return True

    # ------------------------------------------------------------------
    # Files / preview
    # ------------------------------------------------------------------

    def _on_viewpoints_path_changed(
        self,
        path: Path | None,
    ) -> None:

        self._invalidate_validation()

        if path is None:
            self._clear_preview()
            self._set_mapping_columns(
                []
            )
            return

        try:
            columns, rows = _read_preview(
                path,
                max_rows=PREVIEW_ROWS,
            )

        except Exception as exc:
            self._clear_preview()

            self.preview_status.setText(
                f"Unable to preview file: {exc}"
            )

            self._set_mapping_columns(
                []
            )

            return

        self._viewpoint_columns = columns
        self._viewpoint_preview = rows

        self._populate_preview(
            columns,
            rows,
        )

        self._apply_crs_column(path)

        self._set_mapping_columns(
            columns
        )

    def _populate_preview(
        self,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> None:

        self.preview_table.clear()

        self.preview_table.setColumnCount(
            len(columns)
        )

        self.preview_table.setHorizontalHeaderLabels(
            columns
        )

        self.preview_table.setRowCount(
            len(rows)
        )

        for row_index, row in enumerate(
            rows
        ):
            for column_index, column in enumerate(
                columns
            ):
                self.preview_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(
                        str(
                            row.get(
                                column,
                                "",
                            )
                        )
                    ),
                )

        self.preview_status.setText(
            f"{len(columns)} columns detected. "
            f"Showing first {len(rows)} rows."
        )

        self.preview_table.resizeColumnsToContents()

    def _clear_preview(
        self,
    ) -> None:

        self._viewpoint_columns = []
        self._viewpoint_preview = []

        self.preview_table.clear()
        self.preview_table.setRowCount(
            0
        )
        self.preview_table.setColumnCount(
            0
        )

        self.preview_status.setText(
            "Select a Viewpoint CSV to preview its contents."
        )

    # ------------------------------------------------------------------
    # CRS
    # ------------------------------------------------------------------

    def _on_transform_toggled(
        self,
        enabled: bool,
    ) -> None:

        self.target_crs_edit.setEnabled(
            enabled
        )

        self._invalidate_validation()

    def _validate_crs_step(
        self,
    ) -> bool:

        source_text = (
            self.source_crs_edit
            .text()
            .strip()
        )

        try:
            source_crs = (
                CRS.from_user_input(
                    source_text
                )
            )

        except Exception:
            self.crs_status.setText(
                "Source CRS is not valid."
            )
            return False

        if self._crs_note is not None and self._crs_note[0] == "warning":
            # A contradiction with the file is shown, never silently kept.
            self._show_crs_status(self._crs_note[1], warning=True)

        if self.transform_coordinates.isChecked():

            target_text = (
                self.target_crs_edit
                .text()
                .strip()
            )

            try:
                target_crs = (
                    CRS.from_user_input(
                        target_text
                    )
                )

            except Exception:
                self.crs_status.setText(
                    "Target CRS is not valid."
                )
                return False

            status = (
                f"Coordinates will be transformed from "
                f"{source_crs.to_string()} to "
                f"{target_crs.to_string()}."
            )

        else:
            status = (
                f"Coordinates will remain in "
                f"{source_crs.to_string()}."
            )

        self._show_crs_status(status)
        return True

    def _apply_crs_column(self, path: Path) -> None:
        """Use the file's 'crs' column as Source CRS when unambiguous.

        The file's own declaration replaces the default (or a value this
        dialog set earlier from another file), never a CRS the user typed;
        a disagreement or an ambiguous column is reported instead.
        """

        try:
            detection = detect_crs_column(path)
        except Exception:
            detection = None
        self._crs_note = None
        if detection is None or not detection.present:
            self._show_crs_status("")
            return
        if detection.crs is None:
            self._crs_note = ("warning", detection.message or "")
        else:
            current = self.source_crs_edit.text().strip()
            if current in ("", DEFAULT_SOURCE_CRS, self._auto_source_crs or ""):
                self._auto_source_crs = detection.crs
                self.source_crs_edit.setText(detection.crs)
                self._crs_note = (
                    "info",
                    f"Source CRS set from the file's 'crs' column ({detection.crs}).",
                )
            elif not _same_crs(current, detection.crs):
                self._crs_note = (
                    "warning",
                    f"The file's 'crs' column says {detection.crs}, but the Source "
                    f"CRS is {current}. Check which one the coordinates use.",
                )
        self._show_crs_status("")

    def _show_crs_status(self, text: str, *, warning: bool = False) -> None:
        note = self._crs_note
        parts = [note[1]] if note is not None else []
        if text and (not parts or text != parts[0]):
            parts.append(text)
        self.crs_status.setText(" ".join(parts))
        is_warning = warning or (note is not None and note[0] == "warning")
        self.crs_status.setProperty("statusWarning", is_warning)
        refresh_style(self.crs_status)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _set_mapping_columns(
        self,
        columns: list[str],
    ) -> None:

        for row in (
            self.mapping_rows.values()
        ):
            row.set_columns(
                columns
            )

        self._update_mapping_status()

    def _validate_mapping_step(
        self,
    ) -> bool:

        viewpoint_id = (
            self.mapping_rows[
                "viewpoint_id"
            ].source_column
        )

        if viewpoint_id is None:
            self.mapping_status.setText(
                "Viewpoint ID must be mapped."
            )
            return False

        x = self.mapping_rows[
            "x"
        ].source_column

        y = self.mapping_rows[
            "y"
        ].source_column

        longitude = self.mapping_rows[
            "longitude"
        ].source_column

        latitude = self.mapping_rows[
            "latitude"
        ].source_column

        xy_complete = (
            x is not None
            and y is not None
        )

        lonlat_complete = (
            longitude is not None
            and latitude is not None
        )

        if not (
            xy_complete
            or lonlat_complete
        ):
            self.mapping_status.setText(
                "Map either X and Y, or Longitude and Latitude."
            )
            return False

        if (
            xy_complete
            and lonlat_complete
        ):
            self.mapping_status.setText(
                "Both coordinate pairs are mapped. Rivelero will "
                "prefer X/Y according to the generic importer."
            )

        else:
            self.mapping_status.setText(
                "Required field mapping is valid."
            )

        return True

    def _update_mapping_status(
        self,
    ) -> None:

        if not self._viewpoint_columns:
            self.mapping_status.setText(
                "Select a Viewpoint CSV before mapping fields."
            )
            return

        self._validate_mapping_step()

    def _build_viewpoint_mapping(
        self,
    ) -> ViewpointColumnMapping:

        defaults = (
            ViewpointColumnMapping()
        )

        values: dict[
            str,
            str,
        ] = {}

        for (
            canonical_name,
            _label,
            _required,
        ) in VIEWPOINT_MAPPING_FIELDS:

            mapped = (
                self.mapping_rows[
                    canonical_name
                ].source_column
            )

            if mapped is not None:
                values[
                    canonical_name
                ] = mapped

            else:
                # An intentionally unmapped optional field must point to
                # a column name that cannot accidentally match a real
                # source column. The importer treats absent columns as None.
                values[
                    canonical_name
                ] = (
                    f"__rivelero_unmapped_"
                    f"{canonical_name}__"
                )

        # Required coordinate logic is checked separately before this runs.
        return ViewpointColumnMapping(
            **{
                field_name: values.get(
                    field_name,
                    getattr(
                        defaults,
                        field_name,
                    ),
                )
                for (
                    field_name,
                    _label,
                    _required,
                ) in VIEWPOINT_MAPPING_FIELDS
            }
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_import(
        self,
    ) -> bool:
        """Run canonical import without modifying application state."""

        self._result = None
        self._validation_current = False

        if not self._validate_files_step():
            self._show_validation_failure(
                "Source files are not valid."
            )
            return False

        if not self._validate_crs_step():
            self._show_validation_failure(
                "Coordinate reference system is not valid."
            )
            return False

        if not self._validate_mapping_step():
            self._show_validation_failure(
                "Viewpoint field mapping is not valid."
            )
            return False

        configuration_id = (
            self.configuration_id_edit
            .text()
            .strip()
        )

        configuration_name = (
            self.configuration_name_edit
            .text()
            .strip()
        )

        if not configuration_id:
            self._show_validation_failure(
                "Configuration ID cannot be empty."
            )
            return False

        if not configuration_name:
            self._show_validation_failure(
                "Survey name cannot be empty."
            )
            return False

        source_crs = (
            self.source_crs_edit
            .text()
            .strip()
        )

        target_crs = None

        if self.transform_coordinates.isChecked():
            target_crs = (
                self.target_crs_edit
                .text()
                .strip()
            )

        try:
            result = import_survey_csv(
                viewpoints_path=(
                    self.viewpoints_selector.path
                ),
                sensors_path=(
                    self.sensors_selector.path
                ),
                observation_events_path=(
                    self.events_selector.path
                ),
                configuration_id=configuration_id,
                configuration_name=configuration_name,
                options=SurveyImportOptions(
                    source_crs=source_crs,
                    target_crs=target_crs,
                    strict=False,
                    allow_duplicate_coordinates=True,
                ),
                viewpoint_mapping=(
                    self._build_viewpoint_mapping()
                ),
                sensor_mapping=(
                    SensorColumnMapping()
                ),
                event_mapping=(
                    ObservationEventColumnMapping()
                ),
            )

        except Exception as exc:
            self._show_validation_failure(
                str(exc)
            )
            return False

        self._result = result
        self._validation_current = True

        report = result.report

        self.valid_viewpoints.set_value(
            f"{report.viewpoints_imported:,}"
        )

        self.valid_sensors.set_value(
            f"{report.sensors_imported:,}"
        )

        self.valid_events.set_value(
            f"{report.events_imported:,}"
        )

        self.valid_warnings.set_value(
            f"{len(report.warnings):,}"
        )

        self.valid_errors.set_value(
            f"{len(report.errors):,}"
        )

        self.validation_details.show()

        if report.has_errors:
            self.validation_message.setText(
                "Import is possible, but one or more source rows "
                "were rejected. Review the issues before continuing."
            )

        elif report.warnings:
            self.validation_message.setText(
                "Validation succeeded with warnings."
            )

        else:
            self.validation_message.setText(
                "Validation succeeded. The survey is ready to import."
            )

        issues = []

        if report.warnings:
            issues.append(
                "Warnings:\n"
                + "\n".join(
                    f"• {warning}"
                    for warning in report.warnings
                )
            )

        if report.errors:
            issues.append(
                "Errors:\n"
                + "\n".join(
                    f"• {error}"
                    for error in report.errors
                )
            )

        self.issue_text.setText(
            (
                "\n\n".join(
                    issues
                )
                if issues
                else "No import issues detected."
            )
        )

        self._update_navigation()

        return True

    def _show_validation_failure(
        self,
        message: str,
    ) -> None:

        self.validation_message.setText(
            f"Validation failed: {message}"
        )

        self.validation_details.hide()

        self.issue_text.setText(
            message
        )

        self._result = None
        self._validation_current = False

        self._update_navigation()

    def _invalidate_validation(
        self,
        *_args,
    ) -> None:

        self._result = None
        self._validation_current = False

        if hasattr(
            self,
            "validation_message",
        ):
            self.validation_message.setText(
                "Import settings changed. Run validation again."
            )

        if hasattr(
            self,
            "validation_details",
        ):
            self.validation_details.hide()

        if hasattr(
            self,
            "import_button",
        ):
            self._update_navigation()

    # ------------------------------------------------------------------
    # File validation
    # ------------------------------------------------------------------

    def _validate_files_step(
        self,
    ) -> bool:

        viewpoints = (
            self.viewpoints_selector.path
        )

        if viewpoints is None:
            self.preview_status.setText(
                "A Viewpoint CSV is required."
            )
            return False

        if not viewpoints.exists():
            self.preview_status.setText(
                "The selected Viewpoint CSV does not exist."
            )
            return False

        if not viewpoints.is_file():
            self.preview_status.setText(
                "The selected Viewpoint path is not a file."
            )
            return False

        for selector, label in (
            (
                self.sensors_selector,
                "Sensor",
            ),
            (
                self.events_selector,
                "ObservationEvent",
            ),
        ):

            path = selector.path

            if path is None:
                continue

            if not path.exists():
                QMessageBox.warning(
                    self,
                    "Invalid source file",
                    f"The selected {label} CSV does not exist.",
                )
                return False

            if not path.is_file():
                QMessageBox.warning(
                    self,
                    "Invalid source file",
                    f"The selected {label} path is not a file.",
                )
                return False

        if not self._viewpoint_columns:
            self.preview_status.setText(
                "Rivelero could not detect columns in the "
                "Viewpoint CSV."
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _accept_import(
        self,
    ) -> None:

        if (
            not self._validation_current
            or self._result is None
        ):
            if not self.validate_import():
                return

        result = self._result

        if result is None:
            return

        if result.report.has_errors:

            answer = QMessageBox.question(
                self,
                "Import contains rejected rows",
                (
                    f"{len(result.report.errors):,} import errors "
                    "were reported. Valid rows can still be imported.\n\n"
                    "Continue?"
                ),
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                ),
                QMessageBox.StandardButton.No,
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                return

        self.import_completed.emit(
            result
        )

        self.accept()


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------


def _read_preview(
    path: str | Path,
    *,
    max_rows: int,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    """Read a small UTF-8 CSV preview."""

    source = Path(
        path
    ).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    with source.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "CSV contains no header."
            )

        columns = [
            str(column).strip()
            for column in reader.fieldnames
            if column is not None
        ]

        rows = []

        for index, row in enumerate(
            reader
        ):

            if index >= max_rows:
                break

            rows.append(
                {
                    str(key).strip(): (
                        ""
                        if value is None
                        else str(value)
                    )
                    for key, value in row.items()
                    if key is not None
                }
            )

    return (
        columns,
        rows,
    )


def _suggest_column(
    canonical_name: str,
    columns: list[str],
) -> str | None:
    """Suggest a source column without hiding the mapping from the user."""

    normalized = {
        column.strip().lower(): column
        for column in columns
    }

    canonical = (
        canonical_name
        .strip()
        .lower()
    )

    if canonical in normalized:
        return normalized[
            canonical
        ]

    aliases = {
        "viewpoint_id": (
            "id",
            "point_id",
            "view_id",
            "camera_id",
        ),
        "x": (
            "easting",
            "coord_x",
        ),
        "y": (
            "northing",
            "coord_y",
        ),
        "longitude": (
            "lon",
            "lng",
        ),
        "latitude": (
            "lat",
        ),
        "z": (
            "elevation",
            "altitude",
            "height_z",
        ),
        "observer_height_m": (
            "observer_height",
            "camera_height",
            "height_m",
        ),
        "heading_deg": (
            "heading",
            "bearing",
            "azimuth",
            "compass_angle",
        ),
        "horizontal_fov_deg": (
            "horizontal_fov",
            "hfov",
            "fov",
        ),
        "vertical_fov_deg": (
            "vertical_fov",
            "vfov",
        ),
        "sensor_id": (
            "sensor",
            "camera_model_id",
        ),
        "source_id": (
            "source_record_id",
            "external_id",
        ),
    }

    for alias in aliases.get(
        canonical,
        (),
    ):
        if alias in normalized:
            return normalized[
                alias
            ]

    return None


def _same_crs(first: str, second: str) -> bool:
    try:
        return CRS.from_user_input(first) == CRS.from_user_input(second)
    except Exception:
        return False
