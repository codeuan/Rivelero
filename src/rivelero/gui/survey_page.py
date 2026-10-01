"""Survey workflow page for the Rivelero GUI.

The Survey page presents and manages the sampling structure of a Rivelero
project.

Scientific source of truth
--------------------------
The page never creates a parallel GUI-specific Viewpoint model. It reads
canonical Rivelero objects from ApplicationState:

    Sensor
    Viewpoint
    ObservationEvent
    ViewpointConfiguration

This first implementation provides:

- survey overview;
- canonical Viewpoint table;
- metadata-completeness summary;
- selected-Viewpoint inspector;
- Sensor summary;
- ObservationEvent summary;
- signals for import/add/edit/delete workflows.

Actual file parsing and modal editors belong in dedicated import/dialog
modules rather than this page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
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
            QAbstractItemView,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QPushButton,
            QScrollArea,
            QSizePolicy,
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


from rivelero.gui.application_state import (
    ApplicationState,
)
from rivelero.gui.components import (
    ActionBar,
    BadgeType,
    CollapsibleSection,
    ContentCard,
    EmptyState,
    LabeledValue,
    PageHeader,
    StatusBadge,
    make_danger_button,
    make_link_button,
    make_primary_button,
    make_secondary_button,
)
from rivelero.gui.survey_map import SurveyMapWidget
from rivelero.gui.theme import (
    SPACING,
)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetadataCompleteness:
    """Completeness summary for one Viewpoint attribute."""

    label: str
    available: int
    total: int

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0

        return self.available / self.total

    @property
    def percent(self) -> float:
        return self.fraction * 100.0


def _present(
    value: Any,
) -> bool:
    """Return whether a metadata value is meaningfully present."""

    if value is None:
        return False

    if isinstance(
        value,
        str,
    ):
        return bool(
            value.strip()
        )

    return True


def _first_attribute(
    obj: Any,
    names: Iterable[str],
) -> Any:
    """Return the first existing attribute from a list of candidate names."""

    for name in names:
        if hasattr(
            obj,
            name,
        ):
            return getattr(
                obj,
                name,
            )

    return None


def _viewpoint_timestamp(
    viewpoint: Any,
) -> Any:
    """Return timestamp-like Viewpoint metadata when available.

    Temporal information primarily belongs to ObservationEvent, but this
    helper allows source Viewpoints containing acquisition metadata to be
    represented without assuming one exact optional attribute name.
    """

    return _first_attribute(
        viewpoint,
        (
            "timestamp",
            "acquired_at",
            "acquisition_time",
            "datetime",
        ),
    )


def calculate_metadata_completeness(
    viewpoints: Iterable[Any],
) -> tuple[MetadataCompleteness, ...]:
    """Calculate high-level metadata completeness for Viewpoints."""

    items = list(
        viewpoints
    )

    total = len(
        items
    )

    definitions = (
        (
            "Coordinates",
            lambda vp: (
                _present(
                    getattr(
                        vp,
                        "x",
                        None,
                    )
                )
                and _present(
                    getattr(
                        vp,
                        "y",
                        None,
                    )
                )
            ),
        ),
        (
            "Heading",
            lambda vp: _present(
                getattr(
                    vp,
                    "heading_deg",
                    None,
                )
            ),
        ),
        (
            "Horizontal FOV",
            lambda vp: _present(
                getattr(
                    vp,
                    "horizontal_fov_deg",
                    None,
                )
            ),
        ),
        (
            "Observer height",
            lambda vp: _present(
                getattr(
                    vp,
                    "observer_height_m",
                    None,
                )
            ),
        ),
        (
            "Sensor",
            lambda vp: _present(
                getattr(
                    vp,
                    "sensor_id",
                    None,
                )
            ),
        ),
        (
            "Timestamp",
            lambda vp: _present(
                _viewpoint_timestamp(
                    vp
                )
            ),
        ),
    )

    summaries = []

    for label, predicate in definitions:

        available = sum(
            1
            for viewpoint in items
            if predicate(
                viewpoint
            )
        )

        summaries.append(
            MetadataCompleteness(
                label=label,
                available=available,
                total=total,
            )
        )

    return tuple(
        summaries
    )


# ---------------------------------------------------------------------------
# Metadata completeness widget
# ---------------------------------------------------------------------------


class MetadataCompletenessWidget(QWidget):
    """Compact survey metadata-completeness summary."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self._layout = QVBoxLayout(
            self
        )

        self._layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self._layout.setSpacing(
            SPACING.sm
        )

        self._rows: list[
            QWidget
        ] = []

        self.set_summaries(
            ()
        )

    def set_summaries(
        self,
        summaries: Iterable[
            MetadataCompleteness
        ],
    ) -> None:
        """Replace displayed completeness values."""

        self._clear_rows()

        summaries = tuple(
            summaries
        )

        if not summaries:

            empty = QLabel(
                "Import or add Viewpoints to inspect metadata completeness."
            )

            empty.setWordWrap(
                True
            )

            empty.setProperty(
                "secondaryText",
                True,
            )

            self._layout.addWidget(
                empty
            )

            self._rows.append(
                empty
            )

            return

        for summary in summaries:

            row = QWidget()

            row_layout = QHBoxLayout(
                row
            )

            row_layout.setContentsMargins(
                0,
                0,
                0,
                0,
            )

            row_layout.setSpacing(
                SPACING.sm
            )

            label = QLabel(
                summary.label
            )

            value = QLabel(
                f"{summary.percent:.0f}%"
            )

            value.setToolTip(
                f"{summary.available:,} of "
                f"{summary.total:,} Viewpoints"
            )

            if summary.percent >= 99.5:
                value.setProperty(
                    "statusSuccess",
                    True,
                )

            elif summary.percent < 50.0:
                value.setProperty(
                    "statusWarning",
                    True,
                )

            row_layout.addWidget(
                label
            )

            row_layout.addStretch(
                1
            )

            row_layout.addWidget(
                value
            )

            self._layout.addWidget(
                row
            )

            self._rows.append(
                row
            )

    def _clear_rows(
        self,
    ) -> None:

        while self._layout.count():

            item = self._layout.takeAt(
                0
            )

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        self._rows.clear()


# ---------------------------------------------------------------------------
# Survey page
# ---------------------------------------------------------------------------


class SurveyPage(QWidget):
    """Survey definition and inspection page."""

    import_viewpoints_requested = Signal()
    add_viewpoint_requested = Signal()

    edit_viewpoint_requested = Signal(
        str
    )

    delete_viewpoint_requested = Signal(
        str
    )

    sensors_requested = Signal()
    observation_events_requested = Signal()

    survey_metadata_requested = Signal()

    acquisition_requested = Signal()

    viewpoint_selected = Signal(
        object
    )

    continue_requested = Signal()

    def __init__(
        self,
        state: ApplicationState,
        *,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        if not isinstance(
            state,
            ApplicationState,
        ):
            raise TypeError(
                "state must be an ApplicationState."
            )

        self.state = state

        self._displayed_viewpoints: list[
            Any
        ] = []

        self._selected_viewpoint_id: (
            str | None
        ) = None

        self._build_interface()

        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_interface(
        self,
    ) -> None:

        root = QVBoxLayout(
            self
        )

        root.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        root.setSpacing(
            0
        )

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        content = QWidget()

        self.content_layout = QVBoxLayout(
            content
        )

        self.content_layout.setContentsMargins(
            SPACING.page,
            SPACING.xxxl,
            SPACING.page,
            SPACING.xxxl,
        )

        self.content_layout.setSpacing(
            SPACING.xl
        )

        self._build_header()
        self._build_summary_card()
        self._build_viewpoints_card()
        self._build_inspection_row()
        self._build_optional_sections()
        self._build_action_bar()

        self.content_layout.addStretch(
            1
        )

        scroll.setWidget(
            content
        )

        root.addWidget(
            scroll
        )

    def _build_header(
        self,
    ) -> None:

        self.header = PageHeader(
            "Survey",
            (
                "Define the observation opportunities that constitute "
                "your survey. Existing observations, manually defined "
                "Viewpoints and future candidate Viewpoints use the same "
                "canonical Rivelero survey model."
            ),
        )

        self.content_layout.addWidget(
            self.header
        )

    def _build_summary_card(
        self,
    ) -> None:

        self.summary_card = ContentCard(
            title="Survey definition",
            description=(
                "Rivelero separates survey sampling from visibility "
                "assumptions. Define what was observed here; visibility "
                "is configured later."
            ),
        )

        grid = QGridLayout()

        grid.setHorizontalSpacing(
            SPACING.xxl
        )

        grid.setVerticalSpacing(
            SPACING.md
        )

        self.configuration_name = (
            LabeledValue(
                "Configuration",
                "Not defined",
                vertical=True,
            )
        )

        self.viewpoint_count = (
            LabeledValue(
                "Viewpoints",
                "0",
                vertical=True,
            )
        )

        self.event_count = (
            LabeledValue(
                "Observation events",
                "0",
                vertical=True,
            )
        )

        self.sensor_count = (
            LabeledValue(
                "Sensors",
                "0",
                vertical=True,
            )
        )

        grid.addWidget(
            self.configuration_name,
            0,
            0,
        )

        grid.addWidget(
            self.viewpoint_count,
            0,
            1,
        )

        grid.addWidget(
            self.event_count,
            0,
            2,
        )

        grid.addWidget(
            self.sensor_count,
            0,
            3,
        )

        grid.setColumnStretch(
            0,
            2,
        )

        for column in (
            1,
            2,
            3,
        ):
            grid.setColumnStretch(
                column,
                1,
            )

        self.summary_card.add_layout(
            grid
        )

        self.content_layout.addWidget(
            self.summary_card
        )

    def _build_viewpoints_card(
        self,
    ) -> None:

        self.viewpoints_card = ContentCard(
            title="Viewpoints",
            description=(
                "Each row represents one canonical Viewpoint. "
                "Coordinates may repeat because distinct observations or "
                "events can occur at the same physical location."
            ),
        )

        toolbar = QHBoxLayout()

        toolbar.setSpacing(
            SPACING.sm
        )

        self.import_button = (
            make_primary_button(
                "Import Viewpoints"
            )
        )

        self.add_button = (
            make_secondary_button(
                "Add manually"
            )
        )

        self.acquire_button = (
            make_secondary_button(
                "Acquire from platform"
            )
        )

        self.acquire_button.setEnabled(
            False
        )

        self.acquire_button.setToolTip(
            "Direct acquisition from imagery platforms will be "
            "implemented in a future import workflow."
        )

        toolbar.addWidget(
            self.import_button
        )

        toolbar.addWidget(
            self.add_button
        )

        toolbar.addWidget(
            self.acquire_button
        )

        toolbar.addStretch(
            1
        )

        self.search_input = QLineEdit()

        self.search_input.setPlaceholderText(
            "Search Viewpoints…"
        )

        self.search_input.setClearButtonEnabled(
            True
        )

        self.search_input.setMaximumWidth(
            300
        )

        toolbar.addWidget(
            self.search_input
        )

        self.viewpoints_card.add_layout(
            toolbar
        )

        self.empty_state = EmptyState(
            "No Viewpoints yet",
            (
                "Import a survey file or add a Viewpoint manually. "
                "Rivelero will preserve canonical metadata and provenance "
                "rather than reducing observations to coordinate pairs."
            ),
            action_text="Import Viewpoints",
        )

        self.viewpoints_card.add_widget(
            self.empty_state
        )

        self.table = QTableWidget(
            0,
            9,
        )

        self.table.setHorizontalHeaderLabels(
            (
                "ID",
                "X",
                "Y",
                "Heading",
                "FOV",
                "Height",
                "Sensor",
                "Source",
                "Status",
            )
        )

        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.table.setAlternatingRowColors(
            True
        )

        self.table.setSortingEnabled(
            False
        )

        self.table.verticalHeader().setVisible(
            False
        )

        header = (
            self.table.horizontalHeader()
        )

        header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )

        header.setSectionResizeMode(
            7,
            QHeaderView.ResizeMode.Stretch,
        )

        self.table.setMinimumHeight(
            300
        )

        self.view_stack = QStackedWidget()

        self.table_page = QWidget()
        table_layout = QVBoxLayout(self.table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)
        table_layout.addWidget(self.table)
        self.view_stack.addWidget(self.table_page)

        self.map_widget = SurveyMapWidget()
        self.map_widget.viewpoint_selected.connect(
            self._on_map_viewpoint_selected
        )
        self.map_widget.selection_changed.connect(
            self._on_map_selection_changed
        )

        self.map_page = QWidget()
        map_layout = QVBoxLayout(self.map_page)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(0)
        map_layout.addWidget(self.map_widget)
        self.view_stack.addWidget(self.map_page)

        self.viewpoints_card.add_widget(self.view_stack)

        self.view_toggle_layout = QHBoxLayout()
        self.table_toggle = make_secondary_button("Table")
        self.map_toggle = make_secondary_button("Map")
        self.table_toggle.setCheckable(True)
        self.map_toggle.setCheckable(True)
        self.table_toggle.setChecked(True)
        self.map_toggle.setChecked(False)
        self.table_toggle.clicked.connect(
            lambda: self._set_view_mode("table")
        )
        self.map_toggle.clicked.connect(
            lambda: self._set_view_mode("map")
        )
        self.view_toggle_layout.addWidget(self.table_toggle)
        self.view_toggle_layout.addWidget(self.map_toggle)
        self.view_toggle_layout.addStretch(1)
        self.viewpoints_card.add_layout(self.view_toggle_layout)

        selection_actions = QHBoxLayout()

        self.edit_button = (
            make_secondary_button(
                "Edit selected"
            )
        )

        self.delete_button = (
            make_danger_button(
                "Delete selected"
            )
        )

        self.edit_button.setEnabled(
            False
        )

        self.delete_button.setEnabled(
            False
        )

        selection_actions.addWidget(
            self.edit_button
        )

        selection_actions.addWidget(
            self.delete_button
        )

        selection_actions.addStretch(
            1
        )

        self.viewpoints_card.add_layout(
            selection_actions
        )

        self.content_layout.addWidget(
            self.viewpoints_card
        )

        self.import_button.clicked.connect(
            self.import_viewpoints_requested.emit
        )

        self.empty_state.action_requested.connect(
            self.import_viewpoints_requested.emit
        )

        self.add_button.clicked.connect(
            self.add_viewpoint_requested.emit
        )

        self.acquire_button.clicked.connect(
            self.acquisition_requested.emit
        )

        self.search_input.textChanged.connect(
            self._apply_filter
        )

        self.table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )

        self.edit_button.clicked.connect(
            self._emit_edit_selected
        )

        self.delete_button.clicked.connect(
            self._emit_delete_selected
        )

    def _build_inspection_row(
        self,
    ) -> None:

        row = QHBoxLayout()

        row.setSpacing(
            SPACING.xl
        )

        self.metadata_card = ContentCard(
            title="Metadata completeness",
            description=(
                "Completeness describes source metadata only. "
                "Missing values are handled later through explicit "
                "VisibilityConfiguration policies."
            ),
        )

        self.metadata_widget = (
            MetadataCompletenessWidget()
        )

        self.metadata_card.add_widget(
            self.metadata_widget
        )

        row.addWidget(
            self.metadata_card,
            1,
        )

        self.selected_card = ContentCard(
            title="Selected Viewpoint",
            description=(
                "Inspect the canonical metadata that will be passed "
                "to the visibility engine."
            ),
        )

        self.selected_empty = QLabel(
            "Select a Viewpoint in the table to inspect it."
        )

        self.selected_empty.setWordWrap(
            True
        )

        self.selected_empty.setProperty(
            "secondaryText",
            True,
        )

        self.selected_card.add_widget(
            self.selected_empty
        )

        self.selected_details = QWidget()

        selected_grid = QGridLayout(
            self.selected_details
        )

        selected_grid.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        selected_grid.setHorizontalSpacing(
            SPACING.xl
        )

        selected_grid.setVerticalSpacing(
            SPACING.sm
        )

        self.selected_id = LabeledValue(
            "ID"
        )

        self.selected_coordinates = (
            LabeledValue(
                "Coordinates"
            )
        )

        self.selected_crs = LabeledValue(
            "CRS"
        )

        self.selected_heading = (
            LabeledValue(
                "Heading"
            )
        )

        self.selected_fov = LabeledValue(
            "Horizontal FOV"
        )

        self.selected_height = (
            LabeledValue(
                "Observer height"
            )
        )

        self.selected_sensor = (
            LabeledValue(
                "Sensor"
            )
        )

        self.selected_source = (
            LabeledValue(
                "Source"
            )
        )

        selected_grid.addWidget(
            self.selected_id,
            0,
            0,
        )

        selected_grid.addWidget(
            self.selected_coordinates,
            0,
            1,
        )

        selected_grid.addWidget(
            self.selected_crs,
            1,
            0,
        )

        selected_grid.addWidget(
            self.selected_heading,
            1,
            1,
        )

        selected_grid.addWidget(
            self.selected_fov,
            2,
            0,
        )

        selected_grid.addWidget(
            self.selected_height,
            2,
            1,
        )

        selected_grid.addWidget(
            self.selected_sensor,
            3,
            0,
        )

        selected_grid.addWidget(
            self.selected_source,
            3,
            1,
        )

        self.selected_details.hide()

        self.selected_card.add_widget(
            self.selected_details
        )

        row.addWidget(
            self.selected_card,
            1,
        )

        self.content_layout.addLayout(
            row
        )

    def _build_optional_sections(
        self,
    ) -> None:

        self.sensors_section = (
            CollapsibleSection(
                "Sensors",
                description=(
                    "Sensor metadata can provide modality, field of view "
                    "and other acquisition properties used when Viewpoint "
                    "metadata are incomplete."
                ),
                expanded=False,
            )
        )

        sensor_summary_row = QHBoxLayout()

        self.sensor_summary = QLabel(
            "No Sensors defined."
        )

        self.sensor_summary.setProperty(
            "secondaryText",
            True,
        )

        self.manage_sensors_button = (
            make_secondary_button(
                "Manage Sensors"
            )
        )

        sensor_summary_row.addWidget(
            self.sensor_summary,
            1,
        )

        sensor_summary_row.addWidget(
            self.manage_sensors_button
        )

        self.sensors_section.content_layout.addLayout(
            sensor_summary_row
        )

        self.content_layout.addWidget(
            self.sensors_section
        )

        self.events_section = (
            CollapsibleSection(
                "Observation events",
                description=(
                    "ObservationEvents preserve temporal order, repeated "
                    "visits and event-level metadata overrides without "
                    "duplicating physical Viewpoints."
                ),
                expanded=False,
            )
        )

        event_summary_row = QHBoxLayout()

        self.event_summary = QLabel(
            "No ObservationEvents defined."
        )

        self.event_summary.setProperty(
            "secondaryText",
            True,
        )

        self.manage_events_button = (
            make_secondary_button(
                "Manage ObservationEvents"
            )
        )

        event_summary_row.addWidget(
            self.event_summary,
            1,
        )

        event_summary_row.addWidget(
            self.manage_events_button
        )

        self.events_section.content_layout.addLayout(
            event_summary_row
        )

        self.content_layout.addWidget(
            self.events_section
        )

        self.advanced_section = (
            CollapsibleSection(
                "Advanced survey metadata",
                description=(
                    "Inspect provenance and source-specific metadata "
                    "without exposing those details in the basic workflow."
                ),
                badge=BadgeType.ADVANCED,
                expanded=False,
            )
        )

        advanced_text = QLabel(
            "Source-specific provenance and metadata tools will be "
            "connected here as survey import support expands."
        )

        advanced_text.setWordWrap(
            True
        )

        advanced_text.setProperty(
            "secondaryText",
            True,
        )

        self.advanced_metadata_button = (
            make_link_button(
                "Open survey metadata"
            )
        )

        self.advanced_metadata_button.setEnabled(
            False
        )

        self.advanced_section.content_layout.addWidget(
            advanced_text
        )

        self.advanced_section.content_layout.addWidget(
            self.advanced_metadata_button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.content_layout.addWidget(
            self.advanced_section
        )

        self.manage_sensors_button.clicked.connect(
            self.sensors_requested.emit
        )

        self.manage_events_button.clicked.connect(
            self.observation_events_requested.emit
        )

        self.advanced_metadata_button.clicked.connect(
            self.survey_metadata_requested.emit
        )

    def _build_action_bar(
        self,
    ) -> None:

        self.action_bar = ActionBar()

        self.clear_selection_button = (
            make_secondary_button(
                "Clear selection"
            )
        )

        self.clear_selection_button.setEnabled(
            False
        )

        self.continue_button = (
            make_primary_button(
                "Continue to World"
            )
        )

        self.continue_button.setEnabled(
            False
        )

        self.action_bar.add_secondary_action(
            self.clear_selection_button
        )

        self.action_bar.add_primary_action(
            self.continue_button
        )

        self.content_layout.addWidget(
            self.action_bar
        )

        self.clear_selection_button.clicked.connect(
            self.clear_selection
        )

        self.continue_button.clicked.connect(
            self.continue_requested.emit
        )

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh_from_state(
        self,
    ) -> None:
        """Refresh the complete page from ApplicationState."""

        configuration = (
            self.state
            .survey
            .viewpoint_configuration
        )

        viewpoints = (
            []
            if configuration is None
            else list(
                configuration.viewpoints
            )
        )

        if configuration is None:
            self.configuration_name.set_value(
                "Not defined"
            )

        else:
            configuration_name = (
                _first_attribute(
                    configuration,
                    (
                        "name",
                        "configuration_id",
                    ),
                )
            )

            self.configuration_name.set_value(
                configuration_name
            )

        self.viewpoint_count.set_value(
            f"{len(viewpoints):,}"
        )

        self.event_count.set_value(
            f"{self.state.survey.n_observation_events:,}"
        )

        self.sensor_count.set_value(
            f"{self.state.survey.n_sensors:,}"
        )

        self.metadata_widget.set_summaries(
            calculate_metadata_completeness(
                viewpoints
            )
        )

        self.sensor_summary.setText(
            (
                "No Sensors defined."
                if self.state.survey.n_sensors == 0
                else (
                    f"{self.state.survey.n_sensors:,} "
                    "Sensor"
                    + (
                        ""
                        if self.state.survey.n_sensors == 1
                        else "s"
                    )
                    + " available."
                )
            )
        )

        event_count = (
            self.state
            .survey
            .n_observation_events
        )

        self.event_summary.setText(
            (
                "No ObservationEvents defined."
                if event_count == 0
                else (
                    f"{event_count:,} ObservationEvent"
                    + (
                        ""
                        if event_count == 1
                        else "s"
                    )
                    + " defined."
                )
            )
        )

        self._populate_table(
            viewpoints
        )

        self.map_widget.set_configuration(configuration)

        self.empty_state.setVisible(
            len(viewpoints) == 0
        )

        self.view_stack.setVisible(
            len(viewpoints) > 0
        )

        self.table.setVisible(
            len(viewpoints) > 0
        )

        self.edit_button.setVisible(
            len(viewpoints) > 0
        )

        self.delete_button.setVisible(
            len(viewpoints) > 0
        )

        self.continue_button.setEnabled(
            self.state.survey_ready
        )

        self._restore_selection()
        self._set_view_mode(
            "table",
            suppress_update=True,
        )

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate_table(
        self,
        viewpoints: list[Any],
    ) -> None:

        self.table.setUpdatesEnabled(
            False
        )

        try:
            self.table.clearContents()

            self.table.setRowCount(
                len(viewpoints)
            )

            for row, viewpoint in enumerate(
                viewpoints
            ):
                values = self._viewpoint_row(
                    viewpoint
                )

                for column, value in enumerate(
                    values
                ):
                    item = QTableWidgetItem(
                        value
                    )

                    if column in (
                        1,
                        2,
                        3,
                        4,
                        5,
                    ):
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight
                            | Qt.AlignmentFlag.AlignVCenter
                        )

                    self.table.setItem(
                        row,
                        column,
                        item,
                    )

            self._displayed_viewpoints = list(
                viewpoints
            )

        finally:
            self.table.setUpdatesEnabled(
                True
            )

        self._apply_filter(
            self.search_input.text()
        )

    def _viewpoint_row(
        self,
        viewpoint: Any,
    ) -> tuple[str, ...]:

        heading = getattr(
            viewpoint,
            "heading_deg",
            None,
        )

        fov = getattr(
            viewpoint,
            "horizontal_fov_deg",
            None,
        )

        height = getattr(
            viewpoint,
            "observer_height_m",
            None,
        )

        sensor = getattr(
            viewpoint,
            "sensor_id",
            None,
        )

        source = getattr(
            viewpoint,
            "source",
            None,
        )

        status = self._metadata_status(
            viewpoint
        )

        return (
            str(
                viewpoint.viewpoint_id
            ),
            _format_number(
                viewpoint.x,
                decimals=2,
            ),
            _format_number(
                viewpoint.y,
                decimals=2,
            ),
            _format_angle(
                heading
            ),
            _format_angle(
                fov
            ),
            _format_distance(
                height
            ),
            _format_optional(
                sensor
            ),
            _format_optional(
                source
            ),
            status,
        )

    @staticmethod
    def _metadata_status(
        viewpoint: Any,
    ) -> str:
        """Return a concise source-metadata status."""

        missing = []

        if not _present(
            getattr(
                viewpoint,
                "heading_deg",
                None,
            )
        ):
            missing.append(
                "heading"
            )

        if not _present(
            getattr(
                viewpoint,
                "horizontal_fov_deg",
                None,
            )
        ):
            missing.append(
                "FOV"
            )

        if not _present(
            getattr(
                viewpoint,
                "observer_height_m",
                None,
            )
        ):
            missing.append(
                "height"
            )

        if not missing:
            return "Complete"

        if len(missing) == 3:
            return "Basic"

        return (
            "Missing "
            + ", ".join(
                missing
            )
        )

    def _apply_filter(
        self,
        text: str,
    ) -> None:

        query = str(
            text
        ).strip().lower()

        for row in range(
            self.table.rowCount()
        ):
            if not query:
                self.table.setRowHidden(
                    row,
                    False,
                )
                continue

            row_text = " ".join(
                (
                    self.table.item(
                        row,
                        column,
                    ).text()
                    if self.table.item(
                        row,
                        column,
                    )
                    is not None
                    else ""
                )
                for column in range(
                    self.table.columnCount()
                )
            ).lower()

            self.table.setRowHidden(
                row,
                query not in row_text,
            )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _set_view_mode(
        self,
        mode: str,
        *,
        suppress_update: bool = False,
    ) -> None:
        if mode == "table":
            self.view_stack.setCurrentIndex(0)
            self.table_toggle.setChecked(True)
            self.map_toggle.setChecked(False)
        else:
            self.view_stack.setCurrentIndex(1)
            self.table_toggle.setChecked(False)
            self.map_toggle.setChecked(True)

        if not suppress_update:
            self.map_widget.set_selected_viewpoint_id(
                self.state.selection.viewpoint_id,
                emit_signal=False,
            )

    def _on_map_selection_changed(
        self,
        viewpoint_id: str,
    ) -> None:
        if not viewpoint_id:
            self.table.clearSelection()
            return
        for row, viewpoint in enumerate(self._displayed_viewpoints):
            if viewpoint.viewpoint_id == viewpoint_id:
                self.table.clearSelection()
                self.table.selectRow(row)
                return

    def _on_map_viewpoint_selected(
        self,
        viewpoint_id: str,
    ) -> None:
        if not viewpoint_id:
            return
        self.state.select_viewpoint(viewpoint_id)
        self._selected_viewpoint_id = viewpoint_id
        self._show_selected_viewpoint(
            self.state.survey.viewpoint_configuration.get_viewpoint(viewpoint_id)
        )
        self.viewpoint_selected.emit(
            self.state.survey.viewpoint_configuration.get_viewpoint(viewpoint_id)
        )
        self._on_map_selection_changed(viewpoint_id)

    def _on_table_selection_changed(
        self,
    ) -> None:

        selected_rows = (
            self.table
            .selectionModel()
            .selectedRows()
        )

        if not selected_rows:
            self._selected_viewpoint_id = None

            self.edit_button.setEnabled(
                False
            )

            self.delete_button.setEnabled(
                False
            )

            self.clear_selection_button.setEnabled(
                False
            )

            self._show_selected_viewpoint(
                None
            )

            return

        row = selected_rows[
            0
        ].row()

        if (
            row < 0
            or row >= len(
                self._displayed_viewpoints
            )
        ):
            return

        viewpoint = (
            self._displayed_viewpoints[
                row
            ]
        )

        self._selected_viewpoint_id = (
            viewpoint.viewpoint_id
        )

        self.edit_button.setEnabled(
            True
        )

        self.delete_button.setEnabled(
            True
        )

        self.clear_selection_button.setEnabled(
            True
        )

        self.state.select_viewpoint(
            viewpoint.viewpoint_id
        )

        self.map_widget.set_selected_viewpoint_id(
            viewpoint.viewpoint_id,
            emit_signal=False,
        )

        self._show_selected_viewpoint(
            viewpoint
        )

        self.viewpoint_selected.emit(
            viewpoint
        )

    def _restore_selection(
        self,
    ) -> None:

        desired = (
            self.state
            .selection
            .viewpoint_id
        )

        if desired is None:
            self._show_selected_viewpoint(
                None
            )
            return

        for row, viewpoint in enumerate(
            self._displayed_viewpoints
        ):
            if (
                viewpoint.viewpoint_id
                == desired
            ):
                self.table.selectRow(
                    row
                )
                return

        self._show_selected_viewpoint(
            None
        )

    def clear_selection(
        self,
    ) -> None:
        """Clear Viewpoint selection."""

        self.table.clearSelection()

        self.state.clear_selection()

        self._selected_viewpoint_id = None

        self._show_selected_viewpoint(
            None
        )

        self.edit_button.setEnabled(
            False
        )

        self.delete_button.setEnabled(
            False
        )

        self.clear_selection_button.setEnabled(
            False
        )

    def _show_selected_viewpoint(
        self,
        viewpoint: Any | None,
    ) -> None:

        if viewpoint is None:

            self.selected_empty.show()

            self.selected_details.hide()

            return

        self.selected_empty.hide()

        self.selected_details.show()

        self.selected_id.set_value(
            viewpoint.viewpoint_id
        )

        self.selected_coordinates.set_value(
            (
                f"{_format_number(viewpoint.x, decimals=3)}, "
                f"{_format_number(viewpoint.y, decimals=3)}"
            )
        )

        crs = getattr(
            viewpoint,
            "crs",
            None,
        )

        self.selected_crs.set_value(
            (
                crs.to_string()
                if (
                    crs is not None
                    and hasattr(
                        crs,
                        "to_string",
                    )
                )
                else _format_optional(
                    crs
                )
            )
        )

        self.selected_heading.set_value(
            _format_angle(
                getattr(
                    viewpoint,
                    "heading_deg",
                    None,
                )
            )
        )

        self.selected_fov.set_value(
            _format_angle(
                getattr(
                    viewpoint,
                    "horizontal_fov_deg",
                    None,
                )
            )
        )

        self.selected_height.set_value(
            _format_distance(
                getattr(
                    viewpoint,
                    "observer_height_m",
                    None,
                )
            )
        )

        self.selected_sensor.set_value(
            _format_optional(
                getattr(
                    viewpoint,
                    "sensor_id",
                    None,
                )
            )
        )

        self.selected_source.set_value(
            _format_optional(
                getattr(
                    viewpoint,
                    "source",
                    None,
                )
            )
        )

    def _emit_edit_selected(
        self,
    ) -> None:

        if self._selected_viewpoint_id is not None:
            self.edit_viewpoint_requested.emit(
                self._selected_viewpoint_id
            )

    def _emit_delete_selected(
        self,
    ) -> None:

        if self._selected_viewpoint_id is not None:
            self.delete_viewpoint_requested.emit(
                self._selected_viewpoint_id
            )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_optional(
    value: Any,
) -> str:

    if not _present(
        value
    ):
        return "—"

    return str(
        value
    )


def _format_number(
    value: Any,
    *,
    decimals: int = 2,
) -> str:

    if value is None:
        return "—"

    try:
        number = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    return f"{number:.{decimals}f}"


def _format_angle(
    value: Any,
) -> str:

    if value is None:
        return "—"

    try:
        return (
            f"{float(value):.1f}°"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )


def _format_distance(
    value: Any,
) -> str:

    if value is None:
        return "—"

    try:
        return (
            f"{float(value):.2f} m"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )