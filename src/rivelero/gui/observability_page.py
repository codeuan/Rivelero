"""Observability workflow page for Rivelero.

The page covers the complete Observability step:

O1  edit the canonical :class:`VisibilityConfiguration`;
O2  configure the lazy disk-backed VisibilityStore and build the
    SurveyObservabilityField on the shared TaskController;
O3  inspect the SOF spatially, its summary, the build report and the
    visibility of an individual sampling unit;
O4  rehydrate everything from ApplicationState and report readiness.

The page never fills missing source metadata in Viewpoint, Sensor, or
ObservationEvent objects, and never holds the only reference to a scientific
result: the SOF and its build report live in ApplicationState.
"""

from __future__ import annotations

import json
from uuid import uuid4

try:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
        QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
        QProgressBar, QScrollArea, QSpinBox, QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
        QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
        QProgressBar, QScrollArea, QSpinBox, QVBoxLayout,
        QWidget,
    )

from rivelero.gui.application_state import (
    OBSERVABILITY_BUILD_TASK_NAME,
    ApplicationState,
    StaleObservabilityResultError,
    VisualizationLayer,
)
from rivelero.gui.components import (
    ActionBar, BadgeType, CollapsibleSection, ContentCard, LabeledValue,
    PageHeader, make_primary_button, make_secondary_button,
)
from rivelero.gui.observability_map import (
    ObservabilityMapMode, ObservabilityMapWidget,
)
from rivelero.gui.observability_service import (
    DEFAULT_MEMORY_ITEMS, ObservabilityNotReadyError, VisibilityKeyResolver,
    compute_unit_visibility, create_visibility_store,
    default_visibility_cache_directory, install_build_result, prepare_build,
    store_settings_match, unit_outcome_in_report,
)
from rivelero.gui.task_controller import TaskController, make_sof_build_task
from rivelero.gui.theme import SPACING, refresh_style
from rivelero.observability.builder import (
    SOFBuildReport, build_survey_observability_field,
)
from rivelero.observability.masks import ObservabilityState
from rivelero.visibility.configuration import (
    MissingMetadataPolicy, SamplingUnit, VisibilityBackend,
    VisibilityConfiguration,
)


_POLICY_LABELS = {
    MissingMetadataPolicy.USE_DEFAULT: "Use configured default",
    MissingMetadataPolicy.OMNIDIRECTIONAL: "Treat as omnidirectional",
    MissingMetadataPolicy.EXCLUDE: "Exclude sampling unit",
    MissingMetadataPolicy.ERROR: "Stop with an error",
}

UNIT_VISIBILITY_TASK_NAME = "Computing visibility"

# At most this many excluded/failed identifiers are listed inline.
_MAX_LISTED_IDS = 8


class ObservabilityPage(QWidget):
    """Configure, build and inspect survey observability."""

    state_changed = Signal()
    continue_requested = Signal()

    def __init__(
        self,
        state: ApplicationState,
        *,
        task_controller: TaskController | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(state, ApplicationState):
            raise TypeError("state must be an ApplicationState.")
        self.state = state

        # MainWindow passes the shared application controller. A private one
        # is created only when the page is used stand-alone.
        if task_controller is None:
            task_controller = TaskController(state, parent=self)
        if not isinstance(task_controller, TaskController):
            raise TypeError("task_controller must be a TaskController.")
        self.task_controller = task_controller

        self._loading = False

        # Transient bookkeeping for tasks this page started. The scientific
        # result itself is installed into ApplicationState.
        self._build_task_id: str | None = None
        self._build_request = None
        self._unit_task_id: str | None = None

        self._key_resolver = VisibilityKeyResolver()
        self._storage_error: str | None = None

        self._build_interface()
        self._connect_signals()
        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _build_interface(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(
            SPACING.page, SPACING.xxxl, SPACING.page, SPACING.xxxl
        )
        layout.setSpacing(SPACING.xl)
        self.content_layout = layout

        layout.addWidget(PageHeader(
            "Observability",
            "Define the assumptions used to translate survey metadata and "
            "terrain into visibility, then reconstruct where the survey "
            "created an opportunity to observe. Missing source metadata "
            "remain missing; the choices below describe how the analysis "
            "handles them.",
        ))

        self.invalidation_banner = QFrame()
        self.invalidation_banner.setProperty("warningCard", True)
        banner_layout = QVBoxLayout(self.invalidation_banner)
        banner_layout.setContentsMargins(
            SPACING.xl, SPACING.md, SPACING.xl, SPACING.md
        )
        self.invalidation_label = QLabel()
        self.invalidation_label.setWordWrap(True)
        banner_layout.addWidget(self.invalidation_label)
        self.invalidation_banner.setVisible(False)
        layout.addWidget(self.invalidation_banner)

        self._build_configuration_cards(layout)
        self._build_computation_card(layout)
        self._build_result_card(layout)
        self._build_summary_row(layout)
        self._build_selected_unit_card(layout)

        self.action_bar = ActionBar()
        # "&&" renders a literal ampersand; a single "&" is a Qt mnemonic.
        self.continue_button = make_primary_button(
            "Continue to Analysis && Design"
        )
        self.continue_button.setEnabled(False)
        self.action_bar.add_primary_action(self.continue_button)
        layout.addWidget(self.action_bar)

        layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll)
        self._update_control_state()

    def _build_configuration_cards(self, layout: QVBoxLayout) -> None:
        # Basic visibility -------------------------------------------------
        basic = ContentCard(
            title="Visibility model",
            description=(
                "These parameters control the current 2.5D terrain-visibility "
                "calculation. They become part of SOF provenance."
            ),
        )
        form = QFormLayout()
        form.setHorizontalSpacing(SPACING.xl)
        form.setVerticalSpacing(SPACING.md)

        self.name_edit = QLineEdit("Default visibility")
        form.addRow("Configuration name", self.name_edit)

        self.max_distance = QDoubleSpinBox()
        self.max_distance.setRange(0.01, 10_000_000.0)
        self.max_distance.setDecimals(2)
        self.max_distance.setValue(500.0)
        self.max_distance.setSuffix(" m")
        form.addRow("Maximum distance", self.max_distance)

        self.observer_height = QDoubleSpinBox()
        self.observer_height.setRange(0.0, 100_000.0)
        self.observer_height.setDecimals(2)
        self.observer_height.setValue(1.75)
        self.observer_height.setSuffix(" m")
        form.addRow("Default observer height", self.observer_height)

        self.target_height = QDoubleSpinBox()
        self.target_height.setRange(0.0, 100_000.0)
        self.target_height.setDecimals(2)
        self.target_height.setValue(0.0)
        self.target_height.setSuffix(" m")
        form.addRow("Target height", self.target_height)

        self.use_direction = QCheckBox(
            "Use heading and horizontal field-of-view metadata when available"
        )
        self.use_direction.setChecked(True)
        form.addRow("Directionality", self.use_direction)

        self.sampling_unit = QComboBox()
        self.sampling_unit.addItem("One contribution per Viewpoint", SamplingUnit.VIEWPOINT)
        self.sampling_unit.addItem(
            "One contribution per ObservationEvent", SamplingUnit.OBSERVATION_EVENT
        )
        form.addRow("Sampling unit", self.sampling_unit)
        basic.add_layout(form)
        layout.addWidget(basic)

        # Missing metadata -------------------------------------------------
        missing = ContentCard(
            title="Missing metadata",
            description=(
                "These policies do not alter the imported Survey. They state "
                "what the visibility engine should do when metadata cannot be "
                "resolved from Event → Viewpoint → Sensor."
            ),
        )
        missing_form = QFormLayout()
        missing_form.setHorizontalSpacing(SPACING.xl)
        missing_form.setVerticalSpacing(SPACING.md)

        self.heading_policy = self._policy_combo(include_omnidirectional=True)
        self.heading_policy.setCurrentIndex(
            self.heading_policy.findData(MissingMetadataPolicy.OMNIDIRECTIONAL)
        )
        missing_form.addRow("Heading missing", self.heading_policy)

        self.default_heading = QDoubleSpinBox()
        self.default_heading.setRange(0.0, 359.999)
        self.default_heading.setDecimals(1)
        self.default_heading.setSuffix("°")
        missing_form.addRow("Default heading", self.default_heading)

        self.fov_policy = self._policy_combo(include_omnidirectional=True)
        self.fov_policy.setCurrentIndex(
            self.fov_policy.findData(MissingMetadataPolicy.USE_DEFAULT)
        )
        missing_form.addRow("Horizontal FOV missing", self.fov_policy)

        self.default_fov = QDoubleSpinBox()
        self.default_fov.setRange(0.01, 360.0)
        self.default_fov.setDecimals(1)
        self.default_fov.setValue(360.0)
        self.default_fov.setSuffix("°")
        missing_form.addRow("Default horizontal FOV", self.default_fov)

        self.height_policy = self._policy_combo(include_omnidirectional=False)
        self.height_policy.setCurrentIndex(
            self.height_policy.findData(MissingMetadataPolicy.USE_DEFAULT)
        )
        missing_form.addRow("Observer height missing", self.height_policy)

        note = QLabel(
            "‘Exclude’ records the affected sampling unit in the build "
            "report. ‘Stop with an error’ is intended for strict analyses."
        )
        note.setWordWrap(True)
        note.setProperty("secondaryText", True)
        missing.add_layout(missing_form)
        missing.add_widget(note)
        layout.addWidget(missing)

        # Advanced ---------------------------------------------------------
        advanced = CollapsibleSection(
            "Advanced visibility settings",
            description="Backend, curvature and optional vertical-view assumptions.",
            badge=BadgeType.ADVANCED,
            expanded=False,
        )
        advanced_form = QFormLayout()
        advanced_form.setHorizontalSpacing(SPACING.xl)
        advanced_form.setVerticalSpacing(SPACING.md)

        self.backend = QComboBox()
        self.backend.addItem("GDAL viewshed", VisibilityBackend.GDAL)
        advanced_form.addRow("Visibility backend", self.backend)

        self.curvature = QDoubleSpinBox()
        self.curvature.setRange(0.0, 10.0)
        self.curvature.setDecimals(5)
        self.curvature.setValue(0.85714)
        advanced_form.addRow("Curvature coefficient", self.curvature)

        self.use_vertical_fov = QCheckBox("Use vertical field of view — Coming soon")
        self.use_vertical_fov.setEnabled(False)
        self.use_vertical_fov.setToolTip(
            "Vertical FOV is represented by VisibilityConfiguration but the "
            "current visibility engine raises NotImplementedError when enabled."
        )
        advanced_form.addRow("Vertical directionality", self.use_vertical_fov)

        self.default_pitch = QDoubleSpinBox()
        self.default_pitch.setRange(-90.0, 90.0)
        self.default_pitch.setDecimals(1)
        self.default_pitch.setSuffix("°")
        advanced_form.addRow("Default pitch", self.default_pitch)

        self.has_vertical_fov_default = QCheckBox("Provide a default vertical FOV")
        advanced_form.addRow("Vertical FOV default", self.has_vertical_fov_default)

        self.default_vertical_fov = QDoubleSpinBox()
        self.default_vertical_fov.setRange(0.01, 360.0)
        self.default_vertical_fov.setDecimals(1)
        self.default_vertical_fov.setValue(90.0)
        self.default_vertical_fov.setSuffix("°")
        advanced_form.addRow("Default vertical FOV", self.default_vertical_fov)

        self.extra_parameters = QLineEdit("{}")
        self.extra_parameters.setToolTip(
            "Optional JSON object for backend-specific experimental parameters."
        )
        advanced_form.addRow("Extra parameters (JSON)", self.extra_parameters)

        advanced.content_layout.addLayout(advanced_form)

        obstacle_note = QLabel(
            "Environmental obstacle layers are represented in the architecture "
            "but are not supported by the current single-viewpoint engine."
        )
        obstacle_note.setWordWrap(True)
        obstacle_note.setProperty("secondaryText", True)
        advanced.content_layout.addWidget(obstacle_note)
        layout.addWidget(advanced)

        # Configuration status/action -------------------------------------
        status_card = ContentCard(
            title="Configuration status",
            description=(
                "Saving a changed configuration invalidates any SOF built "
                "with different visibility assumptions. Cached viewsheds are "
                "kept and reused only for identical inputs."
            ),
        )
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_card.add_widget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.save_button = make_primary_button("Save visibility configuration")
        actions.addWidget(self.save_button)
        status_card.add_layout(actions)
        layout.addWidget(status_card)

    def _build_computation_card(self, layout: QVBoxLayout) -> None:
        card = ContentCard(
            title="Computation",
            description=(
                "Visibility is computed lazily for each sampling unit, stored "
                "on disk and reused whenever the inputs are identical. The "
                "build runs in the background; the interface stays responsive."
            ),
        )

        self.storage_value = LabeledValue("Visibility cache", "—")
        card.add_widget(self.storage_value)

        self.readiness_label = QLabel()
        self.readiness_label.setWordWrap(True)
        card.add_widget(self.readiness_label)

        buttons = QHBoxLayout()
        self.build_button = make_primary_button("Build observability field")
        self.cancel_button = make_secondary_button("Cancel build")
        self.cancel_button.setVisible(False)
        buttons.addWidget(self.build_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        card.add_layout(buttons)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        card.add_widget(self.progress_bar)

        self.progress_label = QLabel()
        self.progress_label.setProperty("secondaryText", True)
        self.progress_label.setVisible(False)
        card.add_widget(self.progress_label)

        self.build_status_label = QLabel()
        self.build_status_label.setWordWrap(True)
        self.build_status_label.setVisible(False)
        card.add_widget(self.build_status_label)

        self.error_section = CollapsibleSection(
            "Error details",
            expanded=False,
        )
        self.error_details = QPlainTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(180)
        self.error_section.content_layout.addWidget(self.error_details)
        self.error_section.setVisible(False)
        card.add_widget(self.error_section)

        # Advanced storage ------------------------------------------------
        storage = CollapsibleSection(
            "Advanced computation and storage",
            description="Cache location, memory use and failure handling.",
            badge=BadgeType.ADVANCED,
            expanded=False,
        )
        storage_form = QFormLayout()
        storage_form.setHorizontalSpacing(SPACING.xl)
        storage_form.setVerticalSpacing(SPACING.md)

        directory_row = QHBoxLayout()
        self.cache_directory_edit = QLineEdit()
        self.browse_cache_button = make_secondary_button("Browse…")
        directory_row.addWidget(self.cache_directory_edit, 1)
        directory_row.addWidget(self.browse_cache_button)
        storage_form.addRow("Cache directory", directory_row)

        self.memory_items = QSpinBox()
        self.memory_items.setRange(0, 100_000)
        self.memory_items.setValue(DEFAULT_MEMORY_ITEMS)
        self.memory_items.setSuffix(" masks")
        self.memory_items.setToolTip(
            "Number of individual visibility masks kept in RAM. Each mask "
            "uses about one byte per grid cell; older masks are reloaded "
            "from disk when needed."
        )
        storage_form.addRow("Memory cache", self.memory_items)

        self.compressed = QCheckBox("Compress cached masks on disk")
        self.compressed.setChecked(True)
        storage_form.addRow("Compression", self.compressed)

        self.continue_on_error = QCheckBox(
            "Continue when an individual sampling unit fails"
        )
        self.continue_on_error.setToolTip(
            "When enabled, failed units are recorded in the build report and "
            "the SOF represents only the successfully processed units."
        )
        storage_form.addRow("Failure handling", self.continue_on_error)

        storage.content_layout.addLayout(storage_form)

        storage_actions = QHBoxLayout()
        self.apply_storage_button = make_secondary_button("Apply storage settings")
        self.count_cache_button = make_secondary_button("Count cached masks")
        self.clear_memory_button = make_secondary_button("Clear memory cache")
        self.clear_disk_button = make_secondary_button("Delete cached masks…")
        for button in (
            self.apply_storage_button,
            self.count_cache_button,
            self.clear_memory_button,
            self.clear_disk_button,
        ):
            storage_actions.addWidget(button)
        storage_actions.addStretch(1)
        storage.content_layout.addLayout(storage_actions)

        self.storage_status_label = QLabel()
        self.storage_status_label.setWordWrap(True)
        self.storage_status_label.setProperty("secondaryText", True)
        storage.content_layout.addWidget(self.storage_status_label)

        card.add_widget(storage)
        layout.addWidget(card)

    def _build_result_card(self, layout: QVBoxLayout) -> None:
        card = ContentCard(
            title="Result",
            description=(
                "Outside the domain, invalid cells and blind spots are "
                "different states: a blind spot is valid, analysable space "
                "that no sampling unit could observe."
            ),
        )
        self.observability_map = ObservabilityMapWidget()
        self.observability_map.setMinimumHeight(560)
        card.add_widget(self.observability_map)
        layout.addWidget(card)

    def _build_summary_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(SPACING.xl)

        summary = ContentCard(
            title="Observability summary",
            description="Derived from the current Survey Observability Field.",
        )
        self.summary_values = {
            name: LabeledValue(label)
            for name, label in (
                ("analysable", "Analysable cells"),
                ("observable", "Observable cells"),
                ("blind", "Blind-spot cells"),
                ("fraction", "Observable share of analysable cells"),
                ("invalid", "Invalid cells inside domain"),
                ("outside", "Cells outside domain"),
                ("max_exposure", "Maximum exposure"),
                ("active", "Active sampling units"),
                ("unit", "Sampling unit"),
            )
        }
        for widget in self.summary_values.values():
            summary.add_widget(widget)
        row.addWidget(summary, 1)

        report = ContentCard(
            title="Build report",
            description="Computational provenance of the current result.",
        )
        self.report_values = {
            name: LabeledValue(label)
            for name, label in (
                ("requested", "Requested units"),
                ("added", "Added to the SOF"),
                ("computed", "Newly computed"),
                ("cached", "Reused from cache"),
                ("excluded", "Excluded by missing-metadata policy"),
                ("failed", "Failed computation"),
                ("built", "Built"),
                ("configuration", "Visibility configuration"),
            )
        }
        for widget in self.report_values.values():
            report.add_widget(widget)
        self.report_details_label = QLabel()
        self.report_details_label.setWordWrap(True)
        self.report_details_label.setProperty("secondaryText", True)
        report.add_widget(self.report_details_label)
        row.addWidget(report, 1)

        layout.addLayout(row)

    def _build_selected_unit_card(self, layout: QVBoxLayout) -> None:
        card = ContentCard(
            title="Selected sampling unit",
            description=(
                "Select a Viewpoint on the map (or in Survey) to inspect the "
                "effective visibility it contributes. Cached results are read "
                "from disk; nothing is recomputed when switching views."
            ),
        )

        form = QFormLayout()
        form.setHorizontalSpacing(SPACING.xl)
        form.setVerticalSpacing(SPACING.sm)

        self.selected_unit_label = QLabel("No Viewpoint selected")
        form.addRow("Viewpoint", self.selected_unit_label)

        self.event_combo = QComboBox()
        self.event_label = QLabel("ObservationEvent")
        form.addRow(self.event_label, self.event_combo)

        self.unit_status_label = QLabel("—")
        self.unit_status_label.setWordWrap(True)
        form.addRow("Visibility", self.unit_status_label)

        self.unit_parameters_label = QLabel("—")
        self.unit_parameters_label.setWordWrap(True)
        form.addRow("Effective parameters", self.unit_parameters_label)
        card.add_layout(form)

        buttons = QHBoxLayout()
        self.show_unit_button = make_secondary_button("Show on map")
        self.compute_unit_button = make_secondary_button("Compute visibility")
        buttons.addWidget(self.show_unit_button)
        buttons.addWidget(self.compute_unit_button)
        buttons.addStretch(1)
        card.add_layout(buttons)

        layout.addWidget(card)

    def _policy_combo(self, *, include_omnidirectional: bool) -> QComboBox:
        combo = QComboBox()
        policies = [
            MissingMetadataPolicy.USE_DEFAULT,
            MissingMetadataPolicy.EXCLUDE,
            MissingMetadataPolicy.ERROR,
        ]
        if include_omnidirectional:
            policies.insert(1, MissingMetadataPolicy.OMNIDIRECTIONAL)
        for policy in policies:
            combo.addItem(_POLICY_LABELS[policy], policy)
        return combo

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.save_button.clicked.connect(self._save_configuration)
        self.use_direction.toggled.connect(self._update_control_state)
        self.heading_policy.currentIndexChanged.connect(self._update_control_state)
        self.fov_policy.currentIndexChanged.connect(self._update_control_state)
        self.height_policy.currentIndexChanged.connect(self._update_control_state)
        self.use_vertical_fov.toggled.connect(self._update_control_state)
        self.has_vertical_fov_default.toggled.connect(self._update_control_state)

        # Any form edit may create unsaved differences that block building.
        for spin in (
            self.max_distance, self.observer_height, self.target_height,
            self.default_heading, self.default_fov, self.curvature,
            self.default_pitch, self.default_vertical_fov,
        ):
            spin.valueChanged.connect(self._on_form_edited)
        for combo in (
            self.sampling_unit, self.heading_policy, self.fov_policy,
            self.height_policy, self.backend,
        ):
            combo.currentIndexChanged.connect(self._on_form_edited)
        for check in (
            self.use_direction, self.use_vertical_fov,
            self.has_vertical_fov_default,
        ):
            check.toggled.connect(self._on_form_edited)
        self.name_edit.textChanged.connect(self._on_form_edited)
        self.extra_parameters.textChanged.connect(self._on_form_edited)

        self.build_button.clicked.connect(self._start_build)
        self.cancel_button.clicked.connect(self._cancel_build)

        self.browse_cache_button.clicked.connect(self._browse_cache_directory)
        self.apply_storage_button.clicked.connect(self._apply_storage_settings)
        self.count_cache_button.clicked.connect(self._count_cached_masks)
        self.clear_memory_button.clicked.connect(self._clear_memory_cache)
        self.clear_disk_button.clicked.connect(self._clear_disk_cache)

        self.observability_map.viewpoint_selected.connect(self._on_map_viewpoint_selected)
        self.observability_map.mode_changed.connect(self._on_map_mode_changed)
        self.event_combo.currentIndexChanged.connect(self._on_event_chosen)
        self.show_unit_button.clicked.connect(self._show_unit_on_map)
        self.compute_unit_button.clicked.connect(self._compute_selected_unit)

        self.continue_button.clicked.connect(self.continue_requested.emit)

        controller = self.task_controller
        controller.task_progress.connect(self._on_task_progress)
        controller.task_result.connect(self._on_task_result)
        controller.task_error.connect(self._on_task_error)
        controller.task_cancelled.connect(self._on_task_cancelled)
        controller.task_finished.connect(self._on_task_finished)

    def _update_control_state(self, *_args) -> None:
        directional = self.use_direction.isChecked()
        self.heading_policy.setEnabled(directional)
        self.fov_policy.setEnabled(directional)
        self.default_heading.setEnabled(
            directional
            and self.heading_policy.currentData() == MissingMetadataPolicy.USE_DEFAULT
        )
        self.default_fov.setEnabled(
            directional
            and self.fov_policy.currentData() == MissingMetadataPolicy.USE_DEFAULT
        )
        self.observer_height.setEnabled(
            self.height_policy.currentData() == MissingMetadataPolicy.USE_DEFAULT
        )
        vertical = self.use_vertical_fov.isChecked()
        self.default_pitch.setEnabled(vertical)
        self.has_vertical_fov_default.setEnabled(vertical)
        self.default_vertical_fov.setEnabled(
            vertical and self.has_vertical_fov_default.isChecked()
        )

    def _on_form_edited(self, *_args) -> None:
        if not self._loading:
            self._refresh_build_controls()

    # ------------------------------------------------------------------
    # Visibility configuration (O1)
    # ------------------------------------------------------------------

    def _configuration_from_form(self) -> VisibilityConfiguration:
        try:
            extra = json.loads(self.extra_parameters.text().strip() or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Extra parameters must be valid JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise ValueError("Extra parameters JSON must contain an object.")

        sampling = self.sampling_unit.currentData()
        if (
            sampling == SamplingUnit.OBSERVATION_EVENT
            and self.state.survey.n_observation_events == 0
        ):
            raise ValueError(
                "ObservationEvent sampling requires at least one ObservationEvent."
            )

        existing = self.state.analysis.visibility_configuration
        # Re-saving keeps the configuration identity. Cached viewsheds stay
        # safe because VisibilityKey fingerprints the configuration content.
        configuration_id = (
            existing.configuration_id if existing is not None else uuid4().hex
        )

        heading_policy = self.heading_policy.currentData()
        fov_policy = self.fov_policy.currentData()
        height_policy = self.height_policy.currentData()

        return VisibilityConfiguration(
            configuration_id=configuration_id,
            name=self.name_edit.text().strip(),
            backend=self.backend.currentData(),
            max_distance_m=float(self.max_distance.value()),
            default_observer_height_m=(
                float(self.observer_height.value())
                if height_policy == MissingMetadataPolicy.USE_DEFAULT
                else None
            ),
            default_target_height_m=float(self.target_height.value()),
            use_direction=self.use_direction.isChecked(),
            default_heading_deg=(
                float(self.default_heading.value())
                if heading_policy == MissingMetadataPolicy.USE_DEFAULT
                else None
            ),
            default_horizontal_fov_deg=(
                float(self.default_fov.value())
                if fov_policy == MissingMetadataPolicy.USE_DEFAULT
                else None
            ),
            use_vertical_fov=self.use_vertical_fov.isChecked(),
            default_pitch_deg=(
                float(self.default_pitch.value())
                if self.use_vertical_fov.isChecked()
                else 0.0
            ),
            default_vertical_fov_deg=(
                float(self.default_vertical_fov.value())
                if self.use_vertical_fov.isChecked()
                and self.has_vertical_fov_default.isChecked()
                else None
            ),
            missing_heading_policy=heading_policy,
            missing_fov_policy=fov_policy,
            missing_observer_height_policy=height_policy,
            curvature_coefficient=float(self.curvature.value()),
            use_environment_obstacles=False,
            obstacle_layer_ids=(),
            sampling_unit=sampling,
            extra_parameters=extra,
        )

    def _form_matches_saved(self) -> bool:
        """Whether the form describes exactly the saved configuration."""
        saved = self.state.analysis.visibility_configuration
        if saved is None:
            return False
        try:
            return self._configuration_from_form() == saved
        except (TypeError, ValueError):
            return False

    def _save_configuration(self) -> None:
        if not self.state.world_ready:
            QMessageBox.information(
                self,
                "World required",
                "Complete the World definition before configuring observability.",
            )
            return
        try:
            configuration = self._configuration_from_form()
            if configuration == self.state.analysis.visibility_configuration:
                # Identical assumptions: the current SOF stays valid.
                self.status_label.setText(
                    f"‘{configuration.name}’ is unchanged; the current "
                    "result remains valid."
                )
                return
            self.state.set_visibility_configuration(configuration)
        except (TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Invalid visibility configuration", str(exc))
            return
        self._refresh_view()
        self.status_label.setText(
            f"Saved ‘{configuration.name}’. Build the observability field "
            "to apply it."
        )
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # VisibilityStore (O2)
    # ------------------------------------------------------------------

    def _ensure_store(self) -> None:
        """Attach a default VisibilityStore once Survey and World exist."""
        if (
            self.state.analysis.visibility_store is not None
            or not self.state.survey_ready
            or not self.state.world_ready
        ):
            return
        try:
            self.state.set_visibility_store(create_visibility_store())
            self._storage_error = None
        except OSError as exc:
            self._storage_error = (
                f"The default cache directory could not be created: {exc}"
            )

    def _load_storage_controls(self) -> None:
        store = self.state.analysis.visibility_store
        if store is None:
            self.cache_directory_edit.setText(
                str(default_visibility_cache_directory())
            )
            return
        self.cache_directory_edit.setText(str(store.cache_directory))
        self.memory_items.setValue(store.max_memory_items)
        self.compressed.setChecked(store.compressed)

    def _browse_cache_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose visibility cache directory",
            self.cache_directory_edit.text(),
        )
        if directory:
            self.cache_directory_edit.setText(directory)

    def _apply_storage_settings(self) -> None:
        if self.state.busy:
            return
        directory = self.cache_directory_edit.text().strip()
        if not directory:
            self.storage_status_label.setText("Choose a cache directory.")
            return
        store = self.state.analysis.visibility_store
        settings = {
            "cache_directory": directory,
            "max_memory_items": int(self.memory_items.value()),
            "compressed": self.compressed.isChecked(),
        }
        if store is not None and store_settings_match(store, **settings):
            self.storage_status_label.setText("Storage settings unchanged.")
            return
        try:
            new_store = create_visibility_store(
                settings["cache_directory"],
                max_memory_items=settings["max_memory_items"],
                compressed=settings["compressed"],
            )
        except (OSError, TypeError, ValueError) as exc:
            self.storage_status_label.setText(f"Could not use this directory: {exc}")
            return
        # The store is a computational cache, not a scientific input, so the
        # current SOF remains valid.
        self.state.set_visibility_store(new_store)
        self._storage_error = None
        self.storage_status_label.setText(
            "Storage settings applied. The current result, if any, remains valid."
        )
        self._refresh_view()
        self.state_changed.emit()

    def _count_cached_masks(self) -> None:
        store = self.state.analysis.visibility_store
        if store is None:
            return
        self.storage_status_label.setText(
            f"{store.disk_item_count:,} visibility masks cached on disk · "
            f"{store.memory_item_count:,} in memory."
        )

    def _clear_memory_cache(self) -> None:
        store = self.state.analysis.visibility_store
        if store is None:
            return
        store.clear_memory()
        self.storage_status_label.setText(
            "Memory cache cleared. Masks remain on disk."
        )
        self._refresh_storage_summary()

    def _clear_disk_cache(self, *, confirm: bool = True) -> None:
        store = self.state.analysis.visibility_store
        if store is None or self.state.busy:
            return
        if confirm and QMessageBox.question(
            self,
            "Delete cached visibility",
            "Delete every visibility mask cached in\n"
            f"{store.cache_directory}?\n\n"
            "Other files in the directory are not touched. The current "
            "result stays valid, but future builds must recompute visibility.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        count = store.clear_disk()
        self.storage_status_label.setText(
            f"Deleted {count:,} cached visibility masks."
        )
        self._refresh_view()

    # ------------------------------------------------------------------
    # SOF build (O2)
    # ------------------------------------------------------------------

    def build_blockers(self) -> list[str]:
        """Reasons the Build button is disabled, including unsaved edits."""
        self._ensure_store()
        blockers = self.state.observability_build_blockers()
        if (
            self.state.analysis.visibility_configuration is not None
            and not self._form_matches_saved()
        ):
            blockers.append(
                "The form has unsaved visibility changes. Save them first."
            )
        if self._storage_error:
            blockers.append(self._storage_error)
        return blockers

    def _start_build(self) -> None:
        if self.build_blockers():
            self._refresh_build_controls()
            return
        try:
            request = prepare_build(
                self.state,
                continue_on_error=self.continue_on_error.isChecked(),
            )
        except ObservabilityNotReadyError as exc:
            self._set_build_status(str(exc), kind="error")
            return

        task = make_sof_build_task(
            build_function=build_survey_observability_field,
            build_kwargs=request.build_kwargs(),
        )

        self._build_request = request
        self._build_task_id = self.task_controller.start(
            task_name=OBSERVABILITY_BUILD_TASK_NAME,
            function=task,
            total=request.total_units,
            message=f"Building observability for {request.total_units:,} units",
            inject_context=True,
        )

        self.progress_bar.setRange(0, max(1, request.total_units))
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            f"Processed 0 / {request.total_units:,}"
        )
        self._set_build_status(None)
        self.error_section.setVisible(False)
        self._refresh_view()
        self.state_changed.emit()

    def _cancel_build(self) -> None:
        if self._build_task_id is None:
            return
        if self.task_controller.cancel():
            self.cancel_button.setEnabled(False)
            self._set_build_status(
                "Cancellation requested. Rivelero stops after the sampling "
                "unit currently being computed; a running GDAL viewshed "
                "cannot be interrupted.",
                kind="warning",
            )

    def _on_task_progress(self, task_id, processed, total, unit_id, _message) -> None:
        if task_id != self._build_task_id:
            return
        if total:
            self.progress_bar.setRange(0, int(total))
        self.progress_bar.setValue(int(processed))
        text = f"Processed {processed:,} / {total:,}" if total else f"Processed {processed:,}"
        if unit_id:
            text += f" · last processed {unit_id}"
        self.progress_label.setText(text)

    def _on_task_result(self, task_id, result) -> None:
        if task_id == self._unit_task_id:
            self._refresh_selected_unit()
            return
        if task_id != self._build_task_id:
            return
        request = self._build_request
        try:
            install_build_result(self.state, request, result)
        except StaleObservabilityResultError as exc:
            self._set_build_status(
                f"{exc} Rebuild to reconstruct observability for the current "
                "inputs. Visibility computed during this build stays cached.",
                kind="warning",
            )
            return
        except (TypeError, ValueError, RuntimeError) as exc:
            self._set_build_status(
                f"The build finished but its result could not be installed: {exc}",
                kind="error",
            )
            return

        report = result.report
        message = (
            f"Observability field built from {report.added_units:,} of "
            f"{report.requested_units:,} sampling units."
        )
        kind = "success"
        if report.added_units == 0:
            message += (
                " No sampling unit contributed, so every analysable cell is a "
                "blind spot. Check the build report."
            )
            kind = "warning"
        elif report.excluded_units or report.failed_units:
            message += (
                f" {report.excluded_units:,} excluded by policy and "
                f"{report.failed_units:,} failed; see the build report."
            )
            kind = "warning"
        self._set_build_status(message, kind=kind)
        self.state_changed.emit()

    def _on_task_error(self, task_id, message, details) -> None:
        if task_id == self._unit_task_id:
            self.unit_status_label.setText(f"Visibility could not be computed: {message}")
            return
        if task_id != self._build_task_id:
            return
        self._set_build_status(
            f"Build failed: {message}. No result was installed; any previous "
            "result and cached visibility are unchanged.",
            kind="error",
        )
        self.error_details.setPlainText(details or message)
        self.error_section.setVisible(True)

    def _on_task_cancelled(self, task_id, _message) -> None:
        if task_id == self._unit_task_id:
            return
        if task_id != self._build_task_id:
            return
        task = self.state.task
        self._set_build_status(
            f"Build cancelled after {task.processed:,} of "
            f"{(task.total or 0):,} units. No partial result was installed; "
            "visibility already computed stays cached and will be reused.",
            kind="warning",
        )

    def _on_task_finished(self, task_id) -> None:
        if task_id == self._build_task_id:
            self._build_task_id = None
            self._build_request = None
        elif task_id == self._unit_task_id:
            self._unit_task_id = None
        # Any task finishing may re-enable this page's actions.
        self._refresh_view()

    def _set_build_status(self, text: str | None, *, kind: str = "success") -> None:
        if not text:
            self.build_status_label.setVisible(False)
            return
        for name in ("statusSuccess", "statusWarning", "statusError"):
            self.build_status_label.setProperty(name, False)
        self.build_status_label.setProperty(
            {"success": "statusSuccess", "warning": "statusWarning"}.get(
                kind, "statusError"
            ),
            True,
        )
        refresh_style(self.build_status_label)
        self.build_status_label.setText(text)
        self.build_status_label.setVisible(True)

    # ------------------------------------------------------------------
    # Map and selection (O3)
    # ------------------------------------------------------------------

    def _on_map_viewpoint_selected(self, viewpoint_id: str) -> None:
        # Same canonical selection as the Survey page.
        self.state.select_viewpoint(viewpoint_id)
        self._refresh_selected_unit()

    def _on_map_mode_changed(self, mode: str) -> None:
        if not self._loading:
            self.state.set_active_layer(VisualizationLayer(mode))

    def _on_event_chosen(self, _index: int) -> None:
        if self._loading:
            return
        event_id = self.event_combo.currentData()
        viewpoint_id = self.state.selection.viewpoint_id
        if event_id is not None and viewpoint_id is not None:
            self.state.select_observation_event(event_id, viewpoint_id=viewpoint_id)
            self._refresh_selected_unit()

    def _show_unit_on_map(self) -> None:
        self.observability_map.set_mode(ObservabilityMapMode.INDIVIDUAL_VISIBILITY)

    def _current_selection(self):
        """Resolve the selected unit, or return (None, message)."""
        viewpoint_id = self.state.selection.viewpoint_id
        if viewpoint_id is None:
            return None, "Select a Viewpoint on the map or in Survey."
        configuration = self.state.analysis.visibility_configuration
        event_id = None
        if configuration is not None and configuration.is_event_based:
            event_id = self.event_combo.currentData()
        try:
            return self._key_resolver.resolve(
                self.state, viewpoint_id=viewpoint_id, event_id=event_id
            ), None
        except (KeyError, ValueError) as exc:
            return None, str(exc).strip("'\"")

    def _compute_selected_unit(self) -> None:
        if self.state.busy:
            return
        selection, message = self._current_selection()
        store = self.state.analysis.visibility_store
        if selection is None or store is None:
            self.unit_status_label.setText(message or "Configure visibility storage.")
            return
        self._unit_task_id = self.task_controller.start(
            task_name=UNIT_VISIBILITY_TASK_NAME,
            function=compute_unit_visibility,
            kwargs={
                "selection": selection,
                "environment": self.state.analysis.environment,
                "domain": self.state.analysis.analysis_domain,
                "visibility_configuration": (
                    self.state.analysis.visibility_configuration
                ),
                "store": store,
            },
            message=f"Computing visibility for {selection.unit_id}",
        )
        self.unit_status_label.setText(f"Computing visibility for {selection.unit_id}…")
        self._refresh_view()

    def _refresh_event_choices(self) -> None:
        configuration = self.state.analysis.visibility_configuration
        survey = self.state.survey.viewpoint_configuration
        event_based = configuration is not None and configuration.is_event_based
        self.event_label.setVisible(event_based)
        self.event_combo.setVisible(event_based)
        if not event_based:
            return
        viewpoint_id = self.state.selection.viewpoint_id
        events = (
            survey.events_for_viewpoint(viewpoint_id)
            if survey is not None and viewpoint_id is not None
            else ()
        )
        previous = self.state.selection.observation_event_id
        self.event_combo.clear()
        for event in events:
            label = event.event_id
            if event.timestamp is not None:
                label += f" · {event.timestamp.isoformat(timespec='seconds')}"
            self.event_combo.addItem(label, event.event_id)
        index = self.event_combo.findData(previous)
        if index >= 0:
            self.event_combo.setCurrentIndex(index)

    def _refresh_selected_unit(self) -> None:
        self._loading = True
        try:
            self._refresh_event_choices()
        finally:
            self._loading = False

        viewpoint_id = self.state.selection.viewpoint_id
        self.selected_unit_label.setText(viewpoint_id or "No Viewpoint selected")
        self.observability_map.set_selected_viewpoint(viewpoint_id)

        self.unit_parameters_label.setText("—")
        self.show_unit_button.setEnabled(False)
        self.compute_unit_button.setEnabled(False)

        selection, message = self._current_selection()
        store = self.state.analysis.visibility_store

        if selection is None or store is None:
            self.unit_status_label.setText(message or "Configure visibility storage.")
            self.observability_map.set_individual_visibility(None)
            return

        outcome = unit_outcome_in_report(
            self.state.analysis.build_report, selection.unit_id
        )

        stored = store.get(selection.key) if store.contains(selection.key) else None

        if stored is not None:
            self.observability_map.set_individual_visibility(
                stored.visibility_mask, selection.unit_id
            )
            self.unit_status_label.setText(
                f"{selection.unit_id}: cached · "
                f"{stored.visible_cell_count:,} visible analysable cells."
            )
            self.unit_parameters_label.setText(
                _describe_resolved_parameters(stored.metadata)
            )
            self.show_unit_button.setEnabled(True)
            return

        self.observability_map.set_individual_visibility(None)

        if outcome is not None:
            kind, reason = outcome
            label = "Excluded by missing-metadata policy" if kind == "excluded" else "Failed"
            self.unit_status_label.setText(f"{selection.unit_id}: {label}. {reason}")
            return

        self.unit_status_label.setText(
            f"{selection.unit_id}: visibility not yet computed for the current inputs."
        )
        self.compute_unit_button.setEnabled(not self.state.busy)

    # ------------------------------------------------------------------
    # Rehydration (O4)
    # ------------------------------------------------------------------

    def refresh_from_state(self) -> None:
        """Rehydrate the complete page from canonical ApplicationState."""
        self._ensure_store()
        self._load_form_from_state()
        self._load_storage_controls()
        self._refresh_view()

    def _load_form_from_state(self) -> None:
        configuration = self.state.analysis.visibility_configuration
        self._loading = True
        try:
            if configuration is not None:
                self.name_edit.setText(configuration.name)
                self.max_distance.setValue(configuration.max_distance_m)
                self.target_height.setValue(configuration.default_target_height_m)
                self.use_direction.setChecked(configuration.use_direction)
                self._set_combo_data(self.backend, configuration.backend)
                self._set_combo_data(self.sampling_unit, configuration.sampling_unit)
                self._set_combo_data(
                    self.heading_policy, configuration.missing_heading_policy
                )
                self._set_combo_data(self.fov_policy, configuration.missing_fov_policy)
                self._set_combo_data(
                    self.height_policy, configuration.missing_observer_height_policy
                )
                if configuration.default_observer_height_m is not None:
                    self.observer_height.setValue(configuration.default_observer_height_m)
                if configuration.default_heading_deg is not None:
                    self.default_heading.setValue(configuration.default_heading_deg)
                if configuration.default_horizontal_fov_deg is not None:
                    self.default_fov.setValue(configuration.default_horizontal_fov_deg)
                self.curvature.setValue(configuration.curvature_coefficient)
                self.use_vertical_fov.setChecked(configuration.use_vertical_fov)
                if configuration.default_pitch_deg is not None:
                    self.default_pitch.setValue(configuration.default_pitch_deg)
                has_vfov = configuration.default_vertical_fov_deg is not None
                self.has_vertical_fov_default.setChecked(has_vfov)
                if has_vfov:
                    self.default_vertical_fov.setValue(
                        configuration.default_vertical_fov_deg
                    )
                self.extra_parameters.setText(
                    json.dumps(configuration.extra_parameters, sort_keys=True)
                )
        finally:
            self._loading = False
            self._update_control_state()

    def _refresh_view(self) -> None:
        """Refresh everything except the editable configuration form."""
        self._refresh_configuration_status()
        self._refresh_invalidation_banner()
        self._refresh_storage_summary()
        self._refresh_build_controls()
        self._refresh_map()
        self._refresh_summary()
        self._refresh_report()
        self._refresh_selected_unit()
        self.continue_button.setEnabled(
            self.state.observability_ready and not self.state.busy
        )

    def _refresh_configuration_status(self) -> None:
        configuration = self.state.analysis.visibility_configuration
        if configuration is not None:
            self.status_label.setText(
                f"Active configuration: {configuration.name}."
                + (
                    ""
                    if self.state.observability_ready
                    else " No current Survey Observability Field."
                )
            )
        else:
            self.status_label.setText(
                "No visibility configuration saved yet."
                if self.state.world_ready
                else "Complete Survey and World before saving visibility assumptions."
            )
        self.save_button.setEnabled(self.state.world_ready and not self.state.busy)

    def _refresh_invalidation_banner(self) -> None:
        reason = self.state.analysis.invalidation_reason
        show = reason is not None and not self.state.observability_ready
        if show:
            self.invalidation_label.setText(
                f"Previous observability result invalidated: {reason} "
                "Rebuild the observability field to update it; cached "
                "visibility for unchanged inputs will be reused."
            )
        self.invalidation_banner.setVisible(show)

    def _refresh_storage_summary(self) -> None:
        store = self.state.analysis.visibility_store
        if store is None:
            self.storage_value.set_value(
                self._storage_error or "Created once Survey and World are ready"
            )
            return
        self.storage_value.set_value(
            f"{store.cache_directory} · {store.memory_item_count:,} of "
            f"{store.max_memory_items:,} masks in memory"
        )

    def _refresh_build_controls(self) -> None:
        building = self._build_task_id is not None
        busy = self.state.busy

        blockers = [] if building else self.build_blockers()

        if building:
            self.readiness_label.setText("Building observability…")
        elif blockers:
            self.readiness_label.setText(
                "Before building:\n" + "\n".join(f"• {item}" for item in blockers)
            )
        elif self.state.observability_ready:
            self.readiness_label.setText(
                "The current result matches the saved inputs. Rebuilding "
                "reuses cached visibility."
            )
        else:
            self.readiness_label.setText("Ready to build.")

        self.build_button.setEnabled(not building and not blockers)
        self.build_button.setText(
            "Rebuild observability field"
            if self.state.observability_ready
            else "Build observability field"
        )
        self.cancel_button.setVisible(building)
        self.cancel_button.setEnabled(
            building and not self.task_controller.cancellation_requested
        )
        self.progress_bar.setVisible(building)
        self.progress_label.setVisible(building)

        for widget in (
            self.apply_storage_button,
            self.clear_disk_button,
            self.browse_cache_button,
            self.cache_directory_edit,
            self.memory_items,
            self.compressed,
            self.continue_on_error,
        ):
            widget.setEnabled(not busy)

    def _refresh_map(self) -> None:
        sof = self.state.analysis.survey_observability_field
        environment = self.state.analysis.environment
        survey = self.state.survey.viewpoint_configuration

        self.observability_map.set_terrain(
            None if environment is None else environment.elevation_model.source
        )
        self.observability_map.set_field(sof)
        self.observability_map.set_viewpoints(
            [] if survey is None else survey.viewpoints
        )

        layer = self.state.view.active_layer
        self._loading = True
        try:
            try:
                self.observability_map.set_mode(ObservabilityMapMode(layer.value))
            except ValueError:
                pass  # Not an observability layer; keep the current mode.
        finally:
            self._loading = False

    def _refresh_summary(self) -> None:
        sof = self.state.analysis.survey_observability_field
        if sof is None:
            for widget in self.summary_values.values():
                widget.set_value("—")
            return

        counts = sof.state_counts()
        analysable = sof.n_analysable_cells
        observable = sof.n_observable_cells

        values = {
            "analysable": f"{analysable:,}",
            "observable": f"{observable:,}",
            "blind": f"{sof.n_blindspot_cells:,}",
            # Observable and blind-spot cells partition analysable space.
            "fraction": (
                f"{observable / analysable:.1%}" if analysable else "—"
            ),
            "invalid": f"{counts[ObservabilityState.INVALID]:,}",
            "outside": f"{counts[ObservabilityState.OUTSIDE_DOMAIN]:,}",
            "max_exposure": f"{sof.maximum_exposure:,}",
            "active": f"{sof.n_active_units:,}",
            "unit": (
                "ObservationEvent"
                if sof.sampling_unit == SamplingUnit.OBSERVATION_EVENT
                else "Viewpoint"
            ),
        }
        for name, value in values.items():
            self.summary_values[name].set_value(value)

    def _refresh_report(self) -> None:
        report = self.state.analysis.build_report
        sof = self.state.analysis.survey_observability_field
        if not isinstance(report, SOFBuildReport) or sof is None:
            for widget in self.report_values.values():
                widget.set_value("—")
            self.report_details_label.setText("")
            return

        configuration = self.state.analysis.visibility_configuration
        values = {
            "requested": f"{report.requested_units:,}",
            "added": f"{report.added_units:,}",
            "computed": f"{report.computed_units:,}",
            "cached": f"{report.cache_hits:,}",
            "excluded": f"{report.excluded_units:,}",
            "failed": f"{report.failed_units:,}",
            "built": sof.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": "—" if configuration is None else configuration.name,
        }
        for name, value in values.items():
            self.report_values[name].set_value(value)

        details = []
        if report.excluded_ids:
            details.append("Excluded: " + _format_ids(report.excluded_ids))
        if report.failed_ids:
            details.append("Failed: " + _format_ids(report.failed_ids))
        self.report_details_label.setText("\n".join(details))
        self.report_details_label.setToolTip(
            "\n".join(
                f"{unit_id}: {message}"
                for unit_id, message in list(report.errors.items())[:50]
            )
        )

    @staticmethod
    def _set_combo_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)


def _format_ids(ids: list[str]) -> str:
    shown = ", ".join(ids[:_MAX_LISTED_IDS])
    remaining = len(ids) - _MAX_LISTED_IDS
    return shown if remaining <= 0 else f"{shown} … and {remaining:,} more"


def _describe_resolved_parameters(metadata: dict) -> str:
    """Summarise the effective parameters recorded with a cached mask."""
    resolved = metadata.get("resolved_parameters")
    if not isinstance(resolved, dict):
        return "Not recorded for this cached result."

    parts = [f"observer height {resolved.get('observer_height_m'):g} m"]
    if resolved.get("omnidirectional"):
        parts.append("omnidirectional")
    else:
        heading = resolved.get("heading_deg")
        fov = resolved.get("horizontal_fov_deg")
        if heading is not None:
            parts.append(f"heading {heading:g}°")
        if fov is not None:
            parts.append(f"horizontal FOV {fov:g}°")
    parts.append(f"max distance {resolved.get('max_distance_m'):g} m")

    defaults = [
        name
        for key, name in (
            ("used_default_observer_height", "observer height"),
            ("used_default_heading", "heading"),
            ("used_default_horizontal_fov", "horizontal FOV"),
        )
        if resolved.get(key)
    ]
    text = ", ".join(parts)
    text += (
        f". Configuration defaults applied for: {', '.join(defaults)}."
        if defaults
        else ". No configuration defaults were needed."
    )
    return text
