"""Observability workflow page for Rivelero.

O1 implements the scientific visibility-configuration layer only.  Building
and visualising a SurveyObservabilityField are deliberately left to O2/O3.

The page edits canonical :class:`VisibilityConfiguration` objects and stores
them in :class:`ApplicationState`; it never fills missing source metadata in
Viewpoint, Sensor, or ObservationEvent objects.
"""

from __future__ import annotations

import json
from uuid import uuid4

try:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
        QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
        QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
        QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
        QVBoxLayout, QWidget,
    )

from rivelero.gui.application_state import ApplicationState
from rivelero.gui.components import (
    BadgeType, CollapsibleSection, ContentCard, PageHeader,
    make_primary_button,
)
from rivelero.gui.theme import SPACING
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


class ObservabilityPage(QWidget):
    """Configure the assumptions used by Rivelero visibility analysis."""

    state_changed = Signal()

    def __init__(self, state: ApplicationState, *, parent=None) -> None:
        super().__init__(parent)
        if not isinstance(state, ApplicationState):
            raise TypeError("state must be an ApplicationState.")
        self.state = state
        self._loading = False
        self._build_interface()
        self._connect_signals()
        self.refresh_from_state()

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

        layout.addWidget(PageHeader(
            "Observability",
            "Define the assumptions used to translate survey metadata and "
            "terrain into visibility. Missing source metadata remain missing; "
            "the choices below describe how the analysis handles them.",
        ))

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
            "‘Exclude’ records the affected sampling unit in the future build "
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
                "Saving this configuration invalidates any SOF built with "
                "different visibility assumptions. No viewsheds are computed in O1."
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

        # O2/O3 preview ----------------------------------------------------
        future = ContentCard(
            title="Build survey observability",
            description=(
                "O2 will run the lazy disk-backed visibility store and SOF builder "
                "as a background task. O3 will add exposure, blind-spot and "
                "individual-visibility maps."
            ),
            badge=BadgeType.COMING_SOON,
            subtle=True,
        )
        self.build_button = QPushButton("Build observability field")
        self.build_button.setEnabled(False)
        future.add_widget(self.build_button)
        layout.addWidget(future)
        layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll)
        self._update_control_state()

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

    def _connect_signals(self) -> None:
        self.save_button.clicked.connect(self._save_configuration)
        self.use_direction.toggled.connect(self._update_control_state)
        self.heading_policy.currentIndexChanged.connect(self._update_control_state)
        self.fov_policy.currentIndexChanged.connect(self._update_control_state)
        self.height_policy.currentIndexChanged.connect(self._update_control_state)
        self.use_vertical_fov.toggled.connect(self._update_control_state)
        self.has_vertical_fov_default.toggled.connect(self._update_control_state)

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
            self.state.set_visibility_configuration(configuration)
        except (TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Invalid visibility configuration", str(exc))
            return
        self.status_label.setText(
            f"Saved ‘{configuration.name}’. O2 can now attach a VisibilityStore "
            "and build the Survey Observability Field."
        )
        self.state_changed.emit()

    def refresh_from_state(self) -> None:
        """Rehydrate the form from canonical ApplicationState."""
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
                self.status_label.setText(
                    f"Active configuration: {configuration.name}. "
                    "No Survey Observability Field has been built yet."
                    if not self.state.observability_ready
                    else f"Active configuration: {configuration.name}."
                )
            else:
                self.status_label.setText(
                    "No visibility configuration saved yet."
                    if self.state.world_ready
                    else "Complete Survey and World before saving visibility assumptions."
                )
        finally:
            self._loading = False
            self._update_control_state()
        self.save_button.setEnabled(self.state.world_ready)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
