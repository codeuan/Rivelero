"""World workflow page for the Rivelero GUI.

The World page defines:

- the canonical physical Environment;
- the AnalysisGrid inherited from the elevation source;
- the AnalysisDomain;
- spatial compatibility between Survey, terrain and domain.

Scientific operations are delegated to tested services.
"""

from __future__ import annotations

from uuid import uuid4

try:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QButtonGroup,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QButtonGroup,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )

from rivelero.gui.application_state import ApplicationState
from rivelero.gui.components import (
    ActionBar,
    BadgeType,
    CollapsibleSection,
    ContentCard,
    LabeledValue,
    PageHeader,
    StatusBadge,
    make_primary_button,
    make_secondary_button,
)
from rivelero.gui.domain_import_dialog import DomainImportDialog
from rivelero.gui.domain_service import (
    build_domain_from_drawn_polygon,
    build_domain_from_grid_extent,
    build_domain_from_survey_extent,
)
from rivelero.gui.environment_import_dialog import (
    EnvironmentImportDialog,
)
from rivelero.gui.theme import SPACING
from rivelero.gui.task_controller import TaskController
from rivelero.gui.world_map import WorldMapWidget
from rivelero.gui.world_qc import (
    QCSeverity,
    run_world_qc,
)


class WorldPage(QWidget):
    """Terrain, AnalysisDomain and spatial-QC workflow."""

    continue_requested = Signal()
    state_changed = Signal()

    def __init__(
        self,
        state: ApplicationState,
        *,
        task_controller: TaskController,
        parent=None,
    ) -> None:
        super().__init__(parent)

        if not isinstance(
            state,
            ApplicationState,
        ):
            raise TypeError(
                "state must be an ApplicationState."
            )

        self.state = state
        if not isinstance(task_controller, TaskController):
            raise TypeError("task_controller must be a TaskController.")
        self.task_controller = task_controller

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

        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(
            SPACING.page,
            SPACING.xxxl,
            SPACING.page,
            SPACING.xxxl,
        )
        self.content_layout.setSpacing(SPACING.xl)

        self.content_layout.addWidget(
            PageHeader(
                "World",
                (
                    "Define the physical environment and spatial area "
                    "over which Rivelero will reconstruct observability."
                ),
            )
        )

        self._build_terrain_card()
        self._build_map_card()
        self._build_domain_card()
        self._build_qc_card()
        self._build_validity_section()
        self._build_action_bar()

        self.content_layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll)

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------

    def _build_terrain_card(self) -> None:
        self.terrain_card = ContentCard(
            title="Elevation / surface model",
            description=(
                "Terrain provides the physical elevation surface used "
                "by the visibility engine."
            ),
        )

        actions = QHBoxLayout()

        self.load_button = make_primary_button(
            "Load DEM"
        )

        self.download_button = make_secondary_button(
            "Download DEM"
        )

        self.dsm_button = make_secondary_button(
            "DSM"
        )
        self.dsm_button.setEnabled(False)
        self.dsm_button.setToolTip(
            "DSM support is planned for a future implementation."
        )

        actions.addWidget(self.load_button)
        actions.addWidget(self.download_button)
        actions.addWidget(self.dsm_button)

        future = StatusBadge(
            BadgeType.COMING_SOON,
            text="DSM · Future implementation",
        )
        actions.addWidget(future)

        actions.addStretch(1)

        self.terrain_card.add_layout(actions)

        self.terrain_details = QWidget()
        detail_layout = QHBoxLayout(
            self.terrain_details
        )
        detail_layout.setContentsMargins(0, 0, 0, 0)

        self.terrain_name = LabeledValue(
            "Terrain",
            "Not loaded",
            vertical=True,
        )
        self.terrain_crs = LabeledValue(
            "CRS",
            "—",
            vertical=True,
        )
        self.terrain_resolution = LabeledValue(
            "Resolution",
            "—",
            vertical=True,
        )
        self.terrain_size = LabeledValue(
            "Raster size",
            "—",
            vertical=True,
        )

        for widget in (
            self.terrain_name,
            self.terrain_crs,
            self.terrain_resolution,
            self.terrain_size,
        ):
            detail_layout.addWidget(widget, 1)

        self.terrain_card.add_widget(
            self.terrain_details
        )

        self.content_layout.addWidget(
            self.terrain_card
        )

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------

    def _build_map_card(self) -> None:
        self.map_card = ContentCard(
            title="Terrain and survey",
            description=(
                "Inspect terrain, Survey Viewpoints and the current "
                "AnalysisDomain in the same spatial reference."
            ),
        )

        self.world_map = WorldMapWidget()
        self.world_map.setMinimumHeight(520)

        self.map_card.add_widget(
            self.world_map
        )

        self.content_layout.addWidget(
            self.map_card
        )

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------

    def _build_domain_card(self) -> None:
        self.domain_card = ContentCard(
            title="Analysis area",
            description=(
                "Define where observability will be evaluated. "
                "Observer Viewpoints may legitimately exist outside it."
            ),
        )

        self.domain_group = QButtonGroup(self)

        self.full_domain_radio = QRadioButton(
            "Entire terrain extent"
        )

        self.survey_domain_radio = QRadioButton(
            "Survey extent + buffer"
        )

        self.draw_domain_radio = QRadioButton(
            "Draw on map"
        )

        self.import_domain_radio = QRadioButton(
            "Import polygon"
        )

        for button in (
            self.full_domain_radio,
            self.survey_domain_radio,
            self.draw_domain_radio,
            self.import_domain_radio,
        ):
            self.domain_group.addButton(button)

        self.full_domain_radio.setChecked(True)

        self.domain_card.add_widget(
            self.full_domain_radio
        )

        buffer_row = QHBoxLayout()
        buffer_row.addWidget(
            self.survey_domain_radio
        )

        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0.0, 100000.0)
        self.buffer_spin.setValue(250.0)
        self.buffer_spin.setDecimals(1)
        self.buffer_spin.setSuffix(" m")

        buffer_row.addWidget(
            self.buffer_spin
        )
        buffer_row.addStretch(1)

        self.domain_card.add_layout(
            buffer_row
        )

        self.domain_card.add_widget(
            self.draw_domain_radio
        )
        self.domain_card.add_widget(
            self.import_domain_radio
        )

        domain_actions = QHBoxLayout()

        self.apply_domain_button = (
            make_primary_button(
                "Apply analysis area"
            )
        )

        self.clear_domain_button = (
            make_secondary_button(
                "Clear"
            )
        )

        domain_actions.addWidget(
            self.apply_domain_button
        )
        domain_actions.addWidget(
            self.clear_domain_button
        )
        domain_actions.addStretch(1)

        self.domain_card.add_layout(
            domain_actions
        )

        self.domain_summary = QLabel(
            "No AnalysisDomain defined."
        )
        self.domain_summary.setProperty(
            "secondaryText",
            True,
        )

        self.domain_card.add_widget(
            self.domain_summary
        )

        self.content_layout.addWidget(
            self.domain_card
        )

    # ------------------------------------------------------------------
    # QC
    # ------------------------------------------------------------------

    def _build_qc_card(self) -> None:
        self.qc_card = ContentCard(
            title="Spatial check",
            description=(
                "Rivelero reports spatial compatibility without "
                "silently rewriting survey coordinates."
            ),
        )

        self.qc_summary = QLabel(
            "Load terrain to run spatial checks."
        )

        self.qc_summary.setWordWrap(True)
        self.qc_summary.setTextInteractionFlags(
            self.qc_summary.textInteractionFlags()
        )

        self.qc_card.add_widget(
            self.qc_summary
        )

        self.content_layout.addWidget(
            self.qc_card
        )

    # ------------------------------------------------------------------
    # Advanced validity
    # ------------------------------------------------------------------

    def _build_validity_section(self) -> None:
        self.validity_section = CollapsibleSection(
            "Advanced spatial validity",
            description=(
                "AnalysisDomain, valid terrain and observer/target "
                "regions are distinct concepts."
            ),
            badge=BadgeType.ADVANCED,
            expanded=False,
        )

        text = QLabel(
            "Terrain NoData will become invalid analysis space. "
            "Additional validity masks and explicit observer/target "
            "regions will be connected in a later World refinement."
        )

        text.setWordWrap(True)
        text.setProperty(
            "secondaryText",
            True,
        )

        self.validity_section.content_layout.addWidget(
            text
        )

        self.content_layout.addWidget(
            self.validity_section
        )

    # ------------------------------------------------------------------
    # Bottom action
    # ------------------------------------------------------------------

    def _build_action_bar(self) -> None:
        self.action_bar = ActionBar()

        self.continue_button = make_primary_button(
            "Continue to Observability"
        )

        self.continue_button.setEnabled(False)

        self.action_bar.add_primary_action(
            self.continue_button
        )

        self.content_layout.addWidget(
            self.action_bar
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(
            lambda: self._open_environment_dialog(
                preferred_tab=0
            )
        )

        self.download_button.clicked.connect(
            lambda: self._open_environment_dialog(
                preferred_tab=1
            )
        )

        self.apply_domain_button.clicked.connect(
            self._apply_domain_choice
        )

        self.clear_domain_button.clicked.connect(
            self._clear_domain
        )

        self.world_map.domain_polygon_drawn.connect(
            self._on_polygon_drawn
        )

        self.world_map.viewpoint_selected.connect(
            self._on_viewpoint_selected
        )

        self.continue_button.clicked.connect(
            self.continue_requested.emit
        )

    # ------------------------------------------------------------------
    # Survey helpers
    # ------------------------------------------------------------------

    def _viewpoints(self):
        configuration = (
            self.state
            .survey
            .viewpoint_configuration
        )

        if configuration is None:
            return []

        return list(
            configuration.viewpoints
        )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _open_environment_dialog(
        self,
        *,
        preferred_tab: int,
    ) -> None:
        dialog = EnvironmentImportDialog(
            viewpoints=self._viewpoints(),
            task_controller=self.task_controller,
            parent=self,
        )

        dialog.tabs.setCurrentIndex(
            preferred_tab
        )

        if not dialog.exec():
            return

        result = dialog.environment_result

        if result is None:
            return

        self._install_environment_result(
            result
        )

    def _install_environment_result(
        self,
        result,
    ) -> None:
        self.state.set_environment(result.environment, grid=result.grid)
        self.refresh_from_state()
        self.state_changed.emit()

        # Give the user a useful initial domain.
        self.full_domain_radio.setChecked(True)
        self._apply_domain_choice()

        self._refresh_qc()

    # ------------------------------------------------------------------
    # Domain construction
    # ------------------------------------------------------------------

    def _grid(self):
        return self.state.analysis.analysis_grid

    def _apply_domain_choice(self) -> None:
        grid = self._grid()

        if grid is None:
            QMessageBox.information(
                self,
                "Terrain required",
                "Load terrain before defining an analysis area.",
            )
            return

        try:
            if self.full_domain_radio.isChecked():
                result = build_domain_from_grid_extent(
                    grid=grid,
                    domain_id=uuid4().hex,
                    name="Entire terrain",
                )

                self._install_domain(
                    result.domain
                )

            elif self.survey_domain_radio.isChecked():
                result = build_domain_from_survey_extent(
                    viewpoints=self._viewpoints(),
                    grid=grid,
                    domain_id=uuid4().hex,
                    name="Buffered survey extent",
                    buffer_m=float(
                        self.buffer_spin.value()
                    ),
                )

                self._install_domain(
                    result.domain
                )

            elif self.draw_domain_radio.isChecked():
                self.world_map.start_domain_drawing()

            elif self.import_domain_radio.isChecked():
                dialog = DomainImportDialog(
                    grid=grid,
                    parent=self,
                )

                if dialog.exec():
                    result = dialog.domain_result

                    if result is not None:
                        self._install_domain(
                            result.domain
                        )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to define analysis area",
                str(exc),
            )

    def _on_polygon_drawn(
        self,
        vertices,
    ) -> None:
        grid = self._grid()

        if grid is None:
            return

        try:
            result = build_domain_from_drawn_polygon(
                vertices=vertices,
                map_crs=grid.crs,
                grid=grid,
                domain_id=uuid4().hex,
                name="Drawn analysis area",
            )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid analysis area",
                str(exc),
            )
            return

        self._install_domain(
            result.domain
        )

    def _install_domain(
        self,
        domain,
    ) -> None:
        self.state.set_analysis_domain(domain)
        self.refresh_from_state()
        self.state_changed.emit()


    def _clear_domain(self) -> None:
        self.state.set_analysis_domain(None)
        self.refresh_from_state()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_viewpoint_selected(
        self,
        viewpoint_id: str,
    ) -> None:
        self.state.select_viewpoint(
            viewpoint_id
        )

    # ------------------------------------------------------------------
    # QC
    # ------------------------------------------------------------------

    def _refresh_qc(self) -> None:
        grid = self._grid()

        if grid is None:
            self.qc_summary.setText(
                "Load terrain to run spatial checks."
            )
            self.continue_button.setEnabled(
                False
            )
            return

        report = run_world_qc(
            viewpoints=self._viewpoints(),
            grid=grid,
            domain=self.state.analysis.analysis_domain,
        )

        lines = [
            f"Terrain CRS: {report.terrain_crs}",
            (
                f"Viewpoints inside terrain: "
                f"{report.viewpoints_inside_terrain:,} / "
                f"{report.total_viewpoints:,}"
            ),
        ]

        if (
            report.viewpoints_inside_domain
            is not None
        ):
            lines.append(
                (
                    f"Viewpoints inside analysis area: "
                    f"{report.viewpoints_inside_domain:,} / "
                    f"{report.total_viewpoints:,}"
                )
            )

        if report.issues:
            lines.append("")

            symbols = {
                QCSeverity.SUCCESS: "✓",
                QCSeverity.INFO: "ℹ",
                QCSeverity.WARNING: "⚠",
                QCSeverity.ERROR: "✗",
            }

            for issue in report.issues:
                lines.append(
                    f"{symbols[issue.severity]} "
                    f"{issue.message}"
                )

        self.qc_summary.setText(
            "\n".join(lines)
        )

        # World is ready when terrain/domain exist and there is no blocking
        # spatial error. Viewpoints outside the domain are not themselves
        # blocking.
        self.continue_button.setEnabled(
            self.state.world_ready
            and report.compatible
        )

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh_from_state(self) -> None:
        """Rehydrate all World UI from ApplicationState."""

        environment = self.state.analysis.environment
        grid = self.state.analysis.analysis_grid
        domain = self.state.analysis.analysis_domain

        if environment is None or grid is None:
            self.terrain_name.set_value("Not loaded")
            self.terrain_crs.set_value("—")
            self.terrain_resolution.set_value("—")
            self.terrain_size.set_value("—")
            self.world_map.set_domain(None)
        else:
            source = environment.elevation_model.source
            if self.world_map.raster_path != source:
                self.world_map.set_raster(source)
            self.terrain_name.set_value(environment.name)
            self.terrain_crs.set_value(grid.crs.to_string())
            self.terrain_resolution.set_value(
                f"{grid.resolution_x:g} × {grid.resolution_y:g}"
            )
            self.terrain_size.set_value(
                f"{grid.width:,} × {grid.height:,}"
            )
            self.world_map.set_domain(domain)

        self.world_map.set_viewpoints(self._viewpoints())

        self._refresh_qc()

        if domain is None:
            self.domain_summary.setText("No AnalysisDomain defined.")
        else:
            cells = int(domain.effective_analysis_mask.sum())
            self.domain_summary.setText(
                f"{domain.name} · {cells:,} analysis cells"
            )