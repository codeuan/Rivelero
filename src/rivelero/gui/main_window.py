"""Main application window for the new Rivelero GUI.

This module defines the visual shell of Rivelero.

Scientific page implementations live in separate modules. MainWindow is
responsible only for:

- application-level layout;
- workflow navigation;
- project-status presentation;
- global branding;
- page switching;
- connection to ApplicationState and TaskController.

Scientific calculations must never be implemented here.
"""

from __future__ import annotations

from pathlib import Path
from rivelero.gui.survey_page import SurveyPage
from rivelero.gui.survey_import_dialog import (
    SurveyImportDialog,
)
from rivelero.gui.viewpoint_dialog import ViewpointDialog
from rivelero.gui.sensor_dialog import SensorManagerDialog
from rivelero.gui.observation_event_dialog import ObservationEventManagerDialog
from rivelero.gui.world_page import WorldPage

from rivelero.gui.theme import (
    SIZES,
    refresh_style,
)

try:
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QFont, QIcon, QPixmap
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QStackedWidget,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"

except ImportError:
    from PyQt6.QtCore import Qt, QSize
    from PyQt6.QtGui import QFont, QIcon, QPixmap
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QStackedWidget,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt6"


from rivelero.gui.application_state import (
    ApplicationState,
    WorkflowPage,
)
from rivelero.gui.task_controller import (
    TaskController,
)


# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------


WINDOW_MINIMUM_WIDTH = SIZES.minimum_window_width
WINDOW_MINIMUM_HEIGHT = SIZES.minimum_window_height

SIDEBAR_WIDTH = SIZES.sidebar_width
HEADER_HEIGHT = SIZES.header_height

APP_TITLE = "Rivelero"
APP_SUBTITLE = "Spatial Observability Engine"


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    """Return the Rivelero repository root.

    Expected package layout:

        repository/
            Rivelero Icon.png
            Rivelero Logo.png
            src/
                rivelero/
                    gui/
                        main_window.py
    """

    current = Path(__file__).resolve()

    for parent in current.parents:

        if (
            (parent / "src" / "rivelero").exists()
            and (parent / "Rivelero Icon.png").exists()
        ):
            return parent

    # Development fallback corresponding to src/rivelero/gui/main_window.py.
    return current.parents[3]


def asset_path(
    filename: str,
) -> Path:
    """Return a repository-root GUI asset path."""

    return (
        repository_root()
        / filename
    )


# ---------------------------------------------------------------------------
# Reusable visual widgets
# ---------------------------------------------------------------------------


class NavigationButton(QPushButton):
    """Button representing one Rivelero workflow page."""

    def __init__(
        self,
        text: str,
        *,
        page: WorkflowPage,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            text,
            parent,
        )

        self.page = page

        self.setCheckable(
            True
        )

        self.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.setMinimumHeight(
            46
        )

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.setProperty(
            "navigation",
            True,
        )


class StatusIndicator(QWidget):
    """Compact project-readiness indicator."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self._ready = False

        layout = QHBoxLayout(
            self
        )

        layout.setContentsMargins(
            0,
            2,
            0,
            2,
        )

        layout.setSpacing(
            8
        )

        self.dot = QLabel(
            "●"
        )

        self.dot.setFixedWidth(
            14
        )

        self.label = QLabel(
            title
        )

        self.label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout.addWidget(
            self.dot
        )

        layout.addWidget(
            self.label
        )

        self.set_ready(
            False
        )

    def set_ready(
        self,
        ready: bool,
        detail: str | None = None,
    ) -> None:
        """Update readiness state."""

        self._ready = bool(
            ready
        )

        self.dot.setProperty(
            "ready",
            self._ready,
        )

        # Force Qt stylesheet refresh.
        refresh_style(
            self.dot
        )

        if detail:
            self.label.setText(
                detail
            )


class PlaceholderPage(QWidget):
    """Temporary shell page used before scientific pages are connected."""

    def __init__(
        self,
        *,
        title: str,
        description: str,
        sections: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        outer = QVBoxLayout(
            self
        )

        outer.setContentsMargins(
            38,
            32,
            38,
            32,
        )

        outer.setSpacing(
            18
        )

        title_label = QLabel(
            title
        )

        title_font = QFont()
        title_font.setPointSize(
            22
        )
        title_font.setBold(
            True
        )

        title_label.setFont(
            title_font
        )

        description_label = QLabel(
            description
        )

        description_label.setWordWrap(
            True
        )

        description_label.setProperty(
            "pageDescription",
            True,
        )

        outer.addWidget(
            title_label
        )

        outer.addWidget(
            description_label
        )

        for section in sections:

            card = QFrame()

            card.setProperty(
                "contentCard",
                True,
            )

            card_layout = QVBoxLayout(
                card
            )

            card_layout.setContentsMargins(
                20,
                18,
                20,
                18,
            )

            section_label = QLabel(
                section
            )

            section_font = QFont()
            section_font.setPointSize(
                12
            )
            section_font.setBold(
                True
            )

            section_label.setFont(
                section_font
            )

            coming = QLabel(
                "Interface module will be connected in the next implementation pass."
            )

            coming.setWordWrap(
                True
            )

            coming.setProperty(
                "secondaryText",
                True,
            )

            card_layout.addWidget(
                section_label
            )

            card_layout.addWidget(
                coming
            )

            outer.addWidget(
                card
            )

        outer.addStretch(
            1
        )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Main window of the new Rivelero interface."""

    def __init__(
        self,
        *,
        state: ApplicationState | None = None,
        task_controller: TaskController | None = None,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self.state = (
            ApplicationState()
            if state is None
            else state
        )

        if not isinstance(
            self.state,
            ApplicationState,
        ):
            raise TypeError(
                "state must be an ApplicationState or None."
            )

        self.task_controller = (
            TaskController(
                self.state,
                parent=self,
            )
            if task_controller is None
            else task_controller
        )

        if not isinstance(
            self.task_controller,
            TaskController,
        ):
            raise TypeError(
                "task_controller must be a TaskController or None."
            )

        self._navigation_buttons: dict[
            WorkflowPage,
            NavigationButton,
        ] = {}

        self._page_indices: dict[
            WorkflowPage,
            int,
        ] = {}

        self._configure_window()

        self._build_interface()

        self._connect_signals()

        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def _configure_window(
        self,
    ) -> None:

        self.setWindowTitle(
            f"{APP_TITLE} — {APP_SUBTITLE}"
        )

        self.setMinimumSize(
            WINDOW_MINIMUM_WIDTH,
            WINDOW_MINIMUM_HEIGHT,
        )

        icon = asset_path(
            "Rivelero Icon.png"
        )

        if icon.exists():
            self.setWindowIcon(
                QIcon(
                    str(icon)
                )
            )



    # ------------------------------------------------------------------
    # Main composition
    # ------------------------------------------------------------------

    def _build_interface(
        self,
    ) -> None:

        root = QWidget()

        root_layout = QVBoxLayout(
            root
        )

        root_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        root_layout.setSpacing(
            0
        )

        root_layout.addWidget(
            self._build_header()
        )

        body = QWidget()

        body_layout = QHBoxLayout(
            body
        )

        body_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        body_layout.setSpacing(
            0
        )

        body_layout.addWidget(
            self._build_sidebar()
        )

        self.page_stack = (
            self._build_page_stack()
        )

        body_layout.addWidget(
            self.page_stack,
            1,
        )

        root_layout.addWidget(
            body,
            1,
        )

        self.setCentralWidget(
            root
        )

        self._build_status_bar()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _build_header(
        self,
    ) -> QWidget:

        header = QFrame()

        header.setObjectName(
            "AppHeader"
        )

        header.setFixedHeight(
            HEADER_HEIGHT
        )

        layout = QHBoxLayout(
            header
        )

        layout.setContentsMargins(
            24,
            10,
            24,
            10,
        )

        layout.setSpacing(
            18
        )

        logo_label = QLabel()

        logo_path = asset_path(
            "Rivelero Logo.png"
        )

        if logo_path.exists():

            pixmap = QPixmap(
                str(
                    logo_path
                )
            )

            if not pixmap.isNull():

                pixmap = pixmap.scaled(
                    QSize(
                        190,
                        48,
                    ),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

                logo_label.setPixmap(
                    pixmap
                )

        if logo_label.pixmap() is None:
            logo_label.setText(
                APP_TITLE
            )

            fallback_font = QFont()
            fallback_font.setPointSize(
                20
            )
            fallback_font.setBold(
                True
            )

            logo_label.setFont(
                fallback_font
            )

        layout.addWidget(
            logo_label
        )

        subtitle = QLabel(
            APP_SUBTITLE
        )

        subtitle.setProperty(
            "headerSubtitle",
            True,
        )

        layout.addWidget(
            subtitle
        )

        layout.addStretch(
            1
        )

        self.project_name_label = QLabel()

        self.project_name_label.setProperty(
            "projectName",
            True,
        )

        layout.addWidget(
            self.project_name_label
        )

        self.dirty_indicator = QLabel(
            ""
        )

        self.dirty_indicator.setToolTip(
            "Project contains unsaved changes."
        )

        layout.addWidget(
            self.dirty_indicator
        )

        self.dirty_indicator.setProperty(
            "dirtyIndicator",
            True,
        )

        return header

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _build_sidebar(
        self,
    ) -> QWidget:

        sidebar = QFrame()

        sidebar.setObjectName(
            "Sidebar"
        )

        sidebar.setFixedWidth(
            SIDEBAR_WIDTH
        )

        layout = QVBoxLayout(
            sidebar
        )

        layout.setContentsMargins(
            16,
            22,
            16,
            18,
        )

        layout.setSpacing(
            7
        )

        workflow_label = QLabel(
            "WORKFLOW"
        )

        workflow_label.setProperty(
            "sectionHeading",
            True,
        )

        layout.addWidget(
            workflow_label
        )

        navigation = (
            (
                WorkflowPage.SURVEY,
                "1   Survey",
            ),
            (
                WorkflowPage.WORLD,
                "2   World",
            ),
            (
                WorkflowPage.OBSERVABILITY,
                "3   Observability",
            ),
            (
                WorkflowPage.ANALYSIS_DESIGN,
                "4   Analysis & Design",
            ),
            (
                WorkflowPage.OUTPUT,
                "5   Output",
            ),
        )

        for page, text in navigation:

            button = NavigationButton(
                text,
                page=page,
            )

            button.clicked.connect(
                lambda checked=False, p=page: (
                    self.navigate_to(
                        p
                    )
                )
            )

            self._navigation_buttons[
                page
            ] = button

            layout.addWidget(
                button
            )

        layout.addSpacing(
            24
        )

        separator = QFrame()

        separator.setFrameShape(
            QFrame.Shape.HLine
        )

        separator.setProperty(
            "sidebarSeparator",
            True,
        )

        layout.addWidget(
            separator
        )

        layout.addSpacing(
            12
        )

        project_label = QLabel(
            "PROJECT STATUS"
        )

        project_label.setProperty(
            "sectionHeading",
            True,
        )

        layout.addWidget(
            project_label
        )

        self.survey_status = (
            StatusIndicator(
                "Survey not defined"
            )
        )

        self.world_status = (
            StatusIndicator(
                "World not defined"
            )
        )

        self.visibility_status = (
            StatusIndicator(
                "Visibility not configured"
            )
        )

        self.sof_status = (
            StatusIndicator(
                "Observability not built"
            )
        )

        layout.addWidget(
            self.survey_status
        )

        layout.addWidget(
            self.world_status
        )

        layout.addWidget(
            self.visibility_status
        )

        layout.addWidget(
            self.sof_status
        )

        layout.addStretch(
            1
        )

        version_label = QLabel(
            "Open-source spatial observability"
        )

        version_label.setWordWrap(
            True
        )

        version_label.setProperty(
            "sidebarFooter",
            True,
        )

        layout.addWidget(
            version_label
        )

        return sidebar

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _build_page_stack(
        self,
    ) -> QStackedWidget:

        stack = QStackedWidget()

        self.survey_page = SurveyPage(
            self.state
        )

        self.world_page = WorldPage(
            self.state,
            task_controller=self.task_controller,
        )

        pages = {
            WorkflowPage.SURVEY: self.survey_page,

            WorkflowPage.WORLD: self.world_page,

            WorkflowPage.OBSERVABILITY: PlaceholderPage(
                title="Observability",
                description=(
                    "Configure visibility assumptions, reconstruct visual "
                    "survey effort and inspect individual and collective "
                    "observation opportunity."
                ),
                sections=(
                    "Visibility configuration",
                    "Build observability",
                    "Survey observability results",
                    "Individual visibility inspection",
                ),
            ),

            WorkflowPage.ANALYSIS_DESIGN: PlaceholderPage(
                title="Analysis & Design",
                description=(
                    "Quantify survey quality and evaluate alternative "
                    "viewpoint configurations using the same observability "
                    "engine."
                ),
                sections=(
                    "Coverage",
                    "Redundancy",
                    "Viewpoint contribution",
                    "Survey design",
                    "Experimental / research",
                ),
            ),

            WorkflowPage.OUTPUT: PlaceholderPage(
                title="Output",
                description=(
                    "Export spatial observability products and analysis "
                    "provenance for use in GIS, reproducible workflows and "
                    "downstream inference."
                ),
                sections=(
                    "Scientific raster products",
                    "Supporting layers",
                    "Provenance",
                ),
            ),
        }

        for page in WorkflowPage:

            widget = pages[
                page
            ]

            index = stack.addWidget(
                widget
            )

            self._page_indices[
                page
            ] = index

        return stack

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(
        self,
    ) -> None:

        status = QStatusBar()

        self.setStatusBar(
            status
        )

        self.status_message = QLabel(
            "Ready"
        )

        status.addWidget(
            self.status_message,
            1,
        )

        self.task_status_label = QLabel(
            ""
        )

        status.addPermanentWidget(
            self.task_status_label
        )

        brand = QLabel(
            "Rivelero"
        )

        brand.setProperty(
            "statusBrand",
            True,
        )

        status.addPermanentWidget(
            brand
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _connect_signals(
        self,
    ) -> None:

        self.survey_page.continue_requested.connect(
            lambda: self.navigate_to(
                WorkflowPage.WORLD
            )
        )

        self.world_page.continue_requested.connect(
            lambda: self.navigate_to(WorkflowPage.OBSERVABILITY)
        )
        self.world_page.state_changed.connect(
            self.refresh_from_state
        )

        self.survey_page.import_viewpoints_requested.connect(
            self._open_survey_import
        )

        self.survey_page.add_viewpoint_requested.connect(
            self._add_viewpoint
        )
        self.survey_page.edit_viewpoint_requested.connect(
            self._edit_viewpoint
        )
        self.survey_page.delete_viewpoint_requested.connect(
            self._delete_viewpoint
        )
        self.survey_page.sensors_requested.connect(
            self._manage_sensors
        )
        self.survey_page.observation_events_requested.connect(
            self._manage_observation_events
        )

        self.task_controller.task_started.connect(
            self._on_task_started
        )

        self.task_controller.task_progress.connect(
            self._on_task_progress
        )

        self.task_controller.task_error.connect(
            self._on_task_error
        )

        self.task_controller.task_cancelled.connect(
            self._on_task_cancelled
        )

        self.task_controller.task_finished.connect(
            self._on_task_finished
        )

    def _open_survey_import(
        self,
    ) -> None:
        """Open the canonical survey-import workflow."""

        dialog = SurveyImportDialog(
            parent=self
        )

        if not dialog.exec():
            return

        result = dialog.import_result

        if result is None:
            return

        self.state.set_viewpoint_configuration(
            result.viewpoint_configuration
        )

        self.state.set_sensors(
            result.sensors
        )

        self.survey_page.refresh_from_state()

        self.refresh_from_state()

        self.status_message.setText(
            (
                f"Imported "
                f"{result.report.viewpoints_imported:,} Viewpoints, "
                f"{result.report.sensors_imported:,} Sensors and "
                f"{result.report.events_imported:,} ObservationEvents."
            )
        )

    def _refresh_survey(self, message: str) -> None:
        self.survey_page.refresh_from_state()
        self.refresh_from_state()
        self.status_message.setText(message)

    def _add_viewpoint(self) -> None:
        dialog = ViewpointDialog(
            sensors=self.state.survey.sensors,
            parent=self,
        )
        if not dialog.exec() or dialog.result_viewpoint is None:
            return
        try:
            self.state.add_viewpoint(dialog.result_viewpoint)
        except (RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot add Viewpoint", str(exc))
            return
        self._refresh_survey("Added Viewpoint.")

    def _edit_viewpoint(self, viewpoint_id: str) -> None:
        configuration = self.state.survey.viewpoint_configuration
        if configuration is None:
            return
        try:
            viewpoint = configuration.get_viewpoint(viewpoint_id)
        except KeyError:
            return
        dialog = ViewpointDialog(
            sensors=self.state.survey.sensors,
            viewpoint=viewpoint,
            parent=self,
        )
        if not dialog.exec() or dialog.result_viewpoint is None:
            return
        try:
            self.state.replace_viewpoint(
                dialog.result_viewpoint,
                previous_id=viewpoint_id,
            )
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot edit Viewpoint", str(exc))
            return
        self._refresh_survey("Updated Viewpoint.")

    def _delete_viewpoint(self, viewpoint_id: str) -> None:
        if QMessageBox.question(
            self,
            "Delete Viewpoint",
            f"Delete Viewpoint {viewpoint_id!r}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.state.remove_viewpoint(viewpoint_id)
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot delete Viewpoint", str(exc))
            return
        self._refresh_survey("Deleted Viewpoint.")

    def _manage_sensors(self) -> None:
        configuration = self.state.survey.viewpoint_configuration
        referenced_ids = set()
        if configuration is not None:
            referenced_ids = {
                viewpoint.sensor_id
                for viewpoint in configuration.viewpoints
                if viewpoint.sensor_id is not None
            }
        dialog = SensorManagerDialog(
            sensors=self.state.survey.sensors,
            referenced_ids=referenced_ids,
            parent=self,
        )
        if not dialog.exec():
            return
        try:
            self.state.set_sensors(dialog.sensors)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot update Sensors", str(exc))
            return
        self._refresh_survey("Updated Sensors.")

    def _manage_observation_events(self) -> None:
        configuration = self.state.survey.viewpoint_configuration
        if configuration is None:
            QMessageBox.information(self, "Observation Events", "Import or add a Viewpoint before managing ObservationEvents.")
            return
        dialog = ObservationEventManagerDialog(
            configuration=configuration,
            parent=self,
        )
        if not dialog.exec():
            return
        try:
            self.state.set_observation_events(dialog.events)
        except (RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot update ObservationEvents", str(exc))
            return
        self._refresh_survey("Updated ObservationEvents.")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate_to(
        self,
        page: WorkflowPage | str,
    ) -> None:
        """Navigate to one workflow page."""

        if isinstance(
            page,
            str,
        ):
            page = WorkflowPage(
                page
            )

        self.state.set_active_page(
            page
        )

        widget = self.page_stack.widget(self._page_indices[page])
        refresh = getattr(widget, "refresh_from_state", None)
        if callable(refresh):
            refresh()

        self._apply_navigation_state()

    def _apply_navigation_state(
        self,
    ) -> None:

        active = (
            self.state.view.active_page
        )

        self.page_stack.setCurrentIndex(
            self._page_indices[
                active
            ]
        )

        for page, button in (
            self._navigation_buttons.items()
        ):
            button.setChecked(
                page == active
            )

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_from_state(
        self,
    ) -> None:
        """Refresh shell widgets from ApplicationState."""

        self._apply_navigation_state()

        self.project_name_label.setText(
            self.state.project.name
        )

        self.dirty_indicator.setText(
            "●"
            if self.state.project.dirty
            else ""
        )

        summary = (
            self.state.readiness_summary()
        )

        survey = summary[
            "survey"
        ]

        if survey["ready"]:

            detail = (
                f"{survey['viewpoints']:,} viewpoints"
            )

            if (
                survey[
                    "observation_events"
                ]
                > 0
            ):
                detail += (
                    f" · "
                    f"{survey['observation_events']:,} events"
                )

            self.survey_status.set_ready(
                True,
                detail,
            )

        else:
            self.survey_status.set_ready(
                False,
                "Survey not defined",
            )

        world = summary[
            "world"
        ]

        self.world_status.set_ready(
            world["ready"],
            (
                "World ready"
                if world["ready"]
                else "Terrain loaded"
                if world["environment"]
                else "World not defined"
            ),
        )

        visibility = summary[
            "visibility"
        ]

        self.visibility_status.set_ready(
            visibility["ready"],
            (
                "Visibility configured"
                if visibility["ready"]
                else "Visibility not configured"
            ),
        )

        observability = summary[
            "observability"
        ]

        if observability["ready"]:

            self.sof_status.set_ready(
                True,
                (
                    f"{observability['active_units']:,} "
                    "sampling units"
                ),
            )

        else:
            self.sof_status.set_ready(
                False,
                "Observability not built",
            )

        self._refresh_task_state()

    # ------------------------------------------------------------------
    # Task display
    # ------------------------------------------------------------------

    def _refresh_task_state(
        self,
    ) -> None:

        task = self.state.task

        if not task.busy:

            self.task_status_label.setText(
                ""
            )

            return

        if (
            task.total is not None
            and task.total > 0
        ):
            text = (
                f"{task.task_name}: "
                f"{task.processed:,}/{task.total:,}"
            )

        else:
            text = (
                task.task_name
                or "Working"
            )

        self.task_status_label.setText(
            text
        )

    def _on_task_started(
        self,
        task_id: str,
        task_name: str,
    ) -> None:

        self.status_message.setText(
            task_name
        )

        self.refresh_from_state()

    def _on_task_progress(
        self,
        task_id: str,
        processed: int,
        total,
        unit_id,
        message,
    ) -> None:

        if message:
            self.status_message.setText(
                str(message)
            )

        self._refresh_task_state()

    def _on_task_error(
        self,
        task_id: str,
        message: str,
        details: str,
    ) -> None:

        self.status_message.setText(
            message
        )

        self.refresh_from_state()

    def _on_task_cancelled(
        self,
        task_id: str,
        message,
    ) -> None:

        self.status_message.setText(
            str(
                message
                or "Task cancelled"
            )
        )

        self.refresh_from_state()

    def _on_task_finished(
        self,
        task_id: str,
    ) -> None:

        if (
            self.state.task.status.value
            == "succeeded"
        ):
            self.status_message.setText(
                "Ready"
            )

        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    @staticmethod
    def _stylesheet() -> str:
        """Return the base Rivelero GUI stylesheet.

        Colors are deliberately restrained here. Once the final Rivelero
        visual identity is frozen, these values should move into a central
        theme module.
        """

        return """
        QMainWindow {
            background: #f6f7f8;
        }

        QWidget {
            font-family: "Segoe UI", "Arial", sans-serif;
            font-size: 10pt;
            color: #202428;
        }

        #AppHeader {
            background: white;
            border-bottom: 1px solid #dfe3e6;
        }

        QLabel[headerSubtitle="true"] {
            color: #697177;
            font-size: 10pt;
        }

        QLabel[projectName="true"] {
            font-weight: 600;
            color: #30363a;
        }

        #Sidebar {
            background: #20282d;
            border-right: 1px solid #161c20;
        }

        #Sidebar QLabel {
            color: #e9edef;
        }

        QLabel[sectionHeading="true"] {
            color: #9ca8ae;
            font-size: 8pt;
            font-weight: 700;
        }

        QPushButton[navigation="true"] {
            background: transparent;
            color: #dce3e6;
            border: none;
            border-radius: 6px;
            text-align: left;
            padding: 9px 12px;
            font-weight: 500;
        }

        QPushButton[navigation="true"]:hover {
            background: #2d383e;
        }

        QPushButton[navigation="true"]:checked {
            background: #3a474e;
            color: white;
            font-weight: 700;
        }

        QLabel[ready="false"] {
            color: #738087;
        }

        QLabel[ready="true"] {
            color: #67b279;
        }

        QLabel[sidebarFooter="true"] {
            color: #849198;
            font-size: 8pt;
        }

        QLabel[pageDescription="true"] {
            color: #687177;
            font-size: 10.5pt;
        }

        QLabel[secondaryText="true"] {
            color: #7b8489;
        }

        QFrame[contentCard="true"] {
            background: white;
            border: 1px solid #dfe4e7;
            border-radius: 8px;
        }

        QStatusBar {
            background: white;
            border-top: 1px solid #dfe3e6;
            color: #596268;
        }

        QLabel[statusBrand="true"] {
            font-weight: 700;
            color: #596268;
            padding-left: 12px;
        }
        """