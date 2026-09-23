"""Analysis & Design workflow page for Rivelero.

Overview (A1) provides descriptive analysis of the current Survey
Observability Field: how much valid analysis space was observable, how much
stayed blind, how much was covered by exactly one sampling unit, how much
received repeated observation opportunity, the exposure distribution, and
where those patterns are located.

Contribution (A2) attributes that coverage to individual sampling units; see
rivelero.gui.contribution_panel. Scenario (A3) is a non-destructive what-if
design environment; see rivelero.gui.scenario_panel.

The page never holds its own copy of the SOF. It reads the current result
from ApplicationState on every refresh and only caches the cheap derived
CoverageSummary, keyed by the identity of that SOF. When no current SOF
exists (never built, or invalidated by an upstream change) the analysis is
hidden rather than shown for stale inputs.
"""

from __future__ import annotations

from matplotlib.figure import Figure

try:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QFormLayout, QGridLayout, QHBoxLayout, QLabel, QScrollArea,
        QTabWidget, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QFormLayout, QGridLayout, QHBoxLayout, QLabel, QScrollArea,
        QTabWidget, QVBoxLayout, QWidget,
    )

from rivelero.analysis.coverage import CoverageSummary, summarize_coverage
from rivelero.gui.application_state import ApplicationState
from rivelero.gui.components import (
    ActionBar, BadgeType, CollapsibleSection, ContentCard, EmptyState,
    LabeledValue, PageHeader, StatusBadge, make_primary_button,
)
from rivelero.gui.contribution_panel import ContributionPanel
from rivelero.gui.raster_map import SafeFigureCanvas
from rivelero.gui.scenario_panel import ScenarioPanel
from rivelero.gui.task_controller import TaskController
from rivelero.gui.observability_map import (
    ObservabilityMapMode, ObservabilityMapWidget,
)
from rivelero.gui.theme import SPACING
from rivelero.visibility.configuration import SamplingUnit
from rivelero.visualization.analysis import (
    plot_coverage_composition, plot_exposure_distribution,
)


ANALYSIS_MAP_MODES = (
    ObservabilityMapMode.COVERAGE_CLASS,
    ObservabilityMapMode.EXPOSURE,
    ObservabilityMapMode.BLIND_SPOTS,
    ObservabilityMapMode.OBSERVABILITY_STATE,
)

DENOMINATOR_NOTE = (
    "Percentages are shares of analysable cells: cells inside the "
    "AnalysisDomain with valid terrain. Invalid and outside-domain cells are "
    "reported separately and never enter a percentage. Repeated coverage "
    "means two or more sampling units could observe a cell; it does not "
    "imply that the extra observations are without value."
)

_DESIGN_TOOLS = (
    (
        "Configuration comparison",
        "Coming later",
        "Compare coverage, blind spots and repeated coverage between survey "
        "configurations.",
    ),
)


class AnalysisPage(QWidget):
    """Descriptive analysis of the current Survey Observability Field."""

    observability_requested = Signal()
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

        # MainWindow passes the shared controller; a private one is created
        # only when the page is used stand-alone.
        if task_controller is None:
            task_controller = TaskController(state, parent=self)
        self.task_controller = task_controller

        # Derived, non-persistent cache: (sof object, summary).
        self._summary_source = None
        self._summary: CoverageSummary | None = None

        self._build_interface()
        self._connect_signals()
        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def summary(self) -> CoverageSummary | None:
        """Summary of the current SOF, or None when no current SOF exists."""
        return self._summary

    @property
    def analysis_available(self) -> bool:
        return self._summary is not None

    def refresh_from_state(self) -> None:
        """Rehydrate from ApplicationState; never analyse a stale SOF."""

        sof = self.state.analysis.survey_observability_field

        if sof is None:
            self._summary_source = None
            self._summary = None
            self._show_readiness()
            return

        if sof is not self._summary_source:
            self._summary = summarize_coverage(sof)
            self._summary_source = sof

        self.readiness.setVisible(False)
        self.tabs.setVisible(True)
        self.continue_button.setEnabled(True)

        self._refresh_summary(sof)
        self._refresh_map(sof)
        self._refresh_charts()
        self.contribution_panel.refresh_from_state()
        self.scenario_panel.refresh_from_state()

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

        layout.addWidget(PageHeader(
            "Analysis & Design",
            "Understand how the observation opportunity created by the survey "
            "is distributed: where it is unique, where it is repeated, and "
            "where the survey left blind spots.",
        ))

        self.readiness = EmptyState(
            "Build observability before analysing the survey.",
            "Analysis uses the current Survey Observability Field.",
            action_text="Go to Observability",
        )
        layout.addWidget(self.readiness)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.analysis_content = QWidget()
        content_layout = QVBoxLayout(self.analysis_content)
        content_layout.setContentsMargins(0, SPACING.lg, 0, 0)
        content_layout.setSpacing(SPACING.xl)
        self._build_summary_card(content_layout)
        self._build_map_card(content_layout)
        self._build_chart_row(content_layout)
        self.tabs.addTab(self.analysis_content, "Overview")

        self.contribution_panel = ContributionPanel(
            self.state, task_controller=self.task_controller
        )
        self.tabs.addTab(self.contribution_panel, "Contribution")

        self.scenario_panel = ScenarioPanel(
            self.state, task_controller=self.task_controller
        )
        self.tabs.addTab(self.scenario_panel, "Scenario")
        layout.addWidget(self.tabs)

        self._build_design_tools(layout)

        self.action_bar = ActionBar()
        self.continue_button = make_primary_button("Continue to Output")
        self.continue_button.setEnabled(False)
        self.action_bar.add_primary_action(self.continue_button)
        layout.addWidget(self.action_bar)

        layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll)

    def _build_summary_card(self, layout: QVBoxLayout) -> None:
        card = ContentCard(
            title="Coverage summary",
            description="Derived from the current Survey Observability Field.",
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACING.xxxl)
        grid.setVerticalSpacing(SPACING.sm)

        groups = (
            ("COVERAGE", (
                ("observable", "Observable"),
                ("blind", "Blind spots"),
            )),
            ("EXPOSURE", (
                ("unique", "Unique coverage"),
                ("repeated", "Repeated coverage"),
                ("mean", "Mean exposure"),
                ("mean_observable", "Mean exposure where observable"),
                ("median", "Median exposure"),
                ("maximum", "Maximum exposure"),
            )),
            ("ANALYSIS SPACE", (
                ("analysable", "Analysable cells"),
                ("invalid", "Invalid cells in domain"),
                ("outside", "Cells outside domain"),
                ("active", "Active sampling units"),
                ("unit", "Sampling unit"),
            )),
        )

        self.summary_values: dict[str, LabeledValue] = {}
        for column, (heading, rows) in enumerate(groups):
            title = QLabel(heading)
            title.setProperty("sectionHeading", True)
            grid.addWidget(title, 0, column)
            for row, (name, label) in enumerate(rows, start=1):
                widget = LabeledValue(label)
                self.summary_values[name] = widget
                grid.addWidget(widget, row, column)
        for column in range(len(groups)):
            grid.setColumnStretch(column, 1)
        card.add_layout(grid)

        note = QLabel(DENOMINATOR_NOTE)
        note.setWordWrap(True)
        note.setProperty("secondaryText", True)
        card.add_widget(note)
        layout.addWidget(card)

    def _build_map_card(self, layout: QVBoxLayout) -> None:
        card = ContentCard(
            title="Spatial pattern",
            description=(
                "Global shares can hide spatial bias. The map shows where "
                "blind spots, unique coverage and repeated coverage occur."
            ),
        )
        self.analysis_map = ObservabilityMapWidget(modes=ANALYSIS_MAP_MODES)
        self.analysis_map.setMinimumHeight(560)
        card.add_widget(self.analysis_map)
        layout.addWidget(card)

    def _build_chart_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(SPACING.xl)

        distribution = ContentCard(
            title="Exposure distribution",
            description=(
                "How many sampling units could observe each analysable cell."
            ),
        )
        self.distribution_figure = Figure(figsize=(6, 3.4), dpi=100)
        self.distribution_canvas = SafeFigureCanvas(self.distribution_figure)
        self.distribution_canvas.setMinimumHeight(300)
        self.distribution_axes = self.distribution_figure.add_subplot(111)
        distribution.add_widget(self.distribution_canvas)
        self.distribution_readout = QLabel(" ")
        self.distribution_readout.setProperty("secondaryText", True)
        distribution.add_widget(self.distribution_readout)
        distribution.add_stretch()
        row.addWidget(distribution, 3)

        composition = ContentCard(
            title="Coverage composition",
            description="Blind, unique and repeated shares of analysable space.",
        )
        self.composition_figure = Figure(figsize=(4, 2.2), dpi=100)
        self.composition_canvas = SafeFigureCanvas(self.composition_figure)
        self.composition_canvas.setMinimumHeight(240)
        self.composition_axes = self.composition_figure.add_subplot(111)
        composition.add_widget(self.composition_canvas)
        composition.add_stretch()
        row.addWidget(composition, 2)

        layout.addLayout(row)

    def _build_design_tools(self, layout: QVBoxLayout) -> None:
        section = CollapsibleSection(
            "Survey design tools",
            description="Configuration comparison.",
            expanded=False,
        )
        form = QFormLayout()
        form.setHorizontalSpacing(SPACING.xl)
        form.setVerticalSpacing(SPACING.md)
        for title, badge, description in _DESIGN_TOOLS:
            name = QWidget()
            name_layout = QHBoxLayout(name)
            name_layout.setContentsMargins(0, 0, 0, 0)
            name_layout.addWidget(QLabel(title))
            name_layout.addWidget(StatusBadge(BadgeType.COMING_SOON, text=badge))
            name_layout.addStretch(1)
            text = QLabel(description)
            text.setWordWrap(True)
            text.setProperty("secondaryText", True)
            form.addRow(name, text)
        section.content_layout.addLayout(form)
        layout.addWidget(section)

    def _connect_signals(self) -> None:
        self.readiness.action_requested.connect(self.observability_requested.emit)
        self.continue_button.clicked.connect(self.continue_requested.emit)
        self.analysis_map.viewpoint_selected.connect(self._on_viewpoint_selected)
        self.distribution_canvas.mpl_connect(
            "motion_notify_event", self._on_distribution_hover
        )

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _show_readiness(self) -> None:
        reason = self.state.analysis.invalidation_reason
        if reason is not None:
            title = "Observability is out of date."
            description = (
                f"{reason} Rebuild the observability field before analysing "
                "the survey."
            )
        else:
            title = "Build observability before analysing the survey."
            description = (
                "Analysis uses the current Survey Observability Field, which "
                "has not been built yet."
            )
        self.readiness.title_label.setText(title)
        self.readiness.description_label.setText(description)
        self.readiness.setVisible(True)
        self.tabs.setVisible(False)
        self.continue_button.setEnabled(False)
        self.analysis_map.set_field(None)
        self.contribution_panel.refresh_from_state()
        self.scenario_panel.refresh_from_state()

    def _refresh_summary(self, sof) -> None:
        summary = self._summary

        def share(fraction, cells) -> str:
            if fraction is None:
                return "—"
            return f"{fraction:.1%}  ({cells:,} cells)"

        def number(value, digits: int = 2) -> str:
            return "—" if value is None else f"{value:.{digits}f}"

        values = {
            "observable": share(summary.observable_fraction, summary.observable_cells),
            "blind": share(summary.blind_fraction, summary.blind_cells),
            "unique": share(summary.unique_fraction, summary.unique_cells),
            "repeated": share(summary.repeated_fraction, summary.repeated_cells),
            "mean": number(summary.mean_exposure),
            "mean_observable": number(summary.mean_exposure_observable),
            "median": number(summary.median_exposure, 1),
            "maximum": f"{summary.maximum_exposure:,}",
            "analysable": f"{summary.analysable_cells:,}",
            "invalid": f"{summary.invalid_cells:,}",
            "outside": f"{summary.outside_domain_cells:,}",
            "active": f"{summary.active_units:,}",
            "unit": (
                "ObservationEvent"
                if sof.sampling_unit == SamplingUnit.OBSERVATION_EVENT
                else "Viewpoint"
            ),
        }
        for name, value in values.items():
            self.summary_values[name].set_value(value)

    def _refresh_map(self, sof) -> None:
        environment = self.state.analysis.environment
        survey = self.state.survey.viewpoint_configuration
        self.analysis_map.set_terrain(
            None if environment is None else environment.elevation_model.source
        )
        self.analysis_map.set_field(sof)
        self.analysis_map.set_viewpoints([] if survey is None else survey.viewpoints)
        self.analysis_map.set_selected_viewpoint(self.state.selection.viewpoint_id)

    def _refresh_charts(self) -> None:
        plot_exposure_distribution(
            self._summary, ax=self.distribution_axes, title=""
        )
        self.distribution_figure.tight_layout()
        self.distribution_canvas.draw_idle()

        plot_coverage_composition(
            self._summary, ax=self.composition_axes, title=""
        )
        self.composition_figure.subplots_adjust(
            left=0.04, right=0.96, top=0.95, bottom=0.62
        )
        self.composition_canvas.draw_idle()

    def _on_viewpoint_selected(self, viewpoint_id: str) -> None:
        # Same canonical selection as the Survey and Observability pages.
        # (ApplicationState is a slotted dataclass and cannot be weakly
        # referenced, so its bound methods cannot be connected directly.)
        self.state.select_viewpoint(viewpoint_id)

    def _on_distribution_hover(self, event) -> None:
        if self._summary is None or event.inaxes is not self.distribution_axes:
            return
        for container in self.distribution_axes.containers:
            bins = getattr(container, "_rivelero_bins", None)
            if bins is None:
                continue
            for patch, item in zip(container.patches, bins):
                if patch.contains(event)[0]:
                    self.distribution_readout.setText(
                        f"Exposure {item.label}: {item.cells:,} cells · "
                        f"{item.fraction:.1%} of analysable cells"
                    )
                    return
