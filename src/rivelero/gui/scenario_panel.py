"""What-if survey design panel of the Analysis & Design page (A3).

The panel edits a temporary SurveyDesignScenario owned by ApplicationState
(``analysis.design_scenario``). It never modifies the Survey, the baseline
SOF or its exposure; applying a scenario to the Survey is deliberately not
available yet.

Existing units are deactivated/reactivated through selection-based actions
on one scalable model/view table (no per-row widgets). Their cached masks are
read on the shared TaskController, so bulk selections do not block the GUI.
Candidates are canonical Viewpoints whose visibility is computed lazily
through the canonical engine and VisibilityStore, also on the TaskController.

Every background result is applied only if the scenario it was started for
is still the current scenario of the current baseline; otherwise it is
discarded. Baseline A2 contribution values, when shown, are labelled as
baseline values.

Candidate additions are available for Viewpoint sampling only: with
ObservationEvent sampling a candidate would also need a canonical
ObservationEvent, which is deferred rather than approximated.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import (
        QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView, QComboBox, QDialog, QGridLayout, QHBoxLayout,
        QHeaderView, QInputDialog, QLabel, QMessageBox, QTableView, QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import (
        QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt,
        pyqtSignal as Signal,
    )
    from PyQt6.QtWidgets import (
        QAbstractItemView, QComboBox, QDialog, QGridLayout, QHBoxLayout,
        QHeaderView, QInputDialog, QLabel, QMessageBox, QTableView, QVBoxLayout,
        QWidget,
    )

from rasterio.warp import transform as transform_coordinates

from rivelero.analysis.comparison import ScenarioWorkspace
from rivelero.analysis.scenario import (
    CandidateStatus, DesignCandidate, SurveyDesignScenario, suggest_candidate_id,
)
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.application_state import ApplicationState
from rivelero.gui.components import (
    ContentCard, LabeledValue, make_primary_button, make_secondary_button,
)
from rivelero.gui.observability_map import (
    ObservabilityMapMode, ObservabilityMapWidget,
)
from rivelero.gui.observability_service import (
    VisibilityKeyResolver, candidate_terrain_problem,
    compute_candidate_visibility, load_cached_masks,
)
from rivelero.gui.task_controller import TaskController, make_progress_task
from rivelero.gui.theme import SPACING, refresh_style
from rivelero.gui.viewpoint_dialog import ViewpointDialog
from rivelero.visibility.configuration import SamplingUnit


SCENARIO_MASK_TASK_NAME = "Reading cached visibility"
CANDIDATE_TASK_NAME = "Computing candidate visibility"

SCENARIO_MAP_MODES = (
    ObservabilityMapMode.SCENARIO_CHANGE,
    ObservabilityMapMode.SCENARIO_EXPOSURE,
    ObservabilityMapMode.EXPOSURE,
    ObservabilityMapMode.COVERAGE_CLASS,
)

SCENARIO_MAP_LABELS = {
    ObservabilityMapMode.EXPOSURE: "Baseline exposure",
    ObservabilityMapMode.COVERAGE_CLASS: "Baseline unique / repeated coverage",
}

EVENT_CANDIDATE_NOTE = (
    "Candidate additions are not available with ObservationEvent sampling: a "
    "candidate would also need a canonical ObservationEvent. Existing units "
    "can still be deactivated."
)

_UNIT_ROLE = Qt.ItemDataRole.UserRole + 1
_SORT_ROLE = Qt.ItemDataRole.UserRole

FILTERS = (
    ("all", "All units"),
    ("active", "Active in scenario"),
    ("inactive", "Deactivated in scenario"),
    ("baseline_unique", "Baseline unique coverage (A2)"),
    ("no_baseline_unique", "No baseline unique coverage (A2)"),
)


def _signed(value, *, digits: int = 0, suffix: str = "") -> str:
    if value is None:
        return "—"
    if digits:
        # Avoid "-0.00" for differences that round to zero.
        value = round(value, digits) + 0.0
        return f"{value:+.{digits}f}{suffix}"
    return f"{value:+,}{suffix}"


def _units(count: int, noun: str = "unit") -> str:
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def _percent(value) -> str:
    return "—" if value is None else f"{value:.1%}"


# ---------------------------------------------------------------------------
# Existing-unit table
# ---------------------------------------------------------------------------


class ScenarioUnitsModel(QAbstractTableModel):
    """Baseline active units with their state in the current scenario."""

    HEADERS = (
        "Sampling unit", "Viewpoint", "State in scenario",
        "Baseline unique cells (A2)",
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._keys = ()
        self._scenario: SurveyDesignScenario | None = None
        self._baseline_unique: dict = {}

    def set_source(self, scenario, keys, baseline_unique) -> None:
        self.beginResetModel()
        self._scenario = scenario
        self._keys = tuple(keys)
        self._baseline_unique = dict(baseline_unique)
        self.endResetModel()

    def refresh_states(self) -> None:
        if self._keys:
            self.dataChanged.emit(self.index(0, 2), self.index(len(self._keys) - 1, 2))

    def key_at(self, row: int):
        return self._keys[row]

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._keys)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._scenario is None:
            return None
        key = self._keys[index.row()]
        column = index.column()
        if role == _UNIT_ROLE:
            return key
        if column == 0:
            value = key.sampling_unit_id
        elif column == 1:
            value = key.viewpoint_id
        elif column == 2:
            value = "Active" if self._scenario.is_active(key) else "Deactivated"
        else:
            unique = self._baseline_unique.get(key)
            if role == _SORT_ROLE:
                return float("-inf") if unique is None else unique
            value = "Not analysed" if unique is None else f"{unique:,}"
        if role in (Qt.ItemDataRole.DisplayRole, _SORT_ROLE):
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole and column == 3:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def baseline_unique(self, key):
        return self._baseline_unique.get(key)


class ScenarioUnitsFilter(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mode = "all"
        self.setSortRole(_SORT_ROLE)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.refilter()

    def refilter(self) -> None:
        """Re-apply the filter (Qt >= 6.10 API, fallback for older Qt)."""
        if hasattr(self, "beginFilterChange"):
            self.beginFilterChange()
            self.endFilterChange()
        else:
            self.invalidateFilter()

    def filterAcceptsRow(self, row, parent) -> bool:
        model = self.sourceModel()
        key = model.key_at(row)
        scenario = model._scenario
        if self.mode == "active":
            return scenario.is_active(key)
        if self.mode == "inactive":
            return not scenario.is_active(key)
        unique = model.baseline_unique(key)
        if self.mode == "baseline_unique":
            return unique is not None and unique > 0
        if self.mode == "no_baseline_unique":
            return unique is not None and unique == 0
        return True


# ---------------------------------------------------------------------------
# Candidate table
# ---------------------------------------------------------------------------


class CandidatesModel(QAbstractTableModel):
    HEADERS = (
        "Candidate", "X", "Y", "Status", "Visible cells",
        "Gain vs current scenario", "In scenario",
    )

    _STATUS_TEXT = {
        CandidateStatus.PENDING: "Waiting",
        CandidateStatus.COMPUTING: "Computing…",
        CandidateStatus.READY: "Cached",
        CandidateStatus.FAILED: "Failed",
        CandidateStatus.EXCLUDED: "Excluded by policy",
        CandidateStatus.CANCELLED: "Cancelled",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scenario: SurveyDesignScenario | None = None
        self._rows: tuple[DesignCandidate, ...] = ()

    def set_scenario(self, scenario) -> None:
        self.beginResetModel()
        self._scenario = scenario
        self._rows = () if scenario is None else scenario.candidates
        self.endResetModel()

    def candidate_at(self, row: int) -> DesignCandidate:
        return self._rows[row]

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.ToolTipRole
            and section == 5
        ):
            return (
                "Cells this candidate makes observable given every other "
                "scenario modification (newly observable cells)."
            )
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        candidate = self._rows[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.ToolTipRole:
            return candidate.message
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        viewpoint = candidate.viewpoint
        if column == 0:
            return candidate.candidate_id
        if column == 1:
            return f"{viewpoint.x:,.1f}"
        if column == 2:
            return f"{viewpoint.y:,.1f}"
        if column == 3:
            return self._STATUS_TEXT[candidate.status]
        if candidate.mask is None:
            return "—"
        if column == 4:
            return f"{int(candidate.mask.sum()):,}"
        if column == 5:
            gain = self._scenario.candidate_marginal_gain(candidate.candidate_id)
            return _signed(gain.coverage_cells, suffix=" cells")
        return "Yes" if candidate.included else "No"


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class ScenarioPanel(QWidget):
    """Non-destructive what-if design of the current survey."""

    # The live scenario or the saved scenarios changed (saved with the
    # project, so the project becomes dirty).
    edited = Signal()

    def __init__(
        self,
        state: ApplicationState,
        *,
        task_controller: TaskController,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.task_controller = task_controller
        self._resolver = VisibilityKeyResolver()

        self._mask_task: tuple[str, SurveyDesignScenario] | None = None
        self._candidate_task: tuple[str, SurveyDesignScenario, str, Viewpoint] | None = None
        self._shown_scenario = None
        self._shown_contribution = None

        self._build_interface()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> SurveyDesignScenario | None:
        """Current scenario of the current baseline, created on demand."""
        sof = self.state.analysis.survey_observability_field
        if sof is None:
            return None
        scenario = self.state.analysis.design_scenario
        if scenario is None or scenario.baseline is not sof:
            scenario = SurveyDesignScenario(sof)
            self.state.set_design_scenario(scenario)
        return scenario

    @property
    def busy(self) -> bool:
        return self._mask_task is not None or self._candidate_task is not None

    def selected_keys(self) -> list:
        rows = self.units_table.selectionModel().selectedRows()
        return [self.units_proxy.data(index, _UNIT_ROLE) for index in rows]

    def selected_candidates(self) -> list[DesignCandidate]:
        rows = self.candidates_table.selectionModel().selectedRows()
        return [self.candidates_model.candidate_at(index.row()) for index in rows]

    def refresh_from_state(self) -> None:
        scenario = self.scenario
        contribution = self.state.analysis.contribution_analysis
        if scenario is not self._shown_scenario or contribution is not self._shown_contribution:
            self._shown_scenario = scenario
            self._shown_contribution = contribution
            keys = () if scenario is None else scenario.baseline.active_keys
            baseline_unique = (
                {}
                if contribution is None
                else {unit.key: unit.unique_cells for unit in contribution.units}
            )
            self.units_model.set_source(scenario, keys, baseline_unique)
            event_based = (
                scenario is not None
                and scenario.baseline.sampling_unit == SamplingUnit.OBSERVATION_EVENT
            )
            self.units_table.setColumnHidden(1, not event_based)
            sof = self.state.analysis.survey_observability_field
            self.scenario_map.set_field(sof)
            environment = self.state.analysis.environment
            self.scenario_map.set_terrain(
                None if environment is None else environment.elevation_model.source
            )
            survey = self.state.survey.viewpoint_configuration
            self.scenario_map.set_viewpoints([] if survey is None else survey.viewpoints)
        self._refresh_scenario_views()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _build_interface(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACING.lg, 0, 0)
        layout.setSpacing(SPACING.xl)

        summary = ContentCard(
            title="Scenario",
            description=(
                "Temporarily deactivate existing sampling units or add "
                "candidate Viewpoints and see how observable coverage would "
                "change. The Survey and the baseline observability field are "
                "never modified."
            ),
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACING.xxxl)
        grid.setVerticalSpacing(SPACING.sm)
        self.summary_values: dict[tuple[str, str], QLabel] = {}
        for column, heading in enumerate(("", "BASELINE", "SCENARIO", "DIFFERENCE")):
            label = QLabel(heading)
            label.setProperty("sectionHeading", True)
            grid.addWidget(label, 0, column)
        rows = (
            ("units", "Sampling units"),
            ("coverage", "Coverage (observable share)"),
            ("observable", "Observable cells"),
            ("blind", "Blind-spot cells"),
            ("unique", "Unique-coverage cells"),
            ("repeated", "Repeated-coverage cells"),
            ("mean", "Mean exposure"),
            ("maximum", "Maximum exposure"),
        )
        for row, (name, text) in enumerate(rows, start=1):
            label = QLabel(text)
            label.setProperty("secondaryText", True)
            grid.addWidget(label, row, 0)
            for column, part in enumerate(("baseline", "scenario", "difference"), start=1):
                value = QLabel("—")
                value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.summary_values[(name, part)] = value
                grid.addWidget(value, row, column)
        for column in range(4):
            grid.setColumnStretch(column, 1)
        summary.add_layout(grid)

        note = QLabel(
            "Shares use the baseline's analysable cells: a scenario changes the "
            "Survey, not the World. Effects are evaluated against the current "
            "scenario, so the loss from removing several units can exceed the "
            "sum of their individual (A2) removal impacts."
        )
        note.setWordWrap(True)
        note.setProperty("secondaryText", True)
        summary.add_widget(note)

        actions = QHBoxLayout()
        self.reset_button = make_secondary_button("Reset scenario")
        self.save_button = make_primary_button("Save scenario for comparison")
        self.apply_button = make_secondary_button("Apply scenario to Survey — Coming later")
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip(
            "Committing a design changes Survey identity, provenance and cache "
            "reuse; it will be designed separately."
        )
        actions.addWidget(self.save_button)
        actions.addWidget(self.reset_button)
        actions.addStretch(1)
        actions.addWidget(self.apply_button)
        summary.add_layout(actions)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        summary.add_widget(self.status_label)
        layout.addWidget(summary)

        # Existing units -------------------------------------------------
        units = ContentCard(
            title="Existing sampling units",
            description="Select one or more rows, then deactivate or reactivate them.",
        )
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show"))
        self.filter_combo = QComboBox()
        for value, text in FILTERS:
            self.filter_combo.addItem(text, value)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch(1)
        self.deactivate_button = make_primary_button("Deactivate selected")
        self.reactivate_button = make_secondary_button("Reactivate selected")
        filter_row.addWidget(self.deactivate_button)
        filter_row.addWidget(self.reactivate_button)
        units.add_layout(filter_row)

        self.units_model = ScenarioUnitsModel(self)
        self.units_proxy = ScenarioUnitsFilter(self)
        self.units_proxy.setSourceModel(self.units_model)
        self.units_table = self._make_table(self.units_proxy)
        self.units_table.setSortingEnabled(True)
        self.units_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.units_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        units.add_widget(self.units_table)
        self.unit_effect_label = QLabel()
        self.unit_effect_label.setWordWrap(True)
        self.unit_effect_label.setProperty("secondaryText", True)
        units.add_widget(self.unit_effect_label)
        layout.addWidget(units)

        # Candidates -----------------------------------------------------
        candidates = ContentCard(
            title="Candidate Viewpoints",
            description=(
                "Temporary Viewpoints. Their visibility is computed with the "
                "current visibility configuration and cached like any other."
            ),
        )
        candidate_actions = QHBoxLayout()
        self.add_candidate_button = make_primary_button("Add candidate…")
        self.place_candidate_button = make_secondary_button("Place candidate on map")
        self.toggle_candidate_button = make_secondary_button("Include / exclude selected")
        self.remove_candidate_button = make_secondary_button("Remove selected")
        self.cancel_candidate_button = make_secondary_button("Cancel computation")
        self.cancel_candidate_button.setVisible(False)
        for button in (
            self.add_candidate_button, self.place_candidate_button,
            self.toggle_candidate_button, self.remove_candidate_button,
            self.cancel_candidate_button,
        ):
            candidate_actions.addWidget(button)
        candidate_actions.addStretch(1)
        candidates.add_layout(candidate_actions)
        self.candidate_note = QLabel()
        self.candidate_note.setWordWrap(True)
        self.candidate_note.setProperty("secondaryText", True)
        candidates.add_widget(self.candidate_note)
        self.candidates_model = CandidatesModel(self)
        self.candidates_table = self._make_table(self.candidates_model)
        self.candidates_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.candidates_table.setMinimumHeight(150)
        candidates.add_widget(self.candidates_table)
        layout.addWidget(candidates)

        # Map ------------------------------------------------------------
        map_card = ContentCard(
            title="Scenario map",
            description=(
                "Where coverage would be lost or gained relative to the "
                "baseline. Candidates are shown as diamonds."
            ),
        )
        self.scenario_map = ObservabilityMapWidget(
            modes=SCENARIO_MAP_MODES, labels=SCENARIO_MAP_LABELS
        )
        self.scenario_map.setMinimumHeight(560)
        map_card.add_widget(self.scenario_map)
        layout.addWidget(map_card)

    @staticmethod
    def _make_table(model) -> QTableView:
        table = QTableView()
        table.setModel(model)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(220)
        return table

    def _connect_signals(self) -> None:
        self.reset_button.clicked.connect(self._reset)
        self.save_button.clicked.connect(lambda: self.save_snapshot())
        self.filter_combo.currentIndexChanged.connect(
            lambda _i: self.units_proxy.set_mode(self.filter_combo.currentData())
        )
        self.deactivate_button.clicked.connect(self._deactivate_selected)
        self.reactivate_button.clicked.connect(self._reactivate_selected)
        self.units_table.selectionModel().selectionChanged.connect(
            self._on_units_selection
        )
        self.add_candidate_button.clicked.connect(lambda: self._add_candidate())
        self.place_candidate_button.clicked.connect(self._place_candidate)
        self.toggle_candidate_button.clicked.connect(self._toggle_candidates)
        self.remove_candidate_button.clicked.connect(self._remove_candidates)
        self.cancel_candidate_button.clicked.connect(self._cancel_candidate)
        self.scenario_map.point_picked.connect(self._on_point_picked)
        self.scenario_map.viewpoint_selected.connect(self._on_map_viewpoint)

        controller = self.task_controller
        controller.task_result.connect(self._on_task_result)
        controller.task_error.connect(self._on_task_error)
        controller.task_cancelled.connect(self._on_task_cancelled)
        controller.task_finished.connect(self._on_task_finished)

    # ------------------------------------------------------------------
    # Existing units
    # ------------------------------------------------------------------

    def _deactivate_selected(self) -> None:
        scenario = self.scenario
        store = self.state.analysis.visibility_store
        keys = [key for key in self.selected_keys() if scenario.is_active(key)]
        if not keys or store is None or self.state.busy:
            return
        task = make_progress_task(
            function=load_cached_masks, kwargs={"store": store, "keys": keys}
        )
        task_id = self.task_controller.start(
            task_name=SCENARIO_MASK_TASK_NAME,
            function=task,
            total=len(keys),
            inject_context=True,
        )
        self._mask_task = (task_id, scenario)
        self._set_status(f"Reading cached visibility of {_units(len(keys))}…", kind="info")
        self._refresh_controls()

    def _reactivate_selected(self) -> None:
        scenario = self.scenario
        keys = [key for key in self.selected_keys() if not scenario.is_active(key)]
        if keys:
            scenario.reactivate(keys)
            self._edited()
            self._set_status(f"Reactivated {_units(len(keys))}.", kind="success")
            self._refresh_scenario_views()

    def _on_units_selection(self, *_args) -> None:
        keys = self.selected_keys()
        self._refresh_controls()
        if len(keys) != 1:
            self.unit_effect_label.setText(
                f"{_units(len(keys))} selected." if keys else ""
            )
            return
        key = keys[0]
        if key.sampling_unit_type == SamplingUnit.OBSERVATION_EVENT.value:
            self.state.select_observation_event(
                key.sampling_unit_id, viewpoint_id=key.viewpoint_id
            )
        else:
            self.state.select_viewpoint(key.viewpoint_id)
        self.scenario_map.set_selected_viewpoint(key.viewpoint_id)
        self._show_unit_effect(key)

    def _show_unit_effect(self, key) -> None:
        scenario = self.scenario
        store = self.state.analysis.visibility_store
        if not scenario.is_active(key):
            self.unit_effect_label.setText(
                f"{key.sampling_unit_id} is deactivated in this scenario."
            )
            return
        stored = None if store is None else store.get(key)  # cache read only
        if stored is None:
            self.unit_effect_label.setText(
                f"No cached visibility for {key.sampling_unit_id}."
            )
            return
        effect = scenario.marginal_loss(stored.visibility_mask)
        baseline = self.units_model.baseline_unique(key)
        text = (
            f"Deactivating {key.sampling_unit_id} now would turn "
            f"{effect.coverage_cells:,} cells blind "
            f"({_percent(effect.coverage_share_of_analysable)} of analysable) "
            "given the current scenario"
        )
        if baseline is not None:
            text += f"; its baseline unique contribution (A2) is {baseline:,} cells"
        self.unit_effect_label.setText(text + ".")

    def _on_map_viewpoint(self, viewpoint_id: str) -> None:
        self.state.select_viewpoint(viewpoint_id)
        for row in range(self.units_proxy.rowCount()):
            key = self.units_proxy.data(self.units_proxy.index(row, 0), _UNIT_ROLE)
            if key.viewpoint_id == viewpoint_id:
                self.units_table.selectRow(row)
                return

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def _reserved_ids(self) -> set[str]:
        survey = self.state.survey.viewpoint_configuration
        reserved = set() if survey is None else set(survey.viewpoint_ids)
        scenario = self.scenario
        if scenario is not None:
            reserved |= {candidate.candidate_id for candidate in scenario.candidates}
        return reserved

    def _template_candidate(self, xy=None) -> Viewpoint:
        grid = self.state.analysis.analysis_grid
        if xy is None:
            left, top = grid.transform * (0, 0)
            right, bottom = grid.transform * (grid.width, grid.height)
            xy = ((left + right) / 2.0, (top + bottom) / 2.0)
        return Viewpoint(
            viewpoint_id=suggest_candidate_id(self._reserved_ids()),
            x=float(xy[0]),
            y=float(xy[1]),
            crs=grid.crs.to_string(),
            source="design candidate",
        )

    def _place_candidate(self) -> None:
        self.scenario_map.start_point_pick()
        self._set_status("Click the map to place the candidate.", kind="info")

    def _on_point_picked(self, x: float, y: float) -> None:
        self._add_candidate(xy=(x, y))

    def _edit_candidate(self, template: Viewpoint) -> Viewpoint | None:
        """Open the canonical Viewpoint editor; overridable in tests."""
        dialog = ViewpointDialog(
            sensors=self.state.survey.sensors,
            viewpoint=template,
            title="Add candidate Viewpoint",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.result_viewpoint

    def _add_candidate(self, xy=None, viewpoint: Viewpoint | None = None) -> None:
        scenario = self.scenario
        if scenario is None or self._candidates_blocked_reason() is not None:
            return
        if viewpoint is None:
            viewpoint = self._edit_candidate(self._template_candidate(xy))
            if viewpoint is None:
                return
        try:
            scenario.add_candidate(viewpoint, reserved_ids=self._reserved_ids())
        except ValueError as exc:
            QMessageBox.warning(self, "Candidate not added", str(exc))
            return
        self._edited()
        problem = candidate_terrain_problem(viewpoint, self.state.analysis.analysis_grid)
        if problem is not None:
            scenario.set_candidate_status(viewpoint.viewpoint_id, CandidateStatus.FAILED, problem)
            self._set_status(problem, kind="warning")
        self._refresh_scenario_views()
        self._start_next_candidate()

    def _start_next_candidate(self) -> None:
        if self._candidate_task is not None or self.state.busy:
            return
        scenario = self.scenario
        if scenario is None:
            return
        pending = [c for c in scenario.candidates if c.status == CandidateStatus.PENDING]
        if not pending:
            return
        candidate = pending[0]
        try:
            selection = self._resolver.resolve_viewpoint(self.state, candidate.viewpoint)
        except ValueError as exc:
            scenario.set_candidate_status(candidate.candidate_id, CandidateStatus.FAILED, str(exc))
            self._refresh_scenario_views()
            self._start_next_candidate()
            return
        scenario.set_candidate_status(candidate.candidate_id, CandidateStatus.COMPUTING)
        task_id = self.task_controller.start(
            task_name=CANDIDATE_TASK_NAME,
            function=compute_candidate_visibility,
            kwargs={
                "selection": selection,
                "environment": self.state.analysis.environment,
                "domain": self.state.analysis.analysis_domain,
                "visibility_configuration": self.state.analysis.visibility_configuration,
                "store": self.state.analysis.visibility_store,
            },
            message=f"Computing visibility of {candidate.candidate_id}",
        )
        self._candidate_task = (task_id, scenario, candidate.candidate_id, candidate.viewpoint)
        self._refresh_scenario_views()

    def _cancel_candidate(self) -> None:
        # The GDAL viewshed cannot be interrupted; its result is discarded.
        if self._candidate_task is not None:
            self.task_controller.cancel()
            self.cancel_candidate_button.setEnabled(False)

    def _toggle_candidates(self) -> None:
        scenario = self.scenario
        changed = 0
        for candidate in self.selected_candidates():
            if candidate.status == CandidateStatus.READY:
                try:
                    scenario.set_candidate_included(
                        candidate.candidate_id, not candidate.included
                    )
                    changed += 1
                except OverflowError as exc:
                    self._set_status(str(exc), kind="error")
        if changed:
            self._edited()
            self._refresh_scenario_views()

    def _remove_candidates(self) -> None:
        scenario = self.scenario
        for candidate in self.selected_candidates():
            if candidate.status != CandidateStatus.COMPUTING:
                scenario.remove_candidate(candidate.candidate_id)
        self._edited()
        self._refresh_scenario_views()

    def _candidates_blocked_reason(self) -> str | None:
        scenario = self.scenario
        if scenario is None:
            return "Build observability first."
        if scenario.baseline.sampling_unit == SamplingUnit.OBSERVATION_EVENT:
            return EVENT_CANDIDATE_NOTE
        if self.state.analysis.visibility_store is None:
            return "Configure visibility storage first."
        return None

    # ------------------------------------------------------------------
    # Background results
    # ------------------------------------------------------------------

    def _is_current(self, scenario) -> bool:
        return (
            scenario is self.state.analysis.design_scenario
            and scenario.baseline is self.state.analysis.survey_observability_field
        )

    def _on_task_result(self, task_id, result) -> None:
        if self._mask_task is not None and task_id == self._mask_task[0]:
            scenario = self._mask_task[1]
            if not self._is_current(scenario):
                self._set_status("The baseline changed; the result was discarded.", kind="warning")
                return
            try:
                scenario.deactivate(result)
            except (KeyError, ValueError) as exc:
                self._set_status(str(exc), kind="error")
                return
            self._edited()
            self._set_status(f"Deactivated {_units(len(result))}.", kind="success")
            return

        if self._candidate_task is not None and task_id == self._candidate_task[0]:
            _task, scenario, candidate_id, viewpoint = self._candidate_task
            if not self._is_current(scenario) or not self._candidate_unchanged(
                scenario, candidate_id, viewpoint
            ):
                self._set_status(
                    "The scenario changed while the candidate was computed; "
                    "the result was discarded.",
                    kind="warning",
                )
                return
            if result.excluded:
                scenario.set_candidate_status(candidate_id, CandidateStatus.EXCLUDED, result.message)
                return
            try:
                scenario.set_candidate_visibility(
                    candidate_id,
                    key=result.key,
                    mask=result.stored.visibility_mask,
                    include=scenario.candidate(candidate_id).include_when_ready,
                )
            except (OverflowError, ValueError) as exc:
                scenario.set_candidate_status(candidate_id, CandidateStatus.FAILED, str(exc))

    def _candidate_unchanged(self, scenario, candidate_id, viewpoint) -> bool:
        try:
            return scenario.candidate(candidate_id).viewpoint is viewpoint
        except KeyError:
            return False

    def _on_task_error(self, task_id, message, _details) -> None:
        if self._mask_task is not None and task_id == self._mask_task[0]:
            self._set_status(f"Units were not deactivated: {message}", kind="error")
        elif self._candidate_task is not None and task_id == self._candidate_task[0]:
            _task, scenario, candidate_id, viewpoint = self._candidate_task
            if self._is_current(scenario) and self._candidate_unchanged(scenario, candidate_id, viewpoint):
                scenario.set_candidate_status(candidate_id, CandidateStatus.FAILED, message)
            self._set_status(f"Candidate visibility failed: {message}", kind="error")

    def _on_task_cancelled(self, task_id, _message) -> None:
        if self._candidate_task is not None and task_id == self._candidate_task[0]:
            _task, scenario, candidate_id, viewpoint = self._candidate_task
            if self._is_current(scenario) and self._candidate_unchanged(scenario, candidate_id, viewpoint):
                # Not PENDING: pending candidates start automatically.
                scenario.set_candidate_status(
                    candidate_id, CandidateStatus.CANCELLED,
                    "Computation cancelled; remove and add the candidate to retry.",
                )

    def _on_task_finished(self, task_id) -> None:
        started_next = False
        if self._mask_task is not None and task_id == self._mask_task[0]:
            self._mask_task = None
        elif self._candidate_task is not None and task_id == self._candidate_task[0]:
            self._candidate_task = None
            started_next = True
        self.refresh_from_state()
        if started_next or not self.state.busy:
            self._start_next_candidate()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    @property
    def workspace(self):
        workspace = self.state.analysis.scenario_workspace
        if workspace is None:
            workspace = ScenarioWorkspace()
            self.state.analysis.scenario_workspace = workspace
        return workspace

    def save_snapshot(self, name: str | None = None, description: str | None = None):
        """Save the live scenario for comparison (no visibility recomputed)."""
        scenario = self.scenario
        if scenario is None or self.busy:
            return None
        workspace = self.workspace
        if name is None:
            name, accepted = QInputDialog.getText(
                self, "Save scenario for comparison", "Name",
                text=workspace.suggest_name(),
            )
            if not accepted:
                return None
            description, accepted = QInputDialog.getMultiLineText(
                self, "Save scenario for comparison", "Description (optional)",
            )
            if not accepted:
                return None
        try:
            snapshot = workspace.save(scenario, name=name, description=description or "")
        except ValueError as exc:
            self._set_status(str(exc), kind="error")
            return None
        self._edited()
        self._set_status(
            f"Saved {snapshot.name!r} for comparison. The Survey is unchanged.",
            kind="success",
        )
        return snapshot

    def _reset(self) -> None:
        scenario = self.scenario
        if scenario is None or self.busy:
            return
        scenario.reset()
        self._edited()
        self._set_status("Scenario reset to the baseline.", kind="success")
        self.units_model.set_source(
            scenario, scenario.baseline.active_keys, self.units_model._baseline_unique
        )
        self._refresh_scenario_views()

    def _edited(self) -> None:
        self.state.notify_design_changed()
        self.edited.emit()

    def _refresh_scenario_views(self) -> None:
        scenario = self.scenario
        self.candidates_model.set_scenario(scenario)
        self.units_model.refresh_states()
        self.units_proxy.refilter()
        if scenario is None:
            for label in self.summary_values.values():
                label.setText("—")
            self.scenario_map.set_scenario_layers(None, None)
            self.scenario_map.set_candidate_points([])
            self._refresh_controls()
            return

        summary = scenario.summary()
        base, scen, diff = summary.baseline, summary.scenario, summary.difference
        values = {
            "units": (
                f"{summary.baseline_units:,}",
                f"{summary.active_existing_units:,} existing + "
                f"{_units(summary.included_candidates, 'candidate')}",
                _signed(
                    summary.active_existing_units + summary.included_candidates
                    - summary.baseline_units
                ),
            ),
            "coverage": (
                _percent(base.observable_fraction),
                _percent(scen.observable_fraction),
                _signed(diff.coverage_percentage_points, digits=2, suffix=" pp"),
            ),
            "observable": (f"{base.observable_cells:,}", f"{scen.observable_cells:,}",
                           _signed(diff.observable_cells)),
            "blind": (f"{base.blind_cells:,}", f"{scen.blind_cells:,}",
                      _signed(diff.blind_cells)),
            "unique": (f"{base.unique_cells:,}", f"{scen.unique_cells:,}",
                       _signed(diff.unique_cells)),
            "repeated": (f"{base.repeated_cells:,}", f"{scen.repeated_cells:,}",
                         _signed(diff.repeated_cells)),
            "mean": (
                "—" if base.mean_exposure is None else f"{base.mean_exposure:.2f}",
                "—" if scen.mean_exposure is None else f"{scen.mean_exposure:.2f}",
                _signed(diff.mean_exposure, digits=2),
            ),
            "maximum": (f"{base.maximum_exposure:,}", f"{scen.maximum_exposure:,}",
                        _signed(diff.maximum_exposure)),
        }
        for name, parts in values.items():
            for part, text in zip(("baseline", "scenario", "difference"), parts):
                self.summary_values[(name, part)].setText(text)

        self.scenario_map.set_scenario_layers(scenario.exposure, scenario.change_classes())
        self.scenario_map.set_candidate_points(self._candidate_points(scenario))
        keys = self.selected_keys()
        if len(keys) == 1:
            self._show_unit_effect(keys[0])
        self._refresh_controls()

    def _candidate_points(self, scenario):
        crs = scenario.baseline.crs
        points = []
        for candidate in scenario.candidates:
            viewpoint = candidate.viewpoint
            x, y = viewpoint.x, viewpoint.y
            if viewpoint.crs != crs:
                xs, ys = transform_coordinates(viewpoint.crs, crs, [x], [y])
                x, y = xs[0], ys[0]
            points.append((candidate.candidate_id, x, y))
        return points

    def _refresh_controls(self) -> None:
        scenario = self.scenario
        busy = self.state.busy
        selected = self.selected_keys() if scenario is not None else []
        self.deactivate_button.setEnabled(
            not busy and any(scenario.is_active(key) for key in selected)
        )
        self.reactivate_button.setEnabled(
            any(not scenario.is_active(key) for key in selected)
        )
        self.save_button.setEnabled(scenario is not None and not self.busy)
        self.reset_button.setEnabled(
            scenario is not None and not self.busy
            and not (scenario.summary().is_baseline and not scenario.candidates)
        )
        blocked = self._candidates_blocked_reason()
        for button in (self.add_candidate_button, self.place_candidate_button):
            button.setEnabled(blocked is None)
        self.toggle_candidate_button.setEnabled(blocked is None)
        self.remove_candidate_button.setEnabled(blocked is None)
        computing = self._candidate_task is not None
        self.cancel_candidate_button.setVisible(computing)
        self.cancel_candidate_button.setEnabled(
            computing and not self.task_controller.cancellation_requested
        )
        self.candidate_note.setText(blocked or "")
        self.candidate_note.setVisible(bool(blocked))

    def _set_status(self, text: str | None, *, kind: str = "success") -> None:
        if not text:
            self.status_label.setVisible(False)
            return
        for name in ("statusSuccess", "statusWarning", "statusError"):
            self.status_label.setProperty(name, False)
        prop = {
            "success": "statusSuccess",
            "warning": "statusWarning",
            "error": "statusError",
        }.get(kind)
        if prop:
            self.status_label.setProperty(prop, True)
        refresh_style(self.status_label)
        self.status_label.setText(text)
        self.status_label.setVisible(True)
