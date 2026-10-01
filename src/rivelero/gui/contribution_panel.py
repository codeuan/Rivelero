"""Sampling-unit contribution panel of the Analysis & Design page (A2).

Runs the Qt-free contribution analysis on the shared TaskController and
presents it as a sortable table, a selected-unit inspector and a
selected-unit contribution map.

The panel reads the SOF, VisibilityStore and contribution result from
ApplicationState. The contribution result is installed there only if the
SOF inputs did not change while it was being computed, and it is cleared
automatically whenever the SOF is invalidated or rebuilt.

Selection
---------
With Viewpoint sampling a row selects that Viewpoint in ApplicationState,
shared with Survey and Observability. With ObservationEvent sampling a row
selects the ObservationEvent together with the Viewpoint it references
(``select_observation_event(event, viewpoint_id=...)``), so other pages
highlight the right Viewpoint while the event stays the selected sampling
unit here.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import (
        QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
        QProgressBar, QTableView, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt6.QtCore import (
        QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt,
    )
    from PyQt6.QtWidgets import (
        QAbstractItemView, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
        QProgressBar, QTableView, QVBoxLayout, QWidget,
    )

from rivelero.analysis.contribution import (
    ContributionAnalysis, ContributionStatus, SamplingUnitContribution,
    analyse_contributions, unit_contribution_classes,
)
from rivelero.gui.application_state import (
    ApplicationState, StaleObservabilityResultError,
)
from rivelero.gui.components import (
    ContentCard, LabeledValue, make_primary_button, make_secondary_button,
)
from rivelero.gui.observability_map import (
    ObservabilityMapMode, ObservabilityMapWidget,
)
from rivelero.gui.task_controller import TaskController, make_progress_task
from rivelero.gui.theme import SPACING, refresh_style
from rivelero.visibility.configuration import SamplingUnit


CONTRIBUTION_TASK_NAME = "Analysing contributions"

CONTRIBUTION_MAP_MODES = (
    ObservabilityMapMode.UNIT_CONTRIBUTION,
    ObservabilityMapMode.COVERAGE_CLASS,
    ObservabilityMapMode.EXPOSURE,
)

NO_UNIQUE_NOTE = (
    "No unique spatial coverage under the current observability model: every "
    "cell this unit observes is also observed by another sampling unit. This "
    "does not mean the unit is without value; repeated observation can add "
    "robustness, temporal information or other viewing directions."
)

INVARIANT_NOTE = (
    "Unique cells summed over all units equal the field's unique-coverage "
    "cells, because each exposure-1 cell belongs to exactly one unit. "
    "Repeated cells do not sum to the field's repeated-coverage cells: a cell "
    "seen by k units is counted once for each of them."
)

_SORT_ROLE = Qt.ItemDataRole.UserRole
_UNIT_ROLE = Qt.ItemDataRole.UserRole + 1


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _count(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


# (header, sort value, display text, tooltip)
_COLUMNS = (
    (
        "Sampling unit",
        lambda unit: unit.sampling_unit_id,
        lambda unit: unit.sampling_unit_id,
        "Viewpoint or ObservationEvent contributing to the field.",
    ),
    (
        "Viewpoint",
        lambda unit: unit.viewpoint_id,
        lambda unit: unit.viewpoint_id,
        "Viewpoint referenced by the sampling unit.",
    ),
    (
        "Visible cells",
        lambda unit: unit.visible_cells,
        lambda unit: _count(unit.visible_cells),
        "Analysable cells observable from this unit.",
    ),
    (
        "Unique cells",
        lambda unit: unit.unique_cells,
        lambda unit: _count(unit.unique_cells),
        "Analysable cells observable from this unit only (exposure 1).",
    ),
    (
        "Repeated cells",
        lambda unit: unit.repeated_cells,
        lambda unit: _count(unit.repeated_cells),
        "Analysable cells this unit observes that other units also observe.",
    ),
    (
        "Coverage lost if removed",
        lambda unit: unit.coverage_loss_if_removed,
        lambda unit: _percent(unit.coverage_loss_if_removed),
        "Unique cells as a share of all analysable cells: the drop in "
        "observable share if this unit alone were removed (counterfactual).",
    ),
    (
        "Unique share of own visibility",
        lambda unit: unit.unique_share_of_unit_visibility,
        lambda unit: _percent(unit.unique_share_of_unit_visibility),
        "Unique cells / this unit's visible cells.",
    ),
    (
        "Repeated share of own visibility",
        lambda unit: unit.repeated_share_of_unit_visibility,
        lambda unit: _percent(unit.repeated_share_of_unit_visibility),
        "Repeated cells / this unit's visible cells.",
    ),
    (
        "Status",
        lambda unit: unit.status.value,
        lambda unit: {
            ContributionStatus.AVAILABLE: "Available",
            ContributionStatus.MISSING: "Mask missing",
            ContributionStatus.UNREADABLE: "Mask unreadable",
            ContributionStatus.INCONSISTENT: "Mask inconsistent",
        }[unit.status],
        "Whether the unit's cached visibility could be read.",
    ),
)

VIEWPOINT_COLUMN = 1


class ContributionTableModel(QAbstractTableModel):
    """Read-only table of SamplingUnitContribution rows."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._units: tuple[SamplingUnitContribution, ...] = ()

    def set_units(self, units) -> None:
        self.beginResetModel()
        self._units = tuple(units)
        self.endResetModel()

    def unit_at(self, row: int) -> SamplingUnitContribution:
        return self._units[row]

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._units)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return _COLUMNS[section][0]
        if role == Qt.ItemDataRole.ToolTipRole:
            return _COLUMNS[section][3]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        unit = self._units[index.row()]
        _header, sort_value, display, _tip = _COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return display(unit)
        if role == _SORT_ROLE:
            value = sort_value(unit)
            # Unavailable values sort below every real value.
            return float("-inf") if value is None else value
        if role == _UNIT_ROLE:
            return unit
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() >= 2:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole and unit.message:
            return unit.message
        return None


class ContributionPanel(QWidget):
    """Contribution analysis, table, inspector and map."""

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

        self._task_id: str | None = None
        self._task_revision: int | None = None
        self._syncing = False
        self._shown_analysis: ContributionAnalysis | None = None

        self._build_interface()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def analysis(self) -> ContributionAnalysis | None:
        """Contribution analysis of the current SOF, if one exists."""
        return self.state.analysis.contribution_analysis

    @property
    def running(self) -> bool:
        return self._task_id is not None

    def selected_unit(self) -> SamplingUnitContribution | None:
        index = self.table.currentIndex()
        if not index.isValid():
            return None
        return self.proxy.data(index, _UNIT_ROLE)

    def select_unit(self, sampling_unit_id: str) -> bool:
        """Select a table row by sampling-unit ID."""
        for row in range(self.proxy.rowCount()):
            unit = self.proxy.data(self.proxy.index(row, 0), _UNIT_ROLE)
            if unit.sampling_unit_id == sampling_unit_id:
                self.table.selectRow(row)
                return True
        return False

    def refresh_from_state(self) -> None:
        sof = self.state.analysis.survey_observability_field
        analysis = self.analysis

        self.contribution_map.set_field(sof)
        environment = self.state.analysis.environment
        self.contribution_map.set_terrain(
            None if environment is None else environment.elevation_model.source
        )
        survey = self.state.survey.viewpoint_configuration
        self.contribution_map.set_viewpoints(
            [] if survey is None else survey.viewpoints
        )

        if analysis is not self._shown_analysis:
            self._shown_analysis = analysis
            self.model.set_units(() if analysis is None else analysis.units)
            self.table.setColumnHidden(
                VIEWPOINT_COLUMN,
                analysis is None
                or analysis.sampling_unit != SamplingUnit.OBSERVATION_EVENT,
            )
            self._refresh_summary()

        self._refresh_controls()
        self._sync_selection_from_state()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _build_interface(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACING.lg, 0, 0)
        layout.setSpacing(SPACING.xl)

        run_card = ContentCard(
            title="Sampling-unit contribution",
            description=(
                "For each sampling unit, how much analysable space it observes, "
                "how much of that only it observes, and how much observable "
                "coverage would be lost if it alone were removed. Derived from "
                "cached visibility and the aggregate exposure; nothing is "
                "recomputed or modified."
            ),
        )
        buttons = QHBoxLayout()
        self.analyse_button = make_primary_button("Analyse contributions")
        self.cancel_button = make_secondary_button("Cancel")
        self.cancel_button.setVisible(False)
        buttons.addWidget(self.analyse_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        run_card.add_layout(buttons)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        run_card.add_widget(self.progress_bar)
        self.progress_label = QLabel()
        self.progress_label.setProperty("secondaryText", True)
        self.progress_label.setVisible(False)
        run_card.add_widget(self.progress_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        run_card.add_widget(self.status_label)

        summary = QGridLayout()
        summary.setHorizontalSpacing(SPACING.xxxl)
        summary.setVerticalSpacing(SPACING.sm)
        self.summary_values = {}
        for column, rows in (
            (0, (
                ("analysed", "Sampling units analysed"),
                ("with_unique", "Units with unique coverage"),
                ("without_unique", "Units with no unique coverage"),
                ("without_visible", "Units observing no analysable cells"),
            )),
            (1, (
                ("unavailable", "Units without readable cached visibility"),
                ("unique_total", "Unique cells summed over units"),
                ("field_unique", "Unique-coverage cells in the field"),
                ("consistent", "Consistent with the field"),
            )),
        ):
            for row, (name, label) in enumerate(rows):
                widget = LabeledValue(label)
                self.summary_values[name] = widget
                summary.addWidget(widget, row, column)
        summary.setColumnStretch(0, 1)
        summary.setColumnStretch(1, 1)
        run_card.add_layout(summary)

        note = QLabel(INVARIANT_NOTE)
        note.setWordWrap(True)
        note.setProperty("secondaryText", True)
        run_card.add_widget(note)
        layout.addWidget(run_card)

        table_card = ContentCard(
            title="Contribution by sampling unit",
            description=(
                "Click a column header to sort. Ordering is descriptive; a "
                "high or low value is not in itself better or worse."
            ),
        )
        self.model = ContributionTableModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(_SORT_ROLE)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(260)
        table_card.add_widget(self.table)
        layout.addWidget(table_card)

        row = QHBoxLayout()
        row.setSpacing(SPACING.xl)

        inspector = ContentCard(
            title="Selected contribution",
            description="Counterfactual metrics; nothing is removed.",
        )
        form = QFormLayout()
        form.setHorizontalSpacing(SPACING.xl)
        form.setVerticalSpacing(SPACING.sm)
        self.inspector_values = {}
        for name, label in (
            ("unit", "Sampling unit"),
            ("viewpoint", "Viewpoint"),
            ("event", "ObservationEvent"),
            ("sensor", "Sensor"),
            ("source", "Source"),
            ("visible", "Visible cells"),
            ("unique", "Unique cells"),
            ("repeated", "Repeated cells"),
            ("loss", "Coverage lost if removed"),
            ("unique_own", "Unique share of own visibility"),
            ("repeated_own", "Repeated share of own visibility"),
        ):
            value = QLabel("—")
            value.setWordWrap(True)
            self.inspector_values[name] = value
            form.addRow(label, value)
        inspector.add_layout(form)
        self.inspector_note = QLabel()
        self.inspector_note.setWordWrap(True)
        self.inspector_note.setProperty("secondaryText", True)
        inspector.add_widget(self.inspector_note)
        inspector.add_stretch()
        row.addWidget(inspector, 2)

        map_card = ContentCard(
            title="Contribution map",
            description=(
                "Where the selected unit's unique and repeated contributions "
                "are located."
            ),
        )
        self.contribution_map = ObservabilityMapWidget(modes=CONTRIBUTION_MAP_MODES)
        self.contribution_map.setMinimumHeight(520)
        map_card.add_widget(self.contribution_map)
        row.addWidget(map_card, 3)

        layout.addLayout(row)

    def _connect_signals(self) -> None:
        self.analyse_button.clicked.connect(self._start_analysis)
        self.cancel_button.clicked.connect(self._cancel_analysis)
        self.table.selectionModel().currentRowChanged.connect(self._on_row_changed)
        self.contribution_map.viewpoint_selected.connect(self._on_map_viewpoint)

        controller = self.task_controller
        controller.task_progress.connect(self._on_task_progress)
        controller.task_result.connect(self._on_task_result)
        controller.task_error.connect(self._on_task_error)
        controller.task_cancelled.connect(self._on_task_cancelled)
        controller.task_finished.connect(self._on_task_finished)

    # ------------------------------------------------------------------
    # Background analysis
    # ------------------------------------------------------------------

    def _start_analysis(self) -> None:
        sof = self.state.analysis.survey_observability_field
        store = self.state.analysis.visibility_store
        if sof is None or store is None or self.state.busy:
            self._refresh_controls()
            return

        task = make_progress_task(
            function=analyse_contributions,
            kwargs={"sof": sof, "store": store},
        )
        self._task_revision = self.state.analysis.inputs_revision
        total = sof.n_active_units
        self._task_id = self.task_controller.start(
            task_name=CONTRIBUTION_TASK_NAME,
            function=task,
            total=total,
            message=f"Analysing contributions of {total:,} sampling units",
            inject_context=True,
        )
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Processed 0 / {total:,}")
        self._set_status(None)
        self._refresh_controls()

    def _cancel_analysis(self) -> None:
        if self._task_id is not None and self.task_controller.cancel():
            self.cancel_button.setEnabled(False)
            self._set_status(
                "Cancellation requested; stopping after the current mask.",
                kind="warning",
            )

    def _on_task_progress(self, task_id, processed, total, unit_id, _message) -> None:
        if task_id != self._task_id:
            return
        if total:
            self.progress_bar.setRange(0, int(total))
        self.progress_bar.setValue(int(processed))
        text = f"Processed {processed:,} / {total:,}" if total else f"Processed {processed:,}"
        if unit_id:
            text += f" · current {unit_id}"
        self.progress_label.setText(text)

    def _on_task_result(self, task_id, result) -> None:
        if task_id != self._task_id:
            return
        try:
            self.state.set_contribution_analysis(
                result, inputs_revision=self._task_revision
            )
        except StaleObservabilityResultError as exc:
            self._set_status(f"{exc} The result was discarded.", kind="warning")
            return

        if result.complete:
            self._set_status(
                f"Analysed {result.n_units:,} sampling units.", kind="success"
            )
        else:
            self._set_status(
                f"Analysed {result.n_units:,} sampling units; "
                f"{len(result.unavailable_units):,} have no readable cached "
                "visibility and are listed as unavailable, not as zero. "
                "Rebuild the observability field to restore them.",
                kind="warning",
            )

    def _on_task_error(self, task_id, message, _details) -> None:
        if task_id == self._task_id:
            self._set_status(
                f"Contribution analysis failed: {message}", kind="error"
            )

    def _on_task_cancelled(self, task_id, _message) -> None:
        if task_id == self._task_id:
            self._set_status(
                "Contribution analysis cancelled. No partial result was installed.",
                kind="warning",
            )

    def _on_task_finished(self, task_id) -> None:
        if task_id == self._task_id:
            self._task_id = None
            self._task_revision = None
        self.refresh_from_state()

    def _set_status(self, text: str | None, *, kind: str = "success") -> None:
        if not text:
            self.status_label.setVisible(False)
            return
        for name in ("statusSuccess", "statusWarning", "statusError"):
            self.status_label.setProperty(name, False)
        self.status_label.setProperty(
            {"success": "statusSuccess", "warning": "statusWarning"}.get(
                kind, "statusError"
            ),
            True,
        )
        refresh_style(self.status_label)
        self.status_label.setText(text)
        self.status_label.setVisible(True)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_controls(self) -> None:
        running = self.running
        ready = (
            self.state.analysis.survey_observability_field is not None
            and self.state.analysis.visibility_store is not None
        )
        self.analyse_button.setEnabled(ready and not running and not self.state.busy)
        self.analyse_button.setText(
            "Re-analyse contributions" if self.analysis is not None
            else "Analyse contributions"
        )
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(
            running and not self.task_controller.cancellation_requested
        )
        self.progress_bar.setVisible(running)
        self.progress_label.setVisible(running)

    def _refresh_summary(self) -> None:
        analysis = self.analysis
        if analysis is None:
            for widget in self.summary_values.values():
                widget.set_value("—")
            return
        consistent = analysis.consistent_with_field
        values = {
            "analysed": f"{analysis.n_units:,}",
            "with_unique": f"{analysis.n_with_unique_coverage:,}",
            "without_unique": f"{analysis.n_without_unique_coverage:,}",
            "without_visible": f"{analysis.n_without_visible_cells:,}",
            "unavailable": f"{len(analysis.unavailable_units):,}",
            "unique_total": f"{analysis.total_unique_contribution_cells:,}",
            "field_unique": f"{analysis.field_unique_cells:,}",
            "consistent": (
                "Not checked (units unavailable)" if consistent is None
                else "Yes" if consistent else "No — investigate"
            ),
        }
        for name, value in values.items():
            self.summary_values[name].set_value(value)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_row_changed(self, current, _previous) -> None:
        unit = self.proxy.data(current, _UNIT_ROLE) if current.isValid() else None
        self._show_unit(unit)
        if unit is None or self._syncing:
            return
        if unit.observation_event_id is not None:
            self.state.select_observation_event(
                unit.observation_event_id, viewpoint_id=unit.viewpoint_id
            )
        else:
            self.state.select_viewpoint(unit.viewpoint_id)

    def _on_map_viewpoint(self, viewpoint_id: str) -> None:
        self.state.select_viewpoint(viewpoint_id)
        self._sync_selection_from_state()

    def _sync_selection_from_state(self) -> None:
        """Select the row matching the canonical state selection."""
        selection = self.state.selection
        target = None
        for row in range(self.proxy.rowCount()):
            unit = self.proxy.data(self.proxy.index(row, 0), _UNIT_ROLE)
            if (
                selection.observation_event_id is not None
                and unit.observation_event_id == selection.observation_event_id
            ):
                target = row
                break
            if target is None and unit.viewpoint_id == selection.viewpoint_id:
                target = row
                if unit.observation_event_id is None:
                    break
        current = self.table.currentIndex()
        if target is None:
            if not current.isValid():
                self._show_unit(None)
            return
        if current.isValid() and current.row() == target:
            self._show_unit(self.selected_unit())
            return
        self._syncing = True
        try:
            self.table.selectRow(target)
        finally:
            self._syncing = False

    def _show_unit(self, unit: SamplingUnitContribution | None) -> None:
        values = self.inspector_values
        if unit is None:
            for label in values.values():
                label.setText("—")
            self.inspector_note.setText(
                "Select a sampling unit in the table."
                if self.analysis is not None
                else "Analyse contributions to inspect sampling units."
            )
            self.contribution_map.set_selected_viewpoint(None)
            self.contribution_map.set_unit_contribution(None)
            return

        viewpoint = None
        survey = self.state.survey.viewpoint_configuration
        if survey is not None and unit.viewpoint_id in survey.viewpoint_ids:
            viewpoint = survey.get_viewpoint(unit.viewpoint_id)

        values["unit"].setText(unit.sampling_unit_id)
        values["viewpoint"].setText(unit.viewpoint_id)
        values["event"].setText(unit.observation_event_id or "— (Viewpoint sampling)")
        values["sensor"].setText(
            (viewpoint.sensor_id if viewpoint else None) or "—"
        )
        values["source"].setText((viewpoint.source if viewpoint else None) or "—")

        if unit.available:
            values["visible"].setText(
                f"{unit.visible_cells:,}  ({_percent(unit.visible_share_of_analysable)} "
                "of analysable)"
            )
            values["unique"].setText(_count(unit.unique_cells))
            values["repeated"].setText(_count(unit.repeated_cells))
            values["loss"].setText(
                f"{_percent(unit.coverage_loss_if_removed)} of analysable "
                f"({unit.unique_cells:,} cells would become blind spots)"
            )
            values["unique_own"].setText(_percent(unit.unique_share_of_unit_visibility))
            values["repeated_own"].setText(
                _percent(unit.repeated_share_of_unit_visibility)
            )
            if unit.visible_cells == 0:
                note = "This unit observes no analysable cells."
            elif unit.unique_cells == 0:
                note = NO_UNIQUE_NOTE
            else:
                note = ""
        else:
            for name in ("visible", "unique", "repeated", "loss", "unique_own",
                         "repeated_own"):
                values[name].setText("Unavailable")
            note = unit.message or ""
        self.inspector_note.setText(note)

        self.contribution_map.set_selected_viewpoint(unit.viewpoint_id)
        self._show_unit_map(unit)

    def _show_unit_map(self, unit: SamplingUnitContribution) -> None:
        sof = self.state.analysis.survey_observability_field
        store = self.state.analysis.visibility_store
        if sof is None or store is None or not unit.available:
            self.contribution_map.set_unit_contribution(None)
            return
        # Cache read only; a missing mask is shown as unavailable above.
        stored = store.get(unit.key)
        if stored is None:
            self.contribution_map.set_unit_contribution(None)
            return
        self.contribution_map.set_unit_contribution(
            unit_contribution_classes(
                stored.visibility_mask,
                exposure_count=sof.exposure_count,
                analysis_mask=sof.analysis_mask,
                valid_mask=sof.valid_mask,
            ),
            unit.sampling_unit_id,
        )
