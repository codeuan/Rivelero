"""Design comparison panel of the Analysis & Design page (A4).

Compares two design states of the current baseline - the baseline, the
current (unsaved) scenario, or saved snapshots - and reports signed
differences as RIGHT minus LEFT. Nothing is ranked or scored.

Saved snapshots live in ApplicationState (``analysis.scenario_workspace``)
and persist across navigation and baseline rebuilds. A snapshot saved for an
earlier baseline is listed as out of date and cannot be selected for
comparison or loaded into the Scenario tab.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QInputDialog,
        QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QInputDialog,
        QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
        QWidget,
    )

from rivelero.analysis.comparison import (
    BASELINE_ID,
    LIVE_SCENARIO_ID,
    ScenarioComparison,
    ScenarioWorkspace,
    SnapshotUnavailableError,
    compare_states,
    restore_snapshot,
)
from rivelero.gui.application_state import ApplicationState
from rivelero.gui.components import (
    ContentCard, make_primary_button, make_secondary_button,
)
from rivelero.gui.observability_map import (
    ObservabilityMapMode, ObservabilityMapWidget,
)
from rivelero.gui.theme import SPACING, refresh_style


COMPARE_MAP_MODES = (
    ObservabilityMapMode.COMPARISON_CHANGE,
    ObservabilityMapMode.EXPOSURE_DIFFERENCE,
)

# (label, left/right value, difference)
_METRICS = (
    ("Active existing units",
     lambda s: s.active_existing_units, lambda c: c.active_existing_units_delta, "count"),
    ("Candidates", lambda s: s.included_candidates, lambda c: c.included_candidates_delta, "count"),
    ("Sampling units", lambda s: s.sampling_units, lambda c: c.sampling_unit_count_delta, "count"),
    ("Observable cells", lambda s: s.summary.observable_cells,
     lambda c: c.observable_cells_delta, "count"),
    ("Coverage (observable share)", lambda s: s.summary.observable_fraction,
     lambda c: c.coverage_percentage_point_delta, "share"),
    ("Blind-spot cells", lambda s: s.summary.blind_cells, lambda c: c.blind_cells_delta, "count"),
    ("Unique-coverage cells", lambda s: s.summary.unique_cells,
     lambda c: c.unique_cells_delta, "count"),
    ("Repeated-coverage cells", lambda s: s.summary.repeated_cells,
     lambda c: c.repeated_cells_delta, "count"),
    ("Mean exposure", lambda s: s.summary.mean_exposure,
     lambda c: c.mean_exposure_delta, "decimal"),
    ("Maximum exposure", lambda s: s.summary.maximum_exposure,
     lambda c: c.maximum_exposure_delta, "count"),
)


def _value(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "share":
        return f"{value:.1%}"
    if kind == "decimal":
        return f"{value:.2f}"
    return f"{value:,}"


def _difference(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "share":
        return f"{round(value, 2) + 0.0:+.2f} pp"
    if kind == "decimal":
        return f"{round(value, 2) + 0.0:+.2f}"
    return f"{value:+,}"


class ComparePanel(QWidget):
    """Compare baseline, current scenario and saved snapshots."""

    # Emitted after a snapshot has been loaded into the live scenario.
    scenario_loaded = Signal()

    def __init__(self, state: ApplicationState, *, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._syncing = False
        self._comparison: ScenarioComparison | None = None
        self._build_interface()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> ScenarioWorkspace:
        workspace = self.state.analysis.scenario_workspace
        if workspace is None:
            workspace = ScenarioWorkspace()
            self.state.analysis.scenario_workspace = workspace
        return workspace

    @property
    def comparison(self) -> ScenarioComparison | None:
        return self._comparison

    def select(self, left_id: str, right_id: str | None) -> None:
        self.workspace.left_id = left_id
        self.workspace.right_id = right_id
        self.refresh_from_state()

    def swap(self) -> None:
        workspace = self.workspace
        if workspace.right_id is None:
            return
        workspace.left_id, workspace.right_id = workspace.right_id, workspace.left_id
        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        sof = self.state.analysis.survey_observability_field
        workspace = self.workspace
        workspace.prune(sof)
        if workspace.right_id is None:
            workspace.right_id = self._default_right()

        self.compare_map.set_field(sof)
        environment = self.state.analysis.environment
        self.compare_map.set_terrain(
            None if environment is None else environment.elevation_model.source
        )
        survey = self.state.survey.viewpoint_configuration
        self.compare_map.set_viewpoints([] if survey is None else survey.viewpoints)

        self._populate_selectors()
        self._populate_library()
        self._compare()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _build_interface(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACING.lg, 0, 0)
        layout.setSpacing(SPACING.xl)

        card = ContentCard(
            title="Compare designs",
            description=(
                "Measurable differences between two design states of the same "
                "baseline. Differences are always RIGHT − LEFT; no design is "
                "ranked as better."
            ),
        )
        selectors = QHBoxLayout()
        selectors.addWidget(QLabel("Left"))
        self.left_combo = QComboBox()
        self.left_combo.setMinimumWidth(240)
        selectors.addWidget(self.left_combo)
        self.swap_button = make_secondary_button("⇄ Swap")
        selectors.addWidget(self.swap_button)
        selectors.addWidget(QLabel("Right"))
        self.right_combo = QComboBox()
        self.right_combo.setMinimumWidth(240)
        selectors.addWidget(self.right_combo)
        selectors.addStretch(1)
        card.add_layout(selectors)

        self.direction_label = QLabel()
        self.direction_label.setProperty("sectionHeading", True)
        card.add_widget(self.direction_label)

        self.metrics_table = QTableWidget(len(_METRICS), 4)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Metric", "Left", "Right", "Difference (right − left)"]
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        for row, (label, *_rest) in enumerate(_METRICS):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(label))
        self.metrics_table.setMinimumHeight(360)
        card.add_widget(self.metrics_table)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        card.add_widget(self.status_label)
        layout.addWidget(card)

        map_card = ContentCard(
            title="Spatial difference",
            description=(
                "Where coverage is gained or lost from left to right, and the "
                "signed change in exposure."
            ),
        )
        self.compare_map = ObservabilityMapWidget(modes=COMPARE_MAP_MODES)
        self.compare_map.setMinimumHeight(560)
        map_card.add_widget(self.compare_map)
        self.map_summary_label = QLabel()
        self.map_summary_label.setProperty("secondaryText", True)
        map_card.add_widget(self.map_summary_label)
        layout.addWidget(map_card)

        library = ContentCard(
            title="Saved scenarios",
            description=(
                "Save scenarios from the Scenario tab. Deleting a saved scenario "
                "keeps cached visibility and the current scenario unchanged."
            ),
        )
        self.library_table = QTableWidget(0, 4)
        self.library_table.setHorizontalHeaderLabels(
            ["Name", "Description", "Changes", "Status"]
        )
        self.library_table.verticalHeader().setVisible(False)
        self.library_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.library_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.library_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.library_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.library_table.horizontalHeader().setStretchLastSection(True)
        self.library_table.setMinimumHeight(170)
        library.add_widget(self.library_table)
        actions = QHBoxLayout()
        self.rename_button = make_secondary_button("Rename…")
        self.describe_button = make_secondary_button("Edit description…")
        self.delete_button = make_secondary_button("Delete")
        self.load_button = make_primary_button("Load into Scenario")
        for button in (self.rename_button, self.describe_button, self.delete_button):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.load_button)
        library.add_layout(actions)
        layout.addWidget(library)

    def _connect_signals(self) -> None:
        self.left_combo.currentIndexChanged.connect(self._on_selector_changed)
        self.right_combo.currentIndexChanged.connect(self._on_selector_changed)
        self.swap_button.clicked.connect(self.swap)
        self.rename_button.clicked.connect(lambda: self.rename_selected())
        self.describe_button.clicked.connect(lambda: self.describe_selected())
        self.delete_button.clicked.connect(lambda: self.delete_selected())
        self.load_button.clicked.connect(lambda: self.load_selected())
        self.library_table.itemSelectionChanged.connect(self._refresh_library_buttons)

    # ------------------------------------------------------------------
    # Selectors
    # ------------------------------------------------------------------

    def _options(self) -> list[tuple[str, str, bool]]:
        """(state id, label, available) for every comparable state."""
        sof = self.state.analysis.survey_observability_field
        live = self.state.analysis.design_scenario
        options = [(BASELINE_ID, "Baseline", sof is not None)]
        if live is not None and sof is not None and live.baseline is sof and not live.summary().is_baseline:
            options.append((LIVE_SCENARIO_ID, "Current scenario (unsaved)", True))
        for snapshot in self.workspace.snapshots:
            compatible = snapshot.compatible_with(sof)
            label = snapshot.name if compatible else f"{snapshot.name} (out of date)"
            options.append((snapshot.snapshot_id, label, compatible))
        return options

    def _default_right(self) -> str | None:
        for state_id, _label, available in reversed(self._options()):
            if available and state_id != BASELINE_ID:
                return state_id
        return None

    def _populate_selectors(self) -> None:
        options = self._options()
        workspace = self.workspace
        valid_ids = {state_id for state_id, _l, available in options if available}
        if workspace.left_id not in valid_ids:
            workspace.left_id = BASELINE_ID
        if workspace.right_id not in valid_ids:
            workspace.right_id = self._default_right()

        self._syncing = True
        try:
            for combo, selected in (
                (self.left_combo, workspace.left_id),
                (self.right_combo, workspace.right_id),
            ):
                combo.clear()
                for state_id, label, available in options:
                    combo.addItem(label, state_id)
                    item = combo.model().item(combo.count() - 1)
                    item.setEnabled(available)
                    if not available:
                        item.setToolTip("Saved for an earlier observability field.")
                index = combo.findData(selected)
                combo.setCurrentIndex(index)
        finally:
            self._syncing = False
        self.swap_button.setEnabled(workspace.right_id is not None)

    def _on_selector_changed(self, _index: int) -> None:
        if self._syncing:
            return
        self.workspace.left_id = self.left_combo.currentData()
        self.workspace.right_id = self.right_combo.currentData()
        self._compare()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _compare(self) -> None:
        sof = self.state.analysis.survey_observability_field
        workspace = self.workspace
        self._comparison = None
        left_id, right_id = workspace.left_id, workspace.right_id
        labels = {state_id: label for state_id, label, _a in self._options()}

        if sof is None or right_id is None:
            self._clear_metrics()
            self.direction_label.setText(
                "Save a scenario (Scenario tab) to compare it with the baseline."
            )
            self.compare_map.set_comparison_layers(None, None)
            self.map_summary_label.setText("")
            return

        self.direction_label.setText(
            f"Comparing: {labels.get(left_id, left_id)} → {labels.get(right_id, right_id)}"
            "   ·   differences are right − left"
        )
        try:
            store = self.state.analysis.visibility_store
            live = self.state.analysis.design_scenario
            left = workspace.state(left_id, sof=sof, store=store, live=live)
            right = workspace.state(right_id, sof=sof, store=store, live=live)
            comparison = compare_states(left, right, sof)
        except (SnapshotUnavailableError, ValueError) as exc:
            self._clear_metrics()
            self._set_status(str(exc), kind="warning")
            self.compare_map.set_comparison_layers(None, None)
            return

        self._set_status(None)
        self._comparison = comparison
        for row, (_label, value, delta, kind) in enumerate(_METRICS):
            for column, text in (
                (1, _value(value(left), kind)),
                (2, _value(value(right), kind)),
                (3, _difference(delta(comparison), kind)),
            ):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.metrics_table.setItem(row, column, item)

        self.compare_map.set_comparison_layers(
            comparison.change_classes(), comparison.exposure_difference()
        )
        gained = int(comparison.gained_coverage_mask.sum())
        lost = int(comparison.lost_coverage_mask.sum())
        self.map_summary_label.setText(
            f"Gained {gained:,} cells, lost {lost:,} cells: net "
            f"{gained - lost:+,} observable cells (right − left)."
        )

    def _clear_metrics(self) -> None:
        for row in range(len(_METRICS)):
            for column in (1, 2, 3):
                self.metrics_table.setItem(row, column, QTableWidgetItem("—"))

    # ------------------------------------------------------------------
    # Library
    # ------------------------------------------------------------------

    def _populate_library(self) -> None:
        sof = self.state.analysis.survey_observability_field
        snapshots = self.workspace.snapshots
        selected = self.selected_snapshot_id()
        self.library_table.setRowCount(len(snapshots))
        for row, snapshot in enumerate(snapshots):
            summary = snapshot.summary
            changes = (
                f"{summary.deactivated_units:,} deactivated, "
                f"{summary.included_candidates:,} of {len(snapshot.candidates):,} candidates included"
            )
            status = "Current baseline" if snapshot.compatible_with(sof) else "Out of date (earlier baseline)"
            for column, text in enumerate((snapshot.name, snapshot.description, changes, status)):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, snapshot.snapshot_id)
                self.library_table.setItem(row, column, item)
            if snapshot.snapshot_id == selected:
                self.library_table.selectRow(row)
        self._refresh_library_buttons()

    def selected_snapshot_id(self) -> str | None:
        items = self.library_table.selectedItems()
        return None if not items else items[0].data(Qt.ItemDataRole.UserRole)

    def select_snapshot(self, snapshot_id: str) -> None:
        for row in range(self.library_table.rowCount()):
            if self.library_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == snapshot_id:
                self.library_table.selectRow(row)
                return

    def _refresh_library_buttons(self) -> None:
        snapshot_id = self.selected_snapshot_id()
        has = snapshot_id is not None
        for button in (self.rename_button, self.describe_button, self.delete_button):
            button.setEnabled(has)
        sof = self.state.analysis.survey_observability_field
        compatible = has and self.workspace.get(snapshot_id).compatible_with(sof)
        self.load_button.setEnabled(compatible and not self.state.busy)

    def rename_selected(self, name: str | None = None) -> None:
        snapshot_id = self.selected_snapshot_id()
        if snapshot_id is None:
            return
        if name is None:
            name, accepted = QInputDialog.getText(
                self, "Rename saved scenario", "Name",
                text=self.workspace.get(snapshot_id).name,
            )
            if not accepted:
                return
        try:
            self.workspace.rename(snapshot_id, name)
        except ValueError as exc:
            self._set_status(str(exc), kind="error")
            return
        self.refresh_from_state()

    def describe_selected(self, description: str | None = None) -> None:
        snapshot_id = self.selected_snapshot_id()
        if snapshot_id is None:
            return
        if description is None:
            description, accepted = QInputDialog.getMultiLineText(
                self, "Edit description", "Description",
                self.workspace.get(snapshot_id).description,
            )
            if not accepted:
                return
        self.workspace.describe(snapshot_id, description)
        self.refresh_from_state()

    def delete_selected(self, *, confirm: bool = True) -> None:
        snapshot_id = self.selected_snapshot_id()
        if snapshot_id is None:
            return
        name = self.workspace.get(snapshot_id).name
        if confirm and QMessageBox.question(
            self, "Delete saved scenario",
            f"Delete {name!r}? Cached visibility and the current scenario are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.workspace.delete(snapshot_id)
        self._set_status(f"Deleted {name!r}.", kind="success")
        self.refresh_from_state()

    def load_selected(self) -> None:
        """Replace the live scenario with the selected snapshot."""
        snapshot_id = self.selected_snapshot_id()
        sof = self.state.analysis.survey_observability_field
        store = self.state.analysis.visibility_store
        if snapshot_id is None or sof is None or store is None or self.state.busy:
            return
        snapshot = self.workspace.get(snapshot_id)
        try:
            scenario, report = restore_snapshot(snapshot, sof, store)
            self.state.set_design_scenario(scenario)
        except (SnapshotUnavailableError, ValueError) as exc:
            self._set_status(str(exc), kind="error")
            return
        message = f"Loaded {snapshot.name!r} into the Scenario tab. The Survey is unchanged."
        if report.candidates_needing_computation:
            message += (
                " Visibility must be recomputed for: "
                + ", ".join(report.candidates_needing_computation) + "."
            )
        self._set_status(message, kind="success")
        self.scenario_loaded.emit()
        self.refresh_from_state()

    # ------------------------------------------------------------------

    def _set_status(self, text: str | None, *, kind: str = "success") -> None:
        if not text:
            self.status_label.setVisible(False)
            return
        for name in ("statusSuccess", "statusWarning", "statusError"):
            self.status_label.setProperty(name, False)
        self.status_label.setProperty(
            {"success": "statusSuccess", "warning": "statusWarning"}.get(kind, "statusError"),
            True,
        )
        refresh_style(self.status_label)
        self.status_label.setText(text)
        self.status_label.setVisible(True)
