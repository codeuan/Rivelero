"""A4 tests: Compare tab and saving scenarios for comparison."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from observability_fixtures import make_ready_state, visibility_configuration, wait_for
from rivelero.analysis.comparison import BASELINE_ID, LIVE_SCENARIO_ID
from rivelero.analysis.scenario import CandidateStatus, ScenarioChangeClass
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.analysis_page import AnalysisPage
from rivelero.gui.application_state import WorkflowPage
from rivelero.gui.main_window import MainWindow
from rivelero.gui.observability_map import ObservabilityMapMode
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.observability.builder import build_survey_observability_field
from rivelero.visibility.configuration import SamplingUnit


_WIDGETS = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _keep(widget):
    _WIDGETS.append(widget)
    return widget


def _built_state(tmp_path, **kwargs):
    kwargs.setdefault("viewpoint_ids", None)
    kwargs.setdefault("configuration", visibility_configuration(max_distance_m=250.0))
    state = make_ready_state(tmp_path, **kwargs)
    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    return state


def _deactivate(panel, *unit_ids):
    panel.units_table.clearSelection()
    selection = panel.units_table.selectionModel()
    for row in range(panel.units_proxy.rowCount()):
        index = panel.units_proxy.index(row, 0)
        if panel.units_proxy.data(index) in unit_ids:
            selection.select(index, QItemSelectionModel.SelectionFlag.Select
                             | QItemSelectionModel.SelectionFlag.Rows)
    panel.deactivate_button.click()
    wait_for(lambda: not panel.busy)


def _two_snapshots(tmp_path, **kwargs):
    state = _built_state(tmp_path, **kwargs)
    page = _keep(AnalysisPage(state))
    scenario_panel = page.scenario_panel
    _deactivate(scenario_panel, "vp_missing_heading")
    a = scenario_panel.save_snapshot("Without heading unit", "")
    scenario_panel._reset()
    _deactivate(scenario_panel, "vp_overlap_east")
    scenario_panel._add_candidate(viewpoint=Viewpoint(
        viewpoint_id="candidate_001", x=500800.0, y=4099250.0, crs="EPSG:32633",
        observer_height_m=1.75,
    ))
    wait_for(lambda: not scenario_panel.busy)
    b = scenario_panel.save_snapshot("East swapped for candidate", "B")
    page.compare_panel.refresh_from_state()
    return state, page, page.compare_panel, a, b


def _metric(panel, row_label, column):
    table = panel.metrics_table
    for row in range(table.rowCount()):
        if table.item(row, 0).text() == row_label:
            return table.item(row, column).text()
    raise KeyError(row_label)


def test_compare_tab_exists_with_baseline_option(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    panel = page.compare_panel
    assert page.tabs.tabText(3) == "Compare"
    assert panel.left_combo.itemData(0) == BASELINE_ID
    assert panel.comparison is None  # nothing to compare yet
    assert "Save a scenario" in panel.direction_label.text()


def test_saving_does_not_modify_survey_or_recompute(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    survey = state.survey.viewpoint_configuration
    started = []
    page.task_controller.task_started.connect(lambda _id, name: started.append(name))
    snapshot = page.scenario_panel.save_snapshot("Unchanged", "")
    assert snapshot is not None and started == []
    assert state.survey.viewpoint_configuration is survey
    assert state.analysis.scenario_workspace.snapshots == (snapshot,)
    # Duplicate names are refused, not silently overwritten.
    assert page.scenario_panel.save_snapshot("Unchanged", "") is None
    assert "already called" in page.scenario_panel.status_label.text()


def test_saved_scenarios_appear_and_compare_to_baseline(app, tmp_path):
    state, _page, panel, a, b = _two_snapshots(tmp_path)
    ids = [panel.right_combo.itemData(i) for i in range(panel.right_combo.count())]
    assert ids[0] == BASELINE_ID and a.snapshot_id in ids and b.snapshot_id in ids
    assert panel.library_table.rowCount() == 2

    panel.select(BASELINE_ID, a.snapshot_id)
    comparison = panel.comparison
    assert "Baseline → Without heading unit" in panel.direction_label.text()
    expected = a.summary.difference.observable_cells  # recorded when saved
    assert expected < 0
    assert comparison.observable_cells_delta == expected
    assert _metric(panel, "Observable cells", 3) == f"{expected:+,}"
    assert _metric(panel, "Active existing units", 3) == "-1"
    assert _metric(panel, "Blind-spot cells", 3) == f"{-expected:+,}"


def test_swap_reverses_signs_and_maps(app, tmp_path):
    _state, _page, panel, a, b = _two_snapshots(tmp_path)
    panel.select(a.snapshot_id, b.snapshot_id)
    forward = panel.comparison
    forward_text = _metric(panel, "Observable cells", 3)
    panel.swap_button.click()
    backward = panel.comparison

    assert backward.left.state_id == b.snapshot_id
    assert backward.observable_cells_delta == -forward.observable_cells_delta
    assert _metric(panel, "Observable cells", 3) == f"{-forward.observable_cells_delta:+,}"
    assert forward_text == f"{forward.observable_cells_delta:+,}"
    assert np.array_equal(forward.gained_coverage_mask, backward.lost_coverage_mask)
    # Observable delta = gained - lost.
    assert forward.observable_cells_delta == (
        int(forward.gained_coverage_mask.sum()) - int(forward.lost_coverage_mask.sum())
    )


def test_comparison_maps(app, tmp_path):
    _state, _page, panel, a, b = _two_snapshots(tmp_path)
    panel.select(a.snapshot_id, b.snapshot_id)
    widget = panel.compare_map

    assert widget.mode == ObservabilityMapMode.COMPARISON_CHANGE
    classes = widget.layer_array
    assert int((classes == ScenarioChangeClass.GAINED_COVERAGE).sum()) == int(
        panel.comparison.gained_coverage_mask.sum()
    )
    assert "Blind in both" in widget.legend_labels

    widget.set_mode(ObservabilityMapMode.EXPOSURE_DIFFERENCE)
    assert widget.colorbar_visible
    assert widget.colorbar_label.startswith("Exposure difference (right − left")
    norm = widget._layer_image.norm
    assert norm.vmin == -norm.vmax  # symmetric about zero
    shown = widget.layer_array
    expected = panel.comparison.exposure_difference()
    assert np.array_equal(np.ma.getmaskarray(shown), np.ma.getmaskarray(expected))


def test_live_scenario_is_a_comparison_option(app, tmp_path):
    state, page, panel, a, _b = _two_snapshots(tmp_path)
    ids = [panel.left_combo.itemData(i) for i in range(panel.left_combo.count())]
    assert LIVE_SCENARIO_ID in ids  # the modified, unsaved scenario
    panel.select(LIVE_SCENARIO_ID, a.snapshot_id)
    assert panel.comparison.left.label == "Current scenario (unsaved)"


def test_rename_describe_delete(app, tmp_path):
    state, _page, panel, a, b = _two_snapshots(tmp_path)
    live = state.analysis.design_scenario
    store = state.analysis.visibility_store
    disk = store.disk_item_count

    panel.select_snapshot(a.snapshot_id)
    panel.rename_selected("Plan A")
    panel.select_snapshot(a.snapshot_id)
    panel.describe_selected("no heading unit")
    snapshot = state.analysis.scenario_workspace.get(a.snapshot_id)
    assert (snapshot.name, snapshot.description) == ("Plan A", "no heading unit")

    panel.select_snapshot(b.snapshot_id)
    panel.delete_selected(confirm=False)
    names = [s.name for s in state.analysis.scenario_workspace.snapshots]
    assert names == ["Plan A"]
    assert state.analysis.design_scenario is live
    assert store.disk_item_count == disk


def test_load_into_scenario(app, tmp_path):
    state, page, panel, a, b = _two_snapshots(tmp_path)
    survey = state.survey.viewpoint_configuration
    panel.select_snapshot(a.snapshot_id)
    panel.load_button.click()

    scenario = state.analysis.design_scenario
    assert [k.sampling_unit_id for k in scenario.deactivated_keys] == ["vp_missing_heading"]
    assert scenario.candidates == ()
    assert page.tabs.currentWidget() is page.scenario_panel
    assert state.survey.viewpoint_configuration is survey

    page.tabs.setCurrentWidget(panel)
    panel.select_snapshot(b.snapshot_id)
    panel.load_button.click()
    scenario = state.analysis.design_scenario
    candidate = scenario.candidate("candidate_001")
    assert candidate.status == CandidateStatus.READY and candidate.included
    assert scenario.summary() == state.analysis.scenario_workspace.get(b.snapshot_id).summary


def test_out_of_date_snapshots_are_marked_and_disabled(app, tmp_path):
    state, page, panel, a, _b = _two_snapshots(tmp_path)
    state.set_visibility_configuration(visibility_configuration(max_distance_m=200.0))
    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    page.refresh_from_state()

    workspace = state.analysis.scenario_workspace
    assert len(workspace.snapshots) == 2  # kept for the session
    status = panel.library_table.item(0, 3).text()
    assert status.startswith("Out of date")
    index = panel.right_combo.findData(a.snapshot_id)
    assert not panel.right_combo.model().item(index).isEnabled()
    assert "(out of date)" in panel.right_combo.itemText(index)
    assert panel.comparison is None

    panel.select_snapshot(a.snapshot_id)
    assert not panel.load_button.isEnabled()


def test_event_sampling_comparison(app, tmp_path):
    state = _built_state(
        tmp_path,
        viewpoint_ids=("vp_redundant_a", "vp_overlap_east", "vp_complementary"),
        configuration=visibility_configuration(
            max_distance_m=300.0, sampling_unit=SamplingUnit.OBSERVATION_EVENT
        ),
    )
    page = _keep(AnalysisPage(state))
    _deactivate(page.scenario_panel, "event_003")
    snapshot = page.scenario_panel.save_snapshot("Without event 3", "")
    assert snapshot.sampling_unit == SamplingUnit.OBSERVATION_EVENT.value
    page.compare_panel.select(BASELINE_ID, snapshot.snapshot_id)
    comparison = page.compare_panel.comparison
    assert comparison.active_existing_units_delta == -1
    assert comparison.observable_cells_delta == (
        int(comparison.gained_coverage_mask.sum()) - int(comparison.lost_coverage_mask.sum())
    )


def test_snapshots_survive_navigation(app, tmp_path):
    state = _built_state(tmp_path)
    window = _keep(MainWindow(state=state))
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    scenario_panel = window.analysis_page.scenario_panel
    _deactivate(scenario_panel, "vp_center_east")
    snapshot = scenario_panel.save_snapshot("Kept", "")

    window.navigate_to(WorkflowPage.OBSERVABILITY)
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)

    assert state.analysis.scenario_workspace.snapshots == (snapshot,)
    panel = window.analysis_page.compare_panel
    assert panel.right_combo.findData(snapshot.snapshot_id) >= 0
