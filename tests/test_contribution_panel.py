"""A2 tests: Contribution tab of the Analysis & Design page."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from observability_fixtures import make_ready_state, visibility_configuration, wait_for
from rivelero.analysis.contribution import UnitContributionClass, analyse_contributions
from rivelero.gui.analysis_page import AnalysisPage
from rivelero.gui.application_state import TaskStatus, WorkflowPage
from rivelero.gui.contribution_panel import CONTRIBUTION_TASK_NAME, NO_UNIQUE_NOTE
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
    kwargs.setdefault(
        "configuration", visibility_configuration(max_distance_m=250.0)
    )
    state = make_ready_state(tmp_path, **kwargs)
    request = prepare_build(state)
    result = build_survey_observability_field(**request.build_kwargs())
    install_build_result(state, request, result)
    return state


def _analysed_panel(tmp_path, **kwargs):
    state = _built_state(tmp_path, **kwargs)
    page = _keep(AnalysisPage(state))
    panel = page.contribution_panel
    panel.analyse_button.click()
    wait_for(lambda: not panel.running)
    return state, page, panel


def test_contribution_tab_exists_and_needs_sof(app, tmp_path):
    state = make_ready_state(tmp_path)
    page = _keep(AnalysisPage(state))

    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == [
        "Overview", "Contribution", "Scenario", "Compare",
    ]
    assert not page.tabs.isVisibleTo(page)
    assert not page.contribution_panel.analyse_button.isEnabled()


def test_analysis_runs_in_background_and_populates_table(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    panel = page.contribution_panel
    assert panel.analyse_button.isEnabled()
    progress = []
    page.task_controller.task_progress.connect(
        lambda _id, processed, total, unit, _m: progress.append((processed, total, unit))
    )

    panel.analyse_button.click()
    assert state.task.busy and state.task.task_name == CONTRIBUTION_TASK_NAME
    assert not panel.analyse_button.isEnabled()
    wait_for(lambda: not panel.running)

    analysis = state.analysis.contribution_analysis
    assert analysis is not None and analysis.complete
    assert panel.model.rowCount() == 10
    assert [entry[0] for entry in progress] == list(range(1, 11))
    assert panel.summary_values["analysed"].value == "10"
    assert panel.summary_values["consistent"].value == "Yes"
    assert panel.summary_values["unique_total"].value == (
        panel.summary_values["field_unique"].value
    )
    # Viewpoint sampling: the separate Viewpoint column is hidden.
    assert panel.table.isColumnHidden(1)
    # Read-only: the SOF is unchanged.
    assert state.observability_ready


def test_sorting_by_unique_cells(app, tmp_path):
    _state, _page, panel = _analysed_panel(tmp_path)
    panel.table.sortByColumn(3, Qt.SortOrder.DescendingOrder)
    values = [
        panel.proxy.data(panel.proxy.index(row, 3), Qt.ItemDataRole.UserRole)
        for row in range(panel.proxy.rowCount())
    ]
    assert values == sorted(values, reverse=True)

    panel.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    ids = [
        panel.proxy.data(panel.proxy.index(row, 0))
        for row in range(panel.proxy.rowCount())
    ]
    assert ids == sorted(ids)


def test_row_selection_updates_inspector_map_and_state(app, tmp_path):
    state, _page, panel = _analysed_panel(tmp_path)
    unit = state.analysis.contribution_analysis.get("vp_center_east")

    assert panel.select_unit("vp_center_east")
    assert panel.selected_unit() is unit
    assert state.selection.viewpoint_id == "vp_center_east"
    assert state.selection.observation_event_id is None

    values = panel.inspector_values
    assert values["unit"].text() == "vp_center_east"
    assert values["unique"].text() == f"{unit.unique_cells:,}"
    assert values["repeated"].text() == f"{unit.repeated_cells:,}"
    assert f"{unit.unique_cells:,} cells would become blind spots" in values["loss"].text()
    assert values["sensor"].text() == "sensor_rgb_90"

    widget = panel.contribution_map
    assert widget.mode == ObservabilityMapMode.UNIT_CONTRIBUTION
    classes = widget.layer_array
    assert np.count_nonzero(classes == UnitContributionClass.UNIQUE_CONTRIBUTION) == (
        unit.unique_cells
    )
    assert np.count_nonzero(classes == UnitContributionClass.REPEATED_CONTRIBUTION) == (
        unit.repeated_cells
    )
    assert "Unique contribution (blind if removed)" in widget.legend_labels


def test_zero_unique_unit_uses_neutral_language(app, tmp_path):
    state, _page, panel = _analysed_panel(tmp_path)
    analysis = state.analysis.contribution_analysis
    zero_unique = [
        unit for unit in analysis.units
        if unit.visible_cells > 0 and unit.unique_cells == 0
    ]
    if not zero_unique:
        pytest.skip("fixture has no unit without unique coverage")
    panel.select_unit(zero_unique[0].sampling_unit_id)
    assert panel.inspector_note.text() == NO_UNIQUE_NOTE
    for word in ("useless", "unnecessary", "safe to remove"):
        assert word not in NO_UNIQUE_NOTE


def test_state_selection_selects_row(app, tmp_path):
    state, page, panel = _analysed_panel(tmp_path)
    state.select_viewpoint("vp_overlap_west")
    page.refresh_from_state()
    assert panel.selected_unit().sampling_unit_id == "vp_overlap_west"

    # Map click uses the same canonical selection.
    panel.contribution_map.viewpoint_selected.emit("vp_center_360")
    assert state.selection.viewpoint_id == "vp_center_360"
    assert panel.selected_unit().sampling_unit_id == "vp_center_360"


def test_event_sampling_selects_event_and_its_viewpoint(app, tmp_path):
    state, _page, panel = _analysed_panel(
        tmp_path,
        viewpoint_ids=("vp_redundant_a", "vp_overlap_east", "vp_complementary"),
        configuration=visibility_configuration(
            max_distance_m=300.0, sampling_unit=SamplingUnit.OBSERVATION_EVENT
        ),
    )
    assert panel.model.rowCount() == 4
    assert not panel.table.isColumnHidden(1)

    panel.select_unit("event_003")
    assert state.selection.observation_event_id == "event_003"
    assert state.selection.viewpoint_id == "vp_redundant_a"
    assert panel.inspector_values["event"].text() == "event_003"
    assert panel.inspector_values["viewpoint"].text() == "vp_redundant_a"


def test_result_computed_from_edited_inputs_is_not_installed(app, tmp_path, monkeypatch):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    panel = page.contribution_panel
    release = threading.Event()

    def gated(**kwargs):
        release.wait(10)
        return analyse_contributions(**kwargs)

    monkeypatch.setattr("rivelero.gui.contribution_panel.analyse_contributions", gated)
    panel.analyse_button.click()

    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    state.replace_viewpoint(replace(viewpoint, heading_deg=5.0))
    release.set()
    wait_for(lambda: not panel.running)

    assert state.analysis.contribution_analysis is None
    assert "discarded" in panel.status_label.text()


def test_rebuilt_sof_rejects_old_analysis(app, tmp_path, monkeypatch):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    panel = page.contribution_panel
    release = threading.Event()

    def gated(**kwargs):
        release.wait(10)
        return analyse_contributions(**kwargs)

    monkeypatch.setattr("rivelero.gui.contribution_panel.analyse_contributions", gated)
    # Builds are refused while a task runs, so prepare the rebuild first.
    request = prepare_build(state)
    panel.analyse_button.click()

    # A new SOF with identical inputs (same revision) still differs.
    install_build_result(
        state, request, build_survey_observability_field(**request.build_kwargs())
    )
    release.set()
    wait_for(lambda: not panel.running)

    assert state.analysis.contribution_analysis is None


def test_cancellation_installs_nothing(app, tmp_path, monkeypatch):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    panel = page.contribution_panel

    def slow(*, sof, store, progress_callback):
        for index in range(1, 301):
            time.sleep(0.01)
            progress_callback(index, 300, f"unit-{index}")
        raise AssertionError("cancellation was not honoured")

    monkeypatch.setattr("rivelero.gui.contribution_panel.analyse_contributions", slow)
    panel.analyse_button.click()
    wait_for(lambda: state.task.processed >= 2)
    panel.cancel_button.click()
    wait_for(lambda: not panel.running)

    assert state.task.status == TaskStatus.CANCELLED
    assert state.analysis.contribution_analysis is None
    assert "No partial result" in panel.status_label.text()
    assert panel.analyse_button.isEnabled()


def test_upstream_change_clears_contributions(app, tmp_path):
    state, page, panel = _analysed_panel(tmp_path)
    assert panel.model.rowCount() == 10

    state.set_visibility_configuration(visibility_configuration(max_distance_m=100.0))
    page.refresh_from_state()

    assert state.analysis.contribution_analysis is None
    assert panel.model.rowCount() == 0
    assert not page.tabs.isVisibleTo(page)


def test_contributions_survive_navigation(app, tmp_path):
    state = _built_state(tmp_path)
    window = _keep(MainWindow(state=state))
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    panel = window.analysis_page.contribution_panel
    panel.analyse_button.click()
    wait_for(lambda: not panel.running)
    analysis = state.analysis.contribution_analysis
    overview = window.analysis_page.summary

    window.navigate_to(WorkflowPage.SURVEY)
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)

    assert state.analysis.contribution_analysis is analysis
    assert panel.model.rowCount() == 10
    # A1 metrics are unaffected by contribution analysis.
    assert window.analysis_page.summary is overview
