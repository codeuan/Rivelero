"""A3 tests: Scenario tab (what-if design) of the Analysis & Design page."""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from observability_fixtures import make_ready_state, visibility_configuration, wait_for
from rivelero.analysis.contribution import analyse_contributions
from rivelero.analysis.scenario import CandidateStatus, ScenarioChangeClass
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.analysis_page import AnalysisPage
from rivelero.gui.application_state import WorkflowPage
from rivelero.gui.main_window import MainWindow
from rivelero.gui.observability_map import ObservabilityMapMode
from rivelero.gui.observability_service import (
    compute_candidate_visibility,
    install_build_result,
    prepare_build,
)
from rivelero.gui.scenario_panel import CANDIDATE_TASK_NAME, EVENT_CANDIDATE_NOTE
from rivelero.observability.builder import build_survey_observability_field
from rivelero.visibility.configuration import MissingMetadataPolicy, SamplingUnit


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


def _panel(tmp_path, *, contributions=True, **kwargs):
    state = _built_state(tmp_path, **kwargs)
    if contributions:
        sof = state.analysis.survey_observability_field
        state.set_contribution_analysis(
            analyse_contributions(sof, state.analysis.visibility_store),
            inputs_revision=state.analysis.inputs_revision,
        )
    page = _keep(AnalysisPage(state))
    return state, page, page.scenario_panel


def _select(panel, *unit_ids):
    panel.units_table.clearSelection()
    selection = panel.units_table.selectionModel()
    from PySide6.QtCore import QItemSelectionModel

    for row in range(panel.units_proxy.rowCount()):
        index = panel.units_proxy.index(row, 0)
        if panel.units_proxy.data(index) in unit_ids:
            selection.select(
                index,
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )


def _deactivate(panel, *unit_ids):
    _select(panel, *unit_ids)
    panel.deactivate_button.click()
    wait_for(lambda: not panel.busy)


def _candidate(name="candidate_001", x=500800.0, y=4099200.0, **kwargs):
    kwargs.setdefault("observer_height_m", 1.75)
    return Viewpoint(viewpoint_id=name, x=x, y=y, crs="EPSG:32633", **kwargs)


def _add_and_compute(panel, viewpoint):
    panel._add_candidate(viewpoint=viewpoint)
    wait_for(lambda: not panel.busy)


def test_scenario_tab_requires_current_sof(app, tmp_path):
    state = make_ready_state(tmp_path)
    page = _keep(AnalysisPage(state))
    assert page.tabs.tabText(2) == "Scenario"
    assert not page.tabs.isVisibleTo(page)
    assert page.scenario_panel.scenario is None
    assert state.analysis.design_scenario is None


def test_current_sof_creates_baseline_scenario(app, tmp_path):
    state, page, panel = _panel(tmp_path)
    scenario = state.analysis.design_scenario
    assert scenario is not None
    assert scenario.baseline is state.analysis.survey_observability_field
    assert scenario.summary().is_baseline
    assert panel.units_model.rowCount() == 10
    assert panel.summary_values[("coverage", "difference")].text() == "+0.00 pp"
    assert not panel.reset_button.isEnabled()
    assert scenario.summary().baseline == page.summary  # same A1 semantics


def test_deactivate_and_reactivate_selected(app, tmp_path):
    state, page, panel = _panel(tmp_path)
    sof = state.analysis.survey_observability_field
    survey = state.survey.viewpoint_configuration
    baseline_bytes = sof.exposure_count.tobytes()
    a1 = page.summary
    unique = state.analysis.contribution_analysis.get("vp_missing_heading").unique_cells

    _deactivate(panel, "vp_missing_heading")
    summary = state.analysis.design_scenario.summary()
    assert summary.deactivated_units == 1
    # Single removal == the unit's baseline (A2) removal impact.
    assert summary.difference.observable_cells == -unique
    assert panel.summary_values[("observable", "difference")].text() == f"{-unique:+,}"

    # Non-destructive.
    assert sof.exposure_count.tobytes() == baseline_bytes
    assert state.survey.viewpoint_configuration is survey
    assert page.summary is a1 and state.observability_ready

    _select(panel, "vp_missing_heading")
    assert "deactivated in this scenario" in panel.unit_effect_label.text()
    panel.reactivate_button.click()
    assert state.analysis.design_scenario.summary().is_baseline


def test_second_removal_is_evaluated_against_scenario(app, tmp_path):
    state, _page, panel = _panel(tmp_path)
    contributions = state.analysis.contribution_analysis
    a = contributions.get("vp_overlap_west").unique_cells
    b = contributions.get("vp_center_360").unique_cells

    _deactivate(panel, "vp_overlap_west", "vp_center_360")
    joint = -state.analysis.design_scenario.summary().difference.observable_cells
    # Jointly covered cells become blind too: more than the A2 sum.
    assert joint > a + b


def test_unit_effect_shows_marginal_loss_given_scenario(app, tmp_path):
    state, _page, panel = _panel(tmp_path)
    _select(panel, "vp_center_360")
    assert "given the current scenario" in panel.unit_effect_label.text()
    assert state.selection.viewpoint_id == "vp_center_360"


def test_filters(app, tmp_path):
    _state, _page, panel = _panel(tmp_path)
    _deactivate(panel, "vp_center_east")
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("inactive"))
    assert panel.units_proxy.rowCount() == 1
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("active"))
    assert panel.units_proxy.rowCount() == 9
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("no_baseline_unique"))
    assert panel.units_proxy.rowCount() == 5


def test_reset_restores_baseline(app, tmp_path):
    state, _page, panel = _panel(tmp_path)
    _deactivate(panel, "vp_center_east", "vp_overlap_east")
    _add_and_compute(panel, _candidate())
    assert panel.reset_button.isEnabled()

    panel.reset_button.click()
    scenario = state.analysis.design_scenario
    assert scenario.summary().is_baseline
    assert scenario.candidates == ()
    assert scenario.exposure.tobytes() == scenario.baseline.exposure_count.tobytes()


def test_candidate_computation_success(app, tmp_path):
    state, _page, panel = _panel(tmp_path)
    scenario = state.analysis.design_scenario
    started = []
    panel.task_controller.task_started.connect(lambda _id, name: started.append(name))

    panel._add_candidate(viewpoint=_candidate())
    assert CANDIDATE_TASK_NAME in started
    assert scenario.candidate("candidate_001").status == CandidateStatus.COMPUTING
    assert panel.candidates_model.data(panel.candidates_model.index(0, 3)) == "Computing…"
    wait_for(lambda: not panel.busy)

    candidate = scenario.candidate("candidate_001")
    assert candidate.status == CandidateStatus.READY and candidate.included
    summary = scenario.summary()
    assert summary.included_candidates == 1
    gained = int(np.count_nonzero(scenario.change_classes() == ScenarioChangeClass.GAINED_COVERAGE))
    assert summary.difference.observable_cells == gained > 0
    gain_text = panel.candidates_model.data(panel.candidates_model.index(0, 5))
    assert gain_text == f"{gained:+,} cells"
    # Cached through the canonical store under a fingerprinted key.
    assert state.analysis.visibility_store.contains(candidate.key)

    panel.scenario_map.set_mode(ObservabilityMapMode.SCENARIO_EXPOSURE)
    assert panel.scenario_map.colorbar_visible
    panel.scenario_map.set_mode(ObservabilityMapMode.SCENARIO_CHANGE)
    assert "Gained coverage (becomes observable)" in panel.scenario_map.legend_labels


def test_candidate_restores_removed_coverage(app, tmp_path):
    state, _page, panel = _panel(tmp_path)
    _deactivate(panel, "vp_overlap_east")
    lost = -state.analysis.design_scenario.summary().difference.observable_cells
    removed = state.survey.viewpoint_configuration.get_viewpoint("vp_overlap_east")

    # A candidate at the same place (a legitimate duplicate coordinate).
    _add_and_compute(panel, _candidate(x=removed.x, y=removed.y))
    summary = state.analysis.design_scenario.summary()
    assert lost > 0
    assert summary.difference.observable_cells == 0


def test_candidate_outside_terrain_fails_without_changing_exposure(app, tmp_path):
    state, _page, panel = _panel(tmp_path)
    scenario = state.analysis.design_scenario
    before = scenario.exposure.tobytes()
    panel._add_candidate(viewpoint=_candidate(x=510000.0))

    candidate = scenario.candidate("candidate_001")
    assert candidate.status == CandidateStatus.FAILED
    assert "outside the terrain" in candidate.message
    assert not panel.busy
    assert scenario.exposure.tobytes() == before


def test_candidate_engine_failure_is_not_a_zero_mask(app, tmp_path, monkeypatch):
    state, _page, panel = _panel(tmp_path)
    scenario = state.analysis.design_scenario
    before = scenario.exposure.tobytes()

    def failing(**_kwargs):
        raise RuntimeError("viewshed exploded")

    monkeypatch.setattr("rivelero.gui.scenario_panel.compute_candidate_visibility", failing)
    _add_and_compute(panel, _candidate())

    candidate = scenario.candidate("candidate_001")
    assert candidate.status == CandidateStatus.FAILED
    assert candidate.mask is None
    assert "viewshed exploded" in candidate.message
    assert scenario.exposure.tobytes() == before


def test_candidate_excluded_by_policy(app, tmp_path):
    state, _page, panel = _panel(
        tmp_path,
        contributions=False,
        viewpoint_ids=("vp_center_360",),
        configuration=visibility_configuration(
            max_distance_m=250.0,
            missing_observer_height_policy=MissingMetadataPolicy.EXCLUDE,
            default_observer_height_m=None,
        ),
    )
    _add_and_compute(panel, _candidate(observer_height_m=None))
    candidate = state.analysis.design_scenario.candidate("candidate_001")
    assert candidate.status == CandidateStatus.EXCLUDED
    assert "observer height" in candidate.message


def test_duplicate_candidate_id_rejected(app, tmp_path, monkeypatch):
    state, _page, panel = _panel(tmp_path)
    warnings = []
    monkeypatch.setattr(
        "rivelero.gui.scenario_panel.QMessageBox.warning",
        lambda *args: warnings.append(args[2]),
    )
    panel._add_candidate(viewpoint=_candidate(name="vp_center_360"))
    assert warnings and "already used" in warnings[0]
    assert state.analysis.design_scenario.candidates == ()


def test_stale_candidate_result_is_discarded(app, tmp_path, monkeypatch):
    state, _page, panel = _panel(tmp_path)
    release = threading.Event()

    def gated(**kwargs):
        release.wait(10)
        return compute_candidate_visibility(**kwargs)

    monkeypatch.setattr("rivelero.gui.scenario_panel.compute_candidate_visibility", gated)
    panel._add_candidate(viewpoint=_candidate())

    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    state.replace_viewpoint(replace(viewpoint, heading_deg=5.0))  # invalidates SOF
    release.set()
    wait_for(lambda: not panel.busy)

    assert state.analysis.design_scenario is None
    assert "discarded" in panel.status_label.text()


def test_cancelled_candidate_is_not_installed(app, tmp_path, monkeypatch):
    state, _page, panel = _panel(tmp_path)
    release = threading.Event()

    def gated(**kwargs):
        release.wait(10)
        return compute_candidate_visibility(**kwargs)

    monkeypatch.setattr("rivelero.gui.scenario_panel.compute_candidate_visibility", gated)
    panel._add_candidate(viewpoint=_candidate())
    assert panel.cancel_candidate_button.isVisibleTo(panel)
    panel.cancel_candidate_button.click()
    release.set()
    wait_for(lambda: not panel.busy)

    scenario = state.analysis.design_scenario
    candidate = scenario.candidate("candidate_001")
    assert candidate.status == CandidateStatus.CANCELLED
    assert scenario.summary().is_baseline
    assert not panel.busy  # a cancelled candidate is not restarted


def test_event_sampling_supports_deactivation_but_not_candidates(app, tmp_path):
    state, _page, panel = _panel(
        tmp_path,
        viewpoint_ids=("vp_redundant_a", "vp_overlap_east", "vp_complementary"),
        configuration=visibility_configuration(
            max_distance_m=300.0, sampling_unit=SamplingUnit.OBSERVATION_EVENT
        ),
    )
    assert not panel.add_candidate_button.isEnabled()
    assert panel.candidate_note.text() == EVENT_CANDIDATE_NOTE
    assert not panel.units_table.isColumnHidden(1)

    _deactivate(panel, "event_003")
    scenario = state.analysis.design_scenario
    assert scenario.summary().deactivated_units == 1
    # The other event of the same Viewpoint stays active.
    active = [key.sampling_unit_id for key in scenario.baseline.active_keys if scenario.is_active(key)]
    assert "event_001" in active and "event_003" not in active


def test_upstream_change_invalidates_scenario(app, tmp_path):
    state, page, panel = _panel(tmp_path)
    _deactivate(panel, "vp_center_east")
    old = state.analysis.design_scenario

    state.set_visibility_configuration(visibility_configuration(max_distance_m=100.0))
    page.refresh_from_state()
    assert state.analysis.design_scenario is None
    assert not page.tabs.isVisibleTo(page)

    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    page.refresh_from_state()
    fresh = state.analysis.design_scenario
    assert fresh is not old and fresh.summary().is_baseline


def test_navigation_preserves_scenario(app, tmp_path):
    state = _built_state(tmp_path)
    window = _keep(MainWindow(state=state))
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    panel = window.analysis_page.scenario_panel
    _deactivate(panel, "vp_center_east")
    scenario = state.analysis.design_scenario

    window.navigate_to(WorkflowPage.SURVEY)
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)

    assert state.analysis.design_scenario is scenario
    assert scenario.summary().deactivated_units == 1
    assert panel.summary_values[("units", "baseline")].text() == "10"
