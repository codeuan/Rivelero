"""O4 tests: MainWindow integration, rehydration, invalidation and readiness."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from observability_fixtures import make_ready_state, visibility_configuration, wait_for
from rivelero.gui.application_state import WorkflowPage
from rivelero.gui.main_window import MainWindow


_WINDOWS = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _window(state):
    window = MainWindow(state=state)
    _WINDOWS.append(window)
    return window


def _build(window):
    page = window.observability_page
    window.navigate_to(WorkflowPage.OBSERVABILITY)
    assert page.build_button.isEnabled(), page.readiness_label.text()
    page.build_button.click()
    wait_for(lambda: page._build_task_id is None)
    return page


def test_main_window_shares_state_and_task_controller(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = window.observability_page

    assert page.state is window.state
    assert page.task_controller is window.task_controller
    assert window.world_page.task_controller is window.task_controller


def test_sidebar_status_transitions(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    assert window.visibility_status.label.text() == "Visibility configured"
    assert window.sof_status.label.text() == "Observability not built"

    page = window.observability_page
    window.navigate_to(WorkflowPage.OBSERVABILITY)
    page.build_button.click()
    assert window.sof_status.label.text().startswith("Building observability")

    wait_for(lambda: page._build_task_id is None)
    assert window.sof_status.label.text() == "Observability ready · 3 units"

    state.set_visibility_configuration(visibility_configuration(max_distance_m=80.0))
    window.refresh_from_state()
    assert window.sof_status.label.text() == "Observability out of date"


def test_unconfigured_visibility_status(app, tmp_path):
    state = make_ready_state(tmp_path)
    state.set_visibility_configuration(None)
    window = _window(state)
    assert window.visibility_status.label.text() == "Visibility not configured"
    assert not window.observability_page.build_button.isEnabled()
    assert "Save a visibility configuration" in (
        window.observability_page.readiness_label.text()
    )


def test_sof_persists_across_navigation(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = _build(window)
    sof = state.analysis.survey_observability_field
    assert sof is not None

    window.navigate_to(WorkflowPage.SURVEY)
    window.navigate_to(WorkflowPage.WORLD)
    window.navigate_to(WorkflowPage.OBSERVABILITY)

    assert state.analysis.survey_observability_field is sof
    assert page.observability_map.sof is sof
    assert page.summary_values["active"].value == "3"
    assert page.report_values["requested"].value == "3"
    assert not page.invalidation_banner.isVisibleTo(page)
    assert page.continue_button.isEnabled()


def test_selection_does_not_invalidate(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    _build(window)

    state.select_viewpoint("vp_overlap_east")
    window.navigate_to(WorkflowPage.SURVEY)
    window.navigate_to(WorkflowPage.OBSERVABILITY)

    assert state.observability_ready
    assert window.observability_page.observability_map._selected_viewpoint_id == (
        "vp_overlap_east"
    )


@pytest.mark.parametrize(
    "change, reason",
    [
        ("viewpoint", "Survey changed."),
        ("sensor", "Sensors changed."),
        ("events", "Survey changed."),
        ("domain", "Analysis domain changed."),
        ("environment", "Terrain changed."),
    ],
)
def test_upstream_change_shows_invalidated_result(app, tmp_path, change, reason):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = _build(window)
    window.navigate_to(WorkflowPage.SURVEY)

    survey = state.survey.viewpoint_configuration
    if change == "viewpoint":
        viewpoint = survey.viewpoints[0]
        state.replace_viewpoint(replace(viewpoint, heading_deg=10.0))
    elif change == "sensor":
        sensor = next(iter(state.survey.sensors.values()))
        state.replace_sensor(replace(sensor, horizontal_fov_deg=45.0))
    elif change == "events":
        state.set_observation_events([])
    elif change == "domain":
        state.set_analysis_domain(replace(state.analysis.analysis_domain, domain_id="d2"))
    elif change == "environment":
        state.set_environment(
            state.analysis.environment, grid=state.analysis.analysis_grid
        )

    window.navigate_to(WorkflowPage.OBSERVABILITY)

    assert not state.observability_ready
    assert page.invalidation_banner.isVisibleTo(page)
    assert reason in page.invalidation_label.text()
    assert page.observability_map.sof is None
    assert not page.continue_button.isEnabled()
    assert page.summary_values["active"].value == "—"
    assert window.sof_status.label.text() in {
        "Observability out of date",
        "Observability not built",
    }


def test_rebuild_after_invalidation_restores_readiness(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = _build(window)

    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    state.replace_viewpoint(replace(viewpoint, heading_deg=10.0))
    page.refresh_from_state()
    assert page.invalidation_banner.isVisibleTo(page)

    page.build_button.click()
    wait_for(lambda: page._build_task_id is None)

    assert state.observability_ready
    assert not page.invalidation_banner.isVisibleTo(page)
    # Two unchanged Viewpoints are reused from the cache.
    assert page.report_values["cached"].value == "2"
    assert page.report_values["computed"].value == "1"


def test_continue_requires_current_sof_and_navigates(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = window.observability_page
    window.navigate_to(WorkflowPage.OBSERVABILITY)
    assert not page.continue_button.isEnabled()

    _build(window)
    assert page.continue_button.isEnabled()

    page.continue_button.click()
    assert state.view.active_page == WorkflowPage.ANALYSIS_DESIGN
    assert window.page_stack.currentIndex() == window._page_indices[
        WorkflowPage.ANALYSIS_DESIGN
    ]


def test_saving_unchanged_configuration_keeps_result(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = _build(window)

    page._save_configuration()
    assert state.observability_ready
    assert "unchanged" in page.status_label.text()


def test_unsaved_form_edits_block_build_and_saving_invalidates(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = _build(window)

    page.max_distance.setValue(400.0)
    assert not page.build_button.isEnabled()
    assert "unsaved" in page.readiness_label.text()
    assert state.observability_ready  # editing the form alone changes nothing

    page._save_configuration()
    assert not state.observability_ready
    assert state.analysis.visibility_configuration.max_distance_m == 400.0
    assert page.build_button.isEnabled()


def test_storage_settings_apply_without_invalidating(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = _build(window)
    old_store = state.analysis.visibility_store

    page.cache_directory_edit.setText(str(tmp_path / "second_cache"))
    page.memory_items.setValue(7)
    page._apply_storage_settings()

    store = state.analysis.visibility_store
    assert store is not old_store
    assert store.max_memory_items == 7
    assert state.observability_ready

    # The rehydrated page shows the canonical store settings.
    page.refresh_from_state()
    assert page.memory_items.value() == 7
    assert page.cache_directory_edit.text() == str(store.cache_directory)


def test_clear_disk_cache_keeps_result_and_requires_recompute(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _window(state)
    page = _build(window)
    state.select_viewpoint("vp_center_360")
    state.analysis.visibility_store.clear_memory()

    page._clear_disk_cache(confirm=False)

    assert state.observability_ready
    assert state.analysis.visibility_store.disk_item_count == 0
    assert "not yet computed" in page.unit_status_label.text()
    assert page.compute_unit_button.isEnabled()


def test_default_store_created_when_prerequisites_exist(app, tmp_path):
    state = make_ready_state(tmp_path, with_store=False)
    window = _window(state)
    window.navigate_to(WorkflowPage.OBSERVABILITY)

    assert state.analysis.visibility_store is not None
    assert window.observability_page.build_button.isEnabled()
