"""P2 tests: export context from ApplicationState and the Output page."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from observability_fixtures import (
    DEM_NODATA, make_ready_state, visibility_configuration, wait_for,
)
from rivelero.analysis.comparison import BASELINE_ID, LIVE_SCENARIO_ID, ScenarioWorkspace
from rivelero.analysis.contribution import analyse_contributions
from rivelero.analysis.scenario import SurveyDesignScenario
from rivelero.export.catalog import export_catalog
from rivelero.gui.application_state import WorkflowPage
from rivelero.gui.export_service import export_context_from_state
from rivelero.gui.main_window import MainWindow
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.gui.output_page import OutputPage
from rivelero.observability.builder import build_survey_observability_field


_WIDGETS = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _keep(widget):
    _WIDGETS.append(widget)
    return widget


def _ready(tmp_path):
    return make_ready_state(
        tmp_path / "cache", dem=DEM_NODATA, survey_buffer_m=100.0, viewpoint_ids=None,
        configuration=visibility_configuration(max_distance_m=250.0),
    )


def _build(state):
    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    return state


def _with_design(state):
    """Contribution analysis, a modified live scenario and a saved snapshot."""
    sof = state.analysis.survey_observability_field
    store = state.analysis.visibility_store
    state.set_contribution_analysis(
        analyse_contributions(sof, store), inputs_revision=state.analysis.inputs_revision
    )
    scenario = SurveyDesignScenario(sof)
    key = sof.active_keys[0]
    scenario.deactivate({key: store.get(key).visibility_mask})
    state.set_design_scenario(scenario)
    workspace = state.analysis.scenario_workspace = ScenarioWorkspace()
    snapshot = workspace.save(scenario, name="Without first unit")
    workspace.left_id, workspace.right_id = BASELINE_ID, snapshot.snapshot_id
    return snapshot


def _available(state):
    return {p.key: p for p in export_catalog(export_context_from_state(state))}


# ---------------------------------------------------------------------------
# Export context (availability reflects state)
# ---------------------------------------------------------------------------


def test_availability_before_and_after_build(app, tmp_path):
    state = _ready(tmp_path)
    products = _available(state)
    assert products["viewpoints"].available and products["sensors"].available
    assert not products["exposure_count"].available
    assert "Build" in products["exposure_count"].reason
    assert not products["comparison_difference"].available

    _build(state)
    products = _available(state)
    assert products["exposure_count"].available and products["coverage_class"].available
    assert products["scenario_summary"].available
    assert not products["contributions"].available
    assert "contribution" in products["contributions"].reason.lower()
    assert not products["scenario_exposure"].available
    assert not products["comparison_summary"].available

    _with_design(state)
    products = _available(state)
    assert all(products[key].available for key in (
        "contributions", "scenario_exposure", "scenario_change",
        "comparison_summary", "comparison_difference", "comparison_change",
    ))
    assert "Baseline_vs_Without_first_unit" in products["comparison_difference"].filename


def test_stale_state_makes_results_unavailable(app, tmp_path):
    state = _build(_ready(tmp_path))
    _with_design(state)
    state.set_visibility_configuration(visibility_configuration(max_distance_m=200.0))
    assert state.analysis.survey_observability_field is None
    products = _available(state)
    for key in ("exposure_count", "contributions", "scenario_exposure", "comparison_difference"):
        assert not products[key].available and products[key].reason
    assert products["viewpoints"].available


def test_out_of_date_snapshot_is_not_compared(app, tmp_path):
    state = _build(_ready(tmp_path))
    snapshot = _with_design(state)
    workspace = state.analysis.scenario_workspace
    _build(state)  # identical rebuild: new sof_id, the snapshot is out of date
    workspace.left_id, workspace.right_id = BASELINE_ID, snapshot.snapshot_id
    products = _available(state)
    assert not products["comparison_difference"].available
    assert "earlier observability field" in products["comparison_difference"].reason
    # The summary table still lists it, flagged as not comparable.
    rows = export_context_from_state(state).scenario_rows
    flagged = [row for row in rows if row["state_id"] == snapshot.snapshot_id]
    assert flagged and flagged[0]["compatible_with_current_baseline"] is False


def test_live_scenario_export_matches_state(app, tmp_path):
    state = _build(_ready(tmp_path))
    _with_design(state)
    workspace = state.analysis.scenario_workspace
    workspace.left_id, workspace.right_id = BASELINE_ID, LIVE_SCENARIO_ID
    context = export_context_from_state(state)
    sof = state.analysis.survey_observability_field
    live = state.analysis.design_scenario
    np.testing.assert_array_equal(context.scenario_exposure, live.exposure)
    assert context.comparison.right.state_id == LIVE_SCENARIO_ID
    assert context.provenance.sof_id == sof.sof_id
    assert context.provenance.analysis_domain_id == sof.analysis_domain_id
    assert context.scenario_record["deactivated_sampling_units"] == [
        sof.active_keys[0].sampling_unit_id
    ]


# ---------------------------------------------------------------------------
# Output page
# ---------------------------------------------------------------------------


def _saved_window(tmp_path):
    state = _build(_ready(tmp_path))
    _with_design(state)
    window = _keep(MainWindow(state=state))
    window.show_message = lambda *args, **kwargs: None
    assert window.save_project_as(tmp_path / "Sicily Survey")
    assert not state.project.dirty
    return window


def _export(page, directory, keys, *, overwrite=False):
    page.directory_edit.setText(str(directory))
    page.set_selection(keys)
    page.overwrite_box.setChecked(overwrite)
    task_id = page.start_export()
    assert task_id is not None
    wait_for(lambda: not page.running)


def test_output_page_replaces_placeholder_and_menu_opens_it(app, tmp_path):
    window = _saved_window(tmp_path)
    assert isinstance(window.output_page, OutputPage)
    window.navigate_to(WorkflowPage.SURVEY)
    window.export_action.trigger()
    assert window.state.view.active_page == WorkflowPage.OUTPUT
    assert window.page_stack.currentWidget() is window.output_page
    # Viewing the Output page does not modify the project.
    assert not window.state.project.dirty


def test_output_page_disables_unavailable_with_reason(app, tmp_path):
    state = _ready(tmp_path)
    page = _keep(OutputPage(state, task_controller=MainWindow(state=state).task_controller))
    assert page.checkboxes["viewpoints"].isEnabled()
    box = page.checkboxes["exposure_count"]
    assert not box.isEnabled() and not box.isChecked()
    assert "Build" in page.reason_labels["exposure_count"].text()
    assert not page.export_button.isEnabled()


def test_export_from_output_page_writes_files_without_dirtying(app, tmp_path):
    window = _saved_window(tmp_path)
    window.navigate_to(WorkflowPage.OUTPUT)
    page = window.output_page
    state = window.state
    sof = state.analysis.survey_observability_field
    revision = state.analysis.inputs_revision
    keys = ["viewpoints", "exposure_count", "observability_state", "contributions",
            "scenario_exposure", "comparison_difference"]
    out = tmp_path / "export"
    _export(page, out, keys)

    result = page.last_result
    assert result is not None and result.successful
    assert set(result.products_written) == set(keys)
    assert len(list(out.iterdir())) == 2 * len(keys)
    assert "Exported 6 product(s)" in page.status_label.text()

    with rasterio.open(out / "Sicily_Survey_exposure_count.tif") as dataset:
        assert dataset.crs == sof.crs and dataset.transform == sof.transform
        assert dataset.shape == sof.exposure_count.shape
        assert dataset.tags()["RIVELERO_PROJECT_NAME"] == "Sicily Survey"
    record = json.loads((out / "Sicily_Survey_viewpoints.csv.json").read_text(encoding="utf-8"))
    assert record["provenance"]["project_saved"] is True

    # Exporting changes nothing in ApplicationState.
    assert not state.project.dirty
    assert "*" not in window.windowTitle()
    assert state.analysis.survey_observability_field is sof
    assert state.analysis.inputs_revision == revision
    assert not state.busy


def test_output_page_conflict_is_reported_and_files_kept(app, tmp_path):
    window = _saved_window(tmp_path)
    page = window.output_page
    out = tmp_path / "export"
    _export(page, out, ["exposure_count"])
    before = {path.name: path.stat().st_mtime_ns for path in out.iterdir()}

    _export(page, out, ["exposure_count", "observability_state"])
    assert page.last_result is None
    assert "already exist" in page.status_label.text()
    assert {path.name: path.stat().st_mtime_ns for path in out.iterdir()} == before
    assert not window.state.project.dirty

    _export(page, out, ["exposure_count", "observability_state"], overwrite=True)
    assert page.last_result.successful
    assert len(list(out.iterdir())) == 4


# ---------------------------------------------------------------------------
# P3: Figures and Report tabs
# ---------------------------------------------------------------------------


def test_output_page_has_data_figures_report_tabs(app, tmp_path):
    state = _ready(tmp_path)
    page = _keep(OutputPage(state, task_controller=MainWindow(state=state).task_controller))
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == ["Data", "Figures", "Report"]
    figures = page.figures_panel
    assert not figures.checkboxes["exposure"].isEnabled()
    assert "Build" in figures.reason_labels["exposure"].text()
    # A Survey-only report is possible before any observability result.
    report = page.report_panel
    report.directory_edit.setText(str(tmp_path / "reports"))
    assert report.run_button.isEnabled()


def test_figure_export_from_panel(app, tmp_path):
    window = _saved_window(tmp_path)
    page = window.output_page
    panel = page.figures_panel
    map_widget = window.analysis_page.analysis_map
    map_widget.current_extent = (500100.0, 500300.0, 4099300.0, 4099500.0)
    zoom = map_widget.current_extent
    panel.set_selection(["analysis_state", "exposure_difference"])
    panel.format_combo.setCurrentIndex(panel.format_combo.findData("svg"))
    assert not panel.dpi_spin.isEnabled()  # DPI applies to PNG only
    panel.directory_edit.setText(str(tmp_path / "figures"))
    assert panel.start() is not None
    wait_for(lambda: not panel.running)
    result = panel.last_result
    assert result is not None and result.successful
    names = sorted(p.name for p in (tmp_path / "figures").iterdir())
    assert names == ["Sicily_Survey_analysis_state.svg", "Sicily_Survey_analysis_state.svg.json",
                     "Sicily_Survey_exposure_difference.svg",
                     "Sicily_Survey_exposure_difference.svg.json"]
    assert "Exported 2 figure(s)" in panel.status_label.text()
    # Figures never touch the GUI map or the project.
    assert map_widget.current_extent == zoom
    assert not window.state.project.dirty and "*" not in window.windowTitle()


def test_report_generation_from_panel(app, tmp_path):
    window = _saved_window(tmp_path)
    window.navigate_to(WorkflowPage.OUTPUT)
    panel = window.output_page.report_panel
    # The default folder name followed the Save As rename.
    assert panel.name_edit.text() == "Sicily_Survey_report"
    panel.set_selection(["analysis_state", "coverage_composition"])
    panel.directory_edit.setText(str(tmp_path))
    assert panel.start() is not None
    wait_for(lambda: not panel.running, timeout=120)
    result = panel.last_result
    assert result is not None and result.figures == ("analysis_state", "coverage_composition")
    report = tmp_path / "Sicily_Survey_report"
    html = (report / "report.html").read_text(encoding="utf-8")
    assert "Sicily Survey" in html and "5. Observability results" in html
    assert (report / "provenance.json").is_file() and (report / "provenance.md").is_file()
    assert panel.open_button.isVisible() or not panel.isVisible()
    assert not window.state.project.dirty

    # A second run refuses to replace the report unless allowed.
    assert panel.start() is not None
    wait_for(lambda: not panel.running, timeout=120)
    assert panel.last_result is None and "already exist" in panel.status_label.text()
    assert (report / "report.html").read_text(encoding="utf-8") == html
