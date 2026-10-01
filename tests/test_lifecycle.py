"""P4 tests: GUI lifecycle - lazy panels, canvases, New/Open/Close, refresh."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # before Matplotlib's Qt backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from observability_fixtures import (
    DEM_FLAT, DEM_NODATA, make_ready_state, visibility_configuration, wait_for,
)
from rivelero.analysis.comparison import BASELINE_ID, ScenarioWorkspace
from rivelero.analysis.contribution import analyse_contributions
from rivelero.analysis.scenario import SurveyDesignScenario
from rivelero.gui.analysis_page import AnalysisPage
from rivelero.gui.application_state import WorkflowPage
from rivelero.gui.canvas import SafeFigureCanvas
from rivelero.gui.main_window import MainWindow
from rivelero.gui.observability_map import ObservabilityMapWidget
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.gui.project_service import save_state
from rivelero.gui.task_controller import make_progress_task
from rivelero.observability.builder import build_survey_observability_field

_KEEP = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _keep(widget):
    _KEEP.append(widget)
    return widget


def _built(tmp_path, *, dem=DEM_NODATA, viewpoint_ids=None, buffer=100.0):
    state = make_ready_state(tmp_path / "cache", dem=dem, survey_buffer_m=buffer,
                             viewpoint_ids=viewpoint_ids,
                             configuration=visibility_configuration(max_distance_m=250.0))
    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    return state


def _with_design(state):
    sof = state.analysis.survey_observability_field
    store = state.analysis.visibility_store
    state.set_contribution_analysis(analyse_contributions(sof, store),
                                    inputs_revision=state.analysis.inputs_revision)
    scenario = SurveyDesignScenario(sof)
    key = sof.active_keys[0]
    scenario.deactivate({key: store.get(key).visibility_mask})
    state.set_design_scenario(scenario)
    workspace = state.analysis.scenario_workspace = ScenarioWorkspace()
    snapshot = workspace.save(scenario, name="A snapshot")
    workspace.left_id, workspace.right_id = BASELINE_ID, snapshot.snapshot_id


def _window(state=None):
    window = _keep(MainWindow(state=state))
    window.show_message = lambda *args, **kwargs: None
    window.ask_unsaved_changes = lambda: "discard"
    return window


def _visit_everything(window):
    for page in WorkflowPage:
        window.navigate_to(page)
        QApplication.processEvents()
    tabs = window.analysis_page.tabs
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        QApplication.processEvents()


def _maps(widget):
    return widget.findChildren(ObservabilityMapWidget)


# ---------------------------------------------------------------------------
# Lazy Analysis panels
# ---------------------------------------------------------------------------


def test_analysis_design_panels_are_built_on_first_open(app, tmp_path):
    state = _built(tmp_path)
    page = _keep(AnalysisPage(state))
    assert page.constructed_panels == ()
    assert len(_maps(page)) == 1  # only the Overview map
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == [
        "Overview", "Contribution", "Scenario", "Compare"]

    page.tabs.setCurrentIndex(1)
    assert page.constructed_panels == ("contribution",)
    panel = page.contribution_panel
    assert page.tabs.currentWidget() is panel
    assert len(_maps(page)) == 2
    # Leaving and returning never rebuilds a panel.
    page.tabs.setCurrentIndex(0)
    page.tabs.setCurrentIndex(1)
    assert page.contribution_panel is panel and len(_maps(page)) == 2

    page.tabs.setCurrentIndex(3)
    assert page.constructed_panels == ("contribution", "compare")
    assert page.tabs.tabText(3) == "Compare" and page.tabs.currentWidget() is page.compare_panel
    # Programmatic access builds a panel too (and places it in its tab).
    scenario = page.scenario_panel
    assert page.tabs.indexOf(scenario) == 2 and len(_maps(page)) == 4


def test_unopened_panels_do_not_refresh_or_create_state(app, tmp_path):
    state = _built(tmp_path)
    page = _keep(AnalysisPage(state))
    page.refresh_from_state()
    assert state.analysis.design_scenario is None
    assert state.analysis.scenario_workspace is None


# ---------------------------------------------------------------------------
# Canvas lifecycle
# ---------------------------------------------------------------------------


def test_every_canvas_is_the_safe_canvas(app):
    window = _window()
    canvases = window.findChildren(FigureCanvasQTAgg)
    assert canvases and all(isinstance(c, SafeFigureCanvas) for c in canvases)


def test_hidden_canvas_defers_drawing_until_shown(app):
    canvas = SafeFigureCanvas(Figure())
    draws = []
    canvas.draw = lambda: draws.append(1)
    canvas.draw_idle()
    QApplication.processEvents()
    assert draws == []  # hidden: nothing rendered
    canvas.draw_idle()  # still queues (Matplotlib's pending flag was reset)
    QApplication.processEvents()
    assert draws == []
    canvas.resize(200, 150)
    canvas.show()
    wait_for(lambda: draws)
    assert len(draws) == 1
    canvas.close()


def test_deleted_canvas_ignores_queued_redraw(app):
    canvas = SafeFigureCanvas(Figure())
    canvas.show()
    canvas.draw_idle()  # queued with a zero-delay timer
    canvas.deleteLater()
    QApplication.processEvents()
    QApplication.processEvents()
    canvas._draw_idle()  # a stale callback must not raise


# ---------------------------------------------------------------------------
# Output follows project changes
# ---------------------------------------------------------------------------


def test_output_names_follow_save_as_while_visible(app, tmp_path):
    window = _window(_built(tmp_path))
    window.navigate_to(WorkflowPage.OUTPUT)
    report = window.output_page.report_panel
    assert report.name_edit.text() == "Untitled_Rivelero_project_report"
    assert window.save_project_as(tmp_path / "Sicily Survey")
    # No navigation: the visible page refreshed after the save.
    assert report.name_edit.text() == "Sicily_Survey_report"
    catalog_names = window.output_page.checkboxes["viewpoints"].toolTip()
    assert catalog_names == "Sicily_Survey_viewpoints.csv"


# ---------------------------------------------------------------------------
# Navigation, New Project, Open Project
# ---------------------------------------------------------------------------


def test_viewing_pages_never_changes_the_project(app, tmp_path):
    state = _built(tmp_path)
    _with_design(state)
    window = _window(state)
    assert window.save_project_as(tmp_path / "A")
    revision = state.analysis.inputs_revision
    sof = state.analysis.survey_observability_field
    _visit_everything(window)
    assert not state.project.dirty and "*" not in window.windowTitle()
    assert state.analysis.inputs_revision == revision
    assert state.analysis.survey_observability_field is sof


def _assert_empty(window):
    state = window.state
    page = window.analysis_page
    assert state.survey.viewpoint_configuration is None and state.selection.viewpoint_id is None
    assert window.survey_page.table.rowCount() == 0
    assert window.survey_page.map_widget.configuration is None
    assert window.world_page.world_map.raster_path is None
    assert window.observability_page.observability_map.sof is None
    assert page.summary is None and page.analysis_map.sof is None
    for name in page.constructed_panels:
        panel = getattr(page, f"{name}_panel")
        if name == "contribution":
            assert panel.model.rowCount() == 0 and panel.contribution_map.sof is None
        if name == "scenario":
            assert panel.units_proxy.rowCount() == 0 and panel.scenario_map.sof is None
        if name == "compare":
            assert panel.comparison is None and panel.library_table.rowCount() == 0
            assert [panel.right_combo.itemData(i) for i in range(panel.right_combo.count())] == [BASELINE_ID]
    output = window.output_page
    assert not any(box.isEnabled() for key, box in output.checkboxes.items())
    assert not any(box.isEnabled() for box in output.figures_panel.checkboxes.values())
    assert output.report_panel.name_edit.text() == "Untitled_Rivelero_project_report"


def test_new_project_leaves_nothing_behind(app, tmp_path):
    state = _built(tmp_path)
    _with_design(state)
    window = _window(state)
    _visit_everything(window)
    window.analysis_page.contribution_panel.table.selectRow(0)
    state.select_viewpoint(state.survey.viewpoint_configuration.viewpoints[0].viewpoint_id)
    assert window.save_project_as(tmp_path / "Project A")
    window.navigate_to(WorkflowPage.OUTPUT)

    assert window.new_project()
    QApplication.processEvents()
    _assert_empty(window)
    assert window.windowTitle() == "Rivelero — Untitled"


def test_open_other_project_shows_only_its_state(app, tmp_path):
    a = _built(tmp_path / "a")
    _with_design(a)
    b = _built(tmp_path / "b", dem=DEM_FLAT, viewpoint_ids=("vp_center_360", "vp_overlap_west"),
               buffer=None)
    save_state(b, tmp_path / "Project B")
    b_viewpoints = b.survey.n_viewpoints

    window = _window(a)
    _visit_everything(window)
    assert window.save_project_as(tmp_path / "Project A")
    assert window.open_project(tmp_path / "Project B.rivelero")
    QApplication.processEvents()
    _visit_everything(window)

    state = window.state
    page = window.analysis_page
    assert state.project.name == "Project B"
    assert window.survey_page.table.rowCount() == b_viewpoints
    assert window.world_page.world_map.raster_path.name == DEM_FLAT.name
    sof = state.analysis.survey_observability_field
    assert page.analysis_map.sof is sof and page.summary.active_units == b_viewpoints
    assert page.contribution_panel.model.rowCount() == 0  # A's table is gone
    assert page.scenario_panel.units_proxy.rowCount() == b_viewpoints
    assert page.compare_panel.library_table.rowCount() == 0  # no A snapshots
    assert page.compare_panel.comparison is None
    assert not window.output_page.checkboxes["contributions"].isEnabled()
    assert window.output_page.report_panel.name_edit.text() == "Project_B_report"

    # ...and back to A: its snapshot returns, B's state does not remain.
    assert window.open_project(tmp_path / "Project A.rivelero")
    QApplication.processEvents()
    assert window.world_page.world_map.raster_path.name == DEM_NODATA.name
    assert page.compare_panel.library_table.rowCount() == 1


def test_repeated_new_open_does_not_accumulate_resources(app, tmp_path):
    state = _built(tmp_path)
    window = _window(state)
    _visit_everything(window)
    assert window.save_project_as(tmp_path / "P")

    def census():
        canvases = window.findChildren(FigureCanvasQTAgg)
        callbacks = sum(
            len(registry) for c in canvases for registry in c.callbacks.callbacks.values()
        )
        axes = sum(len(c.figure.axes) for c in canvases)
        return len(canvases), callbacks, axes

    for _ in range(2):
        window.new_project()
        window.open_project(tmp_path / "P.rivelero")
        _visit_everything(window)
    before = census()
    for _ in range(3):
        window.new_project()
        window.open_project(tmp_path / "P.rivelero")
        _visit_everything(window)
    assert census() == before


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------


class _Close:
    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def test_close_idle_and_dirty(app):
    window = _window()
    event = _Close()
    window.closeEvent(event)
    assert event.accepted is True

    window.state.set_visibility_configuration(visibility_configuration())
    asked = []
    window.ask_unsaved_changes = lambda: asked.append(1) or "cancel"
    event = _Close()
    window.closeEvent(event)
    assert event.accepted is False and asked == [1]  # unsaved-changes guard kept


def test_close_while_task_running(app):
    window = _window()
    stop = threading.Event()

    def long_task(*, progress_callback):
        for index in range(10_000):
            if stop.is_set():
                break
            progress_callback(index, 10_000, "unit")
            time.sleep(0.01)
        return "finished"

    results = []
    window.task_controller.task_result.connect(lambda _id, value: results.append(value))
    window.task_controller.start(
        task_name="Long task", function=make_progress_task(function=long_task, kwargs={}),
        inject_context=True,
    )
    wait_for(lambda: window.state.busy)

    window.ask_stop_running_task = lambda: False
    event = _Close()
    window.closeEvent(event)
    assert event.accepted is False and window.state.busy  # declined: keeps running

    window.ask_stop_running_task = lambda: True
    event = _Close()
    window.closeEvent(event)
    stop.set()
    assert event.accepted is True
    assert not window.state.busy and not window.task_controller.busy
    assert results == []  # cancelled: no completion delivered


def test_task_errors_are_described_for_users():
    from rivelero.export.catalog import ExportConflictError
    from rivelero.gui.task_controller import describe_error
    from pathlib import Path

    conflict = describe_error(ExportConflictError([Path("a.tif")]))
    assert conflict.startswith("1 file(s) already exist") and "ExportConflictError" not in conflict
    denied = describe_error(PermissionError(13, "Permission denied", "C:/locked/out.tif"))
    assert denied == "Permission denied for C:/locked/out.tif. Choose a location you can write to."
    assert describe_error(ValueError("DPI must be between 50 and 1200.")) == "DPI must be between 50 and 1200."
    assert describe_error(KeyError("vp_1")) == "vp_1"
    assert describe_error(ZeroDivisionError()) == "ZeroDivisionError (no message; see the error details)."
    assert describe_error(ZeroDivisionError("division by zero")) == "ZeroDivisionError: division by zero"
