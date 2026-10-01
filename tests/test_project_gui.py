"""P1 tests: project actions in the MainWindow."""

from __future__ import annotations

import shutil

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from observability_fixtures import DEM_NODATA, make_ready_state, visibility_configuration
from rivelero.gui.application_state import WorkflowPage
from rivelero.gui.main_window import MainWindow
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.observability.builder import build_survey_observability_field


_WINDOWS = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class Prompts:
    """Scripted answers for the MainWindow dialog hooks."""

    def __init__(self, window):
        self.answers = []
        self.messages = []
        window.ask_unsaved_changes = lambda: self.answers.pop(0)
        window.show_message = lambda title, text, kind="information": self.messages.append((title, text))
        window.choose_save_path = lambda: None
        window.choose_open_path = lambda: None


def _window(state=None):
    window = MainWindow(state=state)
    _WINDOWS.append(window)
    return window, Prompts(window)


def _built_state(tmp_path):
    dem = tmp_path / "data" / "terrain.tif"
    dem.parent.mkdir(exist_ok=True)
    shutil.copy2(DEM_NODATA, dem)
    state = make_ready_state(
        tmp_path / "cache", dem=dem, survey_buffer_m=300.0, viewpoint_ids=None,
        configuration=visibility_configuration(max_distance_m=250.0),
    )
    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    return state


def test_title_and_dirty_indicator(app, tmp_path):
    window, _prompts = _window()
    assert window.windowTitle() == "Rivelero — Untitled"
    window.state.set_visibility_configuration(visibility_configuration())
    window.refresh_from_state()
    assert window.windowTitle() == "Rivelero — Untitled *"

    assert window.save_project_as(tmp_path / "Sicily Survey")
    assert window.windowTitle() == "Rivelero — Sicily Survey"
    assert window.state.project.project_path.name == "Sicily Survey.rivelero"


def test_save_existing_and_save_as(app, tmp_path):
    window, prompts = _window(_built_state(tmp_path))
    assert not window.save_project()  # no path yet and the dialog was cancelled
    prompts_path = tmp_path / "first.rivelero"
    window.choose_save_path = lambda: str(prompts_path)
    assert window.save_project()
    assert prompts_path.is_file() and not window.state.project.dirty

    window.state.notify_design_changed()
    before = prompts_path.stat().st_mtime_ns
    assert window.save_project()  # saves to the known path without asking
    assert prompts_path.stat().st_mtime_ns >= before
    assert not window.state.project.dirty


def test_open_rehydrates_every_page(app, tmp_path):
    source, _p = _window(_built_state(tmp_path))
    path = tmp_path / "project.rivelero"
    assert source.save_project_as(path)

    window, prompts = _window()
    assert window.open_project(path)
    state = window.state
    assert prompts.messages == []
    assert not state.project.dirty
    assert window.windowTitle() == "Rivelero — project"

    assert window.survey_page.state is state
    assert state.survey.n_viewpoints == 10
    assert window.world_page.terrain_name.value == state.analysis.environment.name
    assert window.observability_page.summary_values["active"].value == "10"
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    assert window.analysis_page.analysis_available
    assert window.sof_status.label.text() == "Observability ready · 10 units"


def test_unsaved_changes_guard(app, tmp_path):
    window, prompts = _window(_built_state(tmp_path))
    window.state.project.mark_dirty()
    other = tmp_path / "other.rivelero"
    helper, _ = _window()
    helper.save_project_as(other)

    prompts.answers = ["cancel"]
    assert not window.new_project()
    assert window.state.survey_ready  # nothing discarded

    prompts.answers = ["cancel"]
    assert not window.open_project(other)
    assert window.state.survey_ready

    event = QCloseEvent()
    prompts.answers = ["cancel"]
    window.closeEvent(event)
    assert not event.isAccepted()

    # "save" without a path asks for one; cancelling the dialog cancels.
    prompts.answers = ["save"]
    assert not window.new_project()
    assert window.state.survey_ready

    prompts.answers = ["discard"]
    assert window.new_project()
    assert not window.state.survey_ready and not window.state.project.dirty
    assert window.windowTitle() == "Rivelero — Untitled"

    event = QCloseEvent()
    window.closeEvent(event)  # clean project: no prompt needed
    assert event.isAccepted()


def test_malformed_project_leaves_session_untouched(app, tmp_path):
    window, prompts = _window(_built_state(tmp_path))
    state = window.state
    survey = state.survey.viewpoint_configuration
    sof = state.analysis.survey_observability_field
    bad = tmp_path / "bad.rivelero"
    bad.write_bytes(b"not a project")

    # Even after agreeing to discard unsaved changes, a failed open must not
    # replace anything.
    prompts.answers = ["discard"]
    assert not window.open_project(bad)
    assert prompts.messages and prompts.messages[0][0] == "Cannot open project"
    assert state.survey.viewpoint_configuration is survey
    assert state.analysis.survey_observability_field is sof


def test_open_reports_resource_warnings(app, tmp_path):
    state = _built_state(tmp_path)
    source, _ = _window(state)
    path = tmp_path / "p.rivelero"
    source.save_project_as(path)
    (tmp_path / "data" / "terrain.tif").unlink()

    window, prompts = _window()
    assert window.open_project(path)
    assert prompts.messages[0][0] == "Project opened with warnings"
    assert "not found" in prompts.messages[0][1]
    assert window.state.survey_ready and not window.state.world_ready
    assert window.world_page.terrain_name.value == "Not loaded"


def test_design_edits_mark_project_dirty(app, tmp_path):
    window, _prompts = _window(_built_state(tmp_path))
    window.save_project_as(tmp_path / "p")
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    panel = window.analysis_page.scenario_panel
    assert not window.state.project.dirty

    panel.save_snapshot("Kept", "")
    assert window.state.project.dirty
    assert window.windowTitle().endswith(" *")
