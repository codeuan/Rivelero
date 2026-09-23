"""A1 tests: Analysis & Design page readiness, content and navigation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from observability_fixtures import DEM_NODATA, make_ready_state, wait_for
from rivelero.analysis.coverage import summarize_coverage
from rivelero.gui.analysis_page import ANALYSIS_MAP_MODES, AnalysisPage
from rivelero.gui.application_state import WorkflowPage
from rivelero.gui.main_window import MainWindow
from rivelero.gui.observability_map import ObservabilityMapMode
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.observability.builder import build_survey_observability_field


_WIDGETS = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _keep(widget):
    _WIDGETS.append(widget)
    return widget


def _built_state(tmp_path, **kwargs):
    state = make_ready_state(tmp_path, **kwargs)
    request = prepare_build(state)
    result = build_survey_observability_field(**request.build_kwargs())
    install_build_result(state, request, result)
    return state


def test_page_without_sof_asks_for_observability(app, tmp_path):
    state = make_ready_state(tmp_path)
    page = _keep(AnalysisPage(state))

    assert not page.analysis_available
    assert page.readiness.isVisibleTo(page)
    assert not page.analysis_content.isVisibleTo(page)
    assert "Build observability" in page.readiness.title_label.text()
    assert not page.continue_button.isEnabled()

    requested = []
    page.observability_requested.connect(lambda: requested.append(True))
    page.readiness.action_button.click()
    assert requested == [True]


def test_page_with_current_sof_shows_summary(app, tmp_path):
    state = _built_state(
        tmp_path, dem=DEM_NODATA, survey_buffer_m=350.0, viewpoint_ids=None
    )
    sof = state.analysis.survey_observability_field
    page = _keep(AnalysisPage(state))
    expected = summarize_coverage(sof)

    assert page.analysis_available
    assert page.analysis_content.isVisibleTo(page)
    assert not page.readiness.isVisibleTo(page)
    assert page.continue_button.isEnabled()

    values = {name: widget.value for name, widget in page.summary_values.items()}
    assert values["observable"].startswith(f"{expected.observable_fraction:.1%}")
    assert f"{expected.observable_cells:,} cells" in values["observable"]
    assert values["blind"].startswith(f"{expected.blind_fraction:.1%}")
    assert values["unique"].startswith(f"{expected.unique_fraction:.1%}")
    assert values["repeated"].startswith(f"{expected.repeated_fraction:.1%}")
    assert values["maximum"] == f"{sof.maximum_exposure:,}"
    assert values["analysable"] == f"{sof.n_analysable_cells:,}"
    assert values["invalid"] == f"{expected.invalid_cells:,}"
    assert values["outside"] == f"{expected.outside_domain_cells:,}"
    assert values["active"] == "10"


def test_page_reuses_observability_map_with_coverage_layer(app, tmp_path):
    state = _built_state(
        tmp_path, dem=DEM_NODATA, survey_buffer_m=350.0, viewpoint_ids=None
    )
    sof = state.analysis.survey_observability_field
    page = _keep(AnalysisPage(state))
    widget = page.analysis_map

    assert widget.sof is sof
    assert widget.modes == ANALYSIS_MAP_MODES
    assert widget.mode == ObservabilityMapMode.COVERAGE_CLASS
    assert widget.legend_labels == [
        "Outside domain",
        "Invalid / unanalysable",
        "Blind spot (exposure 0)",
        "Unique coverage (exposure 1)",
        "Repeated coverage (exposure ≥ 2)",
    ]
    classes = widget.layer_array
    assert np.count_nonzero(classes == 4) == summarize_coverage(sof).repeated_cells

    # Analysis map modes stay local and do not change the Observability
    # page's layer.
    before = state.view.active_layer
    widget.set_mode(ObservabilityMapMode.EXPOSURE)
    assert state.view.active_layer == before
    assert widget.colorbar_visible


def test_charts_are_drawn_from_summary(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    summary = page.summary

    bars = page.distribution_axes.containers[0]
    heights = [patch.get_height() for patch in bars.patches]
    bins = summary.distribution.binned()
    assert heights == pytest.approx([100.0 * item.fraction for item in bins])
    assert sum(heights) == pytest.approx(100.0)

    widths = [patch.get_width() for patch in page.composition_axes.patches]
    assert widths == pytest.approx(
        [
            100.0 * summary.blind_fraction,
            100.0 * summary.unique_fraction,
            100.0 * summary.repeated_fraction,
        ]
    )


def test_summary_cache_follows_sof_identity(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    first = page.summary

    page.refresh_from_state()
    assert page.summary is first  # unchanged SOF: no recomputation

    request = prepare_build(state)
    result = build_survey_observability_field(**request.build_kwargs())
    install_build_result(state, request, result)
    page.refresh_from_state()
    assert page.summary is not first


def test_stale_sof_blocks_analysis(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    assert page.analysis_available

    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    state.replace_viewpoint(replace(viewpoint, heading_deg=15.0))
    page.refresh_from_state()

    assert not page.analysis_available
    assert page.summary is None
    assert "out of date" in page.readiness.title_label.text()
    assert "Survey changed." in page.readiness.description_label.text()
    assert page.analysis_map.sof is None
    assert not page.continue_button.isEnabled()


def test_navigation_to_and_from_analysis(app, tmp_path):
    state = make_ready_state(tmp_path)
    window = _keep(MainWindow(state=state))
    page = window.analysis_page

    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    assert window.page_stack.currentWidget() is page
    assert not page.analysis_available

    page.readiness.action_button.click()
    assert state.view.active_page == WorkflowPage.OBSERVABILITY

    observability = window.observability_page
    observability.build_button.click()
    wait_for(lambda: observability._build_task_id is None)
    observability.continue_button.click()

    assert state.view.active_page == WorkflowPage.ANALYSIS_DESIGN
    assert page.analysis_available
    summary = page.summary

    # Away and back keeps the same analysis of the same SOF.
    window.navigate_to(WorkflowPage.SURVEY)
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    assert page.summary is summary

    page.continue_button.click()
    assert state.view.active_page == WorkflowPage.OUTPUT


def test_summary_agrees_with_observability_page(app, tmp_path):
    state = _built_state(
        tmp_path, dem=DEM_NODATA, survey_buffer_m=350.0, viewpoint_ids=None
    )
    window = _keep(MainWindow(state=state))
    window.navigate_to(WorkflowPage.OBSERVABILITY)
    observability = window.observability_page.summary_values
    window.navigate_to(WorkflowPage.ANALYSIS_DESIGN)
    analysis = window.analysis_page.summary_values

    assert analysis["analysable"].value == observability["analysable"].value
    assert analysis["invalid"].value == observability["invalid"].value
    assert analysis["outside"].value == observability["outside"].value
    assert analysis["maximum"].value == observability["max_exposure"].value
    assert analysis["active"].value == observability["active"].value
    assert analysis["observable"].value.startswith(observability["fraction"].value)


def test_map_selection_updates_canonical_state(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(AnalysisPage(state))
    page.analysis_map.viewpoint_selected.emit("vp_overlap_west")
    assert state.selection.viewpoint_id == "vp_overlap_west"
    assert state.observability_ready
