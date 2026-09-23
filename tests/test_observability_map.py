"""O3 tests: SOF map modes, legends, selection and individual visibility."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from observability_fixtures import (
    DEM_NODATA,
    make_ready_state,
    visibility_configuration,
    wait_for,
)
from rivelero.gui.application_state import VisualizationLayer
from rivelero.gui.domain_service import elevation_valid_mask
from rivelero.gui.observability_map import (
    ObservabilityMapMode,
    ObservabilityMapWidget,
)
from rivelero.gui.observability_page import ObservabilityPage
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.observability.builder import build_survey_observability_field
from rivelero.observability.masks import ObservabilityState
from rivelero.visibility.configuration import SamplingUnit


# Widgets stay referenced for the session so that queued matplotlib redraws
# never reach a canvas whose C++ object Python has already deleted.
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


@pytest.fixture
def four_state_sof(tmp_path):
    # Nodata terrain + buffered survey domain gives all four states.
    state = _built_state(
        tmp_path,
        dem=DEM_NODATA,
        survey_buffer_m=350.0,
        viewpoint_ids=("vp_center_360",),
    )
    return state, state.analysis.survey_observability_field


# ---------------------------------------------------------------------------
# Data/semantics
# ---------------------------------------------------------------------------


def test_elevation_valid_mask_marks_nodata_invalid(tmp_path):
    state = make_ready_state(tmp_path, dem=DEM_NODATA)
    mask = elevation_valid_mask(DEM_NODATA, state.analysis.analysis_grid)
    assert mask.dtype == bool
    assert 0 < np.count_nonzero(~mask) < mask.size
    assert np.array_equal(
        state.analysis.analysis_domain.effective_valid_mask, mask
    )


def test_sof_distinguishes_all_four_states(four_state_sof):
    _, sof = four_state_sof
    counts = sof.state_counts()
    assert all(counts[state] > 0 for state in ObservabilityState)
    assert counts[ObservabilityState.BLIND_SPOT] == sof.n_blindspot_cells
    assert counts[ObservabilityState.OBSERVABLE] == sof.n_observable_cells
    assert sof.maximum_exposure == 1


# ---------------------------------------------------------------------------
# Map widget
# ---------------------------------------------------------------------------


def test_empty_map_shows_message_and_no_colorbar(app):
    widget = _keep(ObservabilityMapWidget())
    assert widget.sof is None
    assert widget.layer_array is None
    assert not widget.colorbar_visible
    assert not widget.display_combo.isEnabled()
    assert "No current observability field" in widget._message_artist.get_text()


def test_state_mode_uses_categorical_legend(app, four_state_sof):
    _, sof = four_state_sof
    widget = _keep(ObservabilityMapWidget())
    widget.set_field(sof)

    assert widget.mode == ObservabilityMapMode.OBSERVABILITY_STATE
    assert np.array_equal(widget.layer_array, sof.observability_state)
    assert widget.legend_labels == [
        "Outside domain",
        "Invalid / unanalysable",
        "Blind spot",
        "Observable",
    ]
    assert not widget.colorbar_visible


def test_exposure_modes_use_continuous_colorbar(app, four_state_sof):
    _, sof = four_state_sof
    widget = _keep(ObservabilityMapWidget())
    widget.set_field(sof)

    widget.set_mode(ObservabilityMapMode.EXPOSURE)
    data = widget.layer_array
    assert widget.colorbar_visible
    assert widget.colorbar_label == "Exposure (number of sampling units)"
    assert widget.legend_labels == []
    assert np.array_equal(np.ma.getmaskarray(data), ~sof.analysable_mask)
    assert widget._layer_image.norm.vmax == sof.maximum_exposure

    widget.set_mode(ObservabilityMapMode.NORMALIZED_EXPOSURE)
    assert widget.colorbar_label.startswith("Normalized exposure")
    assert widget._layer_image.norm.vmin == 0.0
    assert widget._layer_image.norm.vmax == 1.0


def test_blind_spot_mode_never_marks_outside_or_invalid(app, four_state_sof):
    _, sof = four_state_sof
    widget = _keep(ObservabilityMapWidget())
    widget.set_field(sof)
    widget.set_mode(ObservabilityMapMode.BLIND_SPOTS)

    data = widget.layer_array
    shown = ~np.ma.getmaskarray(data)
    assert np.array_equal(shown, sof.analysable_mask)
    assert np.array_equal(np.ma.getdata(data)[shown] == 1, sof.blindspot_mask[shown])
    assert "Blind spot" in widget.legend_labels

    widget.set_mode(ObservabilityMapMode.OBSERVABLE_SPACE)
    data = widget.layer_array
    assert np.array_equal(
        np.ma.getdata(data)[shown] == 1, sof.observable_mask[shown]
    )


def test_colorbar_and_legend_are_never_duplicated(app, four_state_sof):
    _, sof = four_state_sof
    widget = _keep(ObservabilityMapWidget())
    widget.set_field(sof)

    for _ in range(3):
        for mode in widget.modes:
            widget.set_mode(mode)

    # Main axes + the single reusable colourbar axes.
    assert len(widget.figure.axes) == 2
    legends = [child for child in widget.axes.get_children()
               if child.__class__.__name__ == "Legend"]
    assert len(legends) <= 1
    assert len(widget.axes.images) <= 2  # terrain underlay + one layer


def test_viewpoints_use_one_scatter_and_pick_selects(app, tmp_path):
    state = _built_state(tmp_path, viewpoint_ids=None)
    widget = _keep(ObservabilityMapWidget())
    widget.set_field(state.analysis.survey_observability_field)
    widget.set_viewpoints(state.survey.viewpoint_configuration.viewpoints)

    assert widget.viewpoint_count == 10
    assert len(widget.axes.collections) == 1
    assert widget._viewpoint_artist.get_offsets().shape == (10, 2)

    received = []
    widget.viewpoint_selected.connect(received.append)

    class Pick:
        artist = widget._viewpoint_artist
        ind = [3]

    widget._on_pick(Pick())
    assert received == [widget._viewpoint_ids[3]]
    assert widget._selected_artist.get_visible()


def test_zoom_is_preserved_across_mode_changes(app, four_state_sof):
    _, sof = four_state_sof
    widget = _keep(ObservabilityMapWidget())
    widget.set_field(sof)
    left, right, bottom, top = widget.full_extent
    zoomed = (left, left + (right - left) / 4, bottom, bottom + (top - bottom) / 4)
    widget.axes.set_xlim(zoomed[0], zoomed[1])
    widget.axes.set_ylim(zoomed[2], zoomed[3])

    widget.set_mode(ObservabilityMapMode.EXPOSURE)
    assert widget.axes.get_xlim() == pytest.approx(zoomed[:2])
    assert widget.axes.get_ylim() == pytest.approx(zoomed[2:])


# ---------------------------------------------------------------------------
# Page integration
# ---------------------------------------------------------------------------


def test_page_selection_uses_canonical_state(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(ObservabilityPage(state))

    page.observability_map.viewpoint_selected.emit("vp_overlap_west")
    assert state.selection.viewpoint_id == "vp_overlap_west"
    assert page.observability_map._selected_viewpoint_id == "vp_overlap_west"

    # A selection made elsewhere (Survey) is shown after rehydration and
    # does not invalidate the SOF.
    state.select_viewpoint("vp_center_360")
    page.refresh_from_state()
    assert page.observability_map._selected_viewpoint_id == "vp_center_360"
    assert state.observability_ready


def test_individual_visibility_is_read_from_cache_without_recompute(
    app, tmp_path, monkeypatch
):
    state = _built_state(tmp_path)
    state.analysis.visibility_store.clear_memory()

    def fail(**_kwargs):
        raise AssertionError("cached visibility must not be recomputed")

    monkeypatch.setattr(
        "rivelero.gui.observability_service.compute_viewpoint_visibility", fail
    )

    page = _keep(ObservabilityPage(state))
    state.select_viewpoint("vp_center_360")
    page._refresh_selected_unit()

    assert "cached" in page.unit_status_label.text()
    assert "observer height 1.75 m" in page.unit_parameters_label.text()
    assert page.show_unit_button.isEnabled()
    assert not page.compute_unit_button.isEnabled()

    page.show_unit_button.click()
    assert page.observability_map.mode == ObservabilityMapMode.INDIVIDUAL_VISIBILITY
    assert state.view.active_layer == VisualizationLayer.EFFECTIVE_VISIBILITY
    shown = page.observability_map.layer_array
    assert shown is not None
    assert np.count_nonzero(np.ma.getdata(shown)) > 0


def test_uncomputed_unit_can_be_computed_through_store(app, tmp_path):
    state = make_ready_state(tmp_path)
    page = _keep(ObservabilityPage(state))
    state.select_viewpoint("vp_overlap_east")
    page._refresh_selected_unit()

    assert "not yet computed" in page.unit_status_label.text()
    assert page.compute_unit_button.isEnabled()

    page.compute_unit_button.click()
    wait_for(lambda: not state.busy and page._unit_task_id is None)

    assert "cached" in page.unit_status_label.text()
    assert state.analysis.visibility_store.disk_item_count == 1
    assert not state.observability_ready  # computing one unit builds no SOF


def test_excluded_unit_reports_policy_reason(app, tmp_path):
    from rivelero.visibility.configuration import MissingMetadataPolicy

    state = _built_state(
        tmp_path,
        viewpoint_ids=("vp_center_360", "vp_missing_height"),
        configuration=visibility_configuration(
            missing_observer_height_policy=MissingMetadataPolicy.EXCLUDE,
            default_observer_height_m=None,
        ),
    )
    page = _keep(ObservabilityPage(state))
    state.select_viewpoint("vp_missing_height")
    page._refresh_selected_unit()

    assert "Excluded by missing-metadata policy" in page.unit_status_label.text()
    assert not page.compute_unit_button.isEnabled()
    assert "vp_missing_height" in page.report_details_label.text()


def test_event_sampling_offers_events_of_selected_viewpoint(app, tmp_path):
    state = _built_state(
        tmp_path,
        viewpoint_ids=("vp_redundant_a", "vp_overlap_east", "vp_complementary"),
        configuration=visibility_configuration(
            sampling_unit=SamplingUnit.OBSERVATION_EVENT
        ),
    )
    page = _keep(ObservabilityPage(state))
    state.select_viewpoint("vp_redundant_a")
    page._refresh_selected_unit()

    assert page.event_combo.isVisibleTo(page)
    events = [page.event_combo.itemData(i) for i in range(page.event_combo.count())]
    assert events == ["event_001", "event_003"]

    page.event_combo.setCurrentIndex(1)
    assert state.selection.observation_event_id == "event_003"
    assert "event_003: cached" in page.unit_status_label.text()
    assert "heading 180°" in page.unit_parameters_label.text()
    assert page.summary_values["unit"].value == "ObservationEvent"


def test_summary_matches_canonical_sof(app, four_state_sof):
    state, sof = four_state_sof
    page = _keep(ObservabilityPage(state))
    values = {key: widget.value for key, widget in page.summary_values.items()}
    counts = sof.state_counts()

    assert values["analysable"] == f"{sof.n_analysable_cells:,}"
    assert values["observable"] == f"{sof.n_observable_cells:,}"
    assert values["blind"] == f"{sof.n_blindspot_cells:,}"
    assert values["invalid"] == f"{counts[ObservabilityState.INVALID]:,}"
    assert values["outside"] == f"{counts[ObservabilityState.OUTSIDE_DOMAIN]:,}"
    assert values["active"] == "1"
    assert page.report_values["requested"].value == "1"


def test_map_mode_is_restored_from_state(app, tmp_path):
    state = _built_state(tmp_path)
    page = _keep(ObservabilityPage(state))
    page.observability_map.set_mode(ObservabilityMapMode.EXPOSURE)
    assert state.view.active_layer == VisualizationLayer.EXPOSURE

    second = _keep(ObservabilityPage(state))
    assert second.observability_map.mode == ObservabilityMapMode.EXPOSURE
