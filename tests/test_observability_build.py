"""O2 tests: VisibilityStore configuration, cache identity and SOF builds."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from observability_fixtures import (
    make_ready_state,
    visibility_configuration,
    wait_for,
)
from rivelero.gui.application_state import (
    OBSERVABILITY_BUILD_TASK_NAME,
    ApplicationState,
    StaleObservabilityResultError,
    TaskStatus,
)
from rivelero.gui.observability_service import (
    ObservabilityNotReadyError,
    VisibilityKeyResolver,
    compute_unit_visibility,
    create_visibility_store,
    default_visibility_cache_directory,
    install_build_result,
    prepare_build,
    store_settings_match,
)
from rivelero.observability.builder import (
    build_survey_observability_field,
    make_visibility_key,
)
from rivelero.observability.storage import (
    StoredVisibility,
    VisibilityKey,
    VisibilityStore,
)
from rivelero.visibility.configuration import (
    MissingMetadataPolicy,
    SamplingUnit,
)


def _build(state: ApplicationState, **kwargs):
    request = prepare_build(state, **kwargs)
    result = build_survey_observability_field(**request.build_kwargs())
    return request, result


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_empty_state_lists_every_missing_prerequisite():
    blockers = ApplicationState().observability_build_blockers()

    assert any("Survey" in item for item in blockers)
    assert any("World" in item for item in blockers)
    assert any("visibility configuration" in item for item in blockers)
    assert any("storage" in item for item in blockers)

    with pytest.raises(ObservabilityNotReadyError):
        prepare_build(ApplicationState())


def test_ready_state_has_no_blockers(tmp_path):
    state = make_ready_state(tmp_path)
    assert state.observability_build_blockers() == []


def test_missing_store_blocks_build(tmp_path):
    state = make_ready_state(tmp_path, with_store=False)
    assert state.observability_build_blockers() == [
        "Configure visibility storage."
    ]


def test_event_sampling_without_events_is_a_blocker(tmp_path):
    state = make_ready_state(
        tmp_path,
        viewpoint_ids=("vp_center_360",),
        configuration=visibility_configuration(
            sampling_unit=SamplingUnit.OBSERVATION_EVENT
        ),
    )
    assert state.survey.n_observation_events == 0
    assert any(
        "ObservationEvents" in item
        for item in state.observability_build_blockers()
    )


# ---------------------------------------------------------------------------
# VisibilityStore construction and cache semantics
# ---------------------------------------------------------------------------


def test_default_cache_directory_honours_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RIVELERO_CACHE_DIR", str(tmp_path / "custom"))
    assert default_visibility_cache_directory() == (
        tmp_path / "custom" / "visibility_cache"
    )
    store = create_visibility_store()
    assert store.cache_directory == (tmp_path / "custom" / "visibility_cache").resolve()
    assert store.compressed is True


def test_store_settings_match(tmp_path):
    store = VisibilityStore(tmp_path, max_memory_items=5, compressed=False)
    assert store_settings_match(
        store, cache_directory=tmp_path, max_memory_items=5, compressed=False
    )
    assert not store_settings_match(
        store, cache_directory=tmp_path, max_memory_items=6, compressed=False
    )


def test_build_is_lazy_and_second_build_reuses_disk_cache(tmp_path):
    state = make_ready_state(tmp_path)
    store = state.analysis.visibility_store
    assert store.disk_item_count == 0

    _, first = _build(state)
    assert first.report.requested_units == 3
    assert first.report.computed_units == 3
    assert first.report.cache_hits == 0
    assert store.disk_item_count == 3

    store.clear_memory()
    _, second = _build(state)
    assert second.report.computed_units == 0
    assert second.report.cache_hits == 3
    assert np.array_equal(first.sof.exposure_count, second.sof.exposure_count)


def test_changed_configuration_with_same_id_does_not_reuse_stale_cache(tmp_path):
    # O1 keeps configuration_id when a configuration is re-saved. The cache
    # key must still distinguish the new assumptions.
    state = make_ready_state(tmp_path)
    _, short = _build(state)

    state.set_visibility_configuration(
        visibility_configuration(max_distance_m=300.0)
    )
    assert (
        state.analysis.visibility_configuration.configuration_id
        == short.sof.visibility_configuration_id
    )

    _, long = _build(state)
    assert long.report.computed_units == 3
    assert long.report.cache_hits == 0
    assert long.sof.n_observable_cells > short.sof.n_observable_cells

    # Reverting to the original assumptions reuses the original masks.
    state.set_visibility_configuration(visibility_configuration())
    _, reverted = _build(state)
    assert reverted.report.cache_hits == 3
    assert np.array_equal(reverted.sof.exposure_count, short.sof.exposure_count)


def test_edited_viewpoint_with_same_id_is_recomputed(tmp_path):
    state = make_ready_state(tmp_path)
    _, before = _build(state)

    viewpoint = state.survey.viewpoint_configuration.get_viewpoint("vp_center_360")
    state.replace_viewpoint(replace(viewpoint, x=viewpoint.x + 200.0))

    _, after = _build(state)
    assert after.report.computed_units == 1
    assert after.report.cache_hits == 2
    assert not np.array_equal(before.sof.exposure_count, after.sof.exposure_count)


def test_edited_sensor_fov_is_recomputed(tmp_path):
    state = make_ready_state(tmp_path, viewpoint_ids=("vp_missing_heading",))
    state.set_visibility_configuration(
        visibility_configuration(
            missing_heading_policy=MissingMetadataPolicy.USE_DEFAULT,
            default_heading_deg=90.0,
        )
    )
    _, narrow = _build(state)

    sensor = state.survey.sensors["sensor_rgb_70"]
    state.replace_sensor(replace(sensor, horizontal_fov_deg=180.0))

    _, wide = _build(state)
    assert wide.report.computed_units == 1
    assert wide.sof.n_observable_cells > narrow.sof.n_observable_cells


def test_make_visibility_key_requires_referenced_sensor(tmp_path):
    state = make_ready_state(tmp_path)
    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    assert viewpoint.sensor_id is not None

    with pytest.raises(ValueError, match="supply sensor"):
        make_visibility_key(
            viewpoint=viewpoint,
            environment=state.analysis.environment,
            domain=state.analysis.analysis_domain,
            visibility_configuration=state.analysis.visibility_configuration,
        )


def test_key_without_fingerprint_keeps_historical_identity(tmp_path):
    key = VisibilityKey(
        sampling_unit_id="vp",
        sampling_unit_type="viewpoint",
        viewpoint_id="vp",
        environment_id="env",
        analysis_domain_id="domain",
        visibility_configuration_id="config",
    )
    assert key.canonical_string == "viewpoint|vp|vp|env|domain|config"
    assert "input_fingerprint" not in key.as_metadata()

    fingerprinted = replace(key, input_fingerprint="abc")
    assert fingerprinted.digest != key.digest


def test_clear_disk_only_deletes_managed_masks(tmp_path):
    state = make_ready_state(tmp_path)
    _build(state)
    store = state.analysis.visibility_store

    unrelated = tmp_path / "user_data.npz"
    np.savez(unrelated, values=np.arange(3))
    nested = tmp_path / "ab" / "not-a-digest.npz"
    nested.parent.mkdir(exist_ok=True)
    np.savez(nested, values=np.arange(3))

    assert store.disk_item_count == 3
    assert store.clear_disk() == 3
    assert unrelated.is_file()
    assert nested.is_file()
    assert store.disk_item_count == 0


def test_stored_visibility_records_resolved_parameters(tmp_path):
    state = make_ready_state(tmp_path, viewpoint_ids=("vp_missing_height",))
    _build(state)
    store = state.analysis.visibility_store
    store.clear_memory()

    selection = VisibilityKeyResolver().resolve(
        state, viewpoint_id="vp_missing_height"
    )
    stored = store.get(selection.key)
    resolved = stored.metadata["resolved_parameters"]
    assert resolved["used_default_observer_height"] is True
    assert resolved["observer_height_m"] == pytest.approx(1.75)


def test_store_is_safe_for_concurrent_readers(tmp_path):
    state = make_ready_state(tmp_path)
    _build(state)
    store = state.analysis.visibility_store
    store.max_memory_items = 1
    store.clear_memory()
    keys = [
        VisibilityKeyResolver().resolve(state, viewpoint_id=viewpoint_id).key
        for viewpoint_id in ("vp_center_360", "vp_overlap_west", "vp_overlap_east")
    ]
    errors = []

    def reader():
        try:
            for _ in range(60):
                for key in keys:
                    assert store.get(key) is not None
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.memory_item_count <= 1


def test_compute_unit_visibility_uses_store_pathway(tmp_path):
    state = make_ready_state(tmp_path)
    selection = VisibilityKeyResolver().resolve(state, viewpoint_id="vp_center_360")
    store = state.analysis.visibility_store

    assert not store.contains(selection.key)
    stored = compute_unit_visibility(
        selection=selection,
        environment=state.analysis.environment,
        domain=state.analysis.analysis_domain,
        visibility_configuration=state.analysis.visibility_configuration,
        store=store,
    )
    assert isinstance(stored, StoredVisibility)
    assert store.contains(selection.key)

    # A subsequent SOF build reuses the unit computed for inspection.
    _, result = _build(state)
    assert result.report.cache_hits == 1


# ---------------------------------------------------------------------------
# Result installation
# ---------------------------------------------------------------------------


def test_successful_build_installs_sof_and_report(tmp_path):
    state = make_ready_state(tmp_path)
    request, result = _build(state)
    install_build_result(state, request, result)

    assert state.observability_ready
    assert state.analysis.survey_observability_field is result.sof
    assert state.analysis.build_report is result.report
    assert state.analysis.invalidation_reason is None


def test_result_built_from_edited_inputs_is_rejected(tmp_path):
    state = make_ready_state(tmp_path)
    request, result = _build(state)

    # Same ViewpointConfiguration ID, different content.
    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    state.replace_viewpoint(replace(viewpoint, observer_height_m=10.0))

    with pytest.raises(StaleObservabilityResultError):
        install_build_result(state, request, result)
    assert not state.observability_ready


def test_configuration_change_invalidates_and_records_reason(tmp_path):
    state = make_ready_state(tmp_path)
    request, result = _build(state)
    install_build_result(state, request, result)

    state.set_visibility_configuration(visibility_configuration(max_distance_m=90.0))

    assert not state.observability_ready
    assert state.analysis.build_report is None
    assert state.analysis.invalidation_reason == "Visibility configuration changed."
    assert state.readiness_summary()["observability"]["invalidated"]


def test_store_change_and_selection_keep_sof(tmp_path):
    state = make_ready_state(tmp_path)
    request, result = _build(state)
    install_build_result(state, request, result)

    state.select_viewpoint("vp_center_360")
    state.set_visibility_store(create_visibility_store(tmp_path / "other"))
    state.set_map_extent((0.0, 1.0, 0.0, 1.0))

    assert state.observability_ready


def test_excluded_units_are_reported_not_fatal(tmp_path):
    state = make_ready_state(
        tmp_path,
        viewpoint_ids=("vp_center_360", "vp_missing_height"),
        configuration=visibility_configuration(
            missing_observer_height_policy=MissingMetadataPolicy.EXCLUDE,
            default_observer_height_m=None,
        ),
    )
    _, result = _build(state)

    assert result.report.requested_units == 2
    assert result.report.added_units == 1
    assert result.report.excluded_ids == ["vp_missing_height"]
    assert result.report.failed_units == 0
    assert result.sof.n_active_units == 1


# ---------------------------------------------------------------------------
# Asynchronous build through the page and the shared TaskController
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# Pages stay referenced for the session so that queued matplotlib redraws
# never reach a canvas whose C++ object Python has already deleted.
_PAGES = []


def _page(state):
    from rivelero.gui.observability_page import ObservabilityPage

    page = ObservabilityPage(state)
    _PAGES.append(page)
    return page


def test_page_builds_asynchronously_with_real_progress(app, tmp_path):
    state = make_ready_state(tmp_path)
    page = _page(state)
    progress = []
    page.task_controller.task_progress.connect(
        lambda _id, processed, total, unit, _msg: progress.append((processed, total, unit))
    )

    assert page.build_button.isEnabled()
    page.build_button.click()

    # Control returns immediately; the build runs on the worker pool.
    assert state.task.busy
    assert state.task.task_name == OBSERVABILITY_BUILD_TASK_NAME
    assert state.observability_building
    assert not page.build_button.isEnabled()
    assert page.cancel_button.isVisible() or page.cancel_button.isVisibleTo(page)

    wait_for(lambda: not state.task.busy and page._build_task_id is None)

    assert state.observability_ready
    assert state.task.status == TaskStatus.SUCCEEDED
    assert [entry[0] for entry in progress] == [1, 2, 3]
    assert {entry[1] for entry in progress} == {3}
    assert progress[-1][2] == "vp_overlap_east"
    assert page.report_values["requested"].value == "3"
    assert page.report_values["computed"].value == "3"
    assert page.report_values["cached"].value == "0"
    assert page.summary_values["active"].value == "3"
    assert page.continue_button.isEnabled()


def test_page_rebuild_reports_cache_hits(app, tmp_path):
    state = make_ready_state(tmp_path)
    page = _page(state)
    page.build_button.click()
    wait_for(lambda: page._build_task_id is None)

    state.set_visibility_configuration(visibility_configuration(max_distance_m=90.0))
    state.set_visibility_configuration(visibility_configuration())
    page.refresh_from_state()
    page.build_button.click()
    wait_for(lambda: page._build_task_id is None)

    assert page.report_values["cached"].value == "3"
    assert page.report_values["computed"].value == "0"


def test_failed_build_keeps_previous_result(app, tmp_path, monkeypatch):
    state = make_ready_state(tmp_path)
    page = _page(state)
    page.build_button.click()
    wait_for(lambda: page._build_task_id is None)
    previous = state.analysis.survey_observability_field
    assert previous is not None

    def failing_build(**_kwargs):
        raise RuntimeError("GDAL exploded")

    monkeypatch.setattr(
        "rivelero.gui.observability_page.build_survey_observability_field",
        failing_build,
    )
    page.build_button.click()
    wait_for(lambda: page._build_task_id is None)

    assert state.task.status == TaskStatus.FAILED
    assert state.analysis.survey_observability_field is previous
    assert "GDAL exploded" in page.build_status_label.text()
    assert "RuntimeError" in page.error_details.toPlainText()
    assert page.build_button.isEnabled()


def test_cancel_discards_result_and_reenables_controls(app, tmp_path, monkeypatch):
    state = make_ready_state(tmp_path)
    page = _page(state)

    def slow_build(*, progress_callback, **_kwargs):
        for index in range(1, 201):
            time.sleep(0.01)
            progress_callback(index, 200, f"unit-{index}")
        raise AssertionError("cancellation was not honoured")

    monkeypatch.setattr(
        "rivelero.gui.observability_page.build_survey_observability_field",
        slow_build,
    )
    page.build_button.click()
    wait_for(lambda: state.task.processed >= 2)
    page.cancel_button.click()
    assert state.task.status == TaskStatus.CANCELLING

    wait_for(lambda: page._build_task_id is None)

    assert state.task.status == TaskStatus.CANCELLED
    assert not state.observability_ready
    assert "cancelled" in page.build_status_label.text().lower()
    assert page.build_button.isEnabled()


def test_inputs_edited_during_build_discard_result(app, tmp_path, monkeypatch):
    state = make_ready_state(tmp_path)
    page = _page(state)
    release = threading.Event()

    def gated_build(**kwargs):
        release.wait(10)
        return build_survey_observability_field(**kwargs)

    monkeypatch.setattr(
        "rivelero.gui.observability_page.build_survey_observability_field",
        gated_build,
    )
    page.build_button.click()

    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    state.replace_viewpoint(replace(viewpoint, observer_height_m=4.0))
    release.set()

    wait_for(lambda: page._build_task_id is None)

    assert not state.observability_ready
    assert "changed while" in page.build_status_label.text()
