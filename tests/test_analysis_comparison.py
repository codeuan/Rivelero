"""A4 tests: snapshots, comparison and optimisation primitives (Qt-free)."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from rivelero.analysis.comparison import (
    BASELINE_ID,
    LIVE_SCENARIO_ID,
    IncompatibleBaselineError,
    ScenarioWorkspace,
    SnapshotUnavailableError,
    baseline_state,
    compare_states,
    live_state,
    reconstruct_snapshot_exposure,
    restore_snapshot,
)
from rivelero.analysis.objectives import (
    ObjectiveComponent,
    marginal_gain,
    marginal_loss,
    objective_components,
)
from rivelero.analysis.scenario import (
    CandidateStatus,
    ScenarioChangeClass,
    SurveyDesignScenario,
)
from rivelero.observability.storage import StoredVisibility, VisibilityStore
from rivelero.observability.survey_field import SurveyObservabilityField
from test_analysis_scenario import (
    CRS_UTM,
    SHAPE,
    TRANSFORM,
    _baseline,
    _candidate,
    _cells,
    _key,
)

CANDIDATE_MASK = _cells((2, 3), (2, 4), (1, 1))


def _setup(tmp_path):
    sof, masks = _baseline()
    store = VisibilityStore(tmp_path / "store")
    for name, mask in masks.items():
        store.put(StoredVisibility(key=_key(name), visibility_mask=mask,
                                   transform=TRANSFORM, crs=CRS_UTM, metadata={}))
    store.put(StoredVisibility(key=_key("candidate_001"), visibility_mask=CANDIDATE_MASK,
                               transform=TRANSFORM, crs=CRS_UTM, metadata={}))
    return sof, masks, store


def _scenario_a(sof, masks):
    scenario = SurveyDesignScenario(sof)
    scenario.deactivate({_key("A"): masks["A"]})
    return scenario


def _scenario_b(sof, masks):
    scenario = SurveyDesignScenario(sof)
    scenario.deactivate({_key("C"): masks["C"]})
    scenario.add_candidate(_candidate())
    scenario.set_candidate_visibility(
        "candidate_001", key=_key("candidate_001"), mask=CANDIDATE_MASK
    )
    return scenario


def _workspace(tmp_path):
    sof, masks, store = _setup(tmp_path)
    workspace = ScenarioWorkspace()
    a = workspace.save(_scenario_a(sof, masks), name="A")
    b = workspace.save(_scenario_b(sof, masks), name="B", description="Swap C for a candidate")
    return sof, masks, store, workspace, a, b


def _states(workspace, sof, store):
    return {
        state_id: workspace.state(state_id, sof=sof, store=store)
        for state_id in [BASELINE_ID] + [s.snapshot_id for s in workspace.snapshots]
    }


NUMERIC_DELTAS = (
    "observable_cells_delta", "blind_cells_delta", "unique_cells_delta",
    "repeated_cells_delta", "coverage_percentage_point_delta",
    "mean_exposure_delta", "maximum_exposure_delta",
    "active_existing_units_delta", "included_candidates_delta",
    "sampling_unit_count_delta",
)


# ---------------------------------------------------------------------------
# Comparison invariants
# ---------------------------------------------------------------------------


def test_self_comparison_is_zero(tmp_path):
    sof, _masks, store, workspace, _a, _b = _workspace(tmp_path)
    for state in _states(workspace, sof, store).values():
        comparison = compare_states(state, state, sof)
        for name in NUMERIC_DELTAS:
            assert getattr(comparison, name) == 0, name
        assert not comparison.gained_coverage_mask.any()
        assert not comparison.lost_coverage_mask.any()
        assert not np.any(comparison.exposure_difference())
        classes = comparison.change_classes()
        assert not np.isin(classes, [ScenarioChangeClass.GAINED_COVERAGE,
                                     ScenarioChangeClass.LOST_COVERAGE]).any()


def test_swapping_reverses_signs_and_masks(tmp_path):
    sof, _masks, store, workspace, a, b = _workspace(tmp_path)
    states = _states(workspace, sof, store)
    pairs = [(BASELINE_ID, a.snapshot_id), (BASELINE_ID, b.snapshot_id),
             (a.snapshot_id, b.snapshot_id)]
    for left_id, right_id in pairs:
        forward = compare_states(states[left_id], states[right_id], sof)
        backward = compare_states(states[right_id], states[left_id], sof)
        for name in NUMERIC_DELTAS:
            assert getattr(forward, name) == pytest.approx(-getattr(backward, name)), name
        assert np.array_equal(forward.gained_coverage_mask, backward.lost_coverage_mask)
        assert np.array_equal(forward.lost_coverage_mask, backward.gained_coverage_mask)
        assert np.array_equal(
            forward.exposure_difference().filled(0), -backward.exposure_difference().filled(0)
        )


def test_coverage_difference_invariants(tmp_path):
    sof, _masks, store, workspace, a, b = _workspace(tmp_path)
    states = _states(workspace, sof, store)
    comparison = compare_states(states[a.snapshot_id], states[b.snapshot_id], sof)

    gained = int(comparison.gained_coverage_mask.sum())
    lost = int(comparison.lost_coverage_mask.sum())
    assert comparison.observable_cells_delta == gained - lost
    assert comparison.blind_cells_delta == -comparison.observable_cells_delta
    classes = comparison.change_classes()
    assert int((classes == ScenarioChangeClass.GAINED_COVERAGE).sum()) == gained
    assert int((classes == ScenarioChangeClass.LOST_COVERAGE).sum()) == lost


def test_exposure_difference_is_right_minus_left(tmp_path):
    sof, _masks, store, workspace, a, b = _workspace(tmp_path)
    states = _states(workspace, sof, store)
    left, right = states[a.snapshot_id], states[b.snapshot_id]
    difference = compare_states(left, right, sof).exposure_difference()
    analysable = sof.analysable_mask

    expected = right.exposure.astype(int) - left.exposure.astype(int)
    assert np.array_equal(difference[analysable], expected[analysable])
    assert np.array_equal(np.ma.getmaskarray(difference), ~analysable)
    assert difference.min() < 0 < difference.max()  # a genuinely signed field


def test_hand_checked_baseline_to_b():
    sof, masks = _baseline()
    scenario = _scenario_b(sof, masks)
    comparison = compare_states(baseline_state(sof), live_state(scenario), sof)

    # Removing C loses (2,2); the candidate gains (2,3) and (2,4).
    assert comparison.observable_cells_delta == 1
    assert comparison.active_existing_units_delta == -1
    assert comparison.included_candidates_delta == 1
    assert comparison.sampling_unit_count_delta == 0
    assert np.argwhere(comparison.lost_coverage_mask).tolist() == [[2, 2]]
    assert np.argwhere(comparison.gained_coverage_mask).tolist() == [[2, 3], [2, 4]]


def test_incompatible_baselines_are_refused(tmp_path):
    sof, masks, store, workspace, a, _b = _workspace(tmp_path)
    other = SurveyObservabilityField(
        sof_id="other", viewpoint_configuration_id="survey", environment_id="env",
        analysis_domain_id="domain", visibility_configuration_id="config",
        sampling_unit=sof.sampling_unit, exposure_count=sof.exposure_count.copy(),
        analysis_mask=sof.analysis_mask, valid_mask=sof.valid_mask,
        transform=TRANSFORM, crs=CRS_UTM, active_keys=list(sof.active_keys),
    )
    state_a = workspace.state(a.snapshot_id, sof=sof, store=store)
    with pytest.raises(IncompatibleBaselineError):
        compare_states(baseline_state(other), state_a, other)
    with pytest.raises(SnapshotUnavailableError, match="earlier observability field"):
        workspace.state(a.snapshot_id, sof=other, store=store)
    assert not a.compatible_with(other) and a.compatible_with(sof)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_snapshot_is_independent_of_live_scenario(tmp_path):
    sof, masks, _store = _setup(tmp_path)
    live = _scenario_b(sof, masks)
    workspace = ScenarioWorkspace()
    snapshot = workspace.save(live, name="B")
    frozen_summary = snapshot.summary

    live.reactivate([_key("C")])
    live.set_candidate_included("candidate_001", False)
    live.remove_candidate("candidate_001")
    live.reset()

    assert workspace.get(snapshot.snapshot_id) is snapshot
    assert snapshot.summary == frozen_summary
    assert snapshot.deactivated_keys == (_key("C"),)
    assert [c.viewpoint.viewpoint_id for c in snapshot.candidates] == ["candidate_001"]
    assert snapshot.candidates[0].included and snapshot.candidates[0].key == _key("candidate_001")
    with pytest.raises(Exception):
        snapshot.name = "changed"  # frozen


def test_snapshot_holds_no_masks_or_rasters(tmp_path):
    _sof, _masks, _store, _workspace_, a, b = _workspace(tmp_path)
    for snapshot in (a, b):
        for field in fields(snapshot):
            assert not isinstance(getattr(snapshot, field.name), np.ndarray)
        for candidate in snapshot.candidates:
            assert not any(isinstance(getattr(candidate, f.name), np.ndarray) for f in fields(candidate))


def test_library_identity_names_and_deletion(tmp_path):
    sof, masks, store, workspace, a, b = _workspace(tmp_path)
    assert a.snapshot_id != b.snapshot_id
    assert workspace.suggest_name() == "Scenario 3"
    with pytest.raises(ValueError, match="already called"):
        workspace.save(_scenario_a(sof, masks), name="A")

    renamed = workspace.rename(a.snapshot_id, "Without A")
    assert renamed.name == "Without A" and renamed.snapshot_id == a.snapshot_id
    described = workspace.describe(a.snapshot_id, "  first idea ")
    assert described.description == "first idea"
    with pytest.raises(ValueError):
        workspace.rename(a.snapshot_id, "B")

    workspace.right_id = b.snapshot_id
    disk = store.disk_item_count
    workspace.delete(b.snapshot_id)
    assert [s.name for s in workspace.snapshots] == ["Without A"]
    assert workspace.right_id is None
    assert store.disk_item_count == disk  # cached visibility untouched


def test_reconstruction_matches_saved_live_exposure(tmp_path):
    sof, masks, store = _setup(tmp_path)
    live = _scenario_b(sof, masks)
    workspace = ScenarioWorkspace(exposure_cache_size=0)  # force reconstruction
    snapshot = workspace.save(live, name="B")
    before = sof.exposure_count.tobytes()

    state = workspace.state(snapshot.snapshot_id, sof=sof, store=store)
    assert np.array_equal(state.exposure, live.exposure)
    assert state.summary == snapshot.summary.scenario
    assert sof.exposure_count.tobytes() == before


def test_exposure_cache_is_bounded(tmp_path):
    sof, masks, store = _setup(tmp_path)
    workspace = ScenarioWorkspace(exposure_cache_size=2)
    for index in range(4):
        workspace.save(_scenario_a(sof, masks), name=f"S{index}")
    assert workspace.cached_exposure_count == 2
    first = workspace.snapshots[0]
    state = workspace.state(first.snapshot_id, sof=sof, store=store)  # rebuilt
    assert state.summary == first.summary.scenario
    assert workspace.cached_exposure_count == 2


def test_missing_mask_makes_snapshot_unavailable(tmp_path):
    sof, masks, store = _setup(tmp_path)
    workspace = ScenarioWorkspace(exposure_cache_size=0)
    snapshot = workspace.save(_scenario_b(sof, masks), name="B")
    store.remove(_key("candidate_001"))

    with pytest.raises(SnapshotUnavailableError, match="candidate_001"):
        workspace.state(snapshot.snapshot_id, sof=sof, store=store)
    with pytest.raises(SnapshotUnavailableError):
        reconstruct_snapshot_exposure(snapshot, sof, store)
    # The snapshot record itself is unchanged.
    assert workspace.get(snapshot.snapshot_id).summary.included_candidates == 1


def test_live_state_requires_current_scenario(tmp_path):
    sof, masks, store = _setup(tmp_path)
    workspace = ScenarioWorkspace()
    with pytest.raises(SnapshotUnavailableError):
        workspace.state(LIVE_SCENARIO_ID, sof=sof, store=store, live=None)
    live = _scenario_a(sof, masks)
    assert workspace.state(LIVE_SCENARIO_ID, sof=sof, store=store, live=live).active_existing_units == 2


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def test_restore_reproduces_membership_and_candidates(tmp_path):
    sof, masks, store = _setup(tmp_path)
    original = _scenario_b(sof, masks)
    original.set_candidate_included("candidate_001", False)
    snapshot = ScenarioWorkspace().save(original, name="B")
    before = sof.exposure_count.tobytes()

    restored, report = restore_snapshot(snapshot, sof, store)
    assert restored is not original
    assert restored.deactivated_keys == (_key("C"),)
    candidate = restored.candidate("candidate_001")
    assert candidate.status == CandidateStatus.READY and not candidate.included
    assert restored.summary() == snapshot.summary
    assert np.array_equal(restored.exposure, original.exposure)
    assert report.candidates_needing_computation == ()
    assert sof.exposure_count.tobytes() == before


def test_restore_with_missing_candidate_mask_requires_recomputation(tmp_path):
    sof, masks, store = _setup(tmp_path)
    snapshot = ScenarioWorkspace().save(_scenario_b(sof, masks), name="B")
    store.remove(_key("candidate_001"))

    restored, report = restore_snapshot(snapshot, sof, store)
    candidate = restored.candidate("candidate_001")
    assert report.candidates_needing_computation == ("candidate_001",)
    assert candidate.status == CandidateStatus.PENDING and candidate.mask is None
    assert candidate.include_when_ready is True
    # Not treated as a zero mask: the candidate simply is not in the exposure.
    assert restored.summary().included_candidates == 0


def test_restore_refuses_missing_deactivated_mask(tmp_path):
    sof, masks, store = _setup(tmp_path)
    snapshot = ScenarioWorkspace().save(_scenario_a(sof, masks), name="A")
    store.remove(_key("A"))
    with pytest.raises(SnapshotUnavailableError, match="deactivated units"):
        restore_snapshot(snapshot, sof, store)


# ---------------------------------------------------------------------------
# Optimisation foundation
# ---------------------------------------------------------------------------


def test_marginal_primitives_are_pure_and_match_scenario_methods():
    sof, masks = _baseline()
    scenario = _scenario_a(sof, masks)
    exposure = np.array(scenario.exposure)
    before = exposure.tobytes()

    for mask in (masks["B"], masks["C"]):
        assert marginal_loss(exposure, mask, sof.analysable_mask) == scenario.marginal_loss(mask)
    assert marginal_gain(exposure, CANDIDATE_MASK, sof.analysable_mask) == (
        scenario.marginal_gain(CANDIDATE_MASK)
    )
    assert exposure.tobytes() == before
    with pytest.raises(ValueError):
        marginal_loss(exposure, masks["A"], sof.analysable_mask)  # not in scenario


def test_objective_components_are_named_measurements_only():
    sof, masks = _baseline()
    state = live_state(_scenario_b(sof, masks))
    components = objective_components(state)

    assert set(components) == set(ObjectiveComponent)
    assert components[ObjectiveComponent.OBSERVABLE_CELLS] == state.summary.observable_cells
    assert components[ObjectiveComponent.SAMPLING_UNITS] == 3
    assert components[ObjectiveComponent.INCLUDED_CANDIDATES] == 1
    assert components[ObjectiveComponent.COVERAGE_SHARE] == pytest.approx(
        state.summary.observable_fraction
    )
