"""A3 tests: what-if survey design scenarios (Qt-free)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from rivelero.analysis.contribution import analyse_contributions
from rivelero.analysis.coverage import summarize_coverage
from rivelero.analysis.scenario import (
    CandidateStatus,
    ScenarioChangeClass,
    SurveyDesignScenario,
    suggest_candidate_id,
)
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.exposure import add_visibility_to_exposure
from rivelero.observability.storage import StoredVisibility, VisibilityKey, VisibilityStore
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.visibility.configuration import SamplingUnit


TRANSFORM = Affine(10, 0, 500000, 0, -10, 4100000)
CRS_UTM = CRS.from_epsg(32633)
SHAPE = (3, 5)


def _key(name):
    return VisibilityKey(
        sampling_unit_id=name, sampling_unit_type="viewpoint",
        viewpoint_id=name, environment_id="env", analysis_domain_id="domain",
        visibility_configuration_id="config", input_fingerprint=f"fp-{name}",
    )


def _cells(*cells):
    mask = np.zeros(SHAPE, dtype=bool)
    for row, column in cells:
        mask[row, column] = True
    return mask


def _baseline():
    """Row 0 outside the domain; (1, 0) invalid; 9 analysable cells.

    Exposure (analysable cells):
        (1,1) A        -> 1
        (1,2) A, B     -> 2   (shared: the critical cell)
        (1,3) B        -> 1
        (2,1) A, B, C  -> 3
        (2,2) C        -> 1
        (1,4),(2,0),(2,3),(2,4) -> 0 (blind)
    Masks also mark an outside and an invalid cell, which must be ignored.
    """
    analysis = np.ones(SHAPE, dtype=bool)
    analysis[0, :] = False
    valid = np.ones(SHAPE, dtype=bool)
    valid[1, 0] = False
    masks = {
        "A": _cells((1, 1), (1, 2), (2, 1), (0, 0), (1, 0)),
        "B": _cells((1, 2), (1, 3), (2, 1)),
        "C": _cells((2, 1), (2, 2), (0, 3)),
    }
    analysable = analysis & valid
    exposure = np.zeros(SHAPE, dtype=np.uint16)
    for mask in masks.values():
        add_visibility_to_exposure(exposure, mask, analysable_mask=analysable, inplace=True)
    sof = SurveyObservabilityField(
        sof_id="baseline", viewpoint_configuration_id="survey",
        environment_id="env", analysis_domain_id="domain",
        visibility_configuration_id="config", sampling_unit=SamplingUnit.VIEWPOINT,
        exposure_count=exposure, analysis_mask=analysis, valid_mask=valid,
        transform=TRANSFORM, crs=CRS_UTM, active_keys=[_key(n) for n in masks],
    )
    return sof, masks


def _candidate(name="candidate_001"):
    return Viewpoint(viewpoint_id=name, x=500025.0, y=4099975.0, crs="EPSG:32633")


def _assert_invariants(scenario):
    s = scenario.summary().scenario
    assert s.observable_cells + s.blind_cells == s.analysable_cells
    assert s.unique_cells + s.repeated_cells == s.observable_cells
    assert scenario.exposure.min() >= 0
    # Outside/invalid cells never carry exposure.
    assert not np.any(scenario.exposure[~scenario.analysable_mask])


# ---------------------------------------------------------------------------
# Baseline safety
# ---------------------------------------------------------------------------


def test_empty_scenario_equals_baseline():
    sof, _ = _baseline()
    scenario = SurveyDesignScenario(sof)
    summary = scenario.summary()

    assert summary.is_baseline
    assert np.array_equal(scenario.exposure, sof.exposure_count)
    assert summary.scenario == summary.baseline
    difference = summary.difference
    assert difference.observable_cells == 0
    assert difference.coverage_percentage_points == 0.0
    _assert_invariants(scenario)


def test_baseline_exposure_is_never_mutated():
    sof, masks = _baseline()
    original = sof.exposure_count.copy()
    original_bytes = sof.exposure_count.tobytes()
    scenario = SurveyDesignScenario(sof)

    scenario.deactivate({_key("A"): masks["A"], _key("B"): masks["B"]})
    scenario.add_candidate(_candidate())
    scenario.set_candidate_visibility(
        "candidate_001", key=_key("candidate_001"), mask=_cells((2, 3), (2, 4))
    )
    scenario.reactivate([_key("A")])
    scenario.remove_candidate("candidate_001")
    scenario.reset()

    assert sof.exposure_count.tobytes() == original_bytes
    assert np.array_equal(sof.exposure_count, original)
    with pytest.raises(ValueError):
        scenario.exposure[1, 1] = 99  # read-only view


# ---------------------------------------------------------------------------
# Removal arithmetic
# ---------------------------------------------------------------------------


def test_single_removal_matches_baseline_unique_contribution(tmp_path):
    sof, masks = _baseline()
    store = VisibilityStore(tmp_path)
    for name, mask in masks.items():
        store.put(StoredVisibility(key=_key(name), visibility_mask=mask,
                                   transform=TRANSFORM, crs=CRS_UTM, metadata={}))
    contributions = analyse_contributions(sof, store)

    for name, mask in masks.items():
        scenario = SurveyDesignScenario(sof)
        scenario.deactivate({_key(name): mask})
        difference = scenario.summary().difference
        assert difference.observable_cells == -contributions.get(name).unique_cells
        assert difference.blind_cells == contributions.get(name).unique_cells
        _assert_invariants(scenario)


def test_multiple_removals_create_blind_spot_neither_covers_uniquely():
    sof, masks = _baseline()
    critical = (1, 2)
    assert sof.exposure_count[critical] == 2
    scenario = SurveyDesignScenario(sof)

    # A alone: its only unique cell is (1,1); the shared cell stays observable.
    effect_a = scenario.marginal_loss(masks["A"])
    assert effect_a.coverage_cells == 1
    scenario.deactivate({_key("A"): masks["A"]})
    assert scenario.exposure[critical] == 1

    # B given A already removed: loses (1,3) and now also the shared cell.
    effect_b = scenario.marginal_loss(masks["B"])
    assert effect_b.coverage_cells == 2
    scenario.deactivate({_key("B"): masks["B"]})
    assert scenario.exposure[critical] == 0
    assert scenario.change_classes()[critical] == ScenarioChangeClass.LOST_COVERAGE

    # Baseline unique contributions (1 + 1) under-count the joint loss (3).
    assert scenario.summary().difference.observable_cells == -3
    _assert_invariants(scenario)


def test_reactivation_and_reset_restore_exact_baseline():
    sof, masks = _baseline()
    scenario = SurveyDesignScenario(sof)
    scenario.deactivate({_key("A"): masks["A"], _key("C"): masks["C"]})
    assert scenario.summary().deactivated_units == 2

    scenario.reactivate([_key("C")])
    partial = scenario.summary()
    assert partial.deactivated_units == 1
    assert scenario.is_active(_key("C")) and not scenario.is_active(_key("A"))

    for _ in range(3):  # toggling cannot drift
        scenario.deactivate({_key("C"): masks["C"]})
        scenario.reactivate([_key("C")])
    assert scenario.summary() == partial

    scenario.reset()
    assert scenario.summary().is_baseline
    assert scenario.exposure.tobytes() == sof.exposure_count.tobytes()


def test_deactivation_is_all_or_nothing_for_foreign_masks():
    sof, masks = _baseline()
    scenario = SurveyDesignScenario(sof)
    foreign = _cells((2, 4))  # a blind cell: not part of any unit
    with pytest.raises(ValueError, match="does not belong"):
        scenario.deactivate({_key("A"): masks["A"], _key("B"): foreign})
    assert scenario.summary().is_baseline
    with pytest.raises(KeyError):
        scenario.deactivate({_key("unknown"): masks["A"]})


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _with_candidate(scenario, mask, name="candidate_001", include=True):
    scenario.add_candidate(_candidate(name))
    scenario.set_candidate_visibility(name, key=_key(name), mask=mask, include=include)


def test_candidate_marginal_gain_by_exposure_level():
    sof, _ = _baseline()
    scenario = SurveyDesignScenario(sof)
    blind, unique, repeated = (2, 3), (1, 1), (1, 2)

    effect = scenario.marginal_gain(_cells(blind, unique, repeated, (0, 1), (1, 0)))
    assert effect.visible_cells == 3          # outside/invalid excluded
    assert effect.coverage_cells == 1         # blind -> observable
    assert effect.unique_to_repeated == 1     # 1 -> 2
    assert effect.other_exposure_cells == 1   # 2 -> 3, still repeated

    _with_candidate(scenario, _cells(blind, unique, repeated))
    difference = scenario.summary().difference
    assert difference.observable_cells == 1
    assert difference.unique_cells == 1 - 1   # +1 new unique, -1 now repeated
    assert difference.repeated_cells == 1
    assert scenario.change_classes()[blind] == ScenarioChangeClass.GAINED_COVERAGE
    _assert_invariants(scenario)


def test_candidate_restores_coverage_lost_by_removal():
    sof, masks = _baseline()
    scenario = SurveyDesignScenario(sof)
    scenario.deactivate({_key("B"): masks["B"]})
    assert scenario.exposure[1, 3] == 0  # B's unique cell is now blind

    restoring = _cells((1, 3))
    assert scenario.marginal_gain(restoring).coverage_cells == 1  # vs scenario
    _with_candidate(scenario, restoring)

    classes = scenario.change_classes()
    assert classes[1, 3] == ScenarioChangeClass.REMAINS_OBSERVABLE
    assert scenario.summary().difference.observable_cells == 0


def test_candidate_gain_is_relative_to_other_modifications():
    sof, _ = _baseline()
    scenario = SurveyDesignScenario(sof)
    mask = _cells((2, 3), (2, 4))
    _with_candidate(scenario, mask, "candidate_001")
    _with_candidate(scenario, mask, "candidate_002")

    # Each identical candidate adds nothing given the other.
    assert scenario.candidate_marginal_gain("candidate_001").coverage_cells == 0
    scenario.set_candidate_included("candidate_002", False)
    assert scenario.candidate_marginal_gain("candidate_001").coverage_cells == 2
    assert scenario.candidate_marginal_gain("candidate_002").coverage_cells == 0


def test_candidate_include_exclude_remove_and_zero_mask():
    sof, _ = _baseline()
    scenario = SurveyDesignScenario(sof)
    _with_candidate(scenario, _cells((2, 3)))
    assert scenario.summary().included_candidates == 1
    scenario.set_candidate_included("candidate_001", False)
    assert scenario.summary().is_baseline

    # A zero-visibility candidate is a valid READY result.
    _with_candidate(scenario, np.zeros(SHAPE, dtype=bool), "candidate_002")
    candidate = scenario.candidate("candidate_002")
    assert candidate.status == CandidateStatus.READY and candidate.included
    assert scenario.summary().difference.observable_cells == 0

    scenario.remove_candidate("candidate_002")
    scenario.remove_candidate("candidate_001")
    assert scenario.candidates == ()
    assert scenario.exposure.tobytes() == sof.exposure_count.tobytes()


def test_failed_or_excluded_candidate_never_changes_exposure():
    sof, _ = _baseline()
    scenario = SurveyDesignScenario(sof)
    scenario.add_candidate(_candidate())
    for status in (CandidateStatus.FAILED, CandidateStatus.EXCLUDED):
        scenario.set_candidate_status("candidate_001", status, "reason")
        with pytest.raises(ValueError):
            scenario.set_candidate_included("candidate_001", True)
        assert scenario.exposure.tobytes() == sof.exposure_count.tobytes()


def test_candidate_ids_must_be_unique():
    sof, _ = _baseline()
    scenario = SurveyDesignScenario(sof)
    scenario.add_candidate(_candidate(), reserved_ids={"A", "B"})
    with pytest.raises(ValueError, match="already used"):
        scenario.add_candidate(_candidate())
    with pytest.raises(ValueError, match="already used"):
        scenario.add_candidate(_candidate("A"), reserved_ids={"A"})
    assert suggest_candidate_id({"candidate_001", "candidate_002"}) == "candidate_003"


def test_overflow_is_guarded_and_leaves_scenario_unchanged():
    sof, _ = _baseline()
    # 255 units all seeing (1, 1): the uint8 exposure is saturated there.
    exposure = np.zeros(SHAPE, dtype=np.uint8)
    exposure[1, 1] = 255
    saturated = SurveyObservabilityField(
        sof_id="saturated", viewpoint_configuration_id="survey",
        environment_id="env", analysis_domain_id="domain",
        visibility_configuration_id="config", sampling_unit=SamplingUnit.VIEWPOINT,
        exposure_count=exposure, analysis_mask=sof.analysis_mask,
        valid_mask=sof.valid_mask, transform=TRANSFORM, crs=CRS_UTM,
        active_keys=[_key(f"u{index}") for index in range(255)],
    )
    scenario = SurveyDesignScenario(saturated)
    scenario.add_candidate(_candidate())
    with pytest.raises(OverflowError):
        scenario.set_candidate_visibility(
            "candidate_001", key=_key("candidate_001"), mask=_cells((1, 1))
        )
    assert not scenario.candidate("candidate_001").included
    assert scenario.exposure.tobytes() == exposure.tobytes()


def test_removal_underflow_is_guarded():
    sof, masks = _baseline()
    scenario = SurveyDesignScenario(sof)
    scenario.deactivate({_key("A"): masks["A"]})
    # A is no longer part of the scenario: its loss cannot be evaluated.
    with pytest.raises(ValueError, match="not part of the current scenario"):
        scenario.marginal_loss(masks["A"])


def test_scenario_summary_reuses_a1_semantics():
    sof, masks = _baseline()
    scenario = SurveyDesignScenario(sof)
    baseline = scenario.summary().baseline
    assert baseline == summarize_coverage(sof)
    assert baseline.analysable_cells == 9

    scenario.deactivate({_key("C"): masks["C"]})
    summary = scenario.summary()
    assert summary.scenario.analysable_cells == 9  # denominator unchanged
    assert summary.active_existing_units == 2
    assert summary.difference.coverage_percentage_points == pytest.approx(
        100.0 * (summary.scenario.observable_fraction - baseline.observable_fraction)
    )
