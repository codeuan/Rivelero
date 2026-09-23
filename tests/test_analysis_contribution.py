"""A2 tests: sampling-unit contribution analysis (Qt-free)."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from observability_fixtures import make_ready_state, visibility_configuration
from rivelero.analysis.contribution import (
    ContributionStatus,
    UnitContributionClass,
    analyse_contributions,
    unit_contribution_classes,
)
from rivelero.analysis.coverage import summarize_coverage
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.observability.builder import build_survey_observability_field
from rivelero.observability.exposure import (
    add_visibility_to_exposure,
    remove_visibility_from_exposure,
)
from rivelero.observability.storage import (
    StoredVisibility,
    VisibilityKey,
    VisibilityStore,
)
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.visibility.configuration import SamplingUnit


TRANSFORM = Affine(10, 0, 500000, 0, -10, 4100000)
CRS_UTM = CRS.from_epsg(32633)


def _key(unit_id: str, *, unit_type: str = "viewpoint", viewpoint_id=None):
    return VisibilityKey(
        sampling_unit_id=unit_id,
        sampling_unit_type=unit_type,
        viewpoint_id=viewpoint_id or unit_id,
        environment_id="env",
        analysis_domain_id="domain",
        visibility_configuration_id="config",
        input_fingerprint=f"fp-{unit_id}",
    )


def _masks():
    """4 x 6 grid. Row 0 outside the domain; cells (1,0),(1,1) invalid.

    Units (masks deliberately include outside/invalid cells, which must be
    ignored):

      a   : exclusive block + shared cells          -> unique and repeated
      b   : only cells shared with a or c           -> repeated only
      c   : exclusive cells + shared with b          -> unique and repeated
      solo: cells nobody else sees                  -> unique only
      none: sees no analysable cell                  -> zero visible
    """
    shape = (4, 6)
    analysis = np.ones(shape, dtype=bool)
    analysis[0, :] = False
    valid = np.ones(shape, dtype=bool)
    valid[1, :2] = False

    masks = {name: np.zeros(shape, dtype=bool) for name in ("a", "b", "c", "solo", "none")}
    masks["a"][1:3, 0:3] = True       # includes invalid (1,0),(1,1)
    masks["a"][0, :] = True           # outside domain, ignored
    masks["b"][2, 1:5] = True
    masks["c"][2:4, 3:5] = True
    masks["solo"][3, 0:2] = True
    masks["solo"][1, 5] = True
    masks["none"][0, 2] = True        # outside only
    masks["none"][1, 0] = True        # invalid only
    return analysis, valid, masks


def _field(tmp_path, *, store_masks=True, max_memory_items=32):
    analysis, valid, masks = _masks()
    analysable = analysis & valid
    exposure = np.zeros(analysis.shape, dtype=np.uint32)
    keys = []
    for name, mask in masks.items():
        add_visibility_to_exposure(
            exposure, mask, analysable_mask=analysable, inplace=True
        )
        keys.append(_key(name))

    sof = SurveyObservabilityField(
        sof_id="sof-test",
        viewpoint_configuration_id="survey",
        environment_id="env",
        analysis_domain_id="domain",
        visibility_configuration_id="config",
        sampling_unit=SamplingUnit.VIEWPOINT,
        exposure_count=exposure,
        analysis_mask=analysis,
        valid_mask=valid,
        transform=TRANSFORM,
        crs=CRS_UTM,
        active_keys=keys,
    )
    store = VisibilityStore(tmp_path / "store", max_memory_items=max_memory_items)
    if store_masks:
        for key, mask in zip(keys, masks.values()):
            # Stored masks are already restricted to analysable space by the
            # engine; storing the raw ones proves the analysis restricts too.
            store.put(StoredVisibility(
                key=key, visibility_mask=mask, transform=TRANSFORM,
                crs=CRS_UTM, metadata={},
            ))
    return sof, store, masks


def _expected(sof, mask):
    analysable = sof.analysable_mask
    visible = mask & analysable
    exposure = sof.exposure_count
    return (
        int(np.count_nonzero(visible)),
        int(np.count_nonzero(visible & (exposure == 1))),
        int(np.count_nonzero(visible & (exposure >= 2))),
    )


# ---------------------------------------------------------------------------
# Scientific definitions
# ---------------------------------------------------------------------------


def test_per_unit_counts_match_definitions(tmp_path):
    sof, store, masks = _field(tmp_path)
    result = analyse_contributions(sof, store)

    assert result.n_units == 5
    assert result.complete
    for unit in result.units:
        visible, unique, repeated = _expected(sof, masks[unit.sampling_unit_id])
        assert unit.status == ContributionStatus.AVAILABLE
        assert (unit.visible_cells, unit.unique_cells, unit.repeated_cells) == (
            visible, unique, repeated,
        )
        assert unit.visible_cells == unit.unique_cells + unit.repeated_cells


def test_hand_checked_values_and_denominators(tmp_path):
    sof, store, _ = _field(tmp_path)
    result = analyse_contributions(sof, store)
    analysable = sof.n_analysable_cells
    assert analysable == 16 == result.analysable_cells

    a = result.get("a")
    # a sees (1,2) and row 2 cols 0-2 -> 4 analysable cells; (2,1),(2,2)
    # are shared with b.
    assert (a.visible_cells, a.unique_cells, a.repeated_cells) == (4, 2, 2)
    assert a.coverage_loss_if_removed == pytest.approx(2 / analysable)
    assert a.unique_share_of_analysable == pytest.approx(2 / analysable)
    assert a.repeated_share_of_analysable == pytest.approx(2 / analysable)
    assert a.unique_share_of_unit_visibility == pytest.approx(0.5)
    assert a.repeated_share_of_unit_visibility == pytest.approx(0.5)

    b = result.get("b")
    assert (b.visible_cells, b.unique_cells, b.repeated_cells) == (4, 0, 4)
    assert b.has_unique_coverage is False
    assert b.repeated_share_of_unit_visibility == 1.0

    solo = result.get("solo")
    assert (solo.visible_cells, solo.unique_cells, solo.repeated_cells) == (3, 3, 0)
    assert solo.unique_share_of_unit_visibility == 1.0


def test_outside_and_invalid_cells_never_counted(tmp_path):
    sof, store, masks = _field(tmp_path)
    result = analyse_contributions(sof, store)

    none = result.get("none")
    assert masks["none"].any()  # visible only outside / on invalid cells
    assert none.status == ContributionStatus.AVAILABLE
    assert (none.visible_cells, none.unique_cells, none.repeated_cells) == (0, 0, 0)
    assert none.unique_share_of_unit_visibility is None
    assert none.coverage_loss_if_removed == 0.0

    assert result.n_without_visible_cells == 1
    assert result.n_with_unique_coverage == 3  # a, c, solo
    assert result.n_without_unique_coverage == 1  # b (visible but not unique)


def test_blind_cells_are_not_part_of_any_unit(tmp_path):
    sof, store, masks = _field(tmp_path)
    blind = sof.blindspot_mask
    assert blind.any()
    for mask in masks.values():
        assert not np.any(mask & blind)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_unique_cells_sum_to_field_unique_coverage(tmp_path):
    sof, store, _ = _field(tmp_path)
    result = analyse_contributions(sof, store)
    summary = summarize_coverage(sof)

    assert result.total_unique_contribution_cells == summary.unique_cells
    assert result.field_unique_cells == summary.unique_cells
    assert result.total_visible_contribution_cells == summary.total_exposure
    # Repeated contributions double count shared cells.
    assert sum(unit.repeated_cells for unit in result.units) > summary.repeated_cells
    assert result.consistent_with_field is True


def test_removal_arithmetic_matches_unique_cells(tmp_path):
    sof, store, masks = _field(tmp_path)
    result = analyse_contributions(sof, store)
    analysable = sof.analysable_mask
    observable_before = int(np.count_nonzero(sof.observable_mask))

    for unit in result.units:
        e_minus = remove_visibility_from_exposure(
            sof.exposure_count, masks[unit.sampling_unit_id],
            analysable_mask=analysable,
        )
        newly_blind = analysable & (sof.exposure_count > 0) & (e_minus == 0)
        assert int(np.count_nonzero(newly_blind)) == unit.cells_lost_if_removed
        observable_after = int(np.count_nonzero(analysable & (e_minus > 0)))
        assert observable_after == observable_before - unit.unique_cells
    # The field itself is untouched.
    assert sof.n_observable_cells == observable_before


def test_removal_underflow_is_rejected():
    exposure = np.zeros((1, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="zero exposure"):
        remove_visibility_from_exposure(exposure, np.array([[True, False]]))


def test_addition_overflow_is_rejected():
    exposure = np.full((1, 2), 255, dtype=np.uint8)
    with pytest.raises(OverflowError):
        add_visibility_to_exposure(exposure, np.array([[True, False]]))


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class CountingStore(VisibilityStore):
    """Records reads and fails on any computation or write."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = []
        self.live_masks = []

    def get(self, key):
        self.reads.append(key)
        # With max_memory_items=0 nothing is retained by the LRU, so every
        # earlier mask must already have been released by the analysis.
        gc.collect()
        assert all(ref() is None for ref in self.live_masks), (
            "contribution analysis retained an earlier mask"
        )
        result = super().get(key)
        if result is not None:
            self.live_masks.append(weakref.ref(result.visibility_mask))
        return result

    def get_or_compute(self, key, compute):  # pragma: no cover - must not run
        raise AssertionError("contribution analysis must not compute visibility")


def test_masks_are_read_once_each_and_released_sequentially(tmp_path):
    sof, seed_store, _ = _field(tmp_path)
    store = CountingStore(seed_store.cache_directory, max_memory_items=0)
    disk_before = store.disk_item_count

    result = analyse_contributions(sof, store)

    assert result.complete
    assert store.reads == list(sof.active_keys)
    assert store.disk_item_count == disk_before  # nothing written


def test_missing_mask_is_reported_not_zero(tmp_path):
    sof, store, _ = _field(tmp_path)
    store.remove(_key("a"))

    result = analyse_contributions(sof, store)
    a = result.get("a")

    assert a.status == ContributionStatus.MISSING
    assert a.visible_cells is None and a.unique_cells is None
    assert a.coverage_loss_if_removed is None
    assert "No cached visibility" in a.message
    assert not result.complete
    assert result.consistent_with_field is None
    # A zero-visible unit is a different condition.
    none = result.get("none")
    assert none.status == ContributionStatus.AVAILABLE
    assert none.visible_cells == 0


def test_unreadable_mask_is_reported(tmp_path):
    sof, store, _ = _field(tmp_path)
    path = store.path_for(_key("b"))
    store.clear_memory()
    path.write_bytes(b"not a numpy archive")

    result = analyse_contributions(sof, store)
    assert result.get("b").status == ContributionStatus.UNREADABLE


def test_mask_not_belonging_to_field_is_inconsistent(tmp_path):
    sof, store, masks = _field(tmp_path)
    wrong = np.zeros_like(masks["b"])
    wrong[sof.blindspot_mask] = True  # visible where exposure is zero
    store.put(StoredVisibility(
        key=_key("b"), visibility_mask=wrong, transform=TRANSFORM,
        crs=CRS_UTM, metadata={},
    ))

    result = analyse_contributions(sof, store)
    b = result.get("b")
    assert b.status == ContributionStatus.INCONSISTENT
    assert "zero exposure" in b.message


def test_progress_reports_every_unit(tmp_path):
    sof, store, _ = _field(tmp_path)
    progress = []
    analyse_contributions(sof, store, progress_callback=lambda *a: progress.append(a))
    assert progress == [
        (index, 5, key.sampling_unit_id)
        for index, key in enumerate(sof.active_keys, start=1)
    ]


# ---------------------------------------------------------------------------
# Selected-unit map classes
# ---------------------------------------------------------------------------


def test_unit_contribution_classes(tmp_path):
    sof, _, masks = _field(tmp_path)
    classes = unit_contribution_classes(
        masks["a"],
        exposure_count=sof.exposure_count,
        analysis_mask=sof.analysis_mask,
        valid_mask=sof.valid_mask,
    )
    counts = np.bincount(classes.ravel(), minlength=len(UnitContributionClass))

    assert counts[UnitContributionClass.OUTSIDE_DOMAIN] == 6
    assert counts[UnitContributionClass.INVALID] == 2
    assert counts[UnitContributionClass.UNIQUE_CONTRIBUTION] == 2
    assert counts[UnitContributionClass.REPEATED_CONTRIBUTION] == 2
    assert counts[UnitContributionClass.NOT_VISIBLE_FROM_UNIT] == 16 - 4
    # Derived only: the field is untouched.
    assert sof.n_analysable_cells == 16


# ---------------------------------------------------------------------------
# Real synthetic survey
# ---------------------------------------------------------------------------


def _built_state(tmp_path, **kwargs):
    state = make_ready_state(tmp_path, **kwargs)
    request = prepare_build(state)
    result = build_survey_observability_field(**request.build_kwargs())
    install_build_result(state, request, result)
    return state


def test_real_viewpoint_survey_invariants(tmp_path):
    state = _built_state(
        tmp_path,
        viewpoint_ids=None,
        configuration=visibility_configuration(max_distance_m=250.0),
    )
    sof = state.analysis.survey_observability_field
    store = state.analysis.visibility_store

    result = analyse_contributions(sof, store)

    assert result.n_units == 10
    assert result.complete
    assert result.consistent_with_field is True
    assert result.total_unique_contribution_cells == summarize_coverage(sof).unique_cells
    for unit in result.units:
        mask = store.get(unit.key).visibility_mask
        e_minus = remove_visibility_from_exposure(
            sof.exposure_count, mask, analysable_mask=sof.analysable_mask
        )
        assert int(np.count_nonzero(sof.analysable_mask & (e_minus > 0))) == (
            sof.n_observable_cells - unit.unique_cells
        )


def test_observation_event_units_stay_distinct(tmp_path):
    state = _built_state(
        tmp_path,
        viewpoint_ids=("vp_redundant_a", "vp_overlap_east", "vp_complementary"),
        configuration=visibility_configuration(
            max_distance_m=300.0, sampling_unit=SamplingUnit.OBSERVATION_EVENT
        ),
    )
    sof = state.analysis.survey_observability_field
    result = analyse_contributions(sof, state.analysis.visibility_store)

    assert result.sampling_unit == SamplingUnit.OBSERVATION_EVENT
    assert [unit.sampling_unit_id for unit in result.units] == [
        "event_001", "event_002", "event_003", "event_004",
    ]
    assert [unit.observation_event_id for unit in result.units] == [
        "event_001", "event_002", "event_003", "event_004",
    ]
    assert [unit.viewpoint_id for unit in result.units].count("vp_redundant_a") == 2
    assert result.consistent_with_field is True
