"""A1 tests: descriptive coverage and exposure analysis."""

from __future__ import annotations

import numpy as np
import pytest

from observability_fixtures import DEM_NODATA, make_ready_state
from rivelero.analysis.coverage import (
    CoverageClass,
    coverage_class_raster,
    summarize_coverage,
    summarize_exposure,
)
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.observability.builder import build_survey_observability_field
from rivelero.observability.masks import ObservabilityState


def _grid():
    """4 x 5 grid with every class represented.

    Row 0 is outside the domain (with high exposure that must be ignored),
    row 1 contains two invalid cells (also with exposure) and three valid
    blind cells, and rows 2-3 hold ten more analysable cells with exposures
    0 0 0 1 1 / 1 2 2 3 5: 13 analysable cells in total.
    """
    exposure = np.array(
        [
            [9, 9, 9, 9, 9],
            [7, 7, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 2, 2, 3, 5],
        ],
        dtype=np.uint32,
    )
    analysis = np.ones_like(exposure, dtype=bool)
    analysis[0, :] = False
    valid = np.ones_like(exposure, dtype=bool)
    valid[1, :2] = False
    # Invalid cells cannot legitimately hold exposure in a real SOF (the SOF
    # zeroes them); non-zero values here prove they are excluded anyway.
    return exposure, analysis, valid


def _summary(**overrides):
    exposure, analysis, valid = _grid()
    return summarize_exposure(
        overrides.get("exposure", exposure),
        analysis_mask=overrides.get("analysis", analysis),
        valid_mask=overrides.get("valid", valid),
        active_units=overrides.get("active_units", 9),
    )


def test_cell_accounting_and_denominator():
    summary = _summary()

    assert summary.total_cells == 20
    assert summary.outside_domain_cells == 5
    assert summary.invalid_cells == 2
    # Row 1 (3 valid cells, exposure 0) + rows 2-3 (10 cells).
    assert summary.analysable_cells == 13
    assert (
        summary.outside_domain_cells + summary.invalid_cells
        + summary.analysable_cells == summary.total_cells
    )

    assert summary.blind_cells == 6
    assert summary.unique_cells == 3
    assert summary.repeated_cells == 4
    assert summary.observable_cells == 7
    assert summary.blind_cells + summary.observable_cells == summary.analysable_cells

    # Shares are relative to analysable cells, not the full raster (20) or
    # the domain including invalid cells (15).
    assert summary.observable_fraction == pytest.approx(7 / 13)
    assert summary.blind_fraction == pytest.approx(6 / 13)
    assert summary.unique_fraction == pytest.approx(3 / 13)
    assert summary.repeated_fraction == pytest.approx(4 / 13)
    assert summary.repeated_share_of_observable == pytest.approx(4 / 7)


def test_exposure_statistics_ignore_outside_and_invalid():
    summary = _summary()
    analysable_values = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 5]

    assert summary.maximum_exposure == 5  # not 9 (outside) or 7 (invalid)
    assert summary.total_exposure == sum(analysable_values)
    assert summary.mean_exposure == pytest.approx(np.mean(analysable_values))
    assert summary.median_exposure == pytest.approx(np.median(analysable_values))

    observable = [v for v in analysable_values if v > 0]
    assert summary.mean_exposure_observable == pytest.approx(np.mean(observable))
    assert summary.median_exposure_observable == pytest.approx(np.median(observable))


def test_even_count_median_averages_middle_levels():
    exposure = np.array([[0, 1, 3, 4]], dtype=np.uint16)
    mask = np.ones_like(exposure, dtype=bool)
    summary = summarize_exposure(
        exposure, analysis_mask=mask, valid_mask=mask, active_units=4
    )
    assert summary.median_exposure == pytest.approx(2.0)
    assert summary.median_exposure_observable == pytest.approx(3.0)


def test_exposure_distribution_counts_and_fractions():
    distribution = _summary().distribution

    assert distribution.analysable_cells == 13
    assert list(distribution.cell_counts) == [6, 3, 2, 1, 0, 1]
    assert distribution.cells_at(0) == 6
    assert distribution.cells_at(4) == 0
    assert distribution.cells_at(99) == 0
    assert distribution.fractions.sum() == pytest.approx(1.0)


def test_distribution_binning_keeps_blind_and_unique_levels():
    exposure = np.arange(0, 101, dtype=np.uint32).reshape(1, -1)
    mask = np.ones_like(exposure, dtype=bool)
    summary = summarize_exposure(
        exposure, analysis_mask=mask, valid_mask=mask, active_units=100
    )

    bins = summary.distribution.binned(max_bins=10)
    assert len(bins) <= 10
    assert (bins[0].lower, bins[0].upper, bins[0].cells) == (0, 0, 1)
    assert (bins[1].lower, bins[1].upper, bins[1].cells) == (1, 1, 1)
    assert bins[-1].upper == 100
    assert sum(item.cells for item in bins) == 101
    # Contiguous, non-overlapping ranges.
    for previous, current in zip(bins, bins[1:]):
        assert current.lower == previous.upper + 1


def test_no_observable_cells():
    exposure = np.zeros((3, 3), dtype=np.uint32)
    mask = np.ones_like(exposure, dtype=bool)
    summary = summarize_exposure(
        exposure, analysis_mask=mask, valid_mask=mask, active_units=0
    )

    assert summary.observable_fraction == 0.0
    assert summary.blind_fraction == 1.0
    assert summary.maximum_exposure == 0
    assert summary.mean_exposure == 0.0
    assert summary.median_exposure == 0.0
    assert summary.mean_exposure_observable is None
    assert summary.median_exposure_observable is None
    assert summary.repeated_share_of_observable is None


def test_all_observable_cells():
    exposure = np.full((2, 3), 2, dtype=np.uint32)
    exposure[0, 0] = 1
    mask = np.ones_like(exposure, dtype=bool)
    summary = summarize_exposure(
        exposure, analysis_mask=mask, valid_mask=mask, active_units=2
    )

    assert summary.observable_fraction == 1.0
    assert summary.blind_cells == 0
    assert summary.unique_cells == 1
    assert summary.repeated_cells == 5


def test_no_analysable_cells_gives_undefined_shares():
    exposure = np.zeros((2, 2), dtype=np.uint32)
    analysis = np.zeros_like(exposure, dtype=bool)
    summary = summarize_exposure(
        exposure, analysis_mask=analysis, valid_mask=~analysis, active_units=0
    )

    assert summary.analysable_cells == 0
    assert summary.observable_fraction is None
    assert summary.mean_exposure is None
    assert summary.median_exposure is None
    assert summary.distribution.fractions is None


def test_invalid_inputs_are_rejected():
    exposure, analysis, valid = _grid()
    with pytest.raises(TypeError):
        summarize_exposure(
            exposure.astype(float), analysis_mask=analysis, valid_mask=valid,
            active_units=9,
        )
    with pytest.raises(ValueError, match="active sampling units"):
        summarize_exposure(
            exposure, analysis_mask=analysis, valid_mask=valid, active_units=2
        )


def test_coverage_class_raster_matches_counts():
    exposure, analysis, valid = _grid()
    classes = coverage_class_raster(
        exposure, analysis_mask=analysis, valid_mask=valid
    )
    counts = np.bincount(classes.ravel(), minlength=len(CoverageClass))

    assert counts[CoverageClass.OUTSIDE_DOMAIN] == 5
    assert counts[CoverageClass.INVALID] == 2
    assert counts[CoverageClass.BLIND_SPOT] == 6
    assert counts[CoverageClass.UNIQUE] == 3
    assert counts[CoverageClass.REPEATED] == 4


def test_summary_agrees_with_canonical_sof(tmp_path):
    # Real SOF with all four observability states (terrain NoData inside a
    # buffered survey domain).
    state = make_ready_state(
        tmp_path,
        dem=DEM_NODATA,
        survey_buffer_m=350.0,
        viewpoint_ids=None,
    )
    request = prepare_build(state)
    result = build_survey_observability_field(**request.build_kwargs())
    install_build_result(state, request, result)
    sof = state.analysis.survey_observability_field

    summary = summarize_coverage(sof)
    states = sof.state_counts()

    assert summary.analysable_cells == sof.n_analysable_cells
    assert summary.observable_cells == sof.n_observable_cells
    assert summary.blind_cells == sof.n_blindspot_cells
    assert summary.invalid_cells == states[ObservabilityState.INVALID] > 0
    assert summary.outside_domain_cells == states[ObservabilityState.OUTSIDE_DOMAIN] > 0
    assert summary.maximum_exposure == sof.maximum_exposure
    assert summary.active_units == sof.n_active_units == 10
    assert summary.unique_cells == int(
        np.count_nonzero(sof.analysable_mask & (sof.exposure_count == 1))
    )
    assert summary.repeated_cells == int(
        np.count_nonzero(sof.analysable_mask & (sof.exposure_count >= 2))
    )

    classes = coverage_class_raster(
        sof.exposure_count, analysis_mask=sof.analysis_mask, valid_mask=sof.valid_mask
    )
    # The first three classes coincide with the observability state.
    for state_value in (
        ObservabilityState.OUTSIDE_DOMAIN,
        ObservabilityState.INVALID,
        ObservabilityState.BLIND_SPOT,
    ):
        assert np.array_equal(
            classes == state_value, sof.observability_state == state_value
        )
    assert np.array_equal(
        np.isin(classes, [CoverageClass.UNIQUE, CoverageClass.REPEATED]),
        sof.observable_mask,
    )
