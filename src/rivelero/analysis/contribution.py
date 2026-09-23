"""Sampling-unit contribution analysis (Analysis & Design A2).

Deterministic contribution of each active sampling unit to the aggregate
exposure of a SurveyObservabilityField.

Let ``E`` be the SOF exposure count and ``V_i`` the cached effective
visibility mask of active sampling unit ``i``, both restricted to analysable
cells (inside the AnalysisDomain and valid; see rivelero.analysis.coverage).
Because exposure is additive, ``E = sum_i V_i`` and removing unit ``i``
gives ``E_-i = E - V_i``. Therefore, without rebuilding the survey:

    visible cells       V_i                  (analysable cells unit i observes)
    unique cells        V_i AND E == 1       (observed by unit i only)
    repeated cells      V_i AND E >= 2       (also observed by other units)

    visible = unique + repeated

and the cells that become blind spots if unit ``i`` is removed are exactly
its unique cells. Consequently:

* summed over all units, unique cells equal the field's unique-coverage
  cells, because every exposure-1 cell belongs to exactly one unit;
* summed over all units, visible cells equal total exposure, but summed
  repeated cells do NOT equal the number of repeated-coverage cells: a cell
  with exposure k contributes to k units.

A unit with no unique cells creates no new blind spot when removed *under the
current deterministic observability model*. It is not thereby without value:
repeated observation can provide robustness, temporal information, other
viewing directions or image-quality alternatives. The vocabulary here
("unique", "repeated", "removal impact") deliberately makes no value
judgement, and the per-unit quantities are counts over deterministic masks so
that expected-value counterparts for probabilistic observability can be
added alongside them later.

The analysis is read-only and cache-only: it reads each unit's mask from the
VisibilityStore by its exact SOF VisibilityKey, one at a time, and never
computes missing visibility. A missing mask is reported, never counted as a
zero contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Callable

import numpy as np

from rivelero.analysis.coverage import coverage_masks
from rivelero.observability.masks import invalid_mask
from rivelero.observability.storage import VisibilityKey, VisibilityStore
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.visibility.configuration import SamplingUnit


class ContributionStatus(str, Enum):
    """Whether a unit's contribution could be derived."""

    AVAILABLE = "available"
    # No cached mask exists for the unit's exact VisibilityKey.
    MISSING = "missing"
    # A cache file exists but could not be read or validated.
    UNREADABLE = "unreadable"
    # The mask does not belong to this field (grid mismatch, or visible cells
    # where the field has zero exposure).
    INCONSISTENT = "inconsistent"


class UnitContributionClass(IntEnum):
    """Cell classes of the selected-unit contribution map."""

    OUTSIDE_DOMAIN = 0
    INVALID = 1
    NOT_VISIBLE_FROM_UNIT = 2
    UNIQUE_CONTRIBUTION = 3
    REPEATED_CONTRIBUTION = 4


@dataclass(frozen=True, slots=True)
class SamplingUnitContribution:
    """Deterministic contribution of one active sampling unit.

    Cell counts (``visible_cells``, ``unique_cells``, ``repeated_cells``)
    are over analysable cells and are ``None`` unless ``status`` is
    AVAILABLE. Two denominators are used, always named explicitly:

    ``*_share_of_analysable``
        relative to all analysable cells of the field;
    ``*_share_of_unit_visibility``
        relative to this unit's own visible cells.
    """

    order: int
    key: VisibilityKey

    status: ContributionStatus
    analysable_cells: int

    visible_cells: int | None = None
    unique_cells: int | None = None
    repeated_cells: int | None = None

    message: str | None = None

    @property
    def sampling_unit_id(self) -> str:
        return self.key.sampling_unit_id

    @property
    def viewpoint_id(self) -> str:
        return self.key.viewpoint_id

    @property
    def observation_event_id(self) -> str | None:
        if self.key.sampling_unit_type == SamplingUnit.OBSERVATION_EVENT.value:
            return self.key.sampling_unit_id
        return None

    @property
    def available(self) -> bool:
        return self.status == ContributionStatus.AVAILABLE

    @property
    def has_unique_coverage(self) -> bool | None:
        return None if self.unique_cells is None else self.unique_cells > 0

    def _of_analysable(self, cells: int | None) -> float | None:
        if cells is None or self.analysable_cells == 0:
            return None
        return cells / self.analysable_cells

    def _of_visibility(self, cells: int | None) -> float | None:
        if cells is None or not self.visible_cells:
            return None
        return cells / self.visible_cells

    @property
    def visible_share_of_analysable(self) -> float | None:
        return self._of_analysable(self.visible_cells)

    @property
    def unique_share_of_analysable(self) -> float | None:
        return self._of_analysable(self.unique_cells)

    @property
    def repeated_share_of_analysable(self) -> float | None:
        return self._of_analysable(self.repeated_cells)

    @property
    def unique_share_of_unit_visibility(self) -> float | None:
        return self._of_visibility(self.unique_cells)

    @property
    def repeated_share_of_unit_visibility(self) -> float | None:
        return self._of_visibility(self.repeated_cells)

    @property
    def cells_lost_if_removed(self) -> int | None:
        """Analysable cells that would become blind spots without this unit."""
        return self.unique_cells

    @property
    def coverage_loss_if_removed(self) -> float | None:
        """Drop in observable share of analysable cells if removed.

        Counterfactual metric only; nothing is removed.
        """
        return self.unique_share_of_analysable


@dataclass(frozen=True, slots=True)
class ContributionAnalysis:
    """Contribution of every active sampling unit of one SOF."""

    sof_id: str
    sampling_unit: SamplingUnit

    analysable_cells: int
    field_unique_cells: int
    field_observable_cells: int
    field_total_exposure: int

    units: tuple[SamplingUnitContribution, ...]

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def available_units(self) -> tuple[SamplingUnitContribution, ...]:
        return tuple(unit for unit in self.units if unit.available)

    @property
    def unavailable_units(self) -> tuple[SamplingUnitContribution, ...]:
        return tuple(unit for unit in self.units if not unit.available)

    @property
    def complete(self) -> bool:
        """Whether every active unit's contribution could be derived."""
        return not self.unavailable_units

    @property
    def n_with_unique_coverage(self) -> int:
        return sum(1 for unit in self.available_units if unit.unique_cells > 0)

    @property
    def n_without_unique_coverage(self) -> int:
        """Units that observe something but nothing uniquely."""
        return sum(
            1
            for unit in self.available_units
            if unit.visible_cells > 0 and unit.unique_cells == 0
        )

    @property
    def n_without_visible_cells(self) -> int:
        return sum(1 for unit in self.available_units if unit.visible_cells == 0)

    @property
    def total_unique_contribution_cells(self) -> int:
        return sum(unit.unique_cells for unit in self.available_units)

    @property
    def total_visible_contribution_cells(self) -> int:
        return sum(unit.visible_cells for unit in self.available_units)

    @property
    def consistent_with_field(self) -> bool | None:
        """Whether per-unit results reproduce the field's invariants.

        ``None`` when some units are unavailable (the invariants then cannot
        be checked).
        """
        if not self.complete:
            return None
        return (
            self.total_unique_contribution_cells == self.field_unique_cells
            and self.total_visible_contribution_cells == self.field_total_exposure
        )

    def get(self, sampling_unit_id: str) -> SamplingUnitContribution:
        for unit in self.units:
            if unit.sampling_unit_id == sampling_unit_id:
                return unit
        raise KeyError(sampling_unit_id)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def unit_contribution_counts(
    visibility_mask: np.ndarray,
    *,
    unique_mask: np.ndarray,
    repeated_mask: np.ndarray,
    analysable_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return (visible, unique, repeated, visible-but-unexposed) cell counts.

    The last value counts analysable cells the mask marks visible although
    the field has zero exposure there; it must be zero for a mask that
    belongs to the field.
    """

    visible = np.asarray(visibility_mask, dtype=bool) & analysable_mask
    n_visible = int(np.count_nonzero(visible))
    n_unique = int(np.count_nonzero(visible & unique_mask))
    n_repeated = int(np.count_nonzero(visible & repeated_mask))
    return n_visible, n_unique, n_repeated, n_visible - n_unique - n_repeated


def analyse_contributions(
    sof: SurveyObservabilityField,
    store: VisibilityStore,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ContributionAnalysis:
    """Derive the contribution of every active sampling unit of ``sof``.

    Masks are read from ``store`` one at a time by the SOF's exact
    VisibilityKeys; nothing is computed or written. Working memory is the
    SOF arrays, three shared boolean masks and one unit mask (plus whatever
    the store's LRU retains).

    ``progress_callback(processed, total, sampling_unit_id)`` is called after
    each unit, matching the SOF builder so the same cancellation adapter can
    be used.
    """

    if not isinstance(sof, SurveyObservabilityField):
        raise TypeError("sof must be a SurveyObservabilityField.")
    if not isinstance(store, VisibilityStore):
        raise TypeError("store must be a VisibilityStore.")

    masks = coverage_masks(
        sof.exposure_count,
        analysis_mask=sof.analysis_mask,
        valid_mask=sof.valid_mask,
    )
    analysable_cells = int(np.count_nonzero(masks.analysable))

    keys = list(sof.active_keys)
    total = len(keys)
    units: list[SamplingUnitContribution] = []

    for order, key in enumerate(keys, start=1):
        units.append(
            _analyse_unit(
                order=order,
                key=key,
                sof=sof,
                store=store,
                masks=masks,
                analysable_cells=analysable_cells,
            )
        )
        if progress_callback is not None:
            progress_callback(order, total, key.sampling_unit_id)

    return ContributionAnalysis(
        sof_id=sof.sof_id,
        sampling_unit=sof.sampling_unit,
        analysable_cells=analysable_cells,
        field_unique_cells=int(np.count_nonzero(masks.unique)),
        field_observable_cells=int(np.count_nonzero(masks.observable)),
        field_total_exposure=int(
            sof.exposure_count[masks.analysable].sum(dtype=np.int64)
        ),
        units=tuple(units),
    )


def _analyse_unit(
    *,
    order: int,
    key: VisibilityKey,
    sof: SurveyObservabilityField,
    store: VisibilityStore,
    masks,
    analysable_cells: int,
) -> SamplingUnitContribution:
    def unavailable(status: ContributionStatus, message: str):
        return SamplingUnitContribution(
            order=order,
            key=key,
            status=status,
            analysable_cells=analysable_cells,
            message=message,
        )

    try:
        stored = store.get(key)
    except RuntimeError as exc:
        return unavailable(ContributionStatus.UNREADABLE, str(exc))

    if stored is None:
        return unavailable(
            ContributionStatus.MISSING,
            "No cached visibility exists for this unit in the current "
            "VisibilityStore (for example after the cache was cleared or "
            "the storage location changed). Rebuild the observability field "
            "to restore it.",
        )

    problem = _grid_mismatch(stored, sof)
    if problem is not None:
        return unavailable(ContributionStatus.INCONSISTENT, problem)

    visible, unique, repeated, unexposed = unit_contribution_counts(
        stored.visibility_mask,
        unique_mask=masks.unique,
        repeated_mask=masks.repeated,
        analysable_mask=masks.analysable,
    )
    # Only counts are kept; the mask itself is released here.
    del stored

    if unexposed:
        return unavailable(
            ContributionStatus.INCONSISTENT,
            f"The cached mask marks {unexposed:,} analysable cells visible "
            "where the field has zero exposure, so it is not part of this "
            "field.",
        )

    return SamplingUnitContribution(
        order=order,
        key=key,
        status=ContributionStatus.AVAILABLE,
        analysable_cells=analysable_cells,
        visible_cells=visible,
        unique_cells=unique,
        repeated_cells=repeated,
    )


def _grid_mismatch(stored, sof: SurveyObservabilityField) -> str | None:
    if stored.shape != sof.exposure_count.shape:
        return "Cached mask shape does not match the field grid."
    if stored.transform != sof.transform:
        return "Cached mask transform does not match the field grid."
    if stored.crs != sof.crs:
        return "Cached mask CRS does not match the field grid."
    return None


# ---------------------------------------------------------------------------
# Selected-unit map
# ---------------------------------------------------------------------------


def unit_contribution_classes(
    visibility_mask: np.ndarray,
    *,
    exposure_count: np.ndarray,
    analysis_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Classify cells relative to one unit's contribution (uint8 raster).

    Derived for display only; the SOF is not modified.
    """

    masks = coverage_masks(
        exposure_count, analysis_mask=analysis_mask, valid_mask=valid_mask
    )
    visible = np.asarray(visibility_mask, dtype=bool) & masks.analysable

    classes = np.full(
        masks.analysable.shape,
        UnitContributionClass.OUTSIDE_DOMAIN,
        dtype=np.uint8,
    )
    classes[invalid_mask(analysis_mask, valid_mask)] = UnitContributionClass.INVALID
    classes[masks.analysable] = UnitContributionClass.NOT_VISIBLE_FROM_UNIT
    classes[visible & masks.unique] = UnitContributionClass.UNIQUE_CONTRIBUTION
    classes[visible & masks.repeated] = UnitContributionClass.REPEATED_CONTRIBUTION
    classes[~np.asarray(analysis_mask, dtype=bool)] = UnitContributionClass.OUTSIDE_DOMAIN
    return classes
