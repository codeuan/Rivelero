"""Non-destructive what-if survey design scenarios (Analysis & Design A3).

A scenario is a temporary modification of a *baseline*: the current,
canonical SurveyObservabilityField. It records

* existing active sampling units temporarily deactivated, and
* candidate Viewpoints temporarily added,

and evaluates the resulting exposure. The canonical Survey, the baseline SOF
and its exposure array are never modified.

Exposure arithmetic
-------------------
With baseline exposure ``E_b``, deactivated unit masks ``V_i`` and included
candidate masks ``C_j`` (all restricted to analysable cells)::

    E_s = E_b - sum_i V_i + sum_j C_j

The scenario exposure is **rebuilt from the baseline whenever membership
changes** (copy baseline, subtract every deactivated mask, add every included
candidate mask) using the canonical guarded helpers in
rivelero.observability.exposure. This costs one pass per modification but
cannot drift: toggling a unit on and off any number of times always
reproduces exactly the same exposure, and reset returns the baseline
byte-for-byte. It suits interactive scenarios with a limited number of
modifications; the working set is the baseline array, one scenario array and
the modification masks.

Marginal effects are relative to the *current scenario*, not the baseline:
removing two units that jointly cover a cell can create a blind spot even if
neither covers it uniquely in the baseline, so A2 baseline contributions are
not summed. Metrics reuse A1 (``coverage_masks``/``summarize_exposure``) with
the baseline analysable denominator: a scenario changes the Survey, not the
World.

All quantities are deterministic counts over binary masks, named as such, so
expected-value counterparts can be added for probabilistic observability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

from rivelero.analysis.coverage import (
    CoverageSummary,
    coverage_masks,
    summarize_exposure,
)
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.exposure import (
    add_visibility_to_exposure,
    remove_visibility_from_exposure,
)
from rivelero.observability.masks import invalid_mask
from rivelero.observability.storage import VisibilityKey
from rivelero.observability.survey_field import SurveyObservabilityField


class ScenarioChangeClass(IntEnum):
    """Baseline -> scenario change of each cell."""

    OUTSIDE_DOMAIN = 0
    INVALID = 1
    REMAINS_BLIND = 2
    REMAINS_OBSERVABLE = 3
    LOST_COVERAGE = 4
    GAINED_COVERAGE = 5


class CandidateStatus(str, Enum):
    PENDING = "pending"
    COMPUTING = "computing"
    READY = "ready"
    FAILED = "failed"
    EXCLUDED = "excluded"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class DesignCandidate:
    """A temporary candidate: a canonical Viewpoint plus scenario state.

    ``mask`` exists only when ``status`` is READY. A zero mask (no visible
    analysable cells) is a legitimate READY result, distinct from FAILED or
    EXCLUDED.
    """

    viewpoint: Viewpoint
    key: VisibilityKey | None = None
    status: CandidateStatus = CandidateStatus.PENDING
    mask: np.ndarray | None = None
    included: bool = False
    message: str | None = None

    @property
    def candidate_id(self) -> str:
        return self.viewpoint.viewpoint_id


@dataclass(frozen=True, slots=True)
class MarginalEffect:
    """Effect of one mask relative to the current scenario exposure.

    For a removal (unit active in the scenario):
        ``coverage_cells`` = cells that would become blind (exposure 1).
    For an addition (mask not in the scenario):
        ``coverage_cells`` = cells that would become observable (exposure 0).

    ``unique_to_repeated`` / ``repeated_to_unique`` count cells whose class
    changes between unique and repeated; ``other_exposure_cells`` counts
    visible cells whose exposure changes without changing class.
    """

    visible_cells: int
    coverage_cells: int
    unique_to_repeated: int
    repeated_to_unique: int
    other_exposure_cells: int
    analysable_cells: int

    @property
    def coverage_share_of_analysable(self) -> float | None:
        if self.analysable_cells == 0:
            return None
        return self.coverage_cells / self.analysable_cells


@dataclass(frozen=True, slots=True)
class ScenarioDifference:
    """Signed scenario - baseline differences (no value judgement)."""

    observable_cells: int
    blind_cells: int
    unique_cells: int
    repeated_cells: int
    coverage_percentage_points: float | None
    mean_exposure: float | None
    maximum_exposure: int


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """Comparison-ready description of a scenario against its baseline."""

    baseline_sof_id: str
    baseline: CoverageSummary
    scenario: CoverageSummary

    baseline_units: int
    active_existing_units: int
    deactivated_units: int
    included_candidates: int

    @property
    def is_baseline(self) -> bool:
        return self.deactivated_units == 0 and self.included_candidates == 0

    @property
    def difference(self) -> ScenarioDifference:
        b, s = self.baseline, self.scenario

        def delta(a, c):
            return None if a is None or c is None else c - a

        coverage = delta(b.observable_fraction, s.observable_fraction)
        return ScenarioDifference(
            observable_cells=s.observable_cells - b.observable_cells,
            blind_cells=s.blind_cells - b.blind_cells,
            unique_cells=s.unique_cells - b.unique_cells,
            repeated_cells=s.repeated_cells - b.repeated_cells,
            coverage_percentage_points=(
                None if coverage is None else 100.0 * coverage
            ),
            mean_exposure=delta(b.mean_exposure, s.mean_exposure),
            maximum_exposure=s.maximum_exposure - b.maximum_exposure,
        )


class SurveyDesignScenario:
    """Temporary what-if modifications of one baseline SOF."""

    def __init__(self, baseline: SurveyObservabilityField) -> None:
        if not isinstance(baseline, SurveyObservabilityField):
            raise TypeError("baseline must be a SurveyObservabilityField.")

        self._baseline = baseline
        self._masks = coverage_masks(
            baseline.exposure_count,
            analysis_mask=baseline.analysis_mask,
            valid_mask=baseline.valid_mask,
        )
        self._active_keys = frozenset(baseline.active_keys)

        # Deactivated existing units: key -> analysable-restricted mask.
        self._inactive: dict[VisibilityKey, np.ndarray] = {}
        # Candidates in insertion order.
        self._candidates: dict[str, DesignCandidate] = {}

        self._exposure = self._rebuild()
        self._baseline_summary = self._summarize(baseline.exposure_count, len(self._active_keys))

    # ------------------------------------------------------------------
    # Identity and read-only views
    # ------------------------------------------------------------------

    @property
    def baseline(self) -> SurveyObservabilityField:
        return self._baseline

    @property
    def baseline_sof_id(self) -> str:
        return self._baseline.sof_id

    @property
    def analysable_mask(self) -> np.ndarray:
        return self._masks.analysable

    @property
    def exposure(self) -> np.ndarray:
        """Current scenario exposure (read-only view)."""
        view = self._exposure.view()
        view.flags.writeable = False
        return view

    @property
    def deactivated_keys(self) -> tuple[VisibilityKey, ...]:
        return tuple(self._inactive)

    @property
    def candidates(self) -> tuple[DesignCandidate, ...]:
        return tuple(self._candidates.values())

    def candidate(self, candidate_id: str) -> DesignCandidate:
        return self._candidates[candidate_id]

    def is_active(self, key: VisibilityKey) -> bool:
        return key in self._active_keys and key not in self._inactive

    # ------------------------------------------------------------------
    # Existing units
    # ------------------------------------------------------------------

    def deactivate(self, items: dict[VisibilityKey, np.ndarray]) -> None:
        """Temporarily deactivate existing units given their cached masks.

        All-or-nothing: if any mask is incompatible with the baseline the
        scenario is left unchanged.
        """

        prepared: dict[VisibilityKey, np.ndarray] = {}
        for key, mask in items.items():
            if key not in self._active_keys:
                raise KeyError(f"{key.sampling_unit_id!r} is not a baseline unit.")
            if key in self._inactive:
                continue
            prepared[key] = self._prepare_mask(mask)

        if not prepared:
            return

        previous = dict(self._inactive)
        self._inactive.update(prepared)
        try:
            self._exposure = self._rebuild()
        except ValueError:
            self._inactive = previous
            raise ValueError(
                "A cached mask does not belong to the baseline field; "
                "the scenario was not changed."
            ) from None

    def reactivate(self, keys) -> None:
        for key in keys:
            self._inactive.pop(key, None)
        self._exposure = self._rebuild()

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def add_candidate(
        self,
        viewpoint: Viewpoint,
        *,
        reserved_ids=(),
    ) -> DesignCandidate:
        """Register a candidate Viewpoint (visibility not yet computed)."""

        if not isinstance(viewpoint, Viewpoint):
            raise TypeError("viewpoint must be a Viewpoint.")
        candidate_id = viewpoint.viewpoint_id
        if candidate_id in self._candidates or candidate_id in set(reserved_ids):
            raise ValueError(
                f"Viewpoint ID {candidate_id!r} is already used; candidate IDs "
                "must differ from existing Viewpoints and other candidates."
            )
        candidate = DesignCandidate(viewpoint=viewpoint)
        self._candidates[candidate_id] = candidate
        return candidate

    def set_candidate_visibility(
        self,
        candidate_id: str,
        *,
        key: VisibilityKey,
        mask: np.ndarray,
        include: bool = True,
    ) -> None:
        candidate = self._candidates[candidate_id]
        candidate.mask = self._prepare_mask(mask)
        candidate.key = key
        candidate.status = CandidateStatus.READY
        candidate.message = None
        candidate.included = False
        if include:
            self.set_candidate_included(candidate_id, True)

    def set_candidate_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
        message: str | None = None,
    ) -> None:
        if status == CandidateStatus.READY:
            raise ValueError("Use set_candidate_visibility for READY.")
        candidate = self._candidates[candidate_id]
        was_included = candidate.included
        candidate.status = status
        candidate.message = message
        candidate.mask = None
        candidate.included = False
        if was_included:
            self._exposure = self._rebuild()

    def set_candidate_included(self, candidate_id: str, included: bool) -> None:
        candidate = self._candidates[candidate_id]
        if included and candidate.status != CandidateStatus.READY:
            raise ValueError("Only candidates with computed visibility can be included.")
        if candidate.included == included:
            return
        candidate.included = included
        try:
            self._exposure = self._rebuild()
        except OverflowError:
            candidate.included = not included
            raise

    def remove_candidate(self, candidate_id: str) -> None:
        candidate = self._candidates.pop(candidate_id)
        if candidate.included:
            self._exposure = self._rebuild()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return exactly to the baseline: no deactivations, no candidates."""
        self._inactive.clear()
        self._candidates.clear()
        self._exposure = self._rebuild()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def included_candidates(self) -> tuple[DesignCandidate, ...]:
        return tuple(c for c in self._candidates.values() if c.included)

    def summary(self) -> ScenarioSummary:
        included = len(self.included_candidates)
        active_existing = len(self._active_keys) - len(self._inactive)
        return ScenarioSummary(
            baseline_sof_id=self.baseline_sof_id,
            baseline=self._baseline_summary,
            scenario=self._summarize(self._exposure, active_existing + included),
            baseline_units=len(self._active_keys),
            active_existing_units=active_existing,
            deactivated_units=len(self._inactive),
            included_candidates=included,
        )

    def marginal_loss(self, mask: np.ndarray) -> MarginalEffect:
        """Effect of removing a unit that is active in the current scenario."""

        visible = self._prepare_mask(mask)
        exposure = self._exposure
        if np.any(exposure[visible] == 0):
            raise ValueError("The mask is not part of the current scenario.")
        return MarginalEffect(
            visible_cells=int(np.count_nonzero(visible)),
            coverage_cells=int(np.count_nonzero(visible & (exposure == 1))),
            unique_to_repeated=0,
            repeated_to_unique=int(np.count_nonzero(visible & (exposure == 2))),
            other_exposure_cells=int(np.count_nonzero(visible & (exposure >= 3))),
            analysable_cells=int(np.count_nonzero(self._masks.analysable)),
        )

    def marginal_gain(self, mask: np.ndarray) -> MarginalEffect:
        """Effect of adding a mask that is not part of the current scenario."""

        visible = self._prepare_mask(mask)
        exposure = self._exposure
        return MarginalEffect(
            visible_cells=int(np.count_nonzero(visible)),
            coverage_cells=int(np.count_nonzero(visible & (exposure == 0))),
            unique_to_repeated=int(np.count_nonzero(visible & (exposure == 1))),
            repeated_to_unique=0,
            other_exposure_cells=int(np.count_nonzero(visible & (exposure >= 2))),
            analysable_cells=int(np.count_nonzero(self._masks.analysable)),
        )

    def candidate_marginal_gain(self, candidate_id: str) -> MarginalEffect | None:
        """Gain of a candidate relative to all *other* scenario modifications."""

        candidate = self._candidates[candidate_id]
        if candidate.mask is None:
            return None
        if not candidate.included:
            return self.marginal_gain(candidate.mask)
        # Evaluate against the scenario without this candidate.
        without = remove_visibility_from_exposure(
            self._exposure, candidate.mask, analysable_mask=self._masks.analysable
        )
        visible = candidate.mask
        return MarginalEffect(
            visible_cells=int(np.count_nonzero(visible)),
            coverage_cells=int(np.count_nonzero(visible & (without == 0))),
            unique_to_repeated=int(np.count_nonzero(visible & (without == 1))),
            repeated_to_unique=0,
            other_exposure_cells=int(np.count_nonzero(visible & (without >= 2))),
            analysable_cells=int(np.count_nonzero(self._masks.analysable)),
        )

    def change_classes(self) -> np.ndarray:
        """ScenarioChangeClass raster (uint8) for mapping."""

        baseline_observable = self._masks.observable
        scenario_observable = self._masks.analysable & (self._exposure > 0)

        classes = np.full(
            self._masks.analysable.shape,
            ScenarioChangeClass.OUTSIDE_DOMAIN,
            dtype=np.uint8,
        )
        classes[invalid_mask(self._baseline.analysis_mask, self._baseline.valid_mask)] = (
            ScenarioChangeClass.INVALID
        )
        analysable = self._masks.analysable
        classes[analysable & ~baseline_observable & ~scenario_observable] = (
            ScenarioChangeClass.REMAINS_BLIND
        )
        classes[baseline_observable & scenario_observable] = (
            ScenarioChangeClass.REMAINS_OBSERVABLE
        )
        classes[baseline_observable & ~scenario_observable] = (
            ScenarioChangeClass.LOST_COVERAGE
        )
        classes[~baseline_observable & scenario_observable] = (
            ScenarioChangeClass.GAINED_COVERAGE
        )
        classes[~self._baseline.analysis_mask] = ScenarioChangeClass.OUTSIDE_DOMAIN
        return classes

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self._masks.analysable.shape:
            raise ValueError("Mask shape does not match the baseline grid.")
        return mask & self._masks.analysable

    def _rebuild(self) -> np.ndarray:
        # Always start from a copy: the baseline array is never written.
        exposure = self._baseline.exposure_count.copy()
        analysable = self._masks.analysable
        for mask in self._inactive.values():
            remove_visibility_from_exposure(
                exposure, mask, analysable_mask=analysable, inplace=True
            )
        for candidate in self._candidates.values():
            if candidate.included:
                add_visibility_to_exposure(
                    exposure, candidate.mask, analysable_mask=analysable, inplace=True
                )
        return exposure

    def _summarize(self, exposure: np.ndarray, active_units: int) -> CoverageSummary:
        return summarize_exposure(
            exposure,
            analysis_mask=self._baseline.analysis_mask,
            valid_mask=self._baseline.valid_mask,
            active_units=active_units,
        )


def suggest_candidate_id(reserved_ids, *, prefix: str = "candidate_") -> str:
    """Return the first ``candidate_NNN`` ID not in ``reserved_ids``."""
    reserved = set(reserved_ids)
    index = 1
    while f"{prefix}{index:03d}" in reserved:
        index += 1
    return f"{prefix}{index:03d}"
