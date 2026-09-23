"""Comparison of survey design states (Analysis & Design A4).

Compares two *design states* over the same baseline: the baseline itself,
the live what-if scenario, or saved scenario snapshots. Comparison is
descriptive: it reports measurable, signed differences and never ranks
alternatives or combines metrics into a score.

Direction convention
--------------------
Every difference is **RIGHT minus LEFT**. Swapping the sides negates every
numeric difference and exchanges the gained and lost masks.

Snapshots
---------
A ScenarioSnapshot is an immutable, lightweight record of a scenario:
the baseline SOF identity, the VisibilityKeys of deactivated units, the
candidate Viewpoints with their VisibilityKeys and inclusion flags, and the
ScenarioSummary. It holds no visibility masks and no exposure raster: every
mask it needs is addressable in the VisibilityStore by its fingerprinted
key. Exposure is reconstructed on demand from the baseline (copy baseline,
subtract deactivated masks, add included candidate masks - the A3 rebuild),
and a ScenarioWorkspace keeps only a small bounded cache of recently used
reconstructed exposures. A missing cached mask makes the snapshot
unavailable for spatial comparison rather than silently changing it.

Baseline compatibility
----------------------
Cell-by-cell comparison is meaningful only between modifications of the same
baseline. Every baseline build receives a unique ``sof_id`` and any upstream
change produces a new SOF, so two states are compatible exactly when their
``baseline_sof_id`` values are equal. Snapshots of an earlier baseline are
kept for the session but reported as out of date and cannot be compared or
restored.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

import numpy as np

from rivelero.analysis.coverage import CoverageSummary, summarize_exposure
from rivelero.analysis.scenario import (
    CandidateStatus,
    ScenarioSummary,
    SurveyDesignScenario,
    coverage_change_classes,
)
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.exposure import (
    add_visibility_to_exposure,
    remove_visibility_from_exposure,
)
from rivelero.observability.storage import VisibilityKey, VisibilityStore
from rivelero.observability.survey_field import SurveyObservabilityField


BASELINE_ID = "baseline"
LIVE_SCENARIO_ID = "live"


class IncompatibleBaselineError(ValueError):
    """Raised when two design states do not share one baseline."""


class SnapshotUnavailableError(RuntimeError):
    """Raised when a snapshot's exposure cannot be reconstructed."""


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateDefinition:
    """Candidate as saved in a snapshot (definition, not visibility)."""

    viewpoint: Viewpoint
    key: VisibilityKey | None
    status: CandidateStatus
    included: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioSnapshot:
    """Immutable, lightweight record of one what-if scenario."""

    snapshot_id: str
    name: str
    description: str
    created_at: datetime

    baseline_sof_id: str
    sampling_unit: str
    visibility_configuration_id: str

    deactivated_keys: tuple[VisibilityKey, ...]
    candidates: tuple[CandidateDefinition, ...]
    summary: ScenarioSummary

    @property
    def included_candidates(self) -> tuple[CandidateDefinition, ...]:
        return tuple(c for c in self.candidates if c.included)

    def compatible_with(self, sof: SurveyObservabilityField | None) -> bool:
        return sof is not None and sof.sof_id == self.baseline_sof_id


def snapshot_scenario(
    scenario: SurveyDesignScenario,
    *,
    name: str,
    description: str = "",
    snapshot_id: str | None = None,
) -> ScenarioSnapshot:
    """Record the current state of a live scenario (no masks copied)."""

    name = name.strip()
    if not name:
        raise ValueError("A snapshot needs a name.")
    baseline = scenario.baseline
    return ScenarioSnapshot(
        snapshot_id=snapshot_id or f"snapshot_{uuid4().hex}",
        name=name,
        description=description.strip(),
        created_at=datetime.now(timezone.utc),
        baseline_sof_id=baseline.sof_id,
        sampling_unit=getattr(baseline.sampling_unit, "value", str(baseline.sampling_unit)),
        visibility_configuration_id=baseline.visibility_configuration_id,
        deactivated_keys=tuple(scenario.deactivated_keys),
        candidates=tuple(
            CandidateDefinition(
                # Copy so later edits of the live candidate cannot reach
                # the snapshot.
                viewpoint=replace(candidate.viewpoint),
                key=candidate.key,
                status=candidate.status,
                included=candidate.included,
                message=candidate.message,
            )
            for candidate in scenario.candidates
        ),
        summary=scenario.summary(),
    )


# ---------------------------------------------------------------------------
# Design states and reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesignState:
    """One side of a comparison: an exposure field over a baseline."""

    state_id: str
    label: str
    baseline_sof_id: str
    exposure: np.ndarray
    summary: CoverageSummary
    active_existing_units: int
    included_candidates: int

    @property
    def sampling_units(self) -> int:
        return self.active_existing_units + self.included_candidates


def baseline_state(sof: SurveyObservabilityField) -> DesignState:
    return DesignState(
        state_id=BASELINE_ID,
        label="Baseline",
        baseline_sof_id=sof.sof_id,
        exposure=_read_only(sof.exposure_count),
        summary=_summarize(sof, sof.exposure_count, sof.n_active_units),
        active_existing_units=sof.n_active_units,
        included_candidates=0,
    )


def live_state(scenario: SurveyDesignScenario, *, label="Current scenario (unsaved)") -> DesignState:
    summary = scenario.summary()
    return DesignState(
        state_id=LIVE_SCENARIO_ID,
        label=label,
        baseline_sof_id=scenario.baseline_sof_id,
        exposure=scenario.exposure,
        summary=summary.scenario,
        active_existing_units=summary.active_existing_units,
        included_candidates=summary.included_candidates,
    )


def reconstruct_snapshot_exposure(
    snapshot: ScenarioSnapshot,
    sof: SurveyObservabilityField,
    store: VisibilityStore,
) -> np.ndarray:
    """Rebuild a snapshot's exposure from the baseline and cached masks.

    Never modifies the baseline. Raises SnapshotUnavailableError if the
    snapshot belongs to another baseline or a required mask is not cached.
    """

    if not snapshot.compatible_with(sof):
        raise SnapshotUnavailableError(
            f"{snapshot.name!r} was saved for a different observability field."
        )

    analysable = sof.analysable_mask
    exposure = sof.exposure_count.copy()
    missing = []

    for key in snapshot.deactivated_keys:
        stored = store.get(key)
        if stored is None:
            missing.append(key.sampling_unit_id)
            continue
        remove_visibility_from_exposure(
            exposure, stored.visibility_mask, analysable_mask=analysable, inplace=True
        )

    for candidate in snapshot.included_candidates:
        stored = None if candidate.key is None else store.get(candidate.key)
        if stored is None:
            missing.append(candidate.viewpoint.viewpoint_id)
            continue
        add_visibility_to_exposure(
            exposure, stored.visibility_mask, analysable_mask=analysable, inplace=True
        )

    if missing:
        raise SnapshotUnavailableError(
            f"Cached visibility is missing for {', '.join(missing[:8])}"
            + (" …" if len(missing) > 8 else "")
            + f"; {snapshot.name!r} cannot be compared spatially until it is "
            "recomputed."
        )
    return _read_only(exposure)


def snapshot_state(
    snapshot: ScenarioSnapshot,
    exposure: np.ndarray,
) -> DesignState:
    summary = snapshot.summary
    return DesignState(
        state_id=snapshot.snapshot_id,
        label=snapshot.name,
        baseline_sof_id=snapshot.baseline_sof_id,
        exposure=exposure,
        summary=summary.scenario,
        active_existing_units=summary.active_existing_units,
        included_candidates=summary.included_candidates,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioComparison:
    """RIGHT minus LEFT comparison of two compatible design states."""

    left: DesignState
    right: DesignState
    analysis_mask: np.ndarray
    valid_mask: np.ndarray

    # -- signed numeric differences (right - left) ----------------------

    @property
    def observable_cells_delta(self) -> int:
        return self.right.summary.observable_cells - self.left.summary.observable_cells

    @property
    def blind_cells_delta(self) -> int:
        return self.right.summary.blind_cells - self.left.summary.blind_cells

    @property
    def unique_cells_delta(self) -> int:
        return self.right.summary.unique_cells - self.left.summary.unique_cells

    @property
    def repeated_cells_delta(self) -> int:
        return self.right.summary.repeated_cells - self.left.summary.repeated_cells

    @property
    def coverage_percentage_point_delta(self) -> float | None:
        left, right = self.left.summary.observable_fraction, self.right.summary.observable_fraction
        return None if left is None or right is None else 100.0 * (right - left)

    @property
    def mean_exposure_delta(self) -> float | None:
        left, right = self.left.summary.mean_exposure, self.right.summary.mean_exposure
        return None if left is None or right is None else right - left

    @property
    def maximum_exposure_delta(self) -> int:
        return self.right.summary.maximum_exposure - self.left.summary.maximum_exposure

    @property
    def active_existing_units_delta(self) -> int:
        return self.right.active_existing_units - self.left.active_existing_units

    @property
    def included_candidates_delta(self) -> int:
        return self.right.included_candidates - self.left.included_candidates

    @property
    def sampling_unit_count_delta(self) -> int:
        return self.right.sampling_units - self.left.sampling_units

    # -- spatial differences --------------------------------------------

    @property
    def analysable_mask(self) -> np.ndarray:
        return self.analysis_mask & self.valid_mask

    @property
    def gained_coverage_mask(self) -> np.ndarray:
        """Blind on the left, observable on the right."""
        analysable = self.analysable_mask
        return analysable & (self.left.exposure == 0) & (self.right.exposure > 0)

    @property
    def lost_coverage_mask(self) -> np.ndarray:
        """Observable on the left, blind on the right."""
        analysable = self.analysable_mask
        return analysable & (self.left.exposure > 0) & (self.right.exposure == 0)

    def change_classes(self) -> np.ndarray:
        return coverage_change_classes(
            self.left.exposure,
            self.right.exposure,
            analysis_mask=self.analysis_mask,
            valid_mask=self.valid_mask,
        )

    def exposure_difference(self) -> np.ma.MaskedArray:
        """right exposure - left exposure, masked outside analysable cells."""
        difference = self.right.exposure.astype(np.int64) - self.left.exposure.astype(np.int64)
        return np.ma.masked_where(~self.analysable_mask, difference)


def compare_states(
    left: DesignState,
    right: DesignState,
    sof: SurveyObservabilityField,
) -> ScenarioComparison:
    """Compare two design states of the baseline ``sof`` (RIGHT - LEFT)."""

    for state in (left, right):
        if state.baseline_sof_id != sof.sof_id:
            raise IncompatibleBaselineError(
                f"{state.label!r} belongs to a different observability field; "
                "cell-by-cell differences would not be meaningful."
            )
        if state.exposure.shape != sof.exposure_count.shape:
            raise IncompatibleBaselineError(f"{state.label!r} has a different grid.")
    return ScenarioComparison(
        left=left,
        right=right,
        analysis_mask=sof.analysis_mask,
        valid_mask=sof.valid_mask,
    )


# ---------------------------------------------------------------------------
# Workspace (saved scenarios)
# ---------------------------------------------------------------------------


class ScenarioWorkspace:
    """Session library of saved snapshots plus a bounded exposure cache.

    The library outlives baseline rebuilds so the user keeps the record of
    what was explored; snapshots of other baselines are reported as out of
    date and are never compared or restored against the new baseline.
    """

    def __init__(self, *, exposure_cache_size: int = 4) -> None:
        self._snapshots: dict[str, ScenarioSnapshot] = {}
        self._exposures: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._cache_size = max(0, int(exposure_cache_size))
        self.left_id: str = BASELINE_ID
        self.right_id: str | None = None

    # -- library ------------------------------------------------------

    @property
    def snapshots(self) -> tuple[ScenarioSnapshot, ...]:
        return tuple(self._snapshots.values())

    def get(self, snapshot_id: str) -> ScenarioSnapshot:
        return self._snapshots[snapshot_id]

    def suggest_name(self) -> str:
        names = {snapshot.name for snapshot in self._snapshots.values()}
        index = len(self._snapshots) + 1
        while f"Scenario {index}" in names:
            index += 1
        return f"Scenario {index}"

    def save(
        self,
        scenario: SurveyDesignScenario,
        *,
        name: str,
        description: str = "",
    ) -> ScenarioSnapshot:
        self._require_unique_name(name)
        snapshot = snapshot_scenario(scenario, name=name, description=description)
        self._snapshots[snapshot.snapshot_id] = snapshot
        # The live exposure is already materialised: seed the cache so the
        # first comparison needs no reconstruction.
        self._remember(snapshot, scenario.exposure.copy())
        return snapshot

    def rename(self, snapshot_id: str, name: str) -> ScenarioSnapshot:
        snapshot = self._snapshots[snapshot_id]
        if name.strip() != snapshot.name:
            self._require_unique_name(name)
        updated = replace(snapshot, name=name.strip())
        self._snapshots[snapshot_id] = updated
        return updated

    def describe(self, snapshot_id: str, description: str) -> ScenarioSnapshot:
        updated = replace(self._snapshots[snapshot_id], description=description.strip())
        self._snapshots[snapshot_id] = updated
        return updated

    def delete(self, snapshot_id: str) -> None:
        """Remove a snapshot; cached visibility and the live scenario stay."""
        self._snapshots.pop(snapshot_id)
        for key in [k for k in self._exposures if k[0] == snapshot_id]:
            del self._exposures[key]
        if self.left_id == snapshot_id:
            self.left_id = BASELINE_ID
        if self.right_id == snapshot_id:
            self.right_id = None

    # -- states -------------------------------------------------------

    def state(
        self,
        state_id: str,
        *,
        sof: SurveyObservabilityField,
        store: VisibilityStore | None,
        live: SurveyDesignScenario | None = None,
    ) -> DesignState:
        """Materialise a design state for comparison against ``sof``."""

        if state_id == BASELINE_ID:
            return baseline_state(sof)
        if state_id == LIVE_SCENARIO_ID:
            if live is None or live.baseline is not sof:
                raise SnapshotUnavailableError("There is no current scenario for this field.")
            return live_state(live)

        snapshot = self._snapshots[state_id]
        if not snapshot.compatible_with(sof):
            raise SnapshotUnavailableError(
                f"{snapshot.name!r} was saved for an earlier observability field."
            )
        cache_key = (snapshot.snapshot_id, sof.sof_id)
        exposure = self._exposures.get(cache_key)
        if exposure is None:
            if store is None:
                raise SnapshotUnavailableError("No visibility store is configured.")
            exposure = reconstruct_snapshot_exposure(snapshot, sof, store)
            self._remember(snapshot, exposure)
        else:
            self._exposures.move_to_end(cache_key)
        return snapshot_state(snapshot, exposure)

    @property
    def cached_exposure_count(self) -> int:
        return len(self._exposures)

    def prune(self, sof: SurveyObservabilityField | None) -> None:
        """Drop cached exposures and selections that do not fit ``sof``."""
        current = None if sof is None else sof.sof_id
        for key in [k for k in self._exposures if k[1] != current]:
            del self._exposures[key]
        for side in ("left_id", "right_id"):
            state_id = getattr(self, side)
            if state_id in self._snapshots and not self._snapshots[state_id].compatible_with(sof):
                setattr(self, side, BASELINE_ID if side == "left_id" else None)

    # -- internals ----------------------------------------------------

    def _remember(self, snapshot: ScenarioSnapshot, exposure: np.ndarray) -> None:
        if self._cache_size == 0:
            return
        key = (snapshot.snapshot_id, snapshot.baseline_sof_id)
        self._exposures[key] = _read_only(exposure)
        self._exposures.move_to_end(key)
        while len(self._exposures) > self._cache_size:
            self._exposures.popitem(last=False)

    def _require_unique_name(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("A snapshot needs a name.")
        if any(snapshot.name == name for snapshot in self._snapshots.values()):
            raise ValueError(f"A saved scenario is already called {name!r}.")


# ---------------------------------------------------------------------------
# Restore into the live scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RestoreReport:
    """What happened when a snapshot was loaded into the live scenario."""

    deactivated_units: int
    candidates_restored: int
    candidates_needing_computation: tuple[str, ...]


def restore_snapshot(
    snapshot: ScenarioSnapshot,
    sof: SurveyObservabilityField,
    store: VisibilityStore,
) -> tuple[SurveyDesignScenario, RestoreReport]:
    """Build a new live scenario reproducing ``snapshot``.

    Deactivated units need their cached masks (baseline units are always
    cached after a build); if any is missing the restore is refused rather
    than producing a different scenario. Candidates whose cached mask is
    missing are restored as definitions in PENDING state, keeping their
    intended inclusion, so their visibility is recomputed - never treated
    as zero. The canonical Survey and the baseline are not modified.
    """

    if not snapshot.compatible_with(sof):
        raise SnapshotUnavailableError(
            f"{snapshot.name!r} was saved for a different observability field."
        )

    masks = {}
    missing = []
    for key in snapshot.deactivated_keys:
        stored = store.get(key)
        if stored is None:
            missing.append(key.sampling_unit_id)
        else:
            masks[key] = stored.visibility_mask
    if missing:
        raise SnapshotUnavailableError(
            "Cached visibility is missing for deactivated units: "
            + ", ".join(missing[:8])
            + ". Rebuild the observability field first."
        )

    scenario = SurveyDesignScenario(sof)
    if masks:
        scenario.deactivate(masks)

    needing = []
    for definition in snapshot.candidates:
        viewpoint = replace(definition.viewpoint)
        scenario.add_candidate(viewpoint)
        candidate_id = viewpoint.viewpoint_id
        stored = None if definition.key is None else store.get(definition.key)
        if definition.status == CandidateStatus.READY and stored is not None:
            scenario.set_candidate_visibility(
                candidate_id,
                key=definition.key,
                mask=stored.visibility_mask,
                include=definition.included,
            )
        elif definition.status in (CandidateStatus.READY, CandidateStatus.COMPUTING,
                                   CandidateStatus.PENDING):
            candidate = scenario.candidate(candidate_id)
            candidate.include_when_ready = definition.included
            candidate.message = "Visibility must be recomputed."
            needing.append(candidate_id)
        else:
            scenario.set_candidate_status(candidate_id, definition.status, definition.message)

    return scenario, RestoreReport(
        deactivated_units=len(masks),
        candidates_restored=len(snapshot.candidates),
        candidates_needing_computation=tuple(needing),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize(sof, exposure, active_units) -> CoverageSummary:
    return summarize_exposure(
        exposure,
        analysis_mask=sof.analysis_mask,
        valid_mask=sof.valid_mask,
        active_units=active_units,
    )


def _read_only(array: np.ndarray) -> np.ndarray:
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


def compatible_snapshots(
    snapshots: Iterable[ScenarioSnapshot],
    sof: SurveyObservabilityField | None,
) -> tuple[ScenarioSnapshot, ...]:
    return tuple(snapshot for snapshot in snapshots if snapshot.compatible_with(sof))
