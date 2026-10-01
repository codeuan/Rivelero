"""Deterministic objective components for future survey-design search (A4).

This module deliberately defines *no objective*. It names the measurable,
deterministic quantities of a design state so that a future optimiser can
combine them with a user-defined objective and constraints, for example
"maximise observable cells subject to at most N sampling units" or a
multi-objective trade-off of coverage against acquisition cost.

Nothing here assigns a value to coverage, a cost to a Viewpoint or a penalty
to repeated coverage. Components are counts and shares under the current
deterministic observability model; probabilistic counterparts (expected
coverage, probability of loss, uncertainty reduction) should be added as
separately named components rather than replace these.

Per-change evaluation uses the stable primitives
``rivelero.analysis.scenario.marginal_gain`` and ``marginal_loss``, which
evaluate one candidate or unit against any exposure raster without touching
a scenario, the Survey or the VisibilityStore.
"""

from __future__ import annotations

from enum import Enum

from rivelero.analysis.comparison import DesignState
from rivelero.analysis.scenario import (  # re-exported primitives
    MarginalEffect,
    marginal_gain,
    marginal_loss,
)

__all__ = [
    "ObjectiveComponent",
    "objective_components",
    "MarginalEffect",
    "marginal_gain",
    "marginal_loss",
]


class ObjectiveComponent(str, Enum):
    """Deterministic, measurable properties of a design state."""

    OBSERVABLE_CELLS = "observable_cells"
    BLIND_CELLS = "blind_cells"
    COVERAGE_SHARE = "coverage_share"            # observable / analysable
    UNIQUE_COVERAGE_CELLS = "unique_coverage_cells"
    REPEATED_COVERAGE_CELLS = "repeated_coverage_cells"
    MEAN_EXPOSURE = "mean_exposure"              # over analysable cells
    MAXIMUM_EXPOSURE = "maximum_exposure"
    ACTIVE_EXISTING_UNITS = "active_existing_units"
    INCLUDED_CANDIDATES = "included_candidates"
    SAMPLING_UNITS = "sampling_units"            # existing + candidates


def objective_components(state: DesignState) -> dict[ObjectiveComponent, float | int | None]:
    """Return every component of ``state``; ``None`` where undefined."""

    summary = state.summary
    return {
        ObjectiveComponent.OBSERVABLE_CELLS: summary.observable_cells,
        ObjectiveComponent.BLIND_CELLS: summary.blind_cells,
        ObjectiveComponent.COVERAGE_SHARE: summary.observable_fraction,
        ObjectiveComponent.UNIQUE_COVERAGE_CELLS: summary.unique_cells,
        ObjectiveComponent.REPEATED_COVERAGE_CELLS: summary.repeated_cells,
        ObjectiveComponent.MEAN_EXPOSURE: summary.mean_exposure,
        ObjectiveComponent.MAXIMUM_EXPOSURE: summary.maximum_exposure,
        ObjectiveComponent.ACTIVE_EXISTING_UNITS: state.active_existing_units,
        ObjectiveComponent.INCLUDED_CANDIDATES: state.included_candidates,
        ObjectiveComponent.SAMPLING_UNITS: state.sampling_units,
    }
