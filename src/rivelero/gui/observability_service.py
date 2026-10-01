"""Observability build and inspection services for the Rivelero GUI.

This module connects ApplicationState to the existing scientific backend:

    ApplicationState
        -> VisibilityStore (disk + LRU cache)
        -> build_survey_observability_field(...)
        -> SOFBuildResult
        -> ApplicationState.set_observability_result(...)

It contains no viewshed, exposure or mask mathematics; those belong to
rivelero.visibility and rivelero.observability. Like domain_service, the
module is deliberately independent of Qt so the orchestration can be tested
without a running event loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import os
import sys

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.domain import AnalysisDomain
from rivelero.core.environment import Environment
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.application_state import ApplicationState
from rivelero.observability.builder import (
    SOFBuildReport,
    SOFBuildResult,
    make_visibility_key,
    visibility_context_fingerprint,
)
from rivelero.observability.storage import (
    StoredVisibility,
    VisibilityKey,
    VisibilityStore,
)
from rivelero.visibility.configuration import (
    SamplingUnit,
    VisibilityConfiguration,
)
from rivelero.core.domain import AnalysisGrid
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coordinates
from rivelero.visibility.engine import (
    ViewpointExcludedError,
    compute_viewpoint_visibility,
)


CACHE_DIRECTORY_ENVIRONMENT_VARIABLE = "RIVELERO_CACHE_DIR"

# Masks retained in RAM. Each mask is one byte per grid cell, so 64 masks of
# a 2000 x 2000 grid occupy roughly 256 MB.
DEFAULT_MEMORY_ITEMS = 64


class ObservabilityNotReadyError(RuntimeError):
    """Raised when upstream prerequisites for an SOF build are missing."""

    def __init__(self, blockers: list[str]) -> None:
        self.blockers = list(blockers)
        super().__init__(" ".join(self.blockers))


# ---------------------------------------------------------------------------
# VisibilityStore configuration
# ---------------------------------------------------------------------------


def default_visibility_cache_directory() -> Path:
    """Return the per-user directory used for cached visibility masks.

    ``RIVELERO_CACHE_DIR`` overrides the platform default. The cache is
    shared between sessions: visibility keys include a fingerprint of every
    scientific input, so only compatible results are ever reused.
    """

    override = os.environ.get(CACHE_DIRECTORY_ENVIRONMENT_VARIABLE)

    if override:
        root = Path(override)

    elif sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "Rivelero"

    else:
        root = Path(
            os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        ) / "rivelero"

    return (root / "visibility_cache").expanduser()


def create_visibility_store(
    cache_directory: Path | str | None = None,
    *,
    max_memory_items: int = DEFAULT_MEMORY_ITEMS,
    compressed: bool = True,
) -> VisibilityStore:
    """Create a VisibilityStore using the GUI defaults."""

    return VisibilityStore(
        (
            default_visibility_cache_directory()
            if cache_directory is None
            else cache_directory
        ),
        max_memory_items=max_memory_items,
        compressed=compressed,
    )


def store_settings_match(
    store: VisibilityStore,
    *,
    cache_directory: Path | str,
    max_memory_items: int,
    compressed: bool,
) -> bool:
    """Whether an existing store already uses the requested settings."""

    return (
        store.cache_directory
        == Path(cache_directory).expanduser().resolve()
        and store.max_memory_items == max_memory_items
        and store.compressed == compressed
    )


# ---------------------------------------------------------------------------
# SOF build
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservabilityBuildRequest:
    """Snapshot of canonical inputs for one background SOF build.

    The snapshot references the canonical objects held by ApplicationState at
    the time the build starts. ``inputs_revision`` lets the state reject the
    result if any input changes before the build completes.
    """

    sof_id: str

    viewpoint_configuration: ViewpointConfiguration
    environment: Environment
    domain: AnalysisDomain
    visibility_configuration: VisibilityConfiguration
    store: VisibilityStore
    sensors: Mapping[str, Sensor]

    continue_on_error: bool
    inputs_revision: int

    @property
    def total_units(self) -> int:
        """Number of sampling units the builder will request."""

        if self.visibility_configuration.is_event_based:
            return len(self.viewpoint_configuration.observation_events)

        return len(self.viewpoint_configuration.viewpoints)

    def build_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for build_survey_observability_field."""

        return {
            "sof_id": self.sof_id,
            "viewpoint_configuration": self.viewpoint_configuration,
            "environment": self.environment,
            "domain": self.domain,
            "visibility_configuration": self.visibility_configuration,
            "store": self.store,
            "sensors": self.sensors,
            # Units excluded by an EXCLUDE policy are recorded, not fatal.
            "skip_excluded": True,
            "continue_on_error": self.continue_on_error,
            "metadata": {
                "created_by": "rivelero.gui",
                "visibility_configuration_name": (
                    self.visibility_configuration.name
                ),
            },
        }


def prepare_build(
    state: ApplicationState,
    *,
    continue_on_error: bool = False,
) -> ObservabilityBuildRequest:
    """Capture the current canonical inputs for an SOF build."""

    blockers = state.observability_build_blockers()

    if blockers:
        raise ObservabilityNotReadyError(blockers)

    return ObservabilityBuildRequest(
        sof_id=f"sof_{uuid4().hex}",
        viewpoint_configuration=state.survey.viewpoint_configuration,
        environment=state.analysis.environment,
        domain=state.analysis.analysis_domain,
        visibility_configuration=state.analysis.visibility_configuration,
        store=state.analysis.visibility_store,
        sensors=dict(state.survey.sensors),
        continue_on_error=bool(continue_on_error),
        inputs_revision=state.analysis.inputs_revision,
    )


def install_build_result(
    state: ApplicationState,
    request: ObservabilityBuildRequest,
    result: SOFBuildResult,
) -> None:
    """Install a completed build through the canonical state API.

    Raises StaleObservabilityResultError when inputs changed mid-build.
    """

    if not isinstance(result, SOFBuildResult):
        raise TypeError("result must be an SOFBuildResult.")

    state.set_observability_result(
        result.sof,
        build_report=result.report,
        inputs_revision=request.inputs_revision,
    )


# ---------------------------------------------------------------------------
# Individual visibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SamplingUnitSelection:
    """One resolved sampling unit and its visibility cache identity."""

    viewpoint: Viewpoint
    event: ObservationEvent | None
    sensor: Sensor | None
    key: VisibilityKey

    @property
    def unit_id(self) -> str:
        return self.key.sampling_unit_id


class VisibilityKeyResolver:
    """Build VisibilityKeys for the current analysis without rehashing.

    The analysis-wide fingerprint hashes the AnalysisDomain masks, which is
    too costly to repeat on every selection change. The resolver keeps a
    strong reference to the objects it fingerprinted and recomputes only when
    ApplicationState holds different Environment, AnalysisDomain or
    VisibilityConfiguration objects.
    """

    def __init__(self) -> None:
        self._inputs: tuple[Any, Any, Any] | None = None
        self._fingerprint: str | None = None

    def context_fingerprint(
        self,
        *,
        environment: Environment,
        domain: AnalysisDomain,
        visibility_configuration: VisibilityConfiguration,
    ) -> str:
        inputs = (environment, domain, visibility_configuration)

        if (
            self._inputs is None
            or any(
                cached is not current
                for cached, current in zip(self._inputs, inputs)
            )
        ):
            self._fingerprint = visibility_context_fingerprint(
                environment=environment,
                domain=domain,
                visibility_configuration=visibility_configuration,
            )
            self._inputs = inputs

        return self._fingerprint

    def resolve(
        self,
        state: ApplicationState,
        *,
        viewpoint_id: str,
        event_id: str | None = None,
    ) -> SamplingUnitSelection:
        """Resolve the selected Viewpoint/ObservationEvent to a cache key.

        Raises ValueError with a user-facing message when the selection
        cannot be mapped to a sampling unit of the current configuration.
        """

        survey = state.survey.viewpoint_configuration
        environment = state.analysis.environment
        domain = state.analysis.analysis_domain
        configuration = state.analysis.visibility_configuration

        if (
            survey is None
            or environment is None
            or domain is None
            or configuration is None
        ):
            raise ValueError(
                "Survey, World and a visibility configuration are required."
            )

        viewpoint = survey.get_viewpoint(viewpoint_id)

        event = None

        if configuration.sampling_unit == SamplingUnit.OBSERVATION_EVENT:
            if event_id is None:
                raise ValueError(
                    "Visibility is sampled per ObservationEvent; choose one "
                    "of this Viewpoint's events."
                )

            event = survey.get_event(event_id)

            if event.viewpoint_id != viewpoint.viewpoint_id:
                raise ValueError(
                    "The selected ObservationEvent references a different "
                    "Viewpoint."
                )

        return self.resolve_viewpoint(state, viewpoint, event=event)

    def resolve_viewpoint(
        self,
        state: ApplicationState,
        viewpoint: Viewpoint,
        *,
        event: ObservationEvent | None = None,
    ) -> SamplingUnitSelection:
        """Resolve any canonical Viewpoint, including design candidates.

        Uses the same fingerprinted key construction as SOF builds, so a
        candidate never reuses a mask computed for different coordinates,
        orientation, field of view, height or Sensor.
        """

        environment = state.analysis.environment
        domain = state.analysis.analysis_domain
        configuration = state.analysis.visibility_configuration

        if environment is None or domain is None or configuration is None:
            raise ValueError(
                "World and a visibility configuration are required."
            )

        sensor = None

        if viewpoint.sensor_id is not None:
            sensor = state.survey.sensors.get(viewpoint.sensor_id)

            if sensor is None:
                raise ValueError(
                    f"Viewpoint {viewpoint.viewpoint_id!r} references "
                    f"Sensor {viewpoint.sensor_id!r}, which is not defined."
                )

        key = make_visibility_key(
            viewpoint=viewpoint,
            event=event,
            sensor=sensor,
            environment=environment,
            domain=domain,
            visibility_configuration=configuration,
            context_fingerprint=self.context_fingerprint(
                environment=environment,
                domain=domain,
                visibility_configuration=configuration,
            ),
        )

        return SamplingUnitSelection(
            viewpoint=viewpoint,
            event=event,
            sensor=sensor,
            key=key,
        )


def compute_unit_visibility(
    *,
    selection: SamplingUnitSelection,
    environment: Environment,
    domain: AnalysisDomain,
    visibility_configuration: VisibilityConfiguration,
    store: VisibilityStore,
) -> StoredVisibility:
    """Return one unit's visibility through the same lazy store pathway.

    A cached result is returned without calling GDAL; otherwise the engine
    computes it and the store persists it for later SOF builds.
    """

    return store.get_or_compute(
        selection.key,
        compute=lambda: compute_viewpoint_visibility(
            viewpoint=selection.viewpoint,
            event=selection.event,
            sensor=selection.sensor,
            environment=environment,
            domain=domain,
            configuration=visibility_configuration,
        ),
    )


def candidate_terrain_problem(
    viewpoint: Viewpoint,
    grid: AnalysisGrid,
) -> str | None:
    """Explain why a candidate cannot be evaluated on the terrain, if so.

    The observer may lie outside the AnalysisDomain (it can look into it),
    but the GDAL viewshed requires the observer on the elevation raster.
    """

    source_crs = CRS.from_user_input(viewpoint.crs)
    if source_crs == grid.crs:
        xs, ys = [viewpoint.x], [viewpoint.y]
    else:
        xs, ys = transform_coordinates(source_crs, grid.crs, [viewpoint.x], [viewpoint.y])

    left, top = grid.transform * (0, 0)
    right, bottom = grid.transform * (grid.width, grid.height)
    x, y = float(xs[0]), float(ys[0])
    if not (min(left, right) <= x <= max(left, right)
            and min(top, bottom) <= y <= max(top, bottom)):
        return (
            "The candidate lies outside the terrain raster. The current "
            "visibility engine needs the observer on the elevation model "
            "(it may lie outside the AnalysisDomain)."
        )
    return None


@dataclass(frozen=True, slots=True)
class CandidateVisibilityOutcome:
    """Result of computing one candidate's visibility.

    ``excluded`` means a missing-metadata policy excluded the candidate;
    computation failures are raised instead, so they cannot be mistaken
    for a zero-visibility result.
    """

    key: VisibilityKey
    stored: StoredVisibility | None
    excluded: bool = False
    message: str | None = None


def compute_candidate_visibility(
    *,
    selection: SamplingUnitSelection,
    environment: Environment,
    domain: AnalysisDomain,
    visibility_configuration: VisibilityConfiguration,
    store: VisibilityStore,
) -> CandidateVisibilityOutcome:
    """Lazy, cached candidate visibility through the canonical engine."""

    try:
        stored = compute_unit_visibility(
            selection=selection,
            environment=environment,
            domain=domain,
            visibility_configuration=visibility_configuration,
            store=store,
        )
    except ViewpointExcludedError as exc:
        return CandidateVisibilityOutcome(
            key=selection.key, stored=None, excluded=True, message=str(exc)
        )
    return CandidateVisibilityOutcome(key=selection.key, stored=stored)


def load_cached_masks(
    *,
    store: VisibilityStore,
    keys: list[VisibilityKey],
    progress_callback=None,
) -> dict[VisibilityKey, Any]:
    """Read baseline unit masks from the cache (never computes).

    Raises KeyError naming the units whose masks are missing, so a missing
    mask can never be treated as an empty one.
    """

    masks = {}
    missing = []
    total = len(keys)
    for index, key in enumerate(keys, start=1):
        stored = store.get(key)
        if stored is None:
            missing.append(key.sampling_unit_id)
        else:
            masks[key] = stored.visibility_mask
        if progress_callback is not None:
            progress_callback(index, total, key.sampling_unit_id)
    if missing:
        raise KeyError(
            "No cached visibility for: " + ", ".join(missing[:10])
            + (" …" if len(missing) > 10 else "")
            + ". Rebuild the observability field to restore it."
        )
    return masks


def unit_outcome_in_report(
    report: SOFBuildReport | None,
    unit_id: str,
) -> tuple[str, str] | None:
    """Return ("excluded" | "failed", message) if the report records one."""

    if not isinstance(report, SOFBuildReport):
        return None

    if unit_id in report.excluded_ids:
        return "excluded", report.errors.get(unit_id, "")

    if unit_id in report.failed_ids:
        return "failed", report.errors.get(unit_id, "")

    return None
