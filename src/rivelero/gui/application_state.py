"""Application state for the Rivelero graphical user interface.

This module defines the central session state used by the new Rivelero GUI.

The GUI must not create parallel scientific data models. Canonical Rivelero
objects remain the scientific source of truth:

    Sensor
    ViewpointConfiguration
    Environment
    AnalysisDomain
    VisibilityConfiguration
    VisibilityStore
    SurveyObservabilityField

ApplicationState coordinates those objects and records GUI/session concerns
such as selection, task status, project metadata, and the currently displayed
layer.

A central responsibility of this module is dependency invalidation. For
example, changing the Environment or VisibilityConfiguration makes an
existing SurveyObservabilityField stale and therefore removes it from the
active analysis state.

The module deliberately has no Qt dependency. This allows state transitions
to be tested independently of the GUI framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.domain import AnalysisDomain, AnalysisGrid
from rivelero.core.environment import Environment
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.storage import (
    VisibilityKey,
    VisibilityStore,
)
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.visibility.configuration import VisibilityConfiguration


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WorkflowPage(str, Enum):
    """Primary pages of the new Rivelero GUI."""

    SURVEY = "survey"
    WORLD = "world"
    OBSERVABILITY = "observability"
    ANALYSIS_DESIGN = "analysis_design"
    OUTPUT = "output"


class VisualizationLayer(str, Enum):
    """Scientific layers that may be displayed in the main map area."""

    NONE = "none"

    # Survey
    SURVEY_OVERVIEW = "survey_overview"
    VIEWPOINTS = "viewpoints"
    OBSERVATION_SEQUENCE = "observation_sequence"

    # Individual visibility
    GEOMETRIC_VISIBILITY = "geometric_visibility"
    EFFECTIVE_VISIBILITY = "effective_visibility"
    VISIBILITY_DIFFERENCE = "visibility_difference"

    # Survey observability
    OBSERVABLE_SPACE = "observable_space"
    EXPOSURE = "exposure"
    NORMALIZED_EXPOSURE = "normalized_exposure"
    OBSERVABILITY_STATE = "observability_state"
    BLIND_SPOTS = "blind_spots"

    # World
    TERRAIN = "terrain"
    ANALYSIS_MASK = "analysis_mask"
    VALID_MASK = "valid_mask"

    # Reserved for future analysis products.
    ANALYSIS_RESULT = "analysis_result"


class TaskStatus(str, Enum):
    """Lifecycle state of a long-running GUI task."""

    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateChange(str, Enum):
    """Categories of state transition.

    These values are useful to a future Qt adapter or controller that needs
    to know which parts of the interface should refresh.
    """

    PROJECT = "project"
    SURVEY = "survey"
    ENVIRONMENT = "environment"
    DOMAIN = "domain"
    VISIBILITY_CONFIGURATION = "visibility_configuration"
    VISIBILITY_STORE = "visibility_store"
    OBSERVABILITY = "observability"
    SELECTION = "selection"
    VIEW = "view"
    TASK = "task"
    LEGACY_CONTEXT = "legacy_context"


class StaleObservabilityResultError(ValueError):
    """Raised when an SOF was built from inputs that have since changed."""


# Task name used for SOF builds so status displays can recognise them.
OBSERVABILITY_BUILD_TASK_NAME = "Building observability"


# ---------------------------------------------------------------------------
# Project state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProjectState:
    """Metadata describing the current Rivelero project.

    This is GUI/project metadata rather than a scientific analysis object.
    """

    name: str = "Untitled Rivelero project"

    project_path: Path | None = None

    description: str | None = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    modified_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    dirty: bool = False

    def __post_init__(self) -> None:
        self.name = _required_string(
            "name",
            self.name,
        )

        if self.project_path is not None:
            self.project_path = Path(
                self.project_path
            ).expanduser()

        if (
            self.description is not None
            and not isinstance(self.description, str)
        ):
            raise TypeError(
                "description must be a string or None."
            )

        if not isinstance(
            self.created_at,
            datetime,
        ):
            raise TypeError(
                "created_at must be a datetime."
            )

        if not isinstance(
            self.modified_at,
            datetime,
        ):
            raise TypeError(
                "modified_at must be a datetime."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary."
            )

    def mark_dirty(self) -> None:
        """Mark the project as modified."""

        self.dirty = True
        self.modified_at = datetime.now(
            timezone.utc
        )

    def mark_saved(
        self,
        path: str | Path | None = None,
    ) -> None:
        """Mark the project as saved."""

        if path is not None:
            self.project_path = Path(
                path
            ).expanduser()

        self.dirty = False
        self.modified_at = datetime.now(
            timezone.utc
        )


# ---------------------------------------------------------------------------
# Survey state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SurveyState:
    """Current survey definition used by the GUI."""

    viewpoint_configuration: ViewpointConfiguration | None = None

    sensors: dict[str, Sensor] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            self.viewpoint_configuration is not None
            and not isinstance(
                self.viewpoint_configuration,
                ViewpointConfiguration,
            )
        ):
            raise TypeError(
                "viewpoint_configuration must be a "
                "ViewpointConfiguration or None."
            )

        _validate_sensor_mapping(
            self.sensors
        )

    @property
    def has_survey(self) -> bool:
        """Whether a canonical survey configuration exists."""

        return (
            self.viewpoint_configuration
            is not None
        )

    @property
    def n_viewpoints(self) -> int:
        """Number of canonical Viewpoints."""

        if self.viewpoint_configuration is None:
            return 0

        return len(
            self.viewpoint_configuration.viewpoints
        )

    @property
    def n_observation_events(self) -> int:
        """Number of ObservationEvents."""

        if self.viewpoint_configuration is None:
            return 0

        return len(
            self.viewpoint_configuration.observation_events
        )

    @property
    def n_sensors(self) -> int:
        """Number of Sensors available to the survey."""

        return len(
            self.sensors
        )


# ---------------------------------------------------------------------------
# Analysis state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnalysisState:
    """Current scientific analysis definition and derived SOF."""

    environment: Environment | None = None

    analysis_grid: AnalysisGrid | None = None

    analysis_domain: AnalysisDomain | None = None

    visibility_configuration: VisibilityConfiguration | None = None

    visibility_store: VisibilityStore | None = None

    survey_observability_field: SurveyObservabilityField | None = None

    build_report: Any | None = None

    # Incremented whenever an SOF input changes. A background build records
    # the revision it started from so that a result computed from inputs
    # edited mid-build is never installed as current.
    inputs_revision: int = 0

    # Why the most recent SOF was discarded, if it was discarded because an
    # upstream dependency changed. Cleared when a new SOF is installed.
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            self.environment is not None
            and not isinstance(
                self.environment,
                Environment,
            )
        ):
            raise TypeError(
                "environment must be an Environment or None."
            )

        if (
            self.analysis_domain is not None
            and not isinstance(
                self.analysis_domain,
                AnalysisDomain,
            )
        ):
            raise TypeError(
                "analysis_domain must be an AnalysisDomain or None."
            )

        if (
            self.analysis_grid is not None
            and not isinstance(self.analysis_grid, AnalysisGrid)
        ):
            raise TypeError(
                "analysis_grid must be an AnalysisGrid or None."
            )

        if (
            self.visibility_configuration is not None
            and not isinstance(
                self.visibility_configuration,
                VisibilityConfiguration,
            )
        ):
            raise TypeError(
                "visibility_configuration must be a "
                "VisibilityConfiguration or None."
            )

        if (
            self.visibility_store is not None
            and not isinstance(
                self.visibility_store,
                VisibilityStore,
            )
        ):
            raise TypeError(
                "visibility_store must be a VisibilityStore or None."
            )

        if (
            self.survey_observability_field is not None
            and not isinstance(
                self.survey_observability_field,
                SurveyObservabilityField,
            )
        ):
            raise TypeError(
                "survey_observability_field must be a "
                "SurveyObservabilityField or None."
            )

    @property
    def has_environment(self) -> bool:
        return self.environment is not None

    @property
    def has_domain(self) -> bool:
        return self.analysis_domain is not None

    @property
    def has_visibility_configuration(self) -> bool:
        return (
            self.visibility_configuration
            is not None
        )

    @property
    def has_store(self) -> bool:
        return self.visibility_store is not None

    @property
    def has_sof(self) -> bool:
        return (
            self.survey_observability_field
            is not None
        )


# ---------------------------------------------------------------------------
# Selection state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SelectionState:
    """Current user selection within the scientific project."""

    sampling_unit_id: str | None = None

    viewpoint_id: str | None = None

    observation_event_id: str | None = None

    visibility_key: VisibilityKey | None = None

    def __post_init__(self) -> None:
        for name in (
            "sampling_unit_id",
            "viewpoint_id",
            "observation_event_id",
        ):
            value = getattr(
                self,
                name,
            )

            if (
                value is not None
                and not isinstance(
                    value,
                    str,
                )
            ):
                raise TypeError(
                    f"{name} must be a string or None."
                )

        if (
            self.visibility_key is not None
            and not isinstance(
                self.visibility_key,
                VisibilityKey,
            )
        ):
            raise TypeError(
                "visibility_key must be a VisibilityKey or None."
            )

    def clear(self) -> None:
        """Clear all current selections."""

        self.sampling_unit_id = None
        self.viewpoint_id = None
        self.observation_event_id = None
        self.visibility_key = None


# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaskState:
    """State of the current long-running GUI task."""

    status: TaskStatus = TaskStatus.IDLE

    task_id: str | None = None

    task_name: str | None = None

    processed: int = 0
    total: int | None = None

    current_unit_id: str | None = None

    message: str | None = None

    error_message: str | None = None
    error_details: str | None = None

    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def busy(self) -> bool:
        """Whether a task currently owns the application worker."""

        return self.status in {
            TaskStatus.RUNNING,
            TaskStatus.CANCELLING,
        }

    @property
    def progress_fraction(self) -> float | None:
        """Progress expressed from 0 to 1 when a total is known."""

        if (
            self.total is None
            or self.total <= 0
        ):
            return None

        return min(
            1.0,
            max(
                0.0,
                self.processed / self.total,
            ),
        )

    def reset(self) -> None:
        """Return task state to idle."""

        self.status = TaskStatus.IDLE
        self.task_id = None
        self.task_name = None
        self.processed = 0
        self.total = None
        self.current_unit_id = None
        self.message = None
        self.error_message = None
        self.error_details = None
        self.started_at = None
        self.finished_at = None


# ---------------------------------------------------------------------------
# View state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ViewState:
    """Non-scientific GUI display state."""

    active_page: WorkflowPage = WorkflowPage.SURVEY

    active_layer: VisualizationLayer = VisualizationLayer.NONE

    map_extent: tuple[
        float,
        float,
        float,
        float,
    ] | None = None

    show_viewpoint_ids: bool = False

    show_orientation: bool = True

    show_domain: bool = True

    visualization_options: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if isinstance(
            self.active_page,
            str,
        ):
            self.active_page = WorkflowPage(
                self.active_page
            )

        if isinstance(
            self.active_layer,
            str,
        ):
            self.active_layer = VisualizationLayer(
                self.active_layer
            )

        if self.map_extent is not None:
            self.map_extent = _validate_extent(
                self.map_extent
            )

        if not isinstance(
            self.visualization_options,
            dict,
        ):
            raise TypeError(
                "visualization_options must be a dictionary."
            )


# ---------------------------------------------------------------------------
# Optional legacy/context state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OptionalContextState:
    """Optional products that are not part of the core SOF definition.

    These fields preserve a clean location for Connor's existing suitability,
    obstacle, and OPF functionality while keeping them outside the canonical
    Stage-3 observability state.
    """

    botanical_field: Any | None = None
    obstacle_field: Any | None = None
    legacy_opf: Any | None = None

    extra_products: dict[str, Any] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ApplicationState:
    """Central state container for the new Rivelero GUI.

    Notes
    -----
    ApplicationState coordinates canonical scientific objects but does not
    replace them.

    Scientific dependencies are invalidated conservatively:

    Survey changes
        invalidate SOF and build report.

    Environment changes
        invalidate SOF and build report.

    AnalysisDomain changes
        invalidate SOF and build report.

    VisibilityConfiguration changes
        invalidate SOF and build report.

    VisibilityStore changes
        do not invalidate the scientific SOF itself, because the store is a
        computational cache rather than a scientific input.

    Pure selections and visualization changes
        never invalidate scientific results.
    """

    project: ProjectState = field(
        default_factory=ProjectState
    )

    survey: SurveyState = field(
        default_factory=SurveyState
    )

    analysis: AnalysisState = field(
        default_factory=AnalysisState
    )

    selection: SelectionState = field(
        default_factory=SelectionState
    )

    task: TaskState = field(
        default_factory=TaskState
    )

    view: ViewState = field(
        default_factory=ViewState
    )

    optional_context: OptionalContextState = field(
        default_factory=OptionalContextState
    )

    # Monotonically increasing state revision.
    revision: int = 0

    # Last categories changed. A future Qt adapter may consume this and emit
    # corresponding signals.
    last_changes: tuple[StateChange, ...] = field(
        default_factory=tuple
    )

    # ------------------------------------------------------------------
    # High-level readiness
    # ------------------------------------------------------------------

    @property
    def survey_ready(self) -> bool:
        """Whether a survey exists and contains at least one Viewpoint."""

        return (
            self.survey.has_survey
            and self.survey.n_viewpoints > 0
        )

    @property
    def world_ready(self) -> bool:
        """Whether compatible Environment, grid and domain are available."""

        return (
            self.analysis.has_environment
            and self.analysis.analysis_grid is not None
            and self.analysis.has_domain
            and _grid_matches_domain(
                self.analysis.analysis_grid,
                self.analysis.analysis_domain,
            )
        )

    @property
    def visibility_ready(self) -> bool:
        """Whether the visibility analysis is sufficiently configured."""

        return (
            self.survey_ready
            and self.world_ready
            and self.analysis.has_visibility_configuration
            and self.analysis.has_store
        )

    @property
    def observability_ready(self) -> bool:
        """Whether a current SOF exists."""

        return (
            self.analysis.has_sof
        )

    @property
    def observability_building(self) -> bool:
        """Whether an SOF build task is currently running."""

        return (
            self.task.busy
            and self.task.task_name == OBSERVABILITY_BUILD_TASK_NAME
        )

    def observability_build_blockers(self) -> list[str]:
        """Return human-readable reasons why an SOF cannot be built yet.

        An empty list means every upstream prerequisite exists. The checks
        mirror what the builder would otherwise reject deep in the engine.
        """

        blockers: list[str] = []

        if not self.survey_ready:
            blockers.append("Import or define a Survey with at least one Viewpoint.")

        if not self.world_ready:
            blockers.append("Define terrain and an AnalysisDomain in World.")

        configuration = self.analysis.visibility_configuration

        if configuration is None:
            blockers.append("Save a visibility configuration.")

        elif (
            configuration.is_event_based
            and self.survey.n_observation_events == 0
        ):
            blockers.append(
                "The visibility configuration samples ObservationEvents, "
                "but the Survey contains none."
            )

        if not self.analysis.has_store:
            blockers.append("Configure visibility storage.")

        if self.task.busy:
            blockers.append("Wait for the running task to finish.")

        return blockers

    @property
    def export_ready(self) -> bool:
        """Whether scientific SOF products can be exported."""

        return self.observability_ready

    @property
    def busy(self) -> bool:
        """Whether a background task is active."""

        return self.task.busy

    # ------------------------------------------------------------------
    # Project transitions
    # ------------------------------------------------------------------

    def rename_project(
        self,
        name: str,
    ) -> None:
        """Rename the current project."""

        self.project.name = _required_string(
            "name",
            name,
        )

        self._mark_changed(
            StateChange.PROJECT,
            scientific=False,
        )

    def set_project_description(
        self,
        description: str | None,
    ) -> None:
        """Set optional project description."""

        if (
            description is not None
            and not isinstance(
                description,
                str,
            )
        ):
            raise TypeError(
                "description must be a string or None."
            )

        self.project.description = description

        self._mark_changed(
            StateChange.PROJECT,
            scientific=False,
        )

    # ------------------------------------------------------------------
    # Survey transitions
    # ------------------------------------------------------------------

    def set_viewpoint_configuration(
        self,
        configuration: ViewpointConfiguration | None,
    ) -> None:
        """Replace the current survey configuration.

        Any existing SOF is invalidated because its survey definition is no
        longer guaranteed to match the active configuration.
        """

        if (
            configuration is not None
            and not isinstance(
                configuration,
                ViewpointConfiguration,
            )
        ):
            raise TypeError(
                "configuration must be a "
                "ViewpointConfiguration or None."
            )

        self.survey.viewpoint_configuration = configuration

        self.selection.clear()

        self._invalidate_observability("Survey replaced.")

        self._mark_changed(
            StateChange.SURVEY,
            StateChange.SELECTION,
            StateChange.OBSERVABILITY,
        )

    def set_sensors(
        self,
        sensors: Mapping[str, Sensor],
    ) -> None:
        """Replace the Sensor lookup used by the current survey."""

        sensor_mapping = dict(
            sensors
        )

        _validate_sensor_mapping(
            sensor_mapping
        )

        self.survey.sensors = sensor_mapping

        # Sensor metadata can affect resolved visibility parameters.
        self._invalidate_observability("Sensors changed.")

        self._mark_changed(
            StateChange.SURVEY,
            StateChange.OBSERVABILITY,
        )

    def add_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Add a canonical Viewpoint to the active configuration."""

        configuration = self._require_configuration()
        updated = replace(
            configuration,
            viewpoints=[*configuration.viewpoints, viewpoint],
        )
        self._replace_survey_configuration(updated)

    def replace_viewpoint(
        self,
        viewpoint: Viewpoint,
        *,
        previous_id: str | None = None,
    ) -> None:
        """Replace one Viewpoint while preserving event order and metadata."""

        configuration = self._require_configuration()
        old_id = viewpoint.viewpoint_id if previous_id is None else previous_id
        index = next(
            (index for index, item in enumerate(configuration.viewpoints)
             if item.viewpoint_id == old_id),
            None,
        )
        if index is None:
            raise KeyError(f"Viewpoint {old_id!r} is not present.")

        viewpoints = list(configuration.viewpoints)
        viewpoints[index] = viewpoint
        events = [
            replace(event, viewpoint_id=viewpoint.viewpoint_id)
            if event.viewpoint_id == old_id else event
            for event in configuration.observation_events
        ]
        updated = replace(
            configuration,
            viewpoints=viewpoints,
            observation_events=events,
        )
        self._replace_survey_configuration(updated)
        if self.selection.viewpoint_id == old_id:
            self.select_viewpoint(viewpoint.viewpoint_id)

    def remove_viewpoint(self, viewpoint_id: str) -> None:
        """Remove an unreferenced Viewpoint from the active configuration."""

        configuration = self._require_configuration()
        references = configuration.events_for_viewpoint(viewpoint_id)
        if references:
            raise ValueError(
                f"Viewpoint {viewpoint_id!r} is referenced by "
                f"{len(references)} ObservationEvent(s)."
            )
        updated = replace(
            configuration,
            viewpoints=[
                viewpoint for viewpoint in configuration.viewpoints
                if viewpoint.viewpoint_id != viewpoint_id
            ],
        )
        self._replace_survey_configuration(updated)
        if self.selection.viewpoint_id == viewpoint_id:
            self.clear_selection()

    def set_observation_events(
        self,
        events: list[ObservationEvent],
    ) -> None:
        """Replace ordered canonical ObservationEvents."""

        configuration = self._require_configuration()
        updated = replace(
            configuration,
            observation_events=list(events),
        )
        self._replace_survey_configuration(updated)

    def add_observation_event(self, event: ObservationEvent) -> None:
        """Append one canonical ObservationEvent."""

        configuration = self._require_configuration()
        updated = replace(
            configuration,
            observation_events=[*configuration.observation_events, event],
        )
        self._replace_survey_configuration(updated)

    def replace_observation_event(self, event: ObservationEvent) -> None:
        """Replace one ObservationEvent without changing sequence order."""

        configuration = self._require_configuration()
        events = list(configuration.observation_events)
        index = next(
            (index for index, item in enumerate(events)
             if item.event_id == event.event_id),
            None,
        )
        if index is None:
            raise KeyError(f"ObservationEvent {event.event_id!r} is not present.")
        events[index] = event
        self._replace_survey_configuration(
            replace(configuration, observation_events=events)
        )

    def remove_observation_event(self, event_id: str) -> None:
        """Remove one ObservationEvent while preserving remaining order."""

        configuration = self._require_configuration()
        if event_id not in configuration.event_ids:
            raise KeyError(f"ObservationEvent {event_id!r} is not present.")
        self._replace_survey_configuration(
            replace(
                configuration,
                observation_events=[
                    event for event in configuration.observation_events
                    if event.event_id != event_id
                ],
            )
        )

    def add_sensor(self, sensor: Sensor) -> None:
        """Add a Sensor to the canonical survey sensor mapping."""

        if sensor.sensor_id in self.survey.sensors:
            raise ValueError(f"Sensor {sensor.sensor_id!r} already exists.")
        sensors = dict(self.survey.sensors)
        sensors[sensor.sensor_id] = sensor
        self.set_sensors(sensors)

    def replace_sensor(self, sensor: Sensor, *, previous_id: str | None = None) -> None:
        """Replace one Sensor and preserve Viewpoint references when renamed."""

        old_id = sensor.sensor_id if previous_id is None else previous_id
        if old_id not in self.survey.sensors:
            raise KeyError(f"Sensor {old_id!r} is not present.")
        if sensor.sensor_id != old_id and sensor.sensor_id in self.survey.sensors:
            raise ValueError(f"Sensor {sensor.sensor_id!r} already exists.")
        sensors = dict(self.survey.sensors)
        del sensors[old_id]
        sensors[sensor.sensor_id] = sensor
        configuration = self.survey.viewpoint_configuration
        if sensor.sensor_id != old_id and configuration is not None:
            configuration = replace(
                configuration,
                viewpoints=[
                    replace(viewpoint, sensor_id=sensor.sensor_id)
                    if viewpoint.sensor_id == old_id else viewpoint
                    for viewpoint in configuration.viewpoints
                ],
            )
        self.survey.sensors = sensors
        if configuration is not None:
            self.survey.viewpoint_configuration = configuration
        self._invalidate_observability("Sensors changed.")
        self._mark_changed(StateChange.SURVEY, StateChange.OBSERVABILITY)

    def remove_sensor(self, sensor_id: str) -> None:
        """Remove an unreferenced Sensor."""

        configuration = self.survey.viewpoint_configuration
        if configuration is not None:
            references = [
                viewpoint for viewpoint in configuration.viewpoints
                if viewpoint.sensor_id == sensor_id
            ]
            if references:
                raise ValueError(
                    f"Sensor {sensor_id!r} is referenced by "
                    f"{len(references)} Viewpoint(s)."
                )
        if sensor_id not in self.survey.sensors:
            raise KeyError(f"Sensor {sensor_id!r} is not present.")
        sensors = dict(self.survey.sensors)
        del sensors[sensor_id]
        self.set_sensors(sensors)

    def _require_configuration(self) -> ViewpointConfiguration:
        if self.survey.viewpoint_configuration is None:
            raise RuntimeError("No survey configuration is loaded.")
        return self.survey.viewpoint_configuration

    def _replace_survey_configuration(
        self,
        configuration: ViewpointConfiguration,
    ) -> None:
        self.survey.viewpoint_configuration = configuration
        selected = self.selection.viewpoint_id
        if selected is not None and selected not in configuration.viewpoint_ids:
            self.selection.clear()
        self._invalidate_observability("Survey changed.")
        self._mark_changed(
            StateChange.SURVEY,
            StateChange.OBSERVABILITY,
            StateChange.SELECTION,
        )

    # ------------------------------------------------------------------
    # World transitions
    # ------------------------------------------------------------------

    def set_environment(
        self,
        environment: Environment | None,
        *,
        grid: AnalysisGrid | None = None,
    ) -> None:
        """Install Environment and its grid as one World transition."""

        if (
            environment is not None
            and not isinstance(
                environment,
                Environment,
            )
        ):
            raise TypeError(
                "environment must be an Environment or None."
            )

        if environment is not None and not isinstance(grid, AnalysisGrid):
            raise TypeError(
                "grid must be an AnalysisGrid when environment is provided."
            )

        if environment is None and grid is not None:
            raise ValueError(
                "grid cannot be provided without an Environment."
            )

        self.analysis.environment = environment
        self.analysis.analysis_grid = grid
        self.analysis.analysis_domain = None

        self._invalidate_observability("Terrain changed.")

        self._mark_changed(
            StateChange.ENVIRONMENT,
            StateChange.DOMAIN,
            StateChange.OBSERVABILITY,
        )

    def set_analysis_domain(
        self,
        domain: AnalysisDomain | None,
    ) -> None:
        """Replace the active AnalysisDomain and invalidate derived SOF."""

        if (
            domain is not None
            and not isinstance(
                domain,
                AnalysisDomain,
            )
        ):
            raise TypeError(
                "domain must be an AnalysisDomain or None."
            )

        if domain is not None:
            grid = self.analysis.analysis_grid
            if grid is None:
                raise ValueError(
                    "An active AnalysisGrid is required before setting a domain."
                )
            if not _grid_matches_domain(grid, domain):
                raise ValueError(
                    "AnalysisDomain grid does not match the active AnalysisGrid."
                )

        self.analysis.analysis_domain = domain

        self._invalidate_observability("Analysis domain changed.")

        self._mark_changed(
            StateChange.DOMAIN,
            StateChange.OBSERVABILITY,
        )

    # ------------------------------------------------------------------
    # Visibility transitions
    # ------------------------------------------------------------------

    def set_visibility_configuration(
        self,
        configuration: VisibilityConfiguration | None,
    ) -> None:
        """Replace visibility assumptions and invalidate derived SOF."""

        if (
            configuration is not None
            and not isinstance(
                configuration,
                VisibilityConfiguration,
            )
        ):
            raise TypeError(
                "configuration must be a "
                "VisibilityConfiguration or None."
            )

        self.analysis.visibility_configuration = configuration

        self._invalidate_observability("Visibility configuration changed.")

        self._mark_changed(
            StateChange.VISIBILITY_CONFIGURATION,
            StateChange.OBSERVABILITY,
        )

    def set_visibility_store(
        self,
        store: VisibilityStore | None,
    ) -> None:
        """Replace the computational visibility cache.

        This does not invalidate an already constructed SOF because the
        VisibilityStore is a computational resource rather than a scientific
        analysis parameter.
        """

        if (
            store is not None
            and not isinstance(
                store,
                VisibilityStore,
            )
        ):
            raise TypeError(
                "store must be a VisibilityStore or None."
            )

        self.analysis.visibility_store = store

        self._mark_changed(
            StateChange.VISIBILITY_STORE,
        )

    # ------------------------------------------------------------------
    # SOF transitions
    # ------------------------------------------------------------------

    def set_observability_result(
        self,
        sof: SurveyObservabilityField,
        *,
        build_report: Any | None = None,
        inputs_revision: int | None = None,
    ) -> None:
        """Install a completed SurveyObservabilityField.

        The SOF is checked against the active survey/environment/domain/
        visibility configuration before it is accepted.

        Identifiers alone cannot detect in-place edits (an edited Viewpoint
        keeps its ViewpointConfiguration ID). Background builds therefore
        pass the ``inputs_revision`` observed when they started; the result
        is rejected with StaleObservabilityResultError if any SOF input has
        changed since.
        """

        if not isinstance(
            sof,
            SurveyObservabilityField,
        ):
            raise TypeError(
                "sof must be a SurveyObservabilityField."
            )

        if (
            inputs_revision is not None
            and inputs_revision != self.analysis.inputs_revision
        ):
            raise StaleObservabilityResultError(
                "Survey, World or visibility inputs changed while the "
                "observability field was being built; the result no longer "
                "describes the current analysis."
            )

        self._validate_sof_against_current_state(
            sof
        )

        self.analysis.survey_observability_field = sof
        self.analysis.build_report = build_report
        self.analysis.invalidation_reason = None

        self._mark_changed(
            StateChange.OBSERVABILITY,
        )

    def clear_observability_result(
        self,
    ) -> None:
        """Explicitly clear the current SOF and build report."""

        self._invalidate_observability()

        self._mark_changed(
            StateChange.OBSERVABILITY,
        )

    # ------------------------------------------------------------------
    # Selection transitions
    # ------------------------------------------------------------------

    def select_viewpoint(
        self,
        viewpoint_id: str | None,
    ) -> None:
        """Select a Viewpoint for inspection."""

        if viewpoint_id is not None:
            viewpoint_id = _required_string(
                "viewpoint_id",
                viewpoint_id,
            )

        self.selection.viewpoint_id = viewpoint_id
        self.selection.observation_event_id = None
        self.selection.sampling_unit_id = viewpoint_id
        self.selection.visibility_key = None

        self._mark_changed(
            StateChange.SELECTION,
            scientific=False,
        )

    def select_observation_event(
        self,
        event_id: str | None,
        *,
        viewpoint_id: str | None = None,
    ) -> None:
        """Select an ObservationEvent for inspection."""

        if event_id is not None:
            event_id = _required_string(
                "event_id",
                event_id,
            )

        if viewpoint_id is not None:
            viewpoint_id = _required_string(
                "viewpoint_id",
                viewpoint_id,
            )

        self.selection.observation_event_id = event_id
        self.selection.viewpoint_id = viewpoint_id
        self.selection.sampling_unit_id = event_id
        self.selection.visibility_key = None

        self._mark_changed(
            StateChange.SELECTION,
            scientific=False,
        )

    def select_visibility_key(
        self,
        key: VisibilityKey | None,
    ) -> None:
        """Select one stored visibility result."""

        if (
            key is not None
            and not isinstance(
                key,
                VisibilityKey,
            )
        ):
            raise TypeError(
                "key must be a VisibilityKey or None."
            )

        self.selection.visibility_key = key

        if key is not None:
            self.selection.sampling_unit_id = (
                key.sampling_unit_id
            )
            self.selection.viewpoint_id = (
                key.viewpoint_id
            )

        self._mark_changed(
            StateChange.SELECTION,
            scientific=False,
        )

    def clear_selection(
        self,
    ) -> None:
        """Clear current survey/visibility selection."""

        self.selection.clear()

        self._mark_changed(
            StateChange.SELECTION,
            scientific=False,
        )

    # ------------------------------------------------------------------
    # View transitions
    # ------------------------------------------------------------------

    def set_active_page(
        self,
        page: WorkflowPage | str,
    ) -> None:
        """Change the visible workflow page."""

        if isinstance(
            page,
            str,
        ):
            page = WorkflowPage(
                page
            )

        if not isinstance(
            page,
            WorkflowPage,
        ):
            raise TypeError(
                "page must be a WorkflowPage or valid string."
            )

        self.view.active_page = page

        self._mark_changed(
            StateChange.VIEW,
            scientific=False,
        )

    def set_active_layer(
        self,
        layer: VisualizationLayer | str,
    ) -> None:
        """Change the scientific layer displayed in the map."""

        if isinstance(
            layer,
            str,
        ):
            layer = VisualizationLayer(
                layer
            )

        if not isinstance(
            layer,
            VisualizationLayer,
        ):
            raise TypeError(
                "layer must be a VisualizationLayer or valid string."
            )

        self.view.active_layer = layer

        self._mark_changed(
            StateChange.VIEW,
            scientific=False,
        )

    def set_map_extent(
        self,
        extent: tuple[
            float,
            float,
            float,
            float,
        ] | None,
    ) -> None:
        """Store the current map extent without changing scientific state."""

        if extent is not None:
            extent = _validate_extent(
                extent
            )

        self.view.map_extent = extent

        self._mark_changed(
            StateChange.VIEW,
            scientific=False,
        )

    # ------------------------------------------------------------------
    # Task transitions
    # ------------------------------------------------------------------

    def start_task(
        self,
        *,
        task_id: str,
        task_name: str,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        """Mark a long-running task as started."""

        if self.task.busy:
            raise RuntimeError(
                "A Rivelero task is already running."
            )

        task_id = _required_string(
            "task_id",
            task_id,
        )

        task_name = _required_string(
            "task_name",
            task_name,
        )

        if total is not None:
            if not isinstance(total, int):
                raise TypeError(
                    "total must be an integer or None."
                )

            if total < 0:
                raise ValueError(
                    "total cannot be negative."
                )

        self.task.status = TaskStatus.RUNNING
        self.task.task_id = task_id
        self.task.task_name = task_name
        self.task.processed = 0
        self.task.total = total
        self.task.current_unit_id = None
        self.task.message = message
        self.task.error_message = None
        self.task.error_details = None
        self.task.started_at = datetime.now(
            timezone.utc
        )
        self.task.finished_at = None

        self._mark_changed(
            StateChange.TASK,
            scientific=False,
        )

    def update_task_progress(
        self,
        *,
        processed: int,
        total: int | None = None,
        unit_id: str | None = None,
        message: str | None = None,
    ) -> None:
        """Update progress for the currently running task."""

        if not self.task.busy:
            raise RuntimeError(
                "No Rivelero task is currently running."
            )

        if not isinstance(
            processed,
            int,
        ):
            raise TypeError(
                "processed must be an integer."
            )

        if processed < 0:
            raise ValueError(
                "processed cannot be negative."
            )

        if total is not None:
            if not isinstance(total, int):
                raise TypeError(
                    "total must be an integer or None."
                )

            if total < 0:
                raise ValueError(
                    "total cannot be negative."
                )

            self.task.total = total

        if (
            self.task.total is not None
            and processed > self.task.total
        ):
            raise ValueError(
                "processed cannot exceed total."
            )

        self.task.processed = processed
        self.task.current_unit_id = unit_id

        if message is not None:
            self.task.message = message

        self._mark_changed(
            StateChange.TASK,
            scientific=False,
            mark_project_dirty=False,
        )

    def request_task_cancellation(
        self,
    ) -> None:
        """Record a cancellation request.

        Actual interruption is the responsibility of TaskController.
        """

        if self.task.status != TaskStatus.RUNNING:
            raise RuntimeError(
                "No cancellable task is currently running."
            )

        self.task.status = TaskStatus.CANCELLING
        self.task.message = "Cancellation requested."

        self._mark_changed(
            StateChange.TASK,
            scientific=False,
            mark_project_dirty=False,
        )

    def finish_task(
        self,
        *,
        message: str | None = None,
    ) -> None:
        """Mark the current task as successfully completed."""

        if not self.task.busy:
            raise RuntimeError(
                "No Rivelero task is currently running."
            )

        self.task.status = TaskStatus.SUCCEEDED
        self.task.finished_at = datetime.now(
            timezone.utc
        )

        if message is not None:
            self.task.message = message

        self._mark_changed(
            StateChange.TASK,
            scientific=False,
            mark_project_dirty=False,
        )

    def fail_task(
        self,
        *,
        error_message: str,
        error_details: str | None = None,
    ) -> None:
        """Mark the current task as failed."""

        if not self.task.busy:
            raise RuntimeError(
                "No Rivelero task is currently running."
            )

        self.task.status = TaskStatus.FAILED
        self.task.error_message = _required_string(
            "error_message",
            error_message,
        )
        self.task.error_details = error_details
        self.task.finished_at = datetime.now(
            timezone.utc
        )

        self._mark_changed(
            StateChange.TASK,
            scientific=False,
            mark_project_dirty=False,
        )

    def mark_task_cancelled(
        self,
        *,
        message: str | None = None,
    ) -> None:
        """Mark the current task as cancelled."""

        if not self.task.busy:
            raise RuntimeError(
                "No Rivelero task is currently running."
            )

        self.task.status = TaskStatus.CANCELLED
        self.task.finished_at = datetime.now(
            timezone.utc
        )

        if message is not None:
            self.task.message = message

        self._mark_changed(
            StateChange.TASK,
            scientific=False,
            mark_project_dirty=False,
        )

    def reset_task(
        self,
    ) -> None:
        """Reset a completed/failed/cancelled task to idle."""

        if self.task.busy:
            raise RuntimeError(
                "Cannot reset a running task."
            )

        self.task.reset()

        self._mark_changed(
            StateChange.TASK,
            scientific=False,
            mark_project_dirty=False,
        )

    # ------------------------------------------------------------------
    # Status summaries for the future GUI
    # ------------------------------------------------------------------

    def readiness_summary(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Return a GUI-friendly summary of project readiness.

        This is intentionally data-only. Presentation, icons, colors, and
        wording remain responsibilities of the GUI widgets.
        """

        return {
            "survey": {
                "ready": self.survey_ready,
                "viewpoints": self.survey.n_viewpoints,
                "observation_events": (
                    self.survey.n_observation_events
                ),
                "sensors": self.survey.n_sensors,
            },
            "world": {
                "ready": self.world_ready,
                "environment": (
                    self.analysis.environment is not None
                ),
                "analysis_domain": (
                    self.analysis.analysis_domain is not None
                ),
            },
            "visibility": {
                "ready": self.visibility_ready,
                "configuration": (
                    self.analysis.visibility_configuration
                    is not None
                ),
                "store": (
                    self.analysis.visibility_store
                    is not None
                ),
            },
            "observability": {
                "ready": self.observability_ready,
                "building": self.observability_building,
                "processed": (
                    self.task.processed
                    if self.observability_building
                    else 0
                ),
                "total": (
                    self.task.total
                    if self.observability_building
                    else None
                ),
                "invalidated": (
                    self.analysis.invalidation_reason is not None
                    and not self.observability_ready
                ),
                "active_units": (
                    0
                    if self.analysis.survey_observability_field
                    is None
                    else (
                        self.analysis
                        .survey_observability_field
                        .n_active_units
                    )
                ),
            },
            "export": {
                "ready": self.export_ready,
            },
        }

    # ------------------------------------------------------------------
    # New project / reset
    # ------------------------------------------------------------------

    def new_project(
        self,
        *,
        name: str = "Untitled Rivelero project",
    ) -> None:
        """Reset the complete session to a new project."""

        if self.task.busy:
            raise RuntimeError(
                "Cannot create a new project while a task is running."
            )

        self.project = ProjectState(
            name=name
        )

        self.survey = SurveyState()
        self.analysis = AnalysisState()
        self.selection = SelectionState()
        self.task = TaskState()
        self.view = ViewState()
        self.optional_context = OptionalContextState()

        self.revision += 1

        self.last_changes = tuple(
            StateChange
        )

    # ------------------------------------------------------------------
    # Internal dependency management
    # ------------------------------------------------------------------

    def _invalidate_observability(
        self,
        reason: str | None = None,
    ) -> None:
        """Invalidate survey-level derived observability products.

        ``reason`` describes the upstream change. It is recorded only when a
        result actually existed, so the Observability page can tell the user
        that a previous result was discarded rather than never built.
        """

        self.analysis.inputs_revision += 1

        if reason is not None and self.analysis.has_sof:
            self.analysis.invalidation_reason = reason

        self.analysis.survey_observability_field = None
        self.analysis.build_report = None

        self.selection.visibility_key = None

        if self.view.active_layer in {
            VisualizationLayer.GEOMETRIC_VISIBILITY,
            VisualizationLayer.EFFECTIVE_VISIBILITY,
            VisualizationLayer.VISIBILITY_DIFFERENCE,
            VisualizationLayer.EXPOSURE,
            VisualizationLayer.NORMALIZED_EXPOSURE,
            VisualizationLayer.OBSERVABILITY_STATE,
            VisualizationLayer.BLIND_SPOTS,
        }:
            self.view.active_layer = VisualizationLayer.NONE

    def _validate_sof_against_current_state(
        self,
        sof: SurveyObservabilityField,
    ) -> None:
        """Ensure a completed SOF belongs to the active analysis."""

        if self.survey.viewpoint_configuration is None:
            raise RuntimeError(
                "Cannot install an SOF without an active "
                "ViewpointConfiguration."
            )

        if self.analysis.environment is None:
            raise RuntimeError(
                "Cannot install an SOF without an active Environment."
            )

        if self.analysis.analysis_domain is None:
            raise RuntimeError(
                "Cannot install an SOF without an active AnalysisDomain."
            )

        if self.analysis.visibility_configuration is None:
            raise RuntimeError(
                "Cannot install an SOF without an active "
                "VisibilityConfiguration."
            )

        if (
            sof.viewpoint_configuration_id
            != self.survey.viewpoint_configuration.configuration_id
        ):
            raise ValueError(
                "SOF ViewpointConfiguration does not match "
                "the active survey."
            )

        if (
            sof.environment_id
            != self.analysis.environment.environment_id
        ):
            raise ValueError(
                "SOF Environment does not match the active Environment."
            )

        if (
            sof.analysis_domain_id
            != self.analysis.analysis_domain.domain_id
        ):
            raise ValueError(
                "SOF AnalysisDomain does not match the active domain."
            )

        if (
            sof.visibility_configuration_id
            != (
                self.analysis
                .visibility_configuration
                .configuration_id
            )
        ):
            raise ValueError(
                "SOF VisibilityConfiguration does not match "
                "the active visibility configuration."
            )

    def _mark_changed(
        self,
        *changes: StateChange,
        scientific: bool = True,
        mark_project_dirty: bool = True,
    ) -> None:
        """Record one application-state transition."""

        unique_changes: list[
            StateChange
        ] = []

        for change in changes:
            if not isinstance(
                change,
                StateChange,
            ):
                raise TypeError(
                    "changes must contain StateChange values."
                )

            if change not in unique_changes:
                unique_changes.append(
                    change
                )

        self.revision += 1
        self.last_changes = tuple(
            unique_changes
        )

        # Both scientific and ordinary editable project changes should mark
        # the project dirty. Pure task/view/selection updates explicitly opt
        # out through mark_project_dirty=False where appropriate.
        if mark_project_dirty:
            self.project.mark_dirty()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _grid_matches_domain(
    grid: AnalysisGrid | None,
    domain: AnalysisDomain | None,
) -> bool:
    if grid is None or domain is None:
        return False
    domain_grid = domain.grid
    return (
        domain_grid.crs == grid.crs
        and domain_grid.transform == grid.transform
        and domain_grid.width == grid.width
        and domain_grid.height == grid.height
        and domain_grid.resolution_x == grid.resolution_x
        and domain_grid.resolution_y == grid.resolution_y
    )


def _validate_sensor_mapping(
    sensors: Mapping[str, Sensor],
) -> None:
    """Validate canonical Sensor lookup."""

    if not isinstance(
        sensors,
        Mapping,
    ):
        raise TypeError(
            "sensors must be a mapping."
        )

    for sensor_id, sensor in sensors.items():

        if not isinstance(
            sensor_id,
            str,
        ):
            raise TypeError(
                "Sensor mapping keys must be strings."
            )

        if not isinstance(
            sensor,
            Sensor,
        ):
            raise TypeError(
                "Sensor mapping values must be Sensor objects."
            )

        if sensor_id != sensor.sensor_id:
            raise ValueError(
                f"Sensor mapping key {sensor_id!r} does not "
                f"match Sensor.sensor_id {sensor.sensor_id!r}."
            )


def _validate_extent(
    extent: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Validate map extent as left, right, bottom, top."""

    if (
        not isinstance(
            extent,
            tuple,
        )
        or len(extent) != 4
    ):
        raise TypeError(
            "extent must be a tuple: "
            "(left, right, bottom, top)."
        )

    values = tuple(
        float(value)
        for value in extent
    )

    left, right, bottom, top = values

    if right <= left:
        raise ValueError(
            "extent right must be greater than left."
        )

    if top <= bottom:
        raise ValueError(
            "extent top must be greater than bottom."
        )

    return values


def _required_string(
    name: str,
    value: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return result