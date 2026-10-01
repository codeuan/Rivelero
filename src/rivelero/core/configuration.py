"""Viewpoint-configuration data model for Rivelero.

A ViewpointConfiguration represents a visual sampling strategy as a
collection of Viewpoints and, where available, ObservationEvents.

The configuration is independent of whether its viewpoints originate from
an existing opportunistic dataset, a hypothetical design, a simulation,
Rivelero's candidate-generation workflow, or an optimisation procedure.

The object describes the survey configuration itself. It does not contain
the Environment, AnalysisDomain, visibility assumptions, or observability
results produced from that configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from rivelero.core.observation import ObservationEvent
from rivelero.core.viewpoint import Viewpoint


class ConfigurationType(str, Enum):
    """Broad provenance categories for viewpoint configurations.

    These values describe how a configuration originated. They must not be
    used to select different visibility algorithms.
    """

    OBSERVED = "observed"
    HYPOTHETICAL = "hypothetical"
    SIMULATED = "simulated"
    GENERATED = "generated"
    OPTIMIZED = "optimized"
    OTHER = "other"


@dataclass(slots=True)
class ViewpointConfiguration:
    """Collection of observation opportunities forming one survey strategy.

    Parameters
    ----------
    configuration_id
        Unique identifier for the configuration.

    name
        Human-readable name used to identify the configuration.

    viewpoints
        Viewpoints belonging to the configuration. A Viewpoint represents a
        spatial observer configuration and may participate in more than one
        ViewpointConfiguration.

    observation_events
        Optional ordered collection of ObservationEvents associated with the
        viewpoints. Multiple events may reference the same Viewpoint, allowing
        repeated visits and temporal sampling to be represented without
        duplicating Viewpoint definitions.

    configuration_type
        Provenance category describing whether the configuration is observed,
        hypothetical, simulated, generated, optimized, or another user-defined
        type. This value does not determine analytical behaviour.

    description
        Optional human-readable description of the configuration.

    parent_configuration_id
        Identifier of the configuration from which this configuration was
        derived, when applicable.

    derivation_operation
        Metadata describing how a derived configuration was generated, for
        example viewpoint removal, candidate selection, or optimisation.

    provenance
        Configuration-level provenance information. Source-specific provenance
        should normally remain attached to individual Viewpoints and
        ObservationEvents.

    created_at
        Time at which the configuration object was created.

    extra_metadata
        Extensible user- or application-specific metadata.
    """

    configuration_id: str
    name: str

    viewpoints: list[Viewpoint] = field(default_factory=list)
    observation_events: list[ObservationEvent] = field(default_factory=list)

    configuration_type: ConfigurationType | str = ConfigurationType.OTHER
    description: str | None = None

    parent_configuration_id: str | None = None
    derivation_operation: dict[str, Any] | None = None

    provenance: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the configuration."""

        self.configuration_id = self._required_string(
            "configuration_id",
            self.configuration_id,
        )
        self.name = self._required_string("name", self.name)

        self.description = self._optional_string(
            "description",
            self.description,
        )
        self.parent_configuration_id = self._optional_string(
            "parent_configuration_id",
            self.parent_configuration_id,
        )

        if isinstance(self.configuration_type, str):
            normalized_type = self.configuration_type.strip().lower()

            try:
                self.configuration_type = ConfigurationType(normalized_type)
            except ValueError:
                # Preserve extensibility for user-defined configuration types.
                if not normalized_type:
                    raise ValueError(
                        "configuration_type cannot be an empty string."
                    )
                self.configuration_type = normalized_type

        if not isinstance(self.viewpoints, list):
            self.viewpoints = list(self.viewpoints)

        if not isinstance(self.observation_events, list):
            self.observation_events = list(self.observation_events)

        self._validate_viewpoints()
        self._validate_events()

        if self.parent_configuration_id == self.configuration_id:
            raise ValueError(
                "A ViewpointConfiguration cannot be its own parent."
            )

        if (
            self.derivation_operation is not None
            and not isinstance(self.derivation_operation, dict)
        ):
            raise TypeError(
                "derivation_operation must be a dictionary or None."
            )

        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary.")

        if not isinstance(self.extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime object.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_viewpoints(self) -> None:
        """Validate Viewpoints and enforce unique Viewpoint IDs."""

        seen_ids: set[str] = set()

        for viewpoint in self.viewpoints:
            if not isinstance(viewpoint, Viewpoint):
                raise TypeError(
                    "Every item in viewpoints must be a Viewpoint."
                )

            if viewpoint.viewpoint_id in seen_ids:
                raise ValueError(
                    "Duplicate viewpoint_id within configuration: "
                    f"{viewpoint.viewpoint_id!r}."
                )

            seen_ids.add(viewpoint.viewpoint_id)

    def _validate_events(self) -> None:
        """Validate events and their references to Viewpoints.

        Event order is deliberately preserved. Static analyses may ignore
        order, while temporal and trajectory analyses can use it.
        """

        viewpoint_ids = {
            viewpoint.viewpoint_id
            for viewpoint in self.viewpoints
        }

        seen_event_ids: set[str] = set()

        for event in self.observation_events:
            if not isinstance(event, ObservationEvent):
                raise TypeError(
                    "Every item in observation_events must be an "
                    "ObservationEvent."
                )

            if event.event_id in seen_event_ids:
                raise ValueError(
                    "Duplicate event_id within configuration: "
                    f"{event.event_id!r}."
                )

            seen_event_ids.add(event.event_id)

            if event.viewpoint_id not in viewpoint_ids:
                raise ValueError(
                    f"ObservationEvent {event.event_id!r} references "
                    f"Viewpoint {event.viewpoint_id!r}, which is not present "
                    f"in configuration {self.configuration_id!r}."
                )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    @property
    def viewpoint_ids(self) -> tuple[str, ...]:
        """Return Viewpoint identifiers in configuration order."""

        return tuple(
            viewpoint.viewpoint_id
            for viewpoint in self.viewpoints
        )

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Return ObservationEvent identifiers in event order."""

        return tuple(
            event.event_id
            for event in self.observation_events
        )

    @property
    def n_viewpoints(self) -> int:
        """Number of unique Viewpoints in the configuration."""

        return len(self.viewpoints)

    @property
    def n_events(self) -> int:
        """Number of ObservationEvents in the configuration."""

        return len(self.observation_events)

    @property
    def has_events(self) -> bool:
        """Return whether the configuration contains ObservationEvents."""

        return bool(self.observation_events)

    @property
    def has_temporal_information(self) -> bool:
        """Return whether at least one event contains a timestamp."""

        return any(
            event.timestamp is not None
            for event in self.observation_events
        )

    @property
    def has_sequence_information(self) -> bool:
        """Return whether event ordering/sequence metadata are available."""

        return any(
            event.is_sequenced
            for event in self.observation_events
        )

    @property
    def sources(self) -> tuple[str, ...]:
        """Return unique known data sources represented by the configuration.

        Sources are derived from Viewpoint and ObservationEvent provenance
        rather than assuming that a configuration has one source.
        """

        sources: set[str] = set()

        for viewpoint in self.viewpoints:
            if viewpoint.source is not None:
                sources.add(viewpoint.source)

        for event in self.observation_events:
            if event.source is not None:
                sources.add(event.source)

        return tuple(sorted(sources))

    def get_viewpoint(self, viewpoint_id: str) -> Viewpoint:
        """Return a Viewpoint by identifier.

        Raises
        ------
        KeyError
            If the Viewpoint is not present in this configuration.
        """

        for viewpoint in self.viewpoints:
            if viewpoint.viewpoint_id == viewpoint_id:
                return viewpoint

        raise KeyError(
            f"Viewpoint {viewpoint_id!r} is not present in configuration "
            f"{self.configuration_id!r}."
        )

    def get_event(self, event_id: str) -> ObservationEvent:
        """Return an ObservationEvent by identifier."""

        for event in self.observation_events:
            if event.event_id == event_id:
                return event

        raise KeyError(
            f"ObservationEvent {event_id!r} is not present in configuration "
            f"{self.configuration_id!r}."
        )

    def events_for_viewpoint(
        self,
        viewpoint_id: str,
    ) -> tuple[ObservationEvent, ...]:
        """Return all events referencing a Viewpoint, preserving event order."""

        # Validate the requested Viewpoint first.
        self.get_viewpoint(viewpoint_id)

        return tuple(
            event
            for event in self.observation_events
            if event.viewpoint_id == viewpoint_id
        )

    # ------------------------------------------------------------------
    # Controlled mutation
    # ------------------------------------------------------------------

    def add_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Add a Viewpoint while preserving identifier uniqueness."""

        if not isinstance(viewpoint, Viewpoint):
            raise TypeError("viewpoint must be a Viewpoint.")

        if viewpoint.viewpoint_id in self.viewpoint_ids:
            raise ValueError(
                f"Viewpoint {viewpoint.viewpoint_id!r} already exists in "
                f"configuration {self.configuration_id!r}."
            )

        self.viewpoints.append(viewpoint)

    def add_event(self, event: ObservationEvent) -> None:
        """Add an ObservationEvent to the end of the event sequence."""

        if not isinstance(event, ObservationEvent):
            raise TypeError("event must be an ObservationEvent.")

        if event.event_id in self.event_ids:
            raise ValueError(
                f"ObservationEvent {event.event_id!r} already exists in "
                f"configuration {self.configuration_id!r}."
            )

        if event.viewpoint_id not in self.viewpoint_ids:
            raise ValueError(
                f"ObservationEvent {event.event_id!r} references "
                f"Viewpoint {event.viewpoint_id!r}, which is not present "
                "in this configuration."
            )

        self.observation_events.append(event)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_viewpoints(
        cls,
        configuration_id: str,
        name: str,
        viewpoints: Iterable[Viewpoint],
        *,
        configuration_type: ConfigurationType | str = ConfigurationType.OTHER,
        description: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> "ViewpointConfiguration":
        """Create a static configuration from a collection of Viewpoints."""

        return cls(
            configuration_id=configuration_id,
            name=name,
            viewpoints=list(viewpoints),
            configuration_type=configuration_type,
            description=description,
            provenance={} if provenance is None else dict(provenance),
        )

    # ------------------------------------------------------------------
    # String helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _required_string(name: str, value: str) -> str:
        """Validate a required non-empty string."""

        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

        result = value.strip()

        if not result:
            raise ValueError(f"{name} must be a non-empty string.")

        return result

    @staticmethod
    def _optional_string(
        name: str,
        value: str | None,
    ) -> str | None:
        """Normalize an optional non-empty string."""

        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None.")

        result = value.strip()

        if not result:
            raise ValueError(
                f"{name} cannot be an empty string when provided."
            )

        return result