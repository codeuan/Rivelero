"""Observation-event data model for Rivelero.

This module defines the canonical representation of an observation event.

A Viewpoint describes where and how visual observation can occur.
An ObservationEvent describes the occurrence of that observation opportunity
within a survey, acquisition sequence, or temporal record.

Keeping these concepts separate allows Rivelero to represent repeated visits
to the same viewpoint, temporal imagery, trajectories, and heterogeneous
opportunistic image collections without duplicating viewpoint definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ObservationEvent:
    """Occurrence of an observation associated with a Rivelero Viewpoint.

    Parameters
    ----------
    event_id
        Unique identifier for this observation event.

    viewpoint_id
        Identifier of the Viewpoint associated with the event. Multiple
        ObservationEvents may reference the same Viewpoint.

    timestamp
        Date and time at which the observation occurred, when known.
        Timezone-aware datetime values are preferred when available.

    sequence_id
        Identifier of the acquisition sequence or trajectory to which the
        event belongs, when applicable.

    sequence_index
        Position of the event within its sequence. This allows acquisition
        order to be preserved independently of timestamps.

    image_id
        Identifier of the image or other observation associated with this
        event, when available.

    source
        Source or platform from which the event metadata originated, such as
        Google Street View, Mapillary, UAV imagery, or a simulation.

    acquisition_conditions
        Extensible dictionary describing conditions that apply specifically
        to this acquisition rather than to the underlying Viewpoint or
        Sensor. Examples include illumination, weather, platform speed, or
        image-specific acquisition settings.

    metadata_uncertainty
        Optional dictionary describing uncertainties associated specifically
        with the event metadata.

    extra_metadata
        Source-specific metadata not represented by the standardized Rivelero
        attributes.
    """

    event_id: str
    viewpoint_id: str

    timestamp: datetime | None = None

    sequence_id: str | None = None
    sequence_index: int | None = None

    image_id: str | None = None

    source: str | None = None

    acquisition_conditions: dict[str, Any] = field(default_factory=dict)
    metadata_uncertainty: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize event metadata."""

        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string.")

        if not isinstance(self.viewpoint_id, str) or not self.viewpoint_id.strip():
            raise ValueError("viewpoint_id must be a non-empty string.")

        self.event_id = self.event_id.strip()
        self.viewpoint_id = self.viewpoint_id.strip()

        if self.sequence_id is not None:
            if not isinstance(self.sequence_id, str):
                raise TypeError("sequence_id must be a string or None.")

            self.sequence_id = self.sequence_id.strip()

            if not self.sequence_id:
                raise ValueError(
                    "sequence_id cannot be an empty string when provided."
                )

        if self.sequence_index is not None:
            if not isinstance(self.sequence_index, int):
                raise TypeError("sequence_index must be an integer or None.")

            if self.sequence_index < 0:
                raise ValueError(
                    "sequence_index must be greater than or equal to zero."
                )

        if self.timestamp is not None and not isinstance(
            self.timestamp, datetime
        ):
            raise TypeError("timestamp must be a datetime object or None.")

        if self.image_id is not None:
            if not isinstance(self.image_id, str):
                raise TypeError("image_id must be a string or None.")

            self.image_id = self.image_id.strip()

            if not self.image_id:
                raise ValueError(
                    "image_id cannot be an empty string when provided."
                )

        if self.source is not None:
            if not isinstance(self.source, str):
                raise TypeError("source must be a string or None.")

            self.source = self.source.strip()

            if not self.source:
                raise ValueError(
                    "source cannot be an empty string when provided."
                )

        self._validate_dictionary(
            "acquisition_conditions",
            self.acquisition_conditions,
        )
        self._validate_dictionary(
            "metadata_uncertainty",
            self.metadata_uncertainty,
        )
        self._validate_dictionary(
            "extra_metadata",
            self.extra_metadata,
        )

    @staticmethod
    def _validate_dictionary(name: str, value: dict[str, Any]) -> None:
        """Validate extensible metadata dictionaries."""

        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dictionary.")

    @property
    def is_temporal(self) -> bool:
        """Return whether the event contains explicit temporal information."""

        return self.timestamp is not None

    @property
    def is_sequenced(self) -> bool:
        """Return whether the event belongs to an ordered sequence."""

        return (
            self.sequence_id is not None
            or self.sequence_index is not None
        )