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
import math


@dataclass(slots=True)
class ObservationEvent:
    """Occurrence of an observation associated with a Viewpoint.

    Event-level geometry represents acquisition-specific overrides. When an
    attribute is None, downstream analysis may fall back to the associated
    Viewpoint, Sensor, and finally VisibilityConfiguration.
    """

    event_id: str
    viewpoint_id: str

    timestamp: datetime | None = None

    sequence_id: str | None = None
    sequence_index: int | None = None

    image_id: str | None = None
    source: str | None = None

    # Acquisition-specific spatial/viewing overrides.
    observer_height_m: float | None = None
    heading_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None

    acquisition_conditions: dict[str, Any] = field(default_factory=dict)
    metadata_uncertainty: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_id = self._required_string(
            "event_id",
            self.event_id,
        )
        self.viewpoint_id = self._required_string(
            "viewpoint_id",
            self.viewpoint_id,
        )

        self.sequence_id = self._optional_string(
            "sequence_id",
            self.sequence_id,
        )
        self.image_id = self._optional_string(
            "image_id",
            self.image_id,
        )
        self.source = self._optional_string(
            "source",
            self.source,
        )

        if self.sequence_index is not None:
            if not isinstance(self.sequence_index, int):
                raise TypeError(
                    "sequence_index must be an integer or None."
                )
            if self.sequence_index < 0:
                raise ValueError(
                    "sequence_index must be greater than or equal to zero."
                )

        if self.timestamp is not None and not isinstance(
            self.timestamp,
            datetime,
        ):
            raise TypeError("timestamp must be a datetime or None.")

        if self.observer_height_m is not None:
            self.observer_height_m = self._finite_float(
                "observer_height_m",
                self.observer_height_m,
            )
            if self.observer_height_m < 0:
                raise ValueError(
                    "observer_height_m must be non-negative."
                )

        if self.heading_deg is not None:
            self.heading_deg = (
                self._finite_float("heading_deg", self.heading_deg)
                % 360.0
            )

        if self.pitch_deg is not None:
            self.pitch_deg = self._finite_float(
                "pitch_deg",
                self.pitch_deg,
            )
            if not -90.0 <= self.pitch_deg <= 90.0:
                raise ValueError(
                    "pitch_deg must be within [-90, 90]."
                )

        if self.roll_deg is not None:
            roll = self._finite_float("roll_deg", self.roll_deg)
            self.roll_deg = ((roll + 180.0) % 360.0) - 180.0

        self.horizontal_fov_deg = self._validate_fov(
            "horizontal_fov_deg",
            self.horizontal_fov_deg,
        )
        self.vertical_fov_deg = self._validate_fov(
            "vertical_fov_deg",
            self.vertical_fov_deg,
        )

        for name in (
            "acquisition_conditions",
            "metadata_uncertainty",
            "extra_metadata",
        ):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"{name} must be a dictionary.")

    @property
    def is_temporal(self) -> bool:
        return self.timestamp is not None

    @property
    def is_sequenced(self) -> bool:
        return (
            self.sequence_id is not None
            or self.sequence_index is not None
        )

    @property
    def has_view_override(self) -> bool:
        """Whether the event overrides any Viewpoint viewing geometry."""

        return any(
            value is not None
            for value in (
                self.observer_height_m,
                self.heading_deg,
                self.pitch_deg,
                self.roll_deg,
                self.horizontal_fov_deg,
                self.vertical_fov_deg,
            )
        )

    @staticmethod
    def _required_string(name: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

        result = value.strip()

        if not result:
            raise ValueError(f"{name} must be non-empty.")

        return result

    @staticmethod
    def _optional_string(
        name: str,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None.")

        result = value.strip()

        if not result:
            raise ValueError(f"{name} cannot be empty when provided.")

        return result

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        result = float(value)

        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite.")

        return result

    @classmethod
    def _validate_fov(
        cls,
        name: str,
        value: float | None,
    ) -> float | None:
        if value is None:
            return None

        result = cls._finite_float(name, value)

        if not 0.0 < result <= 360.0:
            raise ValueError(
                f"{name} must be within (0, 360] degrees."
            )

        return result