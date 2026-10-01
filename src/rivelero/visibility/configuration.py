"""Visibility-analysis configuration for Rivelero.

This module defines the scientific assumptions and parameter values used by
Rivelero when converting Viewpoints and their associated metadata into
2.5D visibility estimates.

VisibilityConfiguration is deliberately separate from the source metadata
stored in Viewpoint, Sensor, and ObservationEvent. Missing source metadata
remain unknown in those objects; assumptions introduced for an analysis are
recorded here instead.

A SurveyObservabilityField is therefore conditional on its
VisibilityConfiguration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math


class MissingMetadataPolicy(str, Enum):
    """Policy applied when required source metadata are unavailable."""

    USE_DEFAULT = "use_default"
    OMNIDIRECTIONAL = "omnidirectional"
    EXCLUDE = "exclude"
    ERROR = "error"


class VisibilityBackend(str, Enum):
    """Visibility-computation backend."""

    GDAL = "gdal"
    OTHER = "other"


class SamplingUnit(str, Enum):
    """Sampling unit used when constructing survey observability."""

    VIEWPOINT = "viewpoint"
    OBSERVATION_EVENT = "observation_event"


@dataclass(slots=True)
class VisibilityConfiguration:
    """Scientific assumptions controlling a Rivelero visibility analysis.

    Source metadata remain stored in Viewpoint, Sensor, and ObservationEvent.
    This object records assumptions introduced by the analysis.

    Parameters here may change the resulting visibility field and therefore
    form part of the provenance of a SurveyObservabilityField.
    """

    configuration_id: str
    name: str

    backend: VisibilityBackend | str = VisibilityBackend.GDAL

    # A finite maximum distance is currently required by the GDAL backend.
    max_distance_m: float = 500.0

    default_observer_height_m: float | None = 1.75
    default_target_height_m: float = 0.0

    use_direction: bool = True

    default_heading_deg: float | None = None
    default_horizontal_fov_deg: float | None = 360.0

    use_vertical_fov: bool = False
    default_pitch_deg: float | None = 0.0
    default_vertical_fov_deg: float | None = None

    missing_heading_policy: MissingMetadataPolicy | str = (
        MissingMetadataPolicy.OMNIDIRECTIONAL
    )
    missing_fov_policy: MissingMetadataPolicy | str = (
        MissingMetadataPolicy.USE_DEFAULT
    )
    missing_observer_height_policy: MissingMetadataPolicy | str = (
        MissingMetadataPolicy.USE_DEFAULT
    )

    # Preserve the convention already used by Connor's GDAL implementation.
    curvature_coefficient: float = 0.85714

    # Declared for the architecture, but not yet supported by the common
    # single-viewpoint engine.
    use_environment_obstacles: bool = False
    obstacle_layer_ids: tuple[str, ...] = ()

    sampling_unit: SamplingUnit | str = SamplingUnit.VIEWPOINT

    extra_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.configuration_id = self._required_string(
            "configuration_id",
            self.configuration_id,
        )
        self.name = self._required_string("name", self.name)

        self.backend = self._normalize_enum(
            "backend",
            self.backend,
            VisibilityBackend,
            allow_custom=True,
        )

        self.sampling_unit = self._normalize_enum(
            "sampling_unit",
            self.sampling_unit,
            SamplingUnit,
            allow_custom=False,
        )

        self.missing_heading_policy = self._normalize_enum(
            "missing_heading_policy",
            self.missing_heading_policy,
            MissingMetadataPolicy,
            allow_custom=False,
        )
        self.missing_fov_policy = self._normalize_enum(
            "missing_fov_policy",
            self.missing_fov_policy,
            MissingMetadataPolicy,
            allow_custom=False,
        )
        self.missing_observer_height_policy = self._normalize_enum(
            "missing_observer_height_policy",
            self.missing_observer_height_policy,
            MissingMetadataPolicy,
            allow_custom=False,
        )

        self.max_distance_m = self._positive_float(
            "max_distance_m",
            self.max_distance_m,
        )

        if self.default_observer_height_m is not None:
            self.default_observer_height_m = self._non_negative_float(
                "default_observer_height_m",
                self.default_observer_height_m,
            )

        self.default_target_height_m = self._non_negative_float(
            "default_target_height_m",
            self.default_target_height_m,
        )

        self.default_heading_deg = self._normalize_heading(
            self.default_heading_deg
        )

        self.default_horizontal_fov_deg = self._validate_fov(
            "default_horizontal_fov_deg",
            self.default_horizontal_fov_deg,
        )

        if self.default_pitch_deg is not None:
            self.default_pitch_deg = self._finite_float(
                "default_pitch_deg",
                self.default_pitch_deg,
            )
            if not -90.0 <= self.default_pitch_deg <= 90.0:
                raise ValueError(
                    "default_pitch_deg must be within [-90, 90] degrees."
                )

        self.default_vertical_fov_deg = self._validate_fov(
            "default_vertical_fov_deg",
            self.default_vertical_fov_deg,
        )

        for name in (
            "use_direction",
            "use_vertical_fov",
            "use_environment_obstacles",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")

        self.curvature_coefficient = self._finite_float(
            "curvature_coefficient",
            self.curvature_coefficient,
        )

        if self.curvature_coefficient < 0:
            raise ValueError(
                "curvature_coefficient must be greater than or equal to zero."
            )

        if not isinstance(self.obstacle_layer_ids, tuple):
            self.obstacle_layer_ids = tuple(self.obstacle_layer_ids)

        normalized_ids = tuple(
            self._required_string("obstacle_layer_id", layer_id)
            for layer_id in self.obstacle_layer_ids
        )

        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError(
                "obstacle_layer_ids cannot contain duplicate identifiers."
            )

        self.obstacle_layer_ids = normalized_ids

        if not isinstance(self.extra_parameters, dict):
            raise TypeError("extra_parameters must be a dictionary.")

        self._validate_policy_consistency()

    def _validate_policy_consistency(self) -> None:
        if (
            self.missing_observer_height_policy
            == MissingMetadataPolicy.USE_DEFAULT
            and self.default_observer_height_m is None
        ):
            raise ValueError(
                "missing_observer_height_policy='use_default' requires "
                "default_observer_height_m."
            )

        if (
            self.missing_heading_policy
            == MissingMetadataPolicy.USE_DEFAULT
            and self.default_heading_deg is None
        ):
            raise ValueError(
                "missing_heading_policy='use_default' requires "
                "default_heading_deg."
            )

        if (
            self.missing_fov_policy
            == MissingMetadataPolicy.USE_DEFAULT
            and self.default_horizontal_fov_deg is None
        ):
            raise ValueError(
                "missing_fov_policy='use_default' requires "
                "default_horizontal_fov_deg."
            )

    @property
    def is_event_based(self) -> bool:
        return self.sampling_unit == SamplingUnit.OBSERVATION_EVENT

    @property
    def is_viewpoint_based(self) -> bool:
        return self.sampling_unit == SamplingUnit.VIEWPOINT

    @staticmethod
    def _normalize_enum(
        name: str,
        value: Enum | str,
        enum_type: type[Enum],
        *,
        allow_custom: bool,
    ) -> Enum | str:
        if isinstance(value, enum_type):
            return value

        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a {enum_type.__name__} or string."
            )

        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(f"{name} cannot be empty.")

        try:
            return enum_type(normalized)
        except ValueError:
            if allow_custom:
                return normalized

            allowed = ", ".join(member.value for member in enum_type)
            raise ValueError(
                f"Invalid {name} {value!r}. Allowed values: {allowed}."
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
    def _finite_float(name: str, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        result = float(value)

        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite.")

        return result

    @classmethod
    def _positive_float(cls, name: str, value: float) -> float:
        result = cls._finite_float(name, value)

        if result <= 0:
            raise ValueError(f"{name} must be greater than zero.")

        return result

    @classmethod
    def _non_negative_float(cls, name: str, value: float) -> float:
        result = cls._finite_float(name, value)

        if result < 0:
            raise ValueError(f"{name} must be non-negative.")

        return result

    @classmethod
    def _normalize_heading(
        cls,
        value: float | None,
    ) -> float | None:
        if value is None:
            return None

        return cls._finite_float("default_heading_deg", value) % 360.0

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