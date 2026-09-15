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
    """Policy used when required viewpoint/sensor metadata are unavailable."""

    USE_DEFAULT = "use_default"
    OMNIDIRECTIONAL = "omnidirectional"
    EXCLUDE = "exclude"
    ERROR = "error"


class VisibilityBackend(str, Enum):
    """Supported visibility-computation backends."""

    GDAL = "gdal"
    OTHER = "other"


class SamplingUnit(str, Enum):
    """Observation unit used when constructing survey observability."""

    VIEWPOINT = "viewpoint"
    OBSERVATION_EVENT = "observation_event"


@dataclass(slots=True)
class VisibilityConfiguration:
    """Scientific configuration for a Rivelero visibility analysis.

    Parameters
    ----------
    configuration_id
        Unique identifier for this visibility configuration.

    name
        Human-readable name.

    backend
        Visibility-computation backend. The current Rivelero 2.5D
        implementation uses GDAL.

    max_distance_m
        Maximum observer-to-target distance considered visible. ``None`` means
        that no additional distance limit is imposed by this configuration.

    default_observer_height_m
        Observer height above the elevation/surface model used when a
        Viewpoint does not provide an observer height.

    default_target_height_m
        Default height of the target above the underlying elevation/surface
        model.

    use_direction
        Whether camera/view orientation should constrain geometric visibility.

    default_heading_deg
        Heading used when heading metadata are missing and
        ``missing_heading_policy`` is ``USE_DEFAULT``.

    default_horizontal_fov_deg
        Horizontal field of view used when no event-, viewpoint-, or
        sensor-specific FOV is available.

    use_vertical_fov
        Whether pitch and vertical FOV should constrain visibility. This may
        remain False for the initial Rivelero implementation.

    default_pitch_deg
        Default pitch used when required by vertical viewing geometry.

    default_vertical_fov_deg
        Default vertical field of view.

    missing_heading_policy
        Rule used when heading is unavailable.

    missing_fov_policy
        Rule used when horizontal FOV is unavailable.

    missing_observer_height_policy
        Rule used when observer height is unavailable.

    use_earth_curvature
        Whether Earth-curvature correction is included in the viewshed.

    refraction_coefficient
        Atmospheric refraction coefficient supplied to the visibility backend
        when curvature/refraction correction is enabled.

    use_environment_obstacles
        Whether additional Environment obstacle layers should participate in
        visibility/occlusion analysis.

    obstacle_layer_ids
        Optional explicit list of EnvironmentLayer identifiers to use. An
        empty tuple means no particular layer IDs are selected here.

    sampling_unit
        Whether the resulting survey analysis treats unique Viewpoints or
        individual ObservationEvents as its fundamental sampling unit.

    extra_parameters
        Extensible parameters for future visibility models or external
        backends.

    Notes
    -----
    This object contains assumptions that may change the scientific result.
    General runtime settings such as worker count, cache location, or
    temporary directories belong to ``RiveleroConfig`` instead.
    """

    configuration_id: str
    name: str

    backend: VisibilityBackend | str = VisibilityBackend.GDAL

    max_distance_m: float | None = None

    default_observer_height_m: float | None = 1.7
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

    use_earth_curvature: bool = False
    refraction_coefficient: float = 0.13

    use_environment_obstacles: bool = False
    obstacle_layer_ids: tuple[str, ...] = ()

    sampling_unit: SamplingUnit | str = SamplingUnit.VIEWPOINT

    extra_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize visibility parameters."""

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

        if self.max_distance_m is not None:
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
            "use_earth_curvature",
            "use_environment_obstacles",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")

        self.refraction_coefficient = self._finite_float(
            "refraction_coefficient",
            self.refraction_coefficient,
        )

        if not isinstance(self.obstacle_layer_ids, tuple):
            self.obstacle_layer_ids = tuple(self.obstacle_layer_ids)

        normalized_layer_ids: list[str] = []

        for layer_id in self.obstacle_layer_ids:
            normalized_layer_ids.append(
                self._required_string("obstacle_layer_id", layer_id)
            )

        if len(normalized_layer_ids) != len(set(normalized_layer_ids)):
            raise ValueError(
                "obstacle_layer_ids cannot contain duplicate identifiers."
            )

        self.obstacle_layer_ids = tuple(normalized_layer_ids)

        if not isinstance(self.extra_parameters, dict):
            raise TypeError("extra_parameters must be a dictionary.")

        self._validate_policy_consistency()

    # ------------------------------------------------------------------
    # Missing-metadata consistency
    # ------------------------------------------------------------------

    def _validate_policy_consistency(self) -> None:
        """Check that policies requiring defaults actually have defaults."""

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

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_event_based(self) -> bool:
        """Return whether ObservationEvents are the sampling unit."""

        return self.sampling_unit == SamplingUnit.OBSERVATION_EVENT

    @property
    def is_viewpoint_based(self) -> bool:
        """Return whether unique Viewpoints are the sampling unit."""

        return self.sampling_unit == SamplingUnit.VIEWPOINT

    @property
    def uses_directional_visibility(self) -> bool:
        """Return whether horizontal viewing direction constrains visibility."""

        return self.use_direction

    @property
    def uses_additional_obstacles(self) -> bool:
        """Return whether Environment obstacle layers are enabled."""

        return self.use_environment_obstacles

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_enum(
        name: str,
        value: Enum | str,
        enum_type: type[Enum],
        *,
        allow_custom: bool,
    ) -> Enum | str:
        """Normalize a string-backed enumeration."""

        if isinstance(value, enum_type):
            return value

        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a {enum_type.__name__} or string."
            )

        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(f"{name} cannot be an empty string.")

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
            raise ValueError(f"{name} must be a non-empty string.")

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
            raise ValueError(
                f"{name} must be greater than or equal to zero."
            )

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