"""Canonical Viewpoint data structure for Rivelero.

This module defines the spatial representation of an observer in Rivelero.

A Viewpoint describes where and how visual observation can potentially occur.
It is independent of whether an observation actually occurred at that
location. Temporal occurrences of viewpoints are represented separately by
ObservationEvent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rasterio.crs import CRS


@dataclass(slots=True)
class Viewpoint:
    """Canonical spatial observer configuration used by Rivelero.

    A Viewpoint represents where an observer or sensor is located and, where
    known, how its visual field is spatially configured.

    Viewpoints may originate from existing observations (for example Google
    Street View, Mapillary, UAV imagery, or georeferenced photographs), from
    manually supplied locations, from simulations, or from Rivelero's own
    survey-design algorithms.

    Coordinates do not define viewpoint identity. Multiple Viewpoints may
    occupy the same spatial location while differing in orientation, sensor
    configuration, observer height, provenance, or other attributes.

    Parameters
    ----------
    viewpoint_id
        Unique identifier for the Viewpoint.

    x, y
        Horizontal coordinates of the observer position expressed in `crs`.

    crs
        Coordinate reference system associated with x and y.

    z
        Absolute observer/camera elevation when explicitly known. The vertical
        datum should be recorded in metadata when relevant.

    observer_height_m
        Height of the observer/sensor above the terrain or surface represented
        by the Environment. This is distinct from absolute elevation.

    heading_deg
        Horizontal viewing direction in degrees, when known. Rivelero assumes
        compass convention: 0 degrees = north, 90 = east, 180 = south,
        270 = west.

    pitch_deg
        Vertical viewing angle in degrees, when known. Positive/negative
        convention should be standardized by import adapters before creating
        the Viewpoint.

    roll_deg
        Camera/sensor roll in degrees, when known.

    horizontal_fov_deg, vertical_fov_deg
        View-specific horizontal and vertical fields of view, when explicitly
        known. These may differ from the sensor's native field of view, for
        example when a panorama is rendered using a specified crop.

    sensor_id
        Identifier of the Sensor associated with the Viewpoint, when known.

    platform
        Platform carrying the sensor, such as street-view vehicle, UAV,
        human, robot, or fixed camera.

    source
        Source dataset/platform from which the Viewpoint originated.

    source_id
        Original identifier assigned to the viewpoint by the source dataset,
        where different from `viewpoint_id`.

    position_uncertainty_m
        Positional uncertainty in metres, when available.

    orientation_uncertainty_deg
        Orientation uncertainty in degrees, when available.

    provenance
        Extensible provenance information describing how the Viewpoint was
        obtained or generated.

    extra_metadata
        Source-specific metadata not represented by standardized attributes.

    Notes
    -----
    Missing source metadata should remain None. Analytical assumptions used
    to replace missing values belong to VisibilityConfiguration and should
    not be silently written into the Viewpoint.
    """

    viewpoint_id: str

    x: float
    y: float
    crs: CRS | str

    z: float | None = None
    observer_height_m: float | None = None

    heading_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None

    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None

    sensor_id: str | None = None
    platform: str | None = None

    source: str | None = None
    source_id: str | None = None

    position_uncertainty_m: float | None = None
    orientation_uncertainty_deg: float | None = None

    provenance: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize Viewpoint metadata."""

        if not isinstance(self.viewpoint_id, str) or not self.viewpoint_id.strip():
            raise ValueError("viewpoint_id must be a non-empty string.")

        self.viewpoint_id = self.viewpoint_id.strip()

        if not isinstance(self.x, (int, float)):
            raise TypeError("x must be numeric.")

        if not isinstance(self.y, (int, float)):
            raise TypeError("y must be numeric.")

        self.x = float(self.x)
        self.y = float(self.y)

        if not np.isfinite(self.x) or not np.isfinite(self.y):
            raise ValueError("x and y must be finite.")

        # Standardize the CRS internally as a rasterio CRS object.
        try:
            self.crs = CRS.from_user_input(self.crs)
        except Exception as exc:
            raise ValueError(
                f"Invalid coordinate reference system: {self.crs!r}"
            ) from exc

        if self.z is not None:
            self.z = self._finite_float("z", self.z)

        if self.observer_height_m is not None:
            self.observer_height_m = self._finite_float(
                "observer_height_m",
                self.observer_height_m,
            )
            if self.observer_height_m < 0:
                raise ValueError(
                    "observer_height_m must be greater than or equal to zero."
                )

        if self.heading_deg is not None:
            self.heading_deg = self._finite_float(
                "heading_deg",
                self.heading_deg,
            )
            # Canonical representation: [0, 360).
            self.heading_deg %= 360.0

        if self.pitch_deg is not None:
            self.pitch_deg = self._finite_float(
                "pitch_deg",
                self.pitch_deg,
            )
            if not -90.0 <= self.pitch_deg <= 90.0:
                raise ValueError(
                    "pitch_deg must be within [-90, 90] degrees."
                )

        if self.roll_deg is not None:
            self.roll_deg = self._finite_float(
                "roll_deg",
                self.roll_deg,
            )
            # Canonical representation: [-180, 180).
            self.roll_deg = (
                (self.roll_deg + 180.0) % 360.0
            ) - 180.0

        self.horizontal_fov_deg = self._validate_fov(
            "horizontal_fov_deg",
            self.horizontal_fov_deg,
        )
        self.vertical_fov_deg = self._validate_fov(
            "vertical_fov_deg",
            self.vertical_fov_deg,
        )

        if self.position_uncertainty_m is not None:
            self.position_uncertainty_m = self._finite_float(
                "position_uncertainty_m",
                self.position_uncertainty_m,
            )
            if self.position_uncertainty_m < 0:
                raise ValueError(
                    "position_uncertainty_m must be greater than or equal "
                    "to zero."
                )

        if self.orientation_uncertainty_deg is not None:
            self.orientation_uncertainty_deg = self._finite_float(
                "orientation_uncertainty_deg",
                self.orientation_uncertainty_deg,
            )
            if self.orientation_uncertainty_deg < 0:
                raise ValueError(
                    "orientation_uncertainty_deg must be greater than or "
                    "equal to zero."
                )

        self.sensor_id = self._optional_string(
            "sensor_id",
            self.sensor_id,
        )
        self.platform = self._optional_string(
            "platform",
            self.platform,
        )
        self.source = self._optional_string(
            "source",
            self.source,
        )
        self.source_id = self._optional_string(
            "source_id",
            self.source_id,
        )

        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary.")

        if not isinstance(self.extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        """Convert a numeric value to a finite float."""

        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        result = float(value)

        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite.")

        return result

    @staticmethod
    def _validate_fov(
        name: str,
        value: float | None,
    ) -> float | None:
        """Validate a field-of-view value expressed in degrees."""

        if value is None:
            return None

        result = Viewpoint._finite_float(name, value)

        if not 0.0 < result <= 360.0:
            raise ValueError(
                f"{name} must be within (0, 360] degrees."
            )

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

    @property
    def has_orientation(self) -> bool:
        """Return whether a horizontal viewing direction is known."""

        return self.heading_deg is not None

    @property
    def has_directional_fov(self) -> bool:
        """Return whether a horizontal viewing cone can be defined."""

        return (
            self.heading_deg is not None
            and self.horizontal_fov_deg is not None
        )

    @property
    def position(self) -> tuple[float, float]:
        """Return the horizontal position as ``(x, y)``."""

        return self.x, self.y
