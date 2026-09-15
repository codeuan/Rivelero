"""Sensor data model for Rivelero.

This module defines the canonical representation of a sensor associated with
a Rivelero Viewpoint.

A Sensor describes the device or sensing system used to acquire visual
information. It intentionally stores source metadata rather than assumptions
introduced during a Rivelero analysis. Analysis-level defaults and missing
metadata policies belong to VisibilityConfiguration.

The model is designed to support heterogeneous opportunistic imagery while
remaining extensible to additional sensor types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SensorModality(str, Enum):
    """Broad sensing modalities supported by the canonical Sensor model.

    OTHER allows data sources with modalities that are not yet represented
    explicitly by Rivelero. More modalities can be added in future versions
    without changing the conceptual Sensor interface.
    """

    RGB = "rgb"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    THERMAL = "thermal"
    INFRARED = "infrared"
    NIGHT_VISION = "night_vision"
    DEPTH = "depth"
    LIDAR = "lidar"
    OTHER = "other"


@dataclass(slots=True)
class Sensor:
    """Description of a sensing system associated with a Viewpoint.

    Parameters
    ----------
    sensor_id
        Unique identifier for the sensor within the Rivelero project or
        imported dataset.

    modality
        Broad sensing modality, such as RGB, thermal, multispectral, or
        hyperspectral.

    model
        Manufacturer/model or other human-readable sensor description.

    manufacturer
        Sensor or camera manufacturer, when known.

    image_width_px, image_height_px
        Native image dimensions in pixels, when available.

    focal_length_mm
        Physical focal length in millimetres, when known.

    sensor_width_mm, sensor_height_mm
        Physical dimensions of the imaging sensor, when known.

    horizontal_fov_deg, vertical_fov_deg
        Native sensor field of view in degrees, when explicitly known from
        metadata or calibration. These values should not contain assumptions
        introduced by Rivelero.

    spectral_bands
        Names or descriptions of spectral bands provided by the sensor.

    wavelength_range_nm
        Overall spectral wavelength range represented as
        ``(minimum_nm, maximum_nm)``.

    spatial_resolution
        Source-reported spatial resolution or ground sampling information.
        Units and interpretation should be recorded in ``extra_metadata`` when
        they are not unambiguous.

    source
        Provenance or dataset/platform from which the sensor description was
        obtained.

    extra_metadata
        Extensible dictionary preserving source-specific sensor metadata not
        represented by the standardized Rivelero attributes.
    """

    sensor_id: str

    modality: SensorModality | str = SensorModality.RGB

    model: str | None = None
    manufacturer: str | None = None

    image_width_px: int | None = None
    image_height_px: int | None = None

    focal_length_mm: float | None = None
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None

    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None

    spectral_bands: tuple[str, ...] = ()
    wavelength_range_nm: tuple[float, float] | None = None

    spatial_resolution: float | None = None

    source: str | None = None

    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize sensor metadata."""

        if not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
            raise ValueError("sensor_id must be a non-empty string.")

        self.sensor_id = self.sensor_id.strip()

        if isinstance(self.modality, str):
            try:
                self.modality = SensorModality(self.modality.lower())
            except ValueError:
                # Preserve extensibility for modalities introduced by external
                # users without requiring changes to Rivelero's core enum.
                self.modality = self.modality.strip().lower()

        if self.image_width_px is not None and self.image_width_px <= 0:
            raise ValueError("image_width_px must be greater than zero.")

        if self.image_height_px is not None and self.image_height_px <= 0:
            raise ValueError("image_height_px must be greater than zero.")

        for name in (
            "focal_length_mm",
            "sensor_width_mm",
            "sensor_height_mm",
            "spatial_resolution",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        self._validate_fov("horizontal_fov_deg", self.horizontal_fov_deg)
        self._validate_fov("vertical_fov_deg", self.vertical_fov_deg)

        if self.wavelength_range_nm is not None:
            if len(self.wavelength_range_nm) != 2:
                raise ValueError(
                    "wavelength_range_nm must contain exactly two values."
                )

            minimum, maximum = self.wavelength_range_nm

            if minimum < 0 or maximum <= minimum:
                raise ValueError(
                    "wavelength_range_nm must be an increasing, "
                    "non-negative (minimum, maximum) pair."
                )

        if not isinstance(self.extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

    @staticmethod
    def _validate_fov(name: str, value: float | None) -> None:
        """Validate a field-of-view value expressed in degrees."""

        if value is None:
            return

        if not 0 < value <= 360:
            raise ValueError(f"{name} must be within (0, 360] degrees.")

    @property
    def image_shape(self) -> tuple[int, int] | None:
        """Return image dimensions as ``(height, width)`` when available."""

        if self.image_width_px is None or self.image_height_px is None:
            return None

        return self.image_height_px, self.image_width_px

    @property
    def has_known_fov(self) -> bool:
        """Return whether at least one native FOV dimension is known."""

        return (
            self.horizontal_fov_deg is not None
            or self.vertical_fov_deg is not None
        )