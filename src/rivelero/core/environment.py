"""Environmental data model for Rivelero.

This module defines the physical spatial environment within which Rivelero
evaluates visibility and observability.

An Environment describes the spatial datasets representing terrain, surface
structure, obstacles, and other environmental information relevant to an
analysis. It does not define the area over which observability is evaluated
(AnalysisDomain), the survey configuration (ViewpointConfiguration), or the
analytical assumptions used to calculate visibility
(VisibilityConfiguration).

The Environment stores references and metadata describing environmental
datasets. Loading and retrieving those datasets belongs to the Rivelero I/O
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from rasterio.crs import CRS


class ElevationModelType(str, Enum):
    """Supported categories of elevation/surface representation."""

    DEM = "dem"
    DTM = "dtm"
    DSM = "dsm"
    OTHER = "other"


class EnvironmentLayerType(str, Enum):
    """Broad categories for additional environmental layers."""

    BUILDINGS = "buildings"
    VEGETATION = "vegetation"
    BARRIERS = "barriers"
    INFRASTRUCTURE = "infrastructure"
    LAND_COVER = "land_cover"
    WATER = "water"
    OBSTACLES = "obstacles"
    OTHER = "other"


@dataclass(slots=True)
class ElevationModel:
    """Description of the elevation surface used by Rivelero.

    Parameters
    ----------
    source
        Path, URI, or other identifier for the elevation dataset.

    model_type
        Type of elevation representation, for example DEM, DTM, or DSM.

    crs
        Coordinate reference system of the elevation model, when known.

    resolution_m
        Nominal horizontal spatial resolution in metres, when known.

    nodata_value
        Source nodata value, when explicitly known.

    vertical_datum
        Description of the vertical datum or height reference, when known.

    vertical_accuracy_m
        Reported vertical accuracy in metres, when available.

    source_name
        Dataset/provider name, for example OpenTopography.

    acquisition_date
        Date or time associated with the elevation dataset, when known.

    provenance
        Information describing where and how the dataset was obtained.

    extra_metadata
        Additional source-specific metadata.
    """

    source: Path | str

    model_type: ElevationModelType | str = ElevationModelType.DEM

    crs: CRS | str | None = None
    resolution_m: float | None = None
    nodata_value: float | None = None

    vertical_datum: str | None = None
    vertical_accuracy_m: float | None = None

    source_name: str | None = None
    acquisition_date: datetime | None = None

    provenance: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize elevation metadata."""

        self.source = self._normalize_source(self.source)

        if isinstance(self.model_type, str):
            value = self.model_type.strip().lower()

            try:
                self.model_type = ElevationModelType(value)
            except ValueError:
                if not value:
                    raise ValueError(
                        "model_type cannot be an empty string."
                    )
                self.model_type = value

        if self.crs is not None:
            try:
                self.crs = CRS.from_user_input(self.crs)
            except Exception as exc:
                raise ValueError(
                    f"Invalid elevation-model CRS: {self.crs!r}"
                ) from exc

        if self.resolution_m is not None:
            self.resolution_m = self._positive_float(
                "resolution_m",
                self.resolution_m,
            )

        if self.vertical_accuracy_m is not None:
            self.vertical_accuracy_m = self._non_negative_float(
                "vertical_accuracy_m",
                self.vertical_accuracy_m,
            )

        self.vertical_datum = self._optional_string(
            "vertical_datum",
            self.vertical_datum,
        )
        self.source_name = self._optional_string(
            "source_name",
            self.source_name,
        )

        if (
            self.acquisition_date is not None
            and not isinstance(self.acquisition_date, datetime)
        ):
            raise TypeError(
                "acquisition_date must be a datetime object or None."
            )

        self._validate_dictionary("provenance", self.provenance)
        self._validate_dictionary(
            "extra_metadata",
            self.extra_metadata,
        )

    @staticmethod
    def _normalize_source(value: Path | str) -> Path | str:
        """Normalize a local path while preserving non-file identifiers."""

        if isinstance(value, Path):
            return value.expanduser()

        if not isinstance(value, str):
            raise TypeError("source must be a string or pathlib.Path.")

        value = value.strip()

        if not value:
            raise ValueError("source must not be empty.")

        # Preserve URI-like identifiers as strings.
        if "://" in value:
            return value

        return Path(value).expanduser()

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
            raise ValueError(
                f"{name} cannot be empty when provided."
            )

        return result

    @staticmethod
    def _positive_float(name: str, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        result = float(value)

        if result <= 0:
            raise ValueError(f"{name} must be greater than zero.")

        return result

    @staticmethod
    def _non_negative_float(
        name: str,
        value: float,
    ) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        result = float(value)

        if result < 0:
            raise ValueError(
                f"{name} must be greater than or equal to zero."
            )

        return result

    @staticmethod
    def _validate_dictionary(
        name: str,
        value: dict[str, Any],
    ) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dictionary.")


@dataclass(slots=True)
class EnvironmentLayer:
    """Reference to an additional environmental spatial layer.

    EnvironmentLayer provides a generic representation for environmental
    datasets that may affect visibility or provide contextual information.

    The meaning of a layer is deliberately separated from the decision about
    how it participates in a visibility calculation. For example, a building
    layer belongs to the Environment, while whether Rivelero uses that layer
    as an occlusion source belongs to VisibilityConfiguration.

    Parameters
    ----------
    layer_id
        Unique identifier for the environmental layer.

    layer_type
        Broad semantic category of the layer.

    source
        Path, URI, or other identifier for the dataset.

    name
        Optional human-readable name.

    crs
        Coordinate reference system, when known.

    resolution_m
        Nominal spatial resolution for raster layers, when applicable.

    acquisition_date
        Date or time associated with the layer.

    provenance
        Information describing the origin and processing of the layer.

    extra_metadata
        Source-specific information, such as height-field names, opacity
        attributes, OSM tags, or temporal validity.
    """

    layer_id: str
    layer_type: EnvironmentLayerType | str
    source: Path | str

    name: str | None = None
    crs: CRS | str | None = None
    resolution_m: float | None = None

    acquisition_date: datetime | None = None

    provenance: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize environmental-layer metadata."""

        self.layer_id = self._required_string(
            "layer_id",
            self.layer_id,
        )

        if isinstance(self.layer_type, str):
            value = self.layer_type.strip().lower()

            try:
                self.layer_type = EnvironmentLayerType(value)
            except ValueError:
                if not value:
                    raise ValueError(
                        "layer_type cannot be an empty string."
                    )
                self.layer_type = value

        self.source = ElevationModel._normalize_source(self.source)

        self.name = ElevationModel._optional_string(
            "name",
            self.name,
        )

        if self.crs is not None:
            try:
                self.crs = CRS.from_user_input(self.crs)
            except Exception as exc:
                raise ValueError(
                    f"Invalid environmental-layer CRS: {self.crs!r}"
                ) from exc

        if self.resolution_m is not None:
            self.resolution_m = ElevationModel._positive_float(
                "resolution_m",
                self.resolution_m,
            )

        if (
            self.acquisition_date is not None
            and not isinstance(self.acquisition_date, datetime)
        ):
            raise TypeError(
                "acquisition_date must be a datetime object or None."
            )

        ElevationModel._validate_dictionary(
            "provenance",
            self.provenance,
        )
        ElevationModel._validate_dictionary(
            "extra_metadata",
            self.extra_metadata,
        )

    @staticmethod
    def _required_string(name: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

        result = value.strip()

        if not result:
            raise ValueError(f"{name} must be a non-empty string.")

        return result


@dataclass(slots=True)
class Environment:
    """Physical spatial environment used by a Rivelero analysis.

    An Environment describes the spatial datasets representing the physical
    world in which visibility is evaluated.

    The current 2.5D Rivelero visibility engine normally requires one primary
    elevation model. Additional environmental layers may represent buildings,
    vegetation, barriers, infrastructure, or other spatial information.

    Parameters
    ----------
    environment_id
        Unique identifier for the Environment.

    name
        Human-readable environment name.

    elevation_model
        Primary DEM, DTM, DSM, or other elevation representation.

    layers
        Optional additional environmental layers.

    description
        Human-readable description.

    provenance
        Environment-level provenance information.

    created_at
        Time at which the Environment representation was created.

    extra_metadata
        Extensible project- or application-specific metadata.
    """

    environment_id: str
    name: str

    elevation_model: ElevationModel

    layers: list[EnvironmentLayer] = field(default_factory=list)

    description: str | None = None

    provenance: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the Environment and its constituent layers."""

        self.environment_id = self._required_string(
            "environment_id",
            self.environment_id,
        )
        self.name = self._required_string("name", self.name)

        if not isinstance(self.elevation_model, ElevationModel):
            raise TypeError(
                "elevation_model must be an ElevationModel."
            )

        if not isinstance(self.layers, list):
            self.layers = list(self.layers)

        seen_ids: set[str] = set()

        for layer in self.layers:
            if not isinstance(layer, EnvironmentLayer):
                raise TypeError(
                    "Every item in layers must be an EnvironmentLayer."
                )

            if layer.layer_id in seen_ids:
                raise ValueError(
                    "Duplicate environmental layer ID: "
                    f"{layer.layer_id!r}."
                )

            seen_ids.add(layer.layer_id)

        self.description = ElevationModel._optional_string(
            "description",
            self.description,
        )

        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary.")

        if not isinstance(self.extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime object.")

    # ------------------------------------------------------------------
    # Layer access
    # ------------------------------------------------------------------

    @property
    def layer_ids(self) -> tuple[str, ...]:
        """Return environmental-layer identifiers."""

        return tuple(layer.layer_id for layer in self.layers)

    def get_layer(self, layer_id: str) -> EnvironmentLayer:
        """Return an environmental layer by identifier."""

        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer

        raise KeyError(
            f"EnvironmentLayer {layer_id!r} is not present in "
            f"Environment {self.environment_id!r}."
        )

    def layers_of_type(
        self,
        layer_type: EnvironmentLayerType | str,
    ) -> tuple[EnvironmentLayer, ...]:
        """Return all environmental layers belonging to a category."""

        if isinstance(layer_type, str):
            requested = layer_type.strip().lower()
        else:
            requested = layer_type.value

        return tuple(
            layer
            for layer in self.layers
            if (
                layer.layer_type.value
                if isinstance(layer.layer_type, EnvironmentLayerType)
                else layer.layer_type
            )
            == requested
        )

    def add_layer(self, layer: EnvironmentLayer) -> None:
        """Add an environmental layer while preserving identifier uniqueness."""

        if not isinstance(layer, EnvironmentLayer):
            raise TypeError("layer must be an EnvironmentLayer.")

        if layer.layer_id in self.layer_ids:
            raise ValueError(
                f"EnvironmentLayer {layer.layer_id!r} already exists in "
                f"Environment {self.environment_id!r}."
            )

        self.layers.append(layer)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def elevation_type(self) -> ElevationModelType | str:
        """Return the type of the primary elevation model."""

        return self.elevation_model.model_type

    @property
    def is_surface_model(self) -> bool:
        """Return whether the primary elevation layer is explicitly a DSM."""

        return self.elevation_model.model_type == ElevationModelType.DSM

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _required_string(name: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

        result = value.strip()

        if not result:
            raise ValueError(f"{name} must be a non-empty string.")

        return result