"""Explicit JSON encoding of canonical Rivelero objects.

Values that JSON cannot represent natively are written as small tagged
objects so that every document stays readable and decoding never guesses:

    {"$type": "datetime", "value": "2026-01-01T10:00:00+00:00"}
    {"$type": "crs", "wkt": "...", "string": "EPSG:32633"}
    {"$type": "affine", "coefficients": [a, b, c, d, e, f]}
    {"$type": "geometry", "geojson": {...}}
    {"$type": "path", "value": "..."}
    {"$type": "tuple", "items": [...]}
    {"$type": "float", "value": "nan" | "inf" | "-inf"}
    {"$type": "array", "name": "..."}          (array stored in an .npz)
    {"$type": "dict", "items": [[key, value], ...]}   (keys not plain JSON)
    {"$enum": "SamplingUnit", "value": "viewpoint"}
    {"$dataclass": "Viewpoint", "fields": {...}}

Dataclasses and enums are resolved only through explicit allow-lists, so a
project file can never name an arbitrary Python class. Decoded dataclasses
are rebuilt through their constructors, reusing canonical validation. Values
that cannot be represented safely raise ProjectSerializationError naming the
offending path instead of being dropped.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from rivelero.analysis.comparison import CandidateDefinition, ScenarioSnapshot
from rivelero.analysis.coverage import CoverageSummary, ExposureDistribution
from rivelero.analysis.scenario import CandidateStatus, ScenarioSummary
from rivelero.core.configuration import ConfigurationType, ViewpointConfiguration
from rivelero.core.domain import AnalysisDomain, AnalysisGrid
from rivelero.core.environment import (
    ElevationModel,
    ElevationModelType,
    Environment,
    EnvironmentLayer,
    EnvironmentLayerType,
)
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor, SensorModality
from rivelero.core.viewpoint import Viewpoint
from rivelero.observability.builder import SOFBuildReport
from rivelero.observability.storage import VisibilityKey
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.project.schema import ProjectFormatError, ProjectSerializationError
from rivelero.visibility.configuration import (
    MissingMetadataPolicy,
    SamplingUnit,
    VisibilityBackend,
    VisibilityConfiguration,
)


DATACLASSES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Viewpoint, Sensor, ObservationEvent, ViewpointConfiguration,
        ElevationModel, EnvironmentLayer, Environment,
        AnalysisGrid, AnalysisDomain,
        VisibilityConfiguration, VisibilityKey, SOFBuildReport,
        SurveyObservabilityField,
        ExposureDistribution, CoverageSummary, ScenarioSummary,
        CandidateDefinition, ScenarioSnapshot,
    )
}

ENUMS: dict[str, type[Enum]] = {
    cls.__name__: cls
    for cls in (
        SensorModality, ConfigurationType, ElevationModelType,
        EnvironmentLayerType, VisibilityBackend, MissingMetadataPolicy,
        SamplingUnit, CandidateStatus,
    )
}

# Array dtypes allowed in projects (no object/pickled arrays).
_ARRAY_KINDS = frozenset("biuf")


class ArraySink:
    """Collects arrays referenced by an encoded document."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.arrays: dict[str, np.ndarray] = {}

    def add(self, array: np.ndarray, path: str) -> str:
        if array.dtype.kind not in _ARRAY_KINDS:
            raise ProjectSerializationError(
                f"{path}: arrays of dtype {array.dtype} cannot be stored in a project."
            )
        name = f"{self.prefix}{len(self.arrays)}"
        self.arrays[name] = np.ascontiguousarray(array)
        return name


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode(value: Any, *, arrays: ArraySink | None = None, path: str = "$") -> Any:
    """Encode ``value`` as JSON-compatible data."""

    if value is None or isinstance(value, bool):
        return value

    # Enums first: str-based enums are also instances of str.
    if isinstance(value, Enum):
        name = type(value).__name__
        if ENUMS.get(name) is not type(value):
            raise ProjectSerializationError(f"{path}: enum {name} is not supported.")
        return {"$enum": name, "value": value.value}

    if isinstance(value, np.generic) and not isinstance(value, np.ndarray):
        return encode(value.item(), arrays=arrays, path=path)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"$type": "float", "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}

    if isinstance(value, str):
        return value

    if isinstance(value, datetime):
        return {"$type": "datetime", "value": value.isoformat()}

    if isinstance(value, PurePath):
        return {"$type": "path", "value": str(value)}

    if isinstance(value, CRS):
        return {"$type": "crs", "wkt": value.to_wkt(), "string": value.to_string()}

    if isinstance(value, Affine):
        return {"$type": "affine", "coefficients": [float(v) for v in tuple(value)[:6]]}

    if isinstance(value, BaseGeometry):
        return {"$type": "geometry", "geojson": _json_ready(mapping(value), path)}

    if isinstance(value, np.ndarray):
        if arrays is None:
            raise ProjectSerializationError(f"{path}: arrays need an array store.")
        return {"$type": "array", "name": arrays.add(value, path)}

    if isinstance(value, tuple):
        return {
            "$type": "tuple",
            "items": [encode(item, arrays=arrays, path=f"{path}[{i}]") for i, item in enumerate(value)],
        }

    if isinstance(value, list):
        return [encode(item, arrays=arrays, path=f"{path}[{i}]") for i, item in enumerate(value)]

    if isinstance(value, dict):
        plain = all(isinstance(key, str) and not key.startswith("$") for key in value)
        if plain:
            return {
                key: encode(item, arrays=arrays, path=f"{path}.{key}")
                for key, item in value.items()
            }
        return {
            "$type": "dict",
            "items": [
                [encode(key, arrays=arrays, path=f"{path}<key>"),
                 encode(item, arrays=arrays, path=f"{path}[{key!r}]")]
                for key, item in value.items()
            ],
        }

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if DATACLASSES.get(name) is not type(value):
            raise ProjectSerializationError(f"{path}: {name} is not a project type.")
        return {
            "$dataclass": name,
            "fields": {
                field.name: encode(
                    getattr(value, field.name), arrays=arrays, path=f"{path}.{field.name}"
                )
                for field in dataclasses.fields(value)
                if field.init
            },
        }

    raise ProjectSerializationError(
        f"{path}: values of type {type(value).__name__} cannot be stored in a "
        "project. Convert them to text, numbers, lists or dictionaries."
    )


def _json_ready(value: Any, path: str) -> Any:
    """GeoJSON mappings contain tuples; convert them to plain lists."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v, path) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v, path) for v in value]
    if isinstance(value, (int, float, str)) or value is None:
        return value
    raise ProjectSerializationError(f"{path}: unsupported geometry value.")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(
    value: Any,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    path: str = "$",
) -> Any:
    """Decode data produced by :func:`encode`."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, list):
        return [decode(item, arrays=arrays, path=f"{path}[{i}]") for i, item in enumerate(value)]

    if not isinstance(value, dict):
        raise ProjectFormatError(f"{path}: unexpected value.")

    if "$enum" in value:
        cls = ENUMS.get(value.get("$enum"))
        if cls is None:
            raise ProjectFormatError(f"{path}: unknown enum {value.get('$enum')!r}.")
        try:
            return cls(value.get("value"))
        except ValueError as exc:
            raise ProjectFormatError(f"{path}: {exc}") from None

    if "$dataclass" in value:
        cls = DATACLASSES.get(value.get("$dataclass"))
        if cls is None:
            raise ProjectFormatError(f"{path}: unknown type {value.get('$dataclass')!r}.")
        fields = value.get("fields")
        if not isinstance(fields, dict):
            raise ProjectFormatError(f"{path}: missing fields.")
        known = {field.name for field in dataclasses.fields(cls) if field.init}
        unknown = set(fields) - known
        if unknown:
            raise ProjectFormatError(
                f"{path}: unexpected fields for {cls.__name__}: {sorted(unknown)}."
            )
        kwargs = {
            name: decode(item, arrays=arrays, path=f"{path}.{name}")
            for name, item in fields.items()
        }
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ProjectFormatError(f"{path}: invalid {cls.__name__}: {exc}") from exc

    tag = value.get("$type")
    if tag is None:
        return {key: decode(item, arrays=arrays, path=f"{path}.{key}") for key, item in value.items()}

    try:
        if tag == "datetime":
            return datetime.fromisoformat(value["value"])
        if tag == "path":
            return Path(value["value"])
        if tag == "crs":
            return CRS.from_wkt(value["wkt"])
        if tag == "affine":
            coefficients = value["coefficients"]
            if len(coefficients) != 6:
                raise ValueError("an affine transform needs six coefficients")
            return Affine(*[float(c) for c in coefficients])
        if tag == "geometry":
            return shape(value["geojson"])
        if tag == "tuple":
            return tuple(decode(item, arrays=arrays, path=f"{path}[{i}]")
                         for i, item in enumerate(value["items"]))
        if tag == "float":
            return float(value["value"])
        if tag == "dict":
            return {
                decode(key, arrays=arrays, path=f"{path}<key>"): decode(item, arrays=arrays, path=path)
                for key, item in value["items"]
            }
        if tag == "array":
            if arrays is None or value["name"] not in arrays:
                raise ValueError(f"array {value.get('name')!r} is missing")
            return arrays[value["name"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectFormatError(f"{path}: invalid {tag} value: {exc}") from None

    raise ProjectFormatError(f"{path}: unknown value type {tag!r}.")
