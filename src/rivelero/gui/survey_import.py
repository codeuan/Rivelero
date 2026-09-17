"""Survey import services for Rivelero.

This module converts external tabular survey data into canonical Rivelero
objects.

The importer is deliberately independent of Qt. GUI pages and dialogs may
collect paths, mappings, CRS information and user choices, but conversion
into scientific objects happens here.

Current scope
-------------
The first implementation supports generic CSV import for:

    Viewpoint
    Sensor
    ObservationEvent
    ViewpointConfiguration

Provider-specific acquisition/import adapters (Google Street View,
Mapillary, KartaView, Flickr, iNaturalist, etc.) should eventually convert
their source metadata into the same canonical objects rather than adding
provider-specific logic to the GUI.

Missing metadata is preserved as missing. The importer does not silently
apply visibility defaults; those decisions belong to VisibilityConfiguration.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.crs import CRS

from rivelero.core.configuration import (
    ConfigurationType,
    ViewpointConfiguration,
)
from rivelero.core.observation import ObservationEvent
from rivelero.core.sensor import Sensor
from rivelero.core.viewpoint import Viewpoint


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SurveyImportError(ValueError):
    """Base exception for invalid survey-import data."""


class SurveyImportRowError(SurveyImportError):
    """Error associated with one source-table row."""

    def __init__(
        self,
        *,
        table: str,
        row_number: int,
        message: str,
    ) -> None:

        self.table = table
        self.row_number = row_number
        self.message = message

        super().__init__(
            f"{table} row {row_number}: {message}"
        )


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewpointColumnMapping:
    """Column names used to import Viewpoints.

    At minimum the table must provide:

        viewpoint_id
        x
        y

    or, for geographic input:

        viewpoint_id
        longitude
        latitude

    ``x`` and ``y`` are deliberately generic. Their CRS is supplied
    separately through ``source_crs``.
    """

    viewpoint_id: str = "viewpoint_id"

    x: str = "x"
    y: str = "y"

    longitude: str = "longitude"
    latitude: str = "latitude"

    observer_height_m: str = "observer_height_m"

    heading_deg: str = "heading_deg"
    pitch_deg: str = "pitch_deg"
    roll_deg: str = "roll_deg"

    horizontal_fov_deg: str = "horizontal_fov_deg"
    vertical_fov_deg: str = "vertical_fov_deg"

    sensor_id: str = "sensor_id"

    platform: str = "platform"
    source: str = "source"

    source_id: str = "source_id"

    z: str = "z"
    position_uncertainty_m: str = "position_uncertainty_m"
    orientation_uncertainty_deg: str = "orientation_uncertainty_deg"


@dataclass(frozen=True, slots=True)
class SensorColumnMapping:
    """Column names used to import Sensors."""

    sensor_id: str = "sensor_id"
    name: str = "name"

    modality: str = "modality"

    image_width_px: str = "image_width_px"
    image_height_px: str = "image_height_px"

    focal_length_mm: str = "focal_length_mm"
    sensor_width_mm: str = "sensor_width_mm"
    sensor_height_mm: str = "sensor_height_mm"

    horizontal_fov_deg: str = "horizontal_fov_deg"
    vertical_fov_deg: str = "vertical_fov_deg"

    spectral_bands: str = "spectral_bands"
    wavelength_range_nm: str = "wavelength_range_nm"
    spatial_resolution: str = "spatial_resolution"

    manufacturer: str = "manufacturer"
    model: str = "model"

    source: str = "source"


@dataclass(frozen=True, slots=True)
class ObservationEventColumnMapping:
    """Column names used to import ObservationEvents."""

    event_id: str = "event_id"
    viewpoint_id: str = "viewpoint_id"

    timestamp: str = "timestamp"

    heading_deg: str = "heading_deg"
    pitch_deg: str = "pitch_deg"
    roll_deg: str = "roll_deg"

    horizontal_fov_deg: str = "horizontal_fov_deg"
    vertical_fov_deg: str = "vertical_fov_deg"

    image_id: str = "image_id"

    sequence_id: str = "sequence_id"
    sequence_index: str = "sequence_index"

    source: str = "source"

    observer_height_m: str = "observer_height_m"


# ---------------------------------------------------------------------------
# Import options
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurveyImportOptions:
    """Options controlling canonical Viewpoint import.

    Parameters
    ----------
    source_crs
        CRS of imported coordinates.

    target_crs
        CRS in which canonical Viewpoints should be stored.

        If omitted, source coordinates are preserved.

    strict
        If True, the first invalid row aborts the import.

        If False, invalid rows are reported and valid rows are retained.

    allow_duplicate_coordinates
        Whether multiple Viewpoints may occupy the same coordinates.

        Rivelero normally allows this because physical location does not
        define Viewpoint identity.

    trim_strings
        Whether textual values are stripped of surrounding whitespace.
    """

    source_crs: CRS | str | int = "EPSG:4326"

    target_crs: CRS | str | int | None = None

    strict: bool = False

    allow_duplicate_coordinates: bool = True

    trim_strings: bool = True

    def __post_init__(self) -> None:
        _coerce_crs(
            self.source_crs
        )

        if self.target_crs is not None:
            _coerce_crs(
                self.target_crs
            )


# ---------------------------------------------------------------------------
# Import report / result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SurveyImportReport:
    """Structured report describing a survey import."""

    viewpoint_rows: int = 0
    viewpoints_imported: int = 0
    viewpoints_skipped: int = 0

    sensor_rows: int = 0
    sensors_imported: int = 0
    sensors_skipped: int = 0

    event_rows: int = 0
    events_imported: int = 0
    events_skipped: int = 0

    transformed_coordinates: bool = False

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    @property
    def successful(self) -> bool:
        """Whether at least one Viewpoint was imported."""

        return self.viewpoints_imported > 0

    @property
    def has_errors(self) -> bool:
        return bool(
            self.errors
        )


@dataclass(slots=True)
class SurveyImportResult:
    """Canonical result returned by survey import."""

    viewpoint_configuration: ViewpointConfiguration

    sensors: dict[str, Sensor]

    report: SurveyImportReport


# ---------------------------------------------------------------------------
# High-level survey import
# ---------------------------------------------------------------------------


def import_survey_csv(
    *,
    viewpoints_path: str | Path,
    configuration_id: str,
    configuration_name: str,
    options: SurveyImportOptions,
    viewpoint_mapping: ViewpointColumnMapping | None = None,
    sensors_path: str | Path | None = None,
    sensor_mapping: SensorColumnMapping | None = None,
    observation_events_path: str | Path | None = None,
    event_mapping: ObservationEventColumnMapping | None = None,
    configuration_type: ConfigurationType | str = ConfigurationType.OBSERVED,
    metadata: dict[str, Any] | None = None,
) -> SurveyImportResult:
    """Import a complete Rivelero survey from generic CSV files.

    Viewpoints are required. Sensor and ObservationEvent tables are optional.
    """

    if not isinstance(
        options,
        SurveyImportOptions,
    ):
        raise TypeError(
            "options must be a SurveyImportOptions."
        )

    viewpoint_mapping = (
        ViewpointColumnMapping()
        if viewpoint_mapping is None
        else viewpoint_mapping
    )

    sensor_mapping = (
        SensorColumnMapping()
        if sensor_mapping is None
        else sensor_mapping
    )

    event_mapping = (
        ObservationEventColumnMapping()
        if event_mapping is None
        else event_mapping
    )

    report = SurveyImportReport()

    viewpoints = import_viewpoints_csv(
        viewpoints_path,
        options=options,
        mapping=viewpoint_mapping,
        report=report,
    )

    if not viewpoints:
        raise SurveyImportError(
            "No valid Viewpoints were imported."
        )

    sensors: dict[str, Sensor] = {}

    if sensors_path is not None:
        sensors = import_sensors_csv(
            sensors_path,
            mapping=sensor_mapping,
            strict=options.strict,
            trim_strings=options.trim_strings,
            report=report,
        )

    events: list[ObservationEvent] = []

    if observation_events_path is not None:
        events = import_observation_events_csv(
            observation_events_path,
            mapping=event_mapping,
            viewpoint_ids={
                viewpoint.viewpoint_id
                for viewpoint in viewpoints
            },
            strict=options.strict,
            trim_strings=options.trim_strings,
            report=report,
        )

    _validate_sensor_references(
        viewpoints=viewpoints,
        sensors=sensors,
        report=report,
    )

    configuration = ViewpointConfiguration(
        configuration_id=_required_string(
            "configuration_id",
            configuration_id,
        ),
        name=_required_string(
            "configuration_name",
            configuration_name,
        ),
        viewpoints=viewpoints,
        observation_events=events,
        configuration_type=configuration_type,
        extra_metadata={
            **(
                {}
                if metadata is None
                else dict(metadata)
            ),
            "import": {
                "viewpoints_path": str(
                    Path(
                        viewpoints_path
                    ).expanduser()
                ),
                "sensors_path": (
                    None
                    if sensors_path is None
                    else str(
                        Path(
                            sensors_path
                        ).expanduser()
                    )
                ),
                "observation_events_path": (
                    None
                    if observation_events_path is None
                    else str(
                        Path(
                            observation_events_path
                        ).expanduser()
                    )
                ),
                "source_crs": (
                    _coerce_crs(
                        options.source_crs
                    ).to_string()
                ),
                "target_crs": (
                    None
                    if options.target_crs is None
                    else _coerce_crs(
                        options.target_crs
                    ).to_string()
                ),
            },
        },
    )

    return SurveyImportResult(
        viewpoint_configuration=configuration,
        sensors=sensors,
        report=report,
    )


# ---------------------------------------------------------------------------
# Viewpoint import
# ---------------------------------------------------------------------------


def import_viewpoints_csv(
    path: str | Path,
    *,
    options: SurveyImportOptions,
    mapping: ViewpointColumnMapping | None = None,
    report: SurveyImportReport | None = None,
) -> list[Viewpoint]:
    """Import canonical Viewpoints from a generic CSV table."""

    if not isinstance(
        options,
        SurveyImportOptions,
    ):
        raise TypeError(
            "options must be a SurveyImportOptions."
        )

    mapping = (
        ViewpointColumnMapping()
        if mapping is None
        else mapping
    )

    if not isinstance(
        mapping,
        ViewpointColumnMapping,
    ):
        raise TypeError(
            "mapping must be a ViewpointColumnMapping."
        )

    report = (
        SurveyImportReport()
        if report is None
        else report
    )

    rows, fieldnames = _read_csv(
        path
    )

    coordinate_mode = _resolve_coordinate_columns(
        fieldnames,
        mapping,
    )

    source_crs = _coerce_crs(
        options.source_crs
    )

    target_crs = (
        source_crs
        if options.target_crs is None
        else _coerce_crs(
            options.target_crs
        )
    )

    transformer = None

    if source_crs != target_crs:
        transformer = Transformer.from_crs(
            PyprojCRS.from_user_input(
                source_crs.to_string()
            ),
            PyprojCRS.from_user_input(
                target_crs.to_string()
            ),
            always_xy=True,
        )

        report.transformed_coordinates = True

    viewpoints: list[
        Viewpoint
    ] = []

    seen_ids: set[
        str
    ] = set()

    seen_coordinates: set[
        tuple[float, float]
    ] = set()

    for source_index, row in enumerate(
        rows,
        start=2,
    ):
        report.viewpoint_rows += 1

        try:
            viewpoint_id = _required_cell(
                row,
                mapping.viewpoint_id,
                trim=options.trim_strings,
            )

            if viewpoint_id in seen_ids:
                raise SurveyImportRowError(
                    table="Viewpoints",
                    row_number=source_index,
                    message=(
                        f"duplicate viewpoint_id "
                        f"{viewpoint_id!r}."
                    ),
                )

            if coordinate_mode == "xy":
                x = _required_float_cell(
                    row,
                    mapping.x,
                )

                y = _required_float_cell(
                    row,
                    mapping.y,
                )

            else:
                x = _required_float_cell(
                    row,
                    mapping.longitude,
                )

                y = _required_float_cell(
                    row,
                    mapping.latitude,
                )

            if transformer is not None:
                x, y = transformer.transform(
                    x,
                    y,
                )

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                raise SurveyImportRowError(
                    table="Viewpoints",
                    row_number=source_index,
                    message=(
                        "coordinates must be finite."
                    ),
                )

            coordinate_key = (
                float(x),
                float(y),
            )

            if (
                not options.allow_duplicate_coordinates
                and coordinate_key in seen_coordinates
            ):
                raise SurveyImportRowError(
                    table="Viewpoints",
                    row_number=source_index,
                    message=(
                        "duplicate coordinates are not allowed "
                        "under the current import options."
                    ),
                )

            viewpoint = Viewpoint(
                viewpoint_id=viewpoint_id,
                x=float(x),
                y=float(y),
                crs=target_crs,
                z=_optional_float_cell(
                    row,
                    mapping.z,
                ),
                observer_height_m=_optional_float_cell(
                    row,
                    mapping.observer_height_m,
                ),
                heading_deg=_optional_float_cell(
                    row,
                    mapping.heading_deg,
                ),
                pitch_deg=_optional_float_cell(
                    row,
                    mapping.pitch_deg,
                ),
                roll_deg=_optional_float_cell(
                    row,
                    mapping.roll_deg,
                ),
                horizontal_fov_deg=_optional_float_cell(
                    row,
                    mapping.horizontal_fov_deg,
                ),
                vertical_fov_deg=_optional_float_cell(
                    row,
                    mapping.vertical_fov_deg,
                ),
                sensor_id=_optional_cell(
                    row,
                    mapping.sensor_id,
                    trim=options.trim_strings,
                ),
                platform=_optional_cell(
                    row,
                    mapping.platform,
                    trim=options.trim_strings,
                ),
                source=_optional_cell(
                    row,
                    mapping.source,
                    trim=options.trim_strings,
                ),
                source_id=_optional_cell(
                    row,
                    mapping.source_id,
                    trim=options.trim_strings,
                ),
                position_uncertainty_m=_optional_float_cell(
                    row,
                    mapping.position_uncertainty_m,
                ),
                orientation_uncertainty_deg=_optional_float_cell(
                    row,
                    mapping.orientation_uncertainty_deg,
                ),
                extra_metadata=_unmapped_metadata(
                    row,
                    (
                        mapping.viewpoint_id,
                        mapping.x,
                        mapping.y,
                        mapping.longitude,
                        mapping.latitude,
                        mapping.observer_height_m,
                        mapping.heading_deg,
                        mapping.pitch_deg,
                        mapping.roll_deg,
                        mapping.horizontal_fov_deg,
                        mapping.vertical_fov_deg,
                        mapping.sensor_id,
                        mapping.platform,
                        mapping.source,
                        mapping.source_id,
                        mapping.z,
                        mapping.position_uncertainty_m,
                        mapping.orientation_uncertainty_deg,
                    ),
                    trim=options.trim_strings,
                ),
            )

            viewpoints.append(
                viewpoint
            )

            seen_ids.add(
                viewpoint_id
            )

            seen_coordinates.add(
                coordinate_key
            )

            report.viewpoints_imported += 1

        except Exception as exc:
            _handle_row_error(
                exc=exc,
                table="Viewpoints",
                row_number=source_index,
                strict=options.strict,
                report=report,
            )

            report.viewpoints_skipped += 1

    return viewpoints


# ---------------------------------------------------------------------------
# Sensor import
# ---------------------------------------------------------------------------


def import_sensors_csv(
    path: str | Path,
    *,
    mapping: SensorColumnMapping | None = None,
    strict: bool = False,
    trim_strings: bool = True,
    report: SurveyImportReport | None = None,
) -> dict[str, Sensor]:
    """Import canonical Sensors from CSV."""

    mapping = (
        SensorColumnMapping()
        if mapping is None
        else mapping
    )

    if not isinstance(
        mapping,
        SensorColumnMapping,
    ):
        raise TypeError(
            "mapping must be a SensorColumnMapping."
        )

    report = (
        SurveyImportReport()
        if report is None
        else report
    )

    rows, _ = _read_csv(
        path
    )

    sensors: dict[
        str,
        Sensor,
    ] = {}

    for source_index, row in enumerate(
        rows,
        start=2,
    ):
        report.sensor_rows += 1

        try:
            sensor_id = _required_cell(
                row,
                mapping.sensor_id,
                trim=trim_strings,
            )

            if sensor_id in sensors:
                raise SurveyImportRowError(
                    table="Sensors",
                    row_number=source_index,
                    message=(
                        f"duplicate sensor_id "
                        f"{sensor_id!r}."
                    ),
                )

            sensor_kwargs: dict[str, Any] = {
                "sensor_id": sensor_id,
                "model": _optional_cell(
                    row,
                    mapping.model,
                    trim=trim_strings,
                ),
                "manufacturer": _optional_cell(
                    row,
                    mapping.manufacturer,
                    trim=trim_strings,
                ),
                "image_width_px": _optional_int_cell(
                    row,
                    mapping.image_width_px,
                ),
                "image_height_px": _optional_int_cell(
                    row,
                    mapping.image_height_px,
                ),
                "focal_length_mm": _optional_float_cell(
                    row,
                    mapping.focal_length_mm,
                ),
                "sensor_width_mm": _optional_float_cell(
                    row,
                    mapping.sensor_width_mm,
                ),
                "sensor_height_mm": _optional_float_cell(
                    row,
                    mapping.sensor_height_mm,
                ),
                "horizontal_fov_deg": _optional_float_cell(
                    row,
                    mapping.horizontal_fov_deg,
                ),
                "vertical_fov_deg": _optional_float_cell(
                    row,
                    mapping.vertical_fov_deg,
                ),
                "spectral_bands": _optional_spectral_bands(
                    row,
                    mapping.spectral_bands,
                    trim=trim_strings,
                ),
                "wavelength_range_nm": _optional_wavelength_range(
                    row,
                    mapping.wavelength_range_nm,
                ),
                "spatial_resolution": _optional_float_cell(
                    row,
                    mapping.spatial_resolution,
                ),
                "source": _optional_cell(
                    row,
                    mapping.source,
                    trim=trim_strings,
                ),
                "extra_metadata": _unmapped_metadata(
                    row,
                    (
                        mapping.sensor_id,
                        mapping.name,
                        mapping.modality,
                        mapping.image_width_px,
                        mapping.image_height_px,
                        mapping.focal_length_mm,
                        mapping.sensor_width_mm,
                        mapping.sensor_height_mm,
                        mapping.horizontal_fov_deg,
                        mapping.vertical_fov_deg,
                        mapping.spectral_bands,
                        mapping.wavelength_range_nm,
                        mapping.spatial_resolution,
                        mapping.manufacturer,
                        mapping.model,
                        mapping.source,
                    ),
                    trim=trim_strings,
                ),
            }

            modality = _optional_cell(
                row,
                mapping.modality,
                trim=trim_strings,
            )

            if modality is not None:
                sensor_kwargs["modality"] = modality

            sensor_name = _optional_cell(
                row,
                mapping.name,
                trim=trim_strings,
            )

            if sensor_name is not None:
                sensor_kwargs["extra_metadata"]["name"] = sensor_name

            sensor = Sensor(
                **sensor_kwargs
            )

            sensors[
                sensor_id
            ] = sensor

            report.sensors_imported += 1

        except Exception as exc:
            _handle_row_error(
                exc=exc,
                table="Sensors",
                row_number=source_index,
                strict=strict,
                report=report,
            )

            report.sensors_skipped += 1

    return sensors


# ---------------------------------------------------------------------------
# ObservationEvent import
# ---------------------------------------------------------------------------


def import_observation_events_csv(
    path: str | Path,
    *,
    viewpoint_ids: set[str],
    mapping: ObservationEventColumnMapping | None = None,
    strict: bool = False,
    trim_strings: bool = True,
    report: SurveyImportReport | None = None,
) -> list[ObservationEvent]:
    """Import canonical ObservationEvents from CSV.

    Event order is preserved exactly as it appears in the source table.
    """

    mapping = (
        ObservationEventColumnMapping()
        if mapping is None
        else mapping
    )

    if not isinstance(
        mapping,
        ObservationEventColumnMapping,
    ):
        raise TypeError(
            "mapping must be an ObservationEventColumnMapping."
        )

    report = (
        SurveyImportReport()
        if report is None
        else report
    )

    rows, _ = _read_csv(
        path
    )

    events: list[
        ObservationEvent
    ] = []

    seen_ids: set[
        str
    ] = set()

    for source_index, row in enumerate(
        rows,
        start=2,
    ):
        report.event_rows += 1

        try:
            event_id = _required_cell(
                row,
                mapping.event_id,
                trim=trim_strings,
            )

            if event_id in seen_ids:
                raise SurveyImportRowError(
                    table="ObservationEvents",
                    row_number=source_index,
                    message=(
                        f"duplicate event_id "
                        f"{event_id!r}."
                    ),
                )

            viewpoint_id = _required_cell(
                row,
                mapping.viewpoint_id,
                trim=trim_strings,
            )

            if viewpoint_id not in viewpoint_ids:
                raise SurveyImportRowError(
                    table="ObservationEvents",
                    row_number=source_index,
                    message=(
                        f"references unknown Viewpoint "
                        f"{viewpoint_id!r}."
                    ),
                )

            event = ObservationEvent(
                event_id=event_id,
                viewpoint_id=viewpoint_id,
                timestamp=parse_optional_iso_datetime(
                    _optional_cell(
                        row,
                        mapping.timestamp,
                        trim=trim_strings,
                    )
                ),
                observer_height_m=_optional_float_cell(
                    row,
                    mapping.observer_height_m,
                ),
                heading_deg=_optional_float_cell(
                    row,
                    mapping.heading_deg,
                ),
                pitch_deg=_optional_float_cell(
                    row,
                    mapping.pitch_deg,
                ),
                roll_deg=_optional_float_cell(
                    row,
                    mapping.roll_deg,
                ),
                horizontal_fov_deg=_optional_float_cell(
                    row,
                    mapping.horizontal_fov_deg,
                ),
                vertical_fov_deg=_optional_float_cell(
                    row,
                    mapping.vertical_fov_deg,
                ),
                image_id=_optional_cell(
                    row,
                    mapping.image_id,
                    trim=trim_strings,
                ),
                sequence_id=_optional_cell(
                    row,
                    mapping.sequence_id,
                    trim=trim_strings,
                ),
                sequence_index=_optional_int_cell(
                    row,
                    mapping.sequence_index,
                ),
                source=_optional_cell(
                    row,
                    mapping.source,
                    trim=trim_strings,
                ),
                extra_metadata=_unmapped_metadata(
                    row,
                    (
                        mapping.event_id,
                        mapping.viewpoint_id,
                        mapping.timestamp,
                        mapping.heading_deg,
                        mapping.pitch_deg,
                        mapping.roll_deg,
                        mapping.horizontal_fov_deg,
                        mapping.vertical_fov_deg,
                        mapping.image_id,
                        mapping.sequence_id,
                        mapping.sequence_index,
                        mapping.source,
                        mapping.observer_height_m,
                    ),
                    trim=trim_strings,
                ),
            )

            events.append(
                event
            )

            seen_ids.add(
                event_id
            )

            report.events_imported += 1

        except Exception as exc:
            _handle_row_error(
                exc=exc,
                table="ObservationEvents",
                row_number=source_index,
                strict=strict,
                report=report,
            )

            report.events_skipped += 1

    return events


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


def _validate_sensor_references(
    *,
    viewpoints: Iterable[Viewpoint],
    sensors: Mapping[str, Sensor],
    report: SurveyImportReport,
) -> None:
    """Report unresolved Sensor references without inventing defaults."""

    referenced = {
        viewpoint.sensor_id
        for viewpoint in viewpoints
        if viewpoint.sensor_id is not None
    }

    unresolved = sorted(
        sensor_id
        for sensor_id in referenced
        if sensor_id not in sensors
    )

    for sensor_id in unresolved:
        report.warnings.append(
            f"Viewpoints reference Sensor {sensor_id!r}, "
            "but that Sensor was not included in this import."
        )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _read_csv(
    path: str | Path,
) -> tuple[
    list[dict[str, str]],
    tuple[str, ...],
]:
    """Read a CSV table with UTF-8 BOM tolerance."""

    source = Path(
        path
    ).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {source}"
        )

    if not source.is_file():
        raise SurveyImportError(
            f"CSV path is not a file: {source}"
        )

    with source.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise SurveyImportError(
                f"CSV contains no header: {source}"
            )

        fieldnames = tuple(
            str(name).strip()
            for name in reader.fieldnames
            if name is not None
        )

        rows = [
            {
                (
                    str(key).strip()
                    if key is not None
                    else ""
                ): (
                    ""
                    if value is None
                    else value
                )
                for key, value in row.items()
            }
            for row in reader
        ]

    return (
        rows,
        fieldnames,
    )


def _resolve_coordinate_columns(
    fieldnames: Iterable[str],
    mapping: ViewpointColumnMapping,
) -> str:
    """Determine whether the table uses X/Y or lon/lat."""

    fields = set(
        fieldnames
    )

    has_xy = (
        mapping.x in fields
        and mapping.y in fields
    )

    has_lonlat = (
        mapping.longitude in fields
        and mapping.latitude in fields
    )

    if has_xy:
        return "xy"

    if has_lonlat:
        return "lonlat"

    raise SurveyImportError(
        "Viewpoint CSV must contain either "
        f"{mapping.x!r}/{mapping.y!r} or "
        f"{mapping.longitude!r}/{mapping.latitude!r} columns."
    )


def _required_cell(
    row: Mapping[str, Any],
    column: str,
    *,
    trim: bool = True,
) -> str:
    """Return one required text value."""

    if column not in row:
        raise SurveyImportError(
            f"Required column {column!r} is missing."
        )

    value = row[
        column
    ]

    if value is None:
        raise SurveyImportError(
            f"Required value {column!r} is empty."
        )

    result = str(
        value
    )

    if trim:
        result = result.strip()

    if not result:
        raise SurveyImportError(
            f"Required value {column!r} is empty."
        )

    return result


def _optional_cell(
    row: Mapping[str, Any],
    column: str,
    *,
    trim: bool = True,
) -> str | None:
    """Return an optional text value."""

    if column not in row:
        return None

    value = row[
        column
    ]

    if value is None:
        return None

    result = str(
        value
    )

    if trim:
        result = result.strip()

    if not result:
        return None

    return result


def _required_float_cell(
    row: Mapping[str, Any],
    column: str,
) -> float:
    """Return one required finite floating-point value."""

    value = _required_cell(
        row,
        column,
    )

    try:
        result = float(
            value
        )

    except ValueError as exc:
        raise SurveyImportError(
            f"{column!r} must be numeric; got {value!r}."
        ) from exc

    if not np.isfinite(
        result
    ):
        raise SurveyImportError(
            f"{column!r} must be finite."
        )

    return result


def _optional_float_cell(
    row: Mapping[str, Any],
    column: str,
) -> float | None:
    """Return an optional finite floating-point value."""

    value = _optional_cell(
        row,
        column,
    )

    if value is None:
        return None

    try:
        result = float(
            value
        )

    except ValueError as exc:
        raise SurveyImportError(
            f"{column!r} must be numeric; got {value!r}."
        ) from exc

    if not np.isfinite(
        result
    ):
        raise SurveyImportError(
            f"{column!r} must be finite."
        )

    return result


def parse_optional_iso_datetime(
    value: str | None,
) -> datetime | None:
    """Parse an optional ISO-8601 timestamp while preserving its timezone."""

    if value is None:
        return None

    normalized = value.strip()

    if not normalized:
        return None

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(
            normalized
        )

    except ValueError as exc:
        raise SurveyImportError(
            f"Invalid ISO-8601 timestamp: {value!r}."
        ) from exc


def _unmapped_metadata(
    row: Mapping[str, Any],
    mapped_columns: Iterable[str],
    *,
    trim: bool,
) -> dict[str, str]:
    """Preserve non-empty source columns without inventing canonical fields."""

    mapped = set(
        mapped_columns
    )

    return {
        str(column): value
        for column, raw_value in row.items()
        if column not in mapped
        and (
            value := _optional_cell(
                {str(column): raw_value},
                str(column),
                trim=trim,
            )
        ) is not None
    }


def _optional_spectral_bands(
    row: Mapping[str, Any],
    column: str,
    *,
    trim: bool,
) -> tuple[str, ...]:
    """Parse a simple comma- or semicolon-separated band list."""

    value = _optional_cell(
        row,
        column,
        trim=trim,
    )

    if value is None:
        return ()

    separator = ";" if ";" in value else ","

    return tuple(
        item.strip()
        for item in value.split(separator)
        if item.strip()
    )


def _optional_wavelength_range(
    row: Mapping[str, Any],
    column: str,
) -> tuple[float, float] | None:
    """Parse a simple two-value wavelength range."""

    value = _optional_cell(
        row,
        column,
    )

    if value is None:
        return None

    normalized = value.replace(
        "-",
        ",",
    ).replace(
        ":",
        ",",
    )

    parts = [
        part.strip()
        for part in normalized.split(",")
        if part.strip()
    ]

    if len(parts) != 2:
        raise SurveyImportError(
            f"{column!r} must contain two wavelength values; "
            f"got {value!r}."
        )

    try:
        result = tuple(
            float(part)
            for part in parts
        )

    except ValueError as exc:
        raise SurveyImportError(
            f"{column!r} must contain numeric wavelength values; "
            f"got {value!r}."
        ) from exc

    if not all(
        np.isfinite(part)
        for part in result
    ):
        raise SurveyImportError(
            f"{column!r} must contain finite wavelength values."
        )

    return result


def _optional_int_cell(
    row: Mapping[str, Any],
    column: str,
) -> int | None:
    """Return an optional integer value."""

    value = _optional_cell(
        row,
        column,
    )

    if value is None:
        return None

    try:
        numeric = float(
            value
        )

    except ValueError as exc:
        raise SurveyImportError(
            f"{column!r} must be an integer; got {value!r}."
        ) from exc

    if (
        not np.isfinite(
            numeric
        )
        or not numeric.is_integer()
    ):
        raise SurveyImportError(
            f"{column!r} must be an integer; got {value!r}."
        )

    return int(
        numeric
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _handle_row_error(
    *,
    exc: Exception,
    table: str,
    row_number: int,
    strict: bool,
    report: SurveyImportReport,
) -> None:
    """Apply strict or permissive row-error policy."""

    if isinstance(
        exc,
        SurveyImportRowError,
    ):
        row_error = exc

    else:
        row_error = SurveyImportRowError(
            table=table,
            row_number=row_number,
            message=str(exc),
        )

    if strict:
        raise row_error from exc

    report.errors.append(
        str(row_error)
    )


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------


def _coerce_crs(
    value: CRS | str | int,
) -> CRS:
    """Return Rasterio CRS from user input."""

    if isinstance(
        value,
        CRS,
    ):
        return value

    try:
        return CRS.from_user_input(
            value
        )

    except Exception as exc:
        raise SurveyImportError(
            f"Invalid CRS: {value!r}."
        ) from exc


# ---------------------------------------------------------------------------
# General validation
# ---------------------------------------------------------------------------


def _required_string(
    name: str,
    value: str,
) -> str:
    """Validate a required string."""

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
            f"{name} must be non-empty."
        )

    return result