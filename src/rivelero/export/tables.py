"""CSV table exports.

Conventions (identical for every table):

* UTF-8, comma separated, one header row, RFC 4180 quoting.
* A missing value is an empty cell - never ``None``, ``nan`` or ``0``.
* Timestamps are ISO 8601; coordinates are written with full precision
  (``repr``), so repeated or nearly identical coordinates are preserved.
* Identifiers and references (``viewpoint_id``, ``sensor_id`` ...) are
  written verbatim.
* Survey tables use the canonical field names, which are also the default
  column names of the survey importer, so they can be re-imported.
  ``extra_metadata`` keys become their own columns (the importer keeps
  unmapped columns as extra metadata again); nested dictionaries such as
  ``provenance`` are written as JSON.
* Shares are fractions in [0, 1] and every column name states its
  denominator.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from rivelero.analysis.comparison import ScenarioComparison, ScenarioSnapshot
from rivelero.analysis.contribution import ContributionAnalysis
from rivelero.analysis.coverage import CoverageSummary
from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.sensor import Sensor
from rivelero.export.files import atomic_output
from rivelero.export.metadata import (
    ExportProvenance,
    product_metadata,
    sidecar_path,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# Cell formatting and writing
# ---------------------------------------------------------------------------


def cell(value: Any) -> str:
    """Format one value for CSV (missing -> empty cell)."""

    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return repr(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) if value else ""
    if isinstance(value, (list, tuple)):
        return ";".join(cell(item) for item in value)
    if hasattr(value, "to_string"):  # rasterio CRS
        return value.to_string()
    return str(value)


def write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write rows atomically; values are formatted with :func:`cell`."""

    target = Path(path)
    with atomic_output(target, overwrite=overwrite) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([cell(row.get(column)) for column in columns])
    return target.expanduser().resolve()


def table_outputs(path: Path) -> tuple[Path, Path]:
    return path, sidecar_path(path)


def _write_table(
    path: Path,
    columns: Sequence[str],
    rows: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    existed = Path(path).exists()
    table = write_csv(path, columns, rows, overwrite=overwrite)
    record = {**record, "rows": len(rows)}
    try:
        sidecar = write_sidecar(table, record, overwrite=overwrite)
    except BaseException:
        if not existed:
            table.unlink(missing_ok=True)
        raise
    return table, sidecar


def _extra_columns(items: Iterable[dict[str, Any]], reserved: Sequence[str]) -> list[str]:
    """Union of extra-metadata keys, in first-seen order, avoiding clashes."""

    columns: list[str] = []
    for extra in items:
        for key in extra:
            name = _extra_name(key, reserved)
            if name not in columns:
                columns.append(name)
    return columns


def _extra_name(key: str, reserved: Sequence[str]) -> str:
    return f"extra_{key}" if key in reserved else str(key)


def _with_extra(row: dict[str, Any], extra: dict[str, Any], reserved: Sequence[str]) -> dict[str, Any]:
    for key, value in extra.items():
        row[_extra_name(key, reserved)] = value
    return row


# ---------------------------------------------------------------------------
# Survey tables
# ---------------------------------------------------------------------------

VIEWPOINT_COLUMNS = (
    "viewpoint_id", "x", "y", "z", "crs", "observer_height_m",
    "heading_deg", "pitch_deg", "roll_deg", "horizontal_fov_deg", "vertical_fov_deg",
    "sensor_id", "platform", "source", "source_id",
    "position_uncertainty_m", "orientation_uncertainty_deg", "provenance",
)

SENSOR_COLUMNS = (
    "sensor_id", "modality", "model", "manufacturer",
    "image_width_px", "image_height_px", "focal_length_mm",
    "sensor_width_mm", "sensor_height_mm", "horizontal_fov_deg", "vertical_fov_deg",
    "spectral_bands", "wavelength_range_nm", "spatial_resolution", "source",
)

EVENT_COLUMNS = (
    "event_id", "viewpoint_id", "timestamp", "sequence_id", "sequence_index",
    "image_id", "source", "observer_height_m",
    "heading_deg", "pitch_deg", "roll_deg", "horizontal_fov_deg", "vertical_fov_deg",
    "acquisition_conditions", "metadata_uncertainty",
)


def write_viewpoints(
    configuration: ViewpointConfiguration,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    viewpoints = list(configuration.viewpoints)
    extras = _extra_columns((vp.extra_metadata for vp in viewpoints), VIEWPOINT_COLUMNS)
    rows = [
        _with_extra(
            {name: getattr(vp, name) for name in VIEWPOINT_COLUMNS},
            vp.extra_metadata,
            VIEWPOINT_COLUMNS,
        )
        for vp in viewpoints
    ]
    record = product_metadata(
        provenance,
        product="viewpoints",
        meaning="Canonical Viewpoints of the Survey, one row per Viewpoint, in Survey order.",
        units="x/y/z in the units of the crs column; angles in degrees; heights in metres",
        nodata="empty cell = value not recorded",
        extra={
            "configuration_name": configuration.name,
            "extra_metadata_columns": extras,
            "reimport": (
                "Import with the survey importer using the default column names and "
                "the CRS in the crs column as source CRS. Extra columns return as "
                "extra metadata; crs and provenance also return as extra metadata."
            ),
        },
    )
    return _write_table(Path(path), [*VIEWPOINT_COLUMNS, *extras], rows, record, overwrite=overwrite)


def write_sensors(
    sensors: Sequence[Sensor],
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    extras = _extra_columns((s.extra_metadata for s in sensors), SENSOR_COLUMNS)
    rows = []
    for sensor in sensors:
        row = {name: getattr(sensor, name) for name in SENSOR_COLUMNS}
        wavelength = sensor.wavelength_range_nm
        row["wavelength_range_nm"] = (
            None if wavelength is None else f"{cell(float(wavelength[0]))},{cell(float(wavelength[1]))}"
        )
        rows.append(_with_extra(row, sensor.extra_metadata, SENSOR_COLUMNS))
    record = product_metadata(
        provenance,
        product="sensors",
        meaning="Sensors of the Survey, one row per Sensor.",
        units="angles in degrees; lengths in mm; wavelength in nm",
        nodata="empty cell = value not recorded",
        extra={
            "list_separators": {"spectral_bands": ";", "wavelength_range_nm": ","},
            "extra_metadata_columns": extras,
        },
    )
    return _write_table(Path(path), [*SENSOR_COLUMNS, *extras], rows, record, overwrite=overwrite)


def write_observation_events(
    configuration: ViewpointConfiguration,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    events = list(configuration.observation_events)
    extras = _extra_columns((e.extra_metadata for e in events), EVENT_COLUMNS)
    rows = [
        _with_extra(
            {name: getattr(event, name) for name in EVENT_COLUMNS},
            event.extra_metadata,
            EVENT_COLUMNS,
        )
        for event in events
    ]
    record = product_metadata(
        provenance,
        product="observation_events",
        meaning=(
            "ObservationEvents of the Survey, one row per event; viewpoint_id "
            "references the viewpoints table."
        ),
        units="angles in degrees; heights in metres; timestamp ISO 8601",
        nodata="empty cell = value not recorded (event inherits the Viewpoint value)",
        extra={"extra_metadata_columns": extras},
    )
    return _write_table(Path(path), [*EVENT_COLUMNS, *extras], rows, record, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------

CONTRIBUTION_COLUMNS = {
    "sampling_unit_id": "Identifier of the sampling unit (Viewpoint or ObservationEvent).",
    "sampling_unit_type": "viewpoint or observation_event.",
    "viewpoint_id": "Viewpoint of the sampling unit.",
    "observation_event_id": "ObservationEvent of the sampling unit (empty for Viewpoints).",
    "status": "available, or why the contribution could not be measured.",
    "visible_cells": "Analysable cells visible from this unit.",
    "unique_cells": "Analysable cells visible from this unit and no other active unit.",
    "repeated_cells": "Analysable cells visible from this unit and at least one other.",
    "cells_lost_if_removed": "Observable cells that become blind if only this unit is removed (= unique_cells).",
    "coverage_lost_if_removed_share_of_analysable": "cells_lost_if_removed / analysable_cells.",
    "visible_share_of_analysable": "visible_cells / analysable_cells.",
    "unique_share_of_unit_visibility": "unique_cells / visible_cells.",
    "repeated_share_of_unit_visibility": "repeated_cells / visible_cells.",
    "analysable_cells": "Denominator: analysable cells of the observability field.",
    "message": "Explanation when status is not available.",
}


def write_contributions(
    analysis: ContributionAnalysis,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    unit_type = getattr(analysis.sampling_unit, "value", str(analysis.sampling_unit))
    rows = [
        {
            "sampling_unit_id": unit.sampling_unit_id,
            "sampling_unit_type": unit_type,
            "viewpoint_id": unit.viewpoint_id,
            "observation_event_id": unit.observation_event_id,
            "status": unit.status,
            "visible_cells": unit.visible_cells,
            "unique_cells": unit.unique_cells,
            "repeated_cells": unit.repeated_cells,
            "cells_lost_if_removed": unit.cells_lost_if_removed,
            "coverage_lost_if_removed_share_of_analysable": unit.coverage_loss_if_removed,
            "visible_share_of_analysable": unit.visible_share_of_analysable,
            "unique_share_of_unit_visibility": unit.unique_share_of_unit_visibility,
            "repeated_share_of_unit_visibility": unit.repeated_share_of_unit_visibility,
            "analysable_cells": unit.analysable_cells,
            "message": unit.message,
        }
        for unit in analysis.units
    ]
    record = product_metadata(
        provenance,
        product="sampling_unit_contributions",
        meaning=(
            "Descriptive contribution of each active sampling unit to the "
            "observability field. A unit without unique cells still contributes "
            "repeated observation; the table makes no judgement of a unit's value."
        ),
        units="cells; shares are fractions in [0, 1]",
        nodata="empty cell = not measurable for this unit (see status and message)",
        columns=CONTRIBUTION_COLUMNS,
        extra={
            "contribution_sof_id": analysis.sof_id,
            "field_totals": {
                "analysable_cells": analysis.analysable_cells,
                "observable_cells": analysis.field_observable_cells,
                "unique_cells": analysis.field_unique_cells,
                "total_exposure": analysis.field_total_exposure,
            },
            "complete": analysis.complete,
        },
    )
    return _write_table(Path(path), list(CONTRIBUTION_COLUMNS), rows, record, overwrite=overwrite)


def _coverage_values(summary: CoverageSummary) -> dict[str, Any]:
    return {
        "analysable_cells": summary.analysable_cells,
        "observable_cells": summary.observable_cells,
        "blind_cells": summary.blind_cells,
        "unique_cells": summary.unique_cells,
        "repeated_cells": summary.repeated_cells,
        "observable_share_of_analysable": summary.observable_fraction,
        "blind_share_of_analysable": summary.blind_fraction,
        "mean_exposure": summary.mean_exposure,
        "maximum_exposure": summary.maximum_exposure,
    }


SCENARIO_COLUMNS = {
    "state_id": "baseline, live, or the saved snapshot id.",
    "kind": "baseline, live_scenario or saved_snapshot.",
    "label": "Name of the design state.",
    "compatible_with_current_baseline": "false = saved for an earlier observability field; not comparable cell by cell.",
    "baseline_sof_id": "Observability field the state modifies.",
    "active_existing_units": "Active sampling units of the Survey.",
    "deactivated_units": "Survey sampling units deactivated in the scenario.",
    "included_candidates": "Candidate Viewpoints included in the scenario.",
    "sampling_units": "active_existing_units + included_candidates.",
    "analysable_cells": "Denominator of every share.",
    "observable_cells": "Analysable cells with exposure >= 1.",
    "blind_cells": "Analysable cells with exposure 0.",
    "unique_cells": "Analysable cells with exposure exactly 1.",
    "repeated_cells": "Analysable cells with exposure >= 2.",
    "observable_share_of_analysable": "observable_cells / analysable_cells.",
    "blind_share_of_analysable": "blind_cells / analysable_cells.",
    "mean_exposure": "Mean exposure over analysable cells.",
    "maximum_exposure": "Maximum exposure.",
    "observable_cells_change_from_baseline": "state minus its baseline.",
    "coverage_change_from_baseline_percentage_points": "100 x (state share - baseline share).",
    "created_at": "When a snapshot was saved.",
    "description": "Snapshot description.",
}


def write_scenario_summary(
    rows: list[dict[str, Any]],
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    record = product_metadata(
        provenance,
        product="scenario_summary",
        meaning=(
            "Descriptive summary of the baseline, the current what-if scenario and "
            "saved scenario snapshots. No ranking or score."
        ),
        units="cells; shares are fractions in [0, 1]; changes in percentage points",
        nodata="empty cell = not applicable",
        columns=SCENARIO_COLUMNS,
    )
    return _write_table(Path(path), list(SCENARIO_COLUMNS), rows, record, overwrite=overwrite)


def scenario_row(
    *,
    state_id: str,
    kind: str,
    label: str,
    compatible: bool,
    baseline_sof_id: str,
    coverage: CoverageSummary,
    baseline: CoverageSummary | None,
    active_existing_units: int,
    deactivated_units: int,
    included_candidates: int,
    snapshot: ScenarioSnapshot | None = None,
) -> dict[str, Any]:
    row = {
        "state_id": state_id,
        "kind": kind,
        "label": label,
        "compatible_with_current_baseline": compatible,
        "baseline_sof_id": baseline_sof_id,
        "active_existing_units": active_existing_units,
        "deactivated_units": deactivated_units,
        "included_candidates": included_candidates,
        "sampling_units": active_existing_units + included_candidates,
        **_coverage_values(coverage),
    }
    if baseline is not None:
        row["observable_cells_change_from_baseline"] = (
            coverage.observable_cells - baseline.observable_cells
        )
        if coverage.observable_fraction is not None and baseline.observable_fraction is not None:
            row["coverage_change_from_baseline_percentage_points"] = 100.0 * (
                coverage.observable_fraction - baseline.observable_fraction
            )
    if snapshot is not None:
        row["created_at"] = snapshot.created_at
        row["description"] = snapshot.description or None
    return row


COMPARISON_COLUMNS = ("metric", "left", "right", "difference_right_minus_left", "unit")


def comparison_rows(comparison: ScenarioComparison) -> list[dict[str, Any]]:
    left, right = comparison.left, comparison.right
    ls, rs = left.summary, right.summary

    def pct(value):
        return None if value is None else 100.0 * value

    rows = [
        ("sampling_units", left.sampling_units, right.sampling_units,
         comparison.sampling_unit_count_delta, "sampling units"),
        ("active_existing_units", left.active_existing_units, right.active_existing_units,
         comparison.active_existing_units_delta, "sampling units"),
        ("included_candidates", left.included_candidates, right.included_candidates,
         comparison.included_candidates_delta, "viewpoints"),
        ("analysable_cells", ls.analysable_cells, rs.analysable_cells,
         rs.analysable_cells - ls.analysable_cells, "cells"),
        ("observable_cells", ls.observable_cells, rs.observable_cells,
         comparison.observable_cells_delta, "cells"),
        ("blind_cells", ls.blind_cells, rs.blind_cells, comparison.blind_cells_delta, "cells"),
        ("unique_cells", ls.unique_cells, rs.unique_cells, comparison.unique_cells_delta, "cells"),
        ("repeated_cells", ls.repeated_cells, rs.repeated_cells,
         comparison.repeated_cells_delta, "cells"),
        ("observable_percent_of_analysable", pct(ls.observable_fraction),
         pct(rs.observable_fraction), comparison.coverage_percentage_point_delta,
         "percent; difference in percentage points"),
        ("mean_exposure", ls.mean_exposure, rs.mean_exposure,
         comparison.mean_exposure_delta, "sampling units per analysable cell"),
        ("maximum_exposure", ls.maximum_exposure, rs.maximum_exposure,
         comparison.maximum_exposure_delta, "sampling units"),
        ("cells_gained_coverage", None, None, int(comparison.gained_coverage_mask.sum()),
         "cells blind on the left and observable on the right"),
        ("cells_lost_coverage", None, None, int(comparison.lost_coverage_mask.sum()),
         "cells observable on the left and blind on the right"),
    ]
    return [dict(zip(COMPARISON_COLUMNS, row)) for row in rows]


def write_comparison(
    comparison: ScenarioComparison,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    record = product_metadata(
        provenance,
        product="comparison_summary",
        meaning=(
            "Descriptive comparison of two design states of one baseline. Every "
            "difference is RIGHT minus LEFT. No ranking or combined score."
        ),
        units="see the unit column",
        nodata="empty cell = not applicable (cell-change counts have no left/right value)",
        columns={
            "metric": "Measured quantity.",
            "left": "Value in the LEFT design state.",
            "right": "Value in the RIGHT design state.",
            "difference_right_minus_left": "right - left; for percentages, percentage points.",
            "unit": "Unit of left, right and the difference.",
        },
        extra={
            "direction": "right_minus_left",
            "left": {"state_id": comparison.left.state_id, "label": comparison.left.label},
            "right": {"state_id": comparison.right.state_id, "label": comparison.right.label},
        },
    )
    return _write_table(
        Path(path), COMPARISON_COLUMNS, comparison_rows(comparison), record, overwrite=overwrite
    )
