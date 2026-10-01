"""Reproducibility manifest (P3): JSON and Markdown.

The manifest describes the analysis that produced a set of outputs, from the
canonical state collected in an ExportContext. It deliberately keeps three
kinds of statement apart:

``source_metadata``
    What the Survey data *contained* (e.g. "37 Viewpoints have no heading").
``assumptions``
    What the VisibilityConfiguration *assumed* for missing values (e.g.
    "missing heading policy: omnidirectional") and how often each
    assumption was applied, counted with the engine's own
    ``resolve_visibility_parameters`` - never re-implemented here.
``observability`` / ``analysis``
    Results derived from the current observability field only. Saved
    scenarios of an earlier field are listed with ``status: out_of_date``.

Warnings and limitations are included only when they apply to the current
analysis.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from rivelero.analysis.coverage import CoverageSummary, summarize_coverage
from rivelero.export.catalog import ExportContext
from rivelero.export.files import atomic_output
from rivelero.project.resources import file_sha256
from rivelero.project.schema import SCHEMA_VERSION, software_metadata
from rivelero.visibility.configuration import MissingMetadataPolicy, SamplingUnit
from rivelero.visibility.engine import ViewpointExcludedError, resolve_visibility_parameters

MANIFEST_FORMAT = "rivelero-provenance"
MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """JSON-safe copy of canonical values (enums, paths, CRS, datetimes...)."""

    if isinstance(value, Enum):  # before str/int: many enums subclass them
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _plain(float(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_string"):
        return value.to_string()
    return str(value)


def _share(part: int, whole: int) -> float | None:
    return None if not whole else part / whole


def coverage_values(summary: CoverageSummary) -> dict[str, Any]:
    return {
        "total_cells": summary.total_cells,
        "outside_domain_cells": summary.outside_domain_cells,
        "invalid_cells": summary.invalid_cells,
        "analysable_cells": summary.analysable_cells,
        "observable_cells": summary.observable_cells,
        "blind_cells": summary.blind_cells,
        "unique_cells": summary.unique_cells,
        "repeated_cells": summary.repeated_cells,
        "observable_share_of_analysable": summary.observable_fraction,
        "blind_share_of_analysable": summary.blind_fraction,
        "active_sampling_units": summary.active_units,
        "mean_exposure": summary.mean_exposure,
        "median_exposure": summary.median_exposure,
        "mean_exposure_observable": summary.mean_exposure_observable,
        "maximum_exposure": summary.maximum_exposure,
        "total_exposure": summary.total_exposure,
    }


def coverage_checks(summary: CoverageSummary) -> list[dict[str, Any]]:
    """Identities every reported coverage summary must satisfy."""

    return [
        {"check": "observable + blind == analysable",
         "holds": summary.observable_cells + summary.blind_cells == summary.analysable_cells},
        {"check": "unique + repeated == observable",
         "holds": summary.unique_cells + summary.repeated_cells == summary.observable_cells},
        {"check": "outside + invalid + analysable == total",
         "holds": summary.outside_domain_cells + summary.invalid_cells
         + summary.analysable_cells == summary.total_cells},
    ]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _survey(context: ExportContext) -> dict[str, Any] | None:
    configuration = context.configuration
    if configuration is None:
        return None
    return {
        "configuration_id": configuration.configuration_id,
        "name": configuration.name,
        "configuration_type": _plain(configuration.configuration_type),
        "description": configuration.description,
        "created_at": _plain(configuration.created_at),
        "viewpoints": configuration.n_viewpoints,
        "observation_events": configuration.n_events,
        "sensors": [
            {"sensor_id": s.sensor_id, "modality": _plain(s.modality), "model": s.model,
             "manufacturer": s.manufacturer, "horizontal_fov_deg": s.horizontal_fov_deg,
             "vertical_fov_deg": s.vertical_fov_deg}
            for s in context.sensors
        ],
        "sources": list(configuration.sources),
        "has_timestamps": configuration.has_temporal_information,
        "provenance": _plain(configuration.provenance),
    }


def _missing(values) -> int:
    return sum(1 for value in values if value is None)


def _source_metadata(context: ExportContext) -> dict[str, Any] | None:
    """What the Survey data contained - no assumption applied."""

    configuration = context.configuration
    if configuration is None:
        return None
    viewpoints = list(configuration.viewpoints)
    events = list(configuration.observation_events)
    sensors = {s.sensor_id: s for s in context.sensors}
    by_id = {vp.viewpoint_id: vp for vp in viewpoints}

    def fov_known(vp) -> bool:
        sensor = sensors.get(vp.sensor_id) if vp.sensor_id else None
        return vp.horizontal_fov_deg is not None or (
            sensor is not None and sensor.horizontal_fov_deg is not None
        )

    event_heading_missing = sum(
        1 for e in events
        if e.heading_deg is None and getattr(by_id.get(e.viewpoint_id), "heading_deg", None) is None
    )
    vertical_values = (
        sum(1 for vp in viewpoints if vp.vertical_fov_deg is not None)
        + sum(1 for e in events if e.vertical_fov_deg is not None)
        + sum(1 for s in sensors.values() if s.vertical_fov_deg is not None)
    )
    return {
        "note": "Counts of values absent from the source data; no assumption applied.",
        "viewpoints": {
            "total": len(viewpoints),
            "missing_heading": _missing(vp.heading_deg for vp in viewpoints),
            "missing_observer_height": _missing(vp.observer_height_m for vp in viewpoints),
            "missing_horizontal_fov_viewpoint_and_sensor": sum(
                1 for vp in viewpoints if not fov_known(vp)),
            "missing_sensor_reference": _missing(vp.sensor_id for vp in viewpoints),
            "unknown_sensor_reference": sum(
                1 for vp in viewpoints if vp.sensor_id and vp.sensor_id not in sensors),
            "missing_elevation_z": _missing(vp.z for vp in viewpoints),
            "missing_position_uncertainty": _missing(vp.position_uncertainty_m for vp in viewpoints),
        },
        "observation_events": {
            "total": len(events),
            "missing_timestamp": _missing(e.timestamp for e in events),
            "missing_heading_event_and_viewpoint": event_heading_missing,
        },
        "vertical_fov_values_present": vertical_values,
    }


def _assumptions(context: ExportContext) -> dict[str, Any] | None:
    """The configured assumptions and how often each one was applied."""

    config = context.visibility_configuration
    if config is None:
        return None
    parameters = {f.name: _plain(getattr(config, f.name)) for f in fields(config)}
    applied = None
    configuration = context.configuration
    if configuration is not None:
        applied = _applied_assumptions(context, config, configuration)
    return {
        "note": ("Values assumed by the VisibilityConfiguration; they are modelling "
                 "choices, not observed metadata."),
        "configuration": parameters,
        "applied": applied,
    }


def _applied_assumptions(context, config, configuration) -> dict[str, Any]:
    sensors = {s.sensor_id: s for s in context.sensors}
    by_id = {vp.viewpoint_id: vp for vp in configuration.viewpoints}
    if config.sampling_unit == SamplingUnit.OBSERVATION_EVENT:
        units = [(by_id.get(e.viewpoint_id), e) for e in configuration.observation_events]
    else:
        units = [(vp, None) for vp in configuration.viewpoints]

    counts = {
        "sampling_units": len(units),
        "used_default_observer_height": 0,
        "heading_missing": 0,
        "heading_missing_treated_omnidirectional": 0,
        "heading_missing_used_default": 0,
        "used_default_horizontal_fov": 0,
        "omnidirectional": 0,
        "excluded_by_policy": 0,
        "unresolvable": 0,
    }
    for viewpoint, event in units:
        if viewpoint is None:
            counts["unresolvable"] += 1
            continue
        heading_missing = (event is None or event.heading_deg is None) and viewpoint.heading_deg is None
        try:
            resolved = resolve_visibility_parameters(
                viewpoint=viewpoint, configuration=config,
                sensor=sensors.get(viewpoint.sensor_id) if viewpoint.sensor_id else None,
                event=event,
            )
        except ViewpointExcludedError:
            counts["excluded_by_policy"] += 1
            continue
        except (ValueError, TypeError):
            counts["unresolvable"] += 1
            continue
        counts["used_default_observer_height"] += resolved.used_default_observer_height
        counts["used_default_horizontal_fov"] += resolved.used_default_horizontal_fov
        counts["omnidirectional"] += resolved.omnidirectional
        if config.use_direction and heading_missing:
            counts["heading_missing"] += 1
            if resolved.used_default_heading:
                counts["heading_missing_used_default"] += 1
            elif resolved.omnidirectional:
                counts["heading_missing_treated_omnidirectional"] += 1
    counts["note"] = ("Counted per sampling unit with the visibility engine's "
                      "parameter resolution (Event -> Viewpoint -> Sensor -> configuration).")
    return counts


def _world(context: ExportContext) -> dict[str, Any] | None:
    environment = context.environment
    domain = context.analysis_domain
    if environment is None and domain is None:
        return None
    world: dict[str, Any] = {}
    if environment is not None:
        model = environment.elevation_model
        source = Path(model.source)
        dem: dict[str, Any] = {
            "path": str(source), "source_name": model.source_name,
            "model_type": _plain(model.model_type), "crs": _plain(model.crs),
            "resolution_m": model.resolution_m, "nodata_value": model.nodata_value,
            "vertical_datum": model.vertical_datum,
            "vertical_accuracy_m": model.vertical_accuracy_m,
            "acquisition_date": _plain(model.acquisition_date),
            "linked_not_embedded": True,
        }
        if source.is_file():
            stat = source.stat()
            dem.update(size_bytes=stat.st_size, sha256=file_sha256(source), status="found")
        else:
            dem.update(size_bytes=None, sha256=None, status="missing")
        world["environment"] = {
            "environment_id": environment.environment_id, "name": environment.name,
            "description": environment.description,
            "layers": [getattr(layer, "layer_id", None) for layer in environment.layers],
            "elevation": dem,
        }
    if domain is not None:
        grid = domain.grid
        entry: dict[str, Any] = {
            "domain_id": domain.domain_id, "name": domain.name,
            "creation_method": domain.creation_method,
            "crs": _plain(domain.crs),
            "grid": {"width": grid.width, "height": grid.height,
                     "transform": list(grid.transform)[:6],
                     "resolution": [grid.resolution_x, grid.resolution_y]},
        }
        sof = context.sof
        if sof is not None and sof.analysis_domain_id == domain.domain_id:
            counts = sof.state_counts()
            entry["cells"] = {
                "outside_domain": int(counts[0]), "invalid": int(counts[1]),
                "analysable": int(counts[2] + counts[3]),
            }
        else:
            analysis = domain.effective_analysis_mask
            valid = domain.effective_valid_mask
            entry["cells"] = {
                "outside_domain": int((~analysis).sum()),
                "invalid": int((analysis & ~valid).sum()),
                "analysable": int((analysis & valid).sum()),
            }
        world["analysis_domain"] = entry
    return world


def _observability(context: ExportContext, summary: CoverageSummary | None) -> dict[str, Any] | None:
    sof = context.sof
    if sof is None:
        if context.invalidation_reason:
            return {"status": "out_of_date", "reason": context.invalidation_reason,
                    "note": "No current observability field; no results are reported."}
        return None
    report = context.build_report
    build = None
    if report is not None:
        build = {
            "requested_units": report.requested_units, "added_units": report.added_units,
            "excluded_units": report.excluded_units, "failed_units": report.failed_units,
            "cache_hits": report.cache_hits, "computed_units": report.computed_units,
            "complete": report.complete,
            "excluded_ids": list(report.excluded_ids)[:200],
            "failed_ids": list(report.failed_ids)[:200],
            "errors": dict(list(report.errors.items())[:50]),
        }
    return {
        "status": "current",
        "sof_id": sof.sof_id,
        "created_at": _plain(sof.created_at),
        "viewpoint_configuration_id": sof.viewpoint_configuration_id,
        "environment_id": sof.environment_id,
        "analysis_domain_id": sof.analysis_domain_id,
        "visibility_configuration_id": sof.visibility_configuration_id,
        "sampling_unit": _plain(sof.sampling_unit),
        "active_sampling_units": sof.n_active_units,
        "grid": {"crs": _plain(sof.crs), "transform": list(sof.transform)[:6],
                 "height": int(sof.exposure_count.shape[0]),
                 "width": int(sof.exposure_count.shape[1])},
        "build_report": build,
        "coverage": coverage_values(summary),
        "exposure_distribution": [
            {"exposure": int(level), "cells": int(summary.distribution.cells_at(int(level)))}
            for level in summary.distribution.levels[:100]
        ],
        "checks": coverage_checks(summary),
        "visibility_cache": context.visibility_store_directory,
    }


def _summary_numbers(summary: CoverageSummary) -> dict[str, Any]:
    return {
        "analysable_cells": summary.analysable_cells,
        "observable_cells": summary.observable_cells,
        "blind_cells": summary.blind_cells,
        "unique_cells": summary.unique_cells,
        "repeated_cells": summary.repeated_cells,
        "observable_share_of_analysable": summary.observable_fraction,
        "mean_exposure": summary.mean_exposure,
        "maximum_exposure": summary.maximum_exposure,
    }


def _analysis(context: ExportContext) -> dict[str, Any]:
    sof = context.sof
    analysis: dict[str, Any] = {}
    contribution = context.contribution
    if contribution is not None:
        analysis["contribution"] = {
            "status": "current",
            "sof_id": contribution.sof_id,
            "sampling_unit": _plain(contribution.sampling_unit),
            "units": contribution.n_units,
            "available_units": len(contribution.available_units),
            "unavailable_units": len(contribution.unavailable_units),
            "units_with_unique_cells": contribution.n_with_unique_coverage,
            "units_without_unique_cells": contribution.n_without_unique_coverage,
            "units_without_visible_cells": contribution.n_without_visible_cells,
            "unique_cells_summed_over_units": contribution.total_unique_contribution_cells,
            "field_unique_cells": contribution.field_unique_cells,
            "consistent_with_field": contribution.consistent_with_field,
        }
    scenario = context.scenario_summary
    if scenario is not None:
        difference = scenario.difference
        analysis["scenario"] = {
            "status": "current",
            "baseline_sof_id": scenario.baseline_sof_id,
            "baseline_units": scenario.baseline_units,
            "active_existing_units": scenario.active_existing_units,
            "deactivated_units": scenario.deactivated_units,
            "included_candidates": scenario.included_candidates,
            "definition": _plain(context.scenario_record),
            "baseline": _summary_numbers(scenario.baseline),
            "scenario": _summary_numbers(scenario.scenario),
            "difference_scenario_minus_baseline": {
                "observable_cells": difference.observable_cells,
                "blind_cells": difference.blind_cells,
                "unique_cells": difference.unique_cells,
                "repeated_cells": difference.repeated_cells,
                "coverage_percentage_points": difference.coverage_percentage_points,
            },
            "checks": coverage_checks(scenario.scenario),
        }
    if context.snapshots:
        analysis["saved_scenarios"] = [
            {
                "snapshot_id": s.snapshot_id, "name": s.name, "description": s.description,
                "created_at": _plain(s.created_at), "baseline_sof_id": s.baseline_sof_id,
                "status": "current" if s.compatible_with(sof) else "out_of_date",
                "deactivated_units": s.summary.deactivated_units,
                "included_candidates": s.summary.included_candidates,
                "summary": _summary_numbers(s.summary.scenario),
            }
            for s in context.snapshots
        ]
    comparison = context.comparison
    if comparison is not None:
        gained = int(comparison.gained_coverage_mask.sum())
        lost = int(comparison.lost_coverage_mask.sum())
        analysis["comparison"] = {
            "status": "current",
            "direction": "right_minus_left",
            "left": {"state_id": comparison.left.state_id, "label": comparison.left.label,
                     **_summary_numbers(comparison.left.summary)},
            "right": {"state_id": comparison.right.state_id, "label": comparison.right.label,
                      **_summary_numbers(comparison.right.summary)},
            "difference_right_minus_left": {
                "observable_cells": comparison.observable_cells_delta,
                "blind_cells": comparison.blind_cells_delta,
                "unique_cells": comparison.unique_cells_delta,
                "repeated_cells": comparison.repeated_cells_delta,
                "coverage_percentage_points": comparison.coverage_percentage_point_delta,
                "mean_exposure": comparison.mean_exposure_delta,
                "sampling_units": comparison.sampling_unit_count_delta,
            },
            "cells_gained_coverage": gained,
            "cells_lost_coverage": lost,
            "checks": [{"check": "observable delta == gained - lost",
                        "holds": comparison.observable_cells_delta == gained - lost}],
        }
    return analysis


def _warnings(context: ExportContext, manifest: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        warnings.append({"code": code, "message": message})

    provenance = context.provenance
    software = manifest["software"]
    if software.get("version") is None:
        add("software_version_unavailable",
            f"Rivelero is not installed as a package, so no version number is available; "
            f"git commit {software.get('git_commit') or 'unknown'} identifies the code."
            if software.get("git_commit") else
            "Neither the Rivelero version nor a git commit could be determined.")
    if not provenance.project_saved:
        add("project_unsaved",
            "The project has unsaved changes or was never saved; the outputs may not be "
            "reproducible from a saved .rivelero file.")
    world = manifest.get("world") or {}
    dem = (world.get("environment") or {}).get("elevation")
    if dem is not None:
        add("terrain_external",
            "Terrain is linked by path, not embedded in the project; keep the DEM file "
            "(checksum recorded) to reproduce the analysis.")
        if dem.get("status") == "missing":
            add("terrain_missing", "The linked DEM file could not be found; no checksum recorded.")
    observability = manifest.get("observability") or {}
    if observability.get("status") == "out_of_date":
        add("observability_out_of_date",
            f"The observability field is out of date ({observability['reason']}); "
            "derived results are omitted.")
    if observability.get("status") == "current":
        add("visibility_cache_external",
            "Cached visibility masks are stored outside the project "
            f"({context.visibility_store_directory or 'unknown location'}); a rebuild "
            "recomputes them.")
        build = observability.get("build_report") or {}
        if build and not build.get("complete", True):
            add("build_incomplete",
                f"{build['excluded_units']} sampling unit(s) were excluded and "
                f"{build['failed_units']} failed during the build.")
        if observability["coverage"]["invalid_cells"]:
            add("dem_nodata_in_domain",
                f"{observability['coverage']['invalid_cells']:,} cells inside the AnalysisDomain "
                "have no valid terrain; sight lines crossing DEM NoData may be unreliable "
                "under the current GDAL backend.")
    stale = [s for s in (manifest.get("analysis") or {}).get("saved_scenarios", ())
             if s["status"] == "out_of_date"]
    if stale:
        add("snapshots_out_of_date",
            f"{len(stale)} saved scenario(s) belong to an earlier observability field; "
            "they are listed but not compared.")
    source = manifest.get("source_metadata") or {}
    if source.get("vertical_fov_values_present"):
        add("vertical_fov_not_modelled",
            f"{source['vertical_fov_values_present']} vertical field-of-view value(s) are "
            "present in the source data but vertical FOV is not modelled.")
    environment = world.get("environment") or {}
    if environment.get("layers"):
        add("obstacles_not_modelled",
            "The Environment contains additional layers, but environmental obstacle "
            "layers are not modelled; only terrain occludes sight lines.")
    contribution = (manifest.get("analysis") or {}).get("contribution")
    if contribution and contribution["unavailable_units"]:
        add("contribution_incomplete",
            f"{contribution['unavailable_units']} sampling unit(s) had no readable cached "
            "visibility in the contribution analysis.")
    return warnings


def _limitations(context: ExportContext, manifest: dict[str, Any]) -> list[str]:
    if context.sof is None:
        return []
    limitations = [
        "Observability represents modelled observation opportunity under the configured "
        "visibility model, not guaranteed detection.",
        "Only terrain elevation occludes sight lines; vegetation, buildings and other "
        "environmental obstacles were not modelled.",
        "Vertical field of view (pitch/vertical FOV) was not modelled.",
        "Normalized exposure divides by the number of active sampling units (global "
        "normalization).",
    ]
    assumptions = manifest.get("assumptions") or {}
    config = assumptions.get("configuration") or {}
    applied = assumptions.get("applied") or {}
    if config and not config.get("use_direction", True):
        limitations.append("Viewing direction was not used: every sampling unit was treated "
                           "as omnidirectional.")
    elif applied.get("heading_missing"):
        policy = config.get("missing_heading_policy")
        limitations.append(
            f"{applied['heading_missing']:,} sampling unit(s) had no heading; they were "
            f"interpreted with the configured missing-heading policy ({policy}).")
    if applied.get("used_default_observer_height"):
        limitations.append(
            f"{applied['used_default_observer_height']:,} sampling unit(s) used the default "
            f"observer height ({config.get('default_observer_height_m')} m).")
    if applied.get("used_default_horizontal_fov"):
        limitations.append(
            f"{applied['used_default_horizontal_fov']:,} sampling unit(s) used the default "
            f"horizontal FOV ({config.get('default_horizontal_fov_deg')}°).")
    observability = manifest.get("observability") or {}
    if (observability.get("coverage") or {}).get("invalid_cells"):
        limitations.append("Sight lines crossing DEM NoData may be unreliable under the "
                           "current GDAL backend.")
    return limitations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_manifest(
    context: ExportContext,
    *,
    outputs: list[dict[str, Any]] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the provenance manifest (a JSON-serialisable dict)."""

    provenance = context.provenance
    summary = None if context.sof is None else summarize_coverage(context.sof)
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "software": software_metadata(),
        "project_schema_version": SCHEMA_VERSION,
        "project": {
            "name": provenance.project_name,
            "path": provenance.project_path,
            "saved": provenance.project_saved,
            "description": context.project_description,
        },
        "survey": _survey(context),
        "source_metadata": _source_metadata(context),
        "world": _world(context),
        "assumptions": _assumptions(context),
        "observability": _observability(context, summary),
        "analysis": _analysis(context),
    }
    manifest["warnings"] = _warnings(context, manifest)
    manifest["limitations"] = _limitations(context, manifest)
    manifest["outputs"] = list(outputs or [])
    return _plain(manifest)


def write_json(manifest: dict[str, Any], path: Path, *, overwrite: bool = False) -> Path:
    with atomic_output(path, overwrite=overwrite) as temporary:
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                        allow_nan=False), encoding="utf-8")
    return Path(path).expanduser().resolve()


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _md(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _kv(title: str, data: dict[str, Any] | None, *, level: int = 2) -> list[str]:
    if not data:
        return []
    lines = [f"{'#' * level} {title}", ""]
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- **{key.replace('_', ' ')}**: {_md(value)}")
    lines.append("")
    return lines


def manifest_markdown(manifest: dict[str, Any]) -> str:
    """Human-readable rendering of the manifest."""

    software = manifest["software"]
    lines = [
        "# Rivelero provenance",
        "",
        f"Generated {manifest['generated_at']} by Rivelero "
        f"{software.get('version') or '(version unavailable)'}"
        f" · git commit {software.get('git_commit') or 'unknown'}"
        f" · project schema {manifest['project_schema_version']}.",
        "",
    ]
    lines += _kv("Project", manifest["project"])
    survey = manifest.get("survey")
    if survey:
        lines += _kv("Survey", survey)
        if survey["sensors"]:
            lines += ["| Sensor | Modality | Model | Horizontal FOV (°) |", "|---|---|---|---|"]
            lines += [f"| {_md(s['sensor_id'])} | {_md(s['modality'])} | {_md(s['model'])} | "
                      f"{_md(s['horizontal_fov_deg'])} |" for s in survey["sensors"]]
            lines.append("")
    source = manifest.get("source_metadata")
    if source:
        lines += ["## Source metadata (as recorded)", "", source["note"], ""]
        lines += _kv("Viewpoints", source["viewpoints"], level=3)
        lines += _kv("ObservationEvents", source["observation_events"], level=3)
    assumptions = manifest.get("assumptions")
    if assumptions:
        lines += ["## Visibility assumptions (configured)", "", assumptions["note"], ""]
        lines += _kv("Configuration", assumptions["configuration"], level=3)
        if assumptions.get("applied"):
            lines += _kv("Assumptions applied to sampling units", assumptions["applied"], level=3)
    world = manifest.get("world")
    if world:
        environment = world.get("environment")
        if environment:
            lines += _kv("Environment", environment)
            lines += _kv("Elevation model", environment["elevation"], level=3)
        domain = world.get("analysis_domain")
        if domain:
            lines += _kv("AnalysisDomain", domain)
            lines += _kv("Cells", domain["cells"], level=3)
    observability = manifest.get("observability")
    if observability:
        lines += _kv("Observability", observability)
        if observability.get("build_report"):
            lines += _kv("Build report", observability["build_report"], level=3)
        if observability.get("coverage"):
            lines += _kv("Coverage", observability["coverage"], level=3)
        for check in observability.get("checks", ()):
            lines.append(f"- check `{check['check']}`: {'holds' if check['holds'] else 'FAILS'}")
        lines.append("")
    analysis = manifest.get("analysis") or {}
    if analysis.get("contribution"):
        lines += _kv("Sampling-unit contribution", analysis["contribution"])
    if analysis.get("scenario"):
        scenario = analysis["scenario"]
        lines += _kv("Current scenario", scenario)
        lines += _kv("Difference (scenario − baseline)",
                     scenario["difference_scenario_minus_baseline"], level=3)
    if analysis.get("saved_scenarios"):
        lines += ["## Saved scenarios", "",
                  "| Name | Status | Deactivated | Candidates | Observable cells |",
                  "|---|---|---|---|---|"]
        lines += [f"| {_md(s['name'])} | {_md(s['status'])} | {_md(s['deactivated_units'])} | "
                  f"{_md(s['included_candidates'])} | {_md(s['summary']['observable_cells'])} |"
                  for s in analysis["saved_scenarios"]]
        lines.append("")
    if analysis.get("comparison"):
        comparison = analysis["comparison"]
        lines += [f"## Comparison (right − left)", "",
                  f"Left: {_md(comparison['left']['label'])} · "
                  f"Right: {_md(comparison['right']['label'])}", ""]
        lines += _kv("Difference (right − left)", comparison["difference_right_minus_left"],
                     level=3)
        lines += [f"- cells gained coverage: {_md(comparison['cells_gained_coverage'])}",
                  f"- cells lost coverage: {_md(comparison['cells_lost_coverage'])}", ""]
    if manifest.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {_md(w['message'])}" for w in manifest["warnings"]] + [""]
    if manifest.get("limitations"):
        lines += ["## Limitations", ""] + [f"- {_md(x)}" for x in manifest["limitations"]] + [""]
    if manifest.get("outputs"):
        lines += ["## Outputs", "", "| File | Meaning |", "|---|---|"]
        lines += [f"| {_md(o.get('file'))} | {_md(o.get('meaning'))} |" for o in manifest["outputs"]]
        lines.append("")
    return "\n".join(lines)


def write_markdown(manifest: dict[str, Any], path: Path, *, overwrite: bool = False) -> Path:
    with atomic_output(path, overwrite=overwrite) as temporary:
        temporary.write_text(manifest_markdown(manifest), encoding="utf-8")
    return Path(path).expanduser().resolve()
