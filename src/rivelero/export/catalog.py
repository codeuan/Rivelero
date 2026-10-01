"""Export catalog and directory export runner.

``export_catalog(context)`` lists every exportable product with whether it
is available now and, if not, why. ``run_export`` writes a selection into a
directory.

Directory-export behaviour
--------------------------
1. **Preflight.** Every file the selection would create (products and their
   JSON sidecars) is computed first. Unless ``overwrite`` is True, if *any*
   of them already exists nothing is written and ExportConflictError lists
   the conflicts. Unavailable products and paths too long for the
   platform are refused before writing, too.
2. **Per-file atomicity.** Each file is written to a temporary file in the
   destination and renamed into place, so no truncated file is ever left.
3. **Partial failure.** If one product fails, the error is recorded and the
   remaining products are still written; the result lists written and failed
   products. Already written files are kept (each is complete and valid).
4. Files that are not part of the selection are never touched or deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rivelero.analysis.comparison import ScenarioComparison
from rivelero.analysis.contribution import ContributionAnalysis
from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.sensor import Sensor
from rivelero.export import rasters, tables
from rivelero.export.files import check_path_length, safe_filename_part
from rivelero.export.metadata import ExportProvenance
from rivelero.observability.survey_field import SurveyObservabilityField


class ExportConflictError(FileExistsError):
    """Raised before writing when target files already exist."""

    def __init__(self, conflicts: list[Path]) -> None:
        self.conflicts = conflicts
        shown = ", ".join(path.name for path in conflicts[:6])
        more = f" and {len(conflicts) - 6} more" if len(conflicts) > 6 else ""
        super().__init__(
            f"{len(conflicts)} file(s) already exist: {shown}{more}. "
            "Choose another folder or allow overwriting."
        )


class ExportUnavailableError(ValueError):
    """Raised when a requested product cannot be exported now."""


@dataclass(frozen=True, slots=True)
class ExportContext:
    """Everything an export may write; built from application state.

    Only current, valid results may be placed here. A ``*_reason`` explains
    why the corresponding result is missing.
    """

    provenance: ExportProvenance
    prefix: str = "rivelero"

    configuration: ViewpointConfiguration | None = None
    sensors: tuple[Sensor, ...] = ()
    survey_reason: str | None = "No Survey has been imported."

    sof: SurveyObservabilityField | None = None
    sof_reason: str | None = "Build the Survey Observability Field first."

    contribution: ContributionAnalysis | None = None
    contribution_reason: str | None = "Run the contribution analysis first."

    # Frozen when the context is built, so later edits of the live scenario
    # cannot change an export that is already running.
    scenario_exposure: np.ndarray | None = None
    scenario_change: np.ndarray | None = None
    scenario_record: dict[str, Any] | None = None
    scenario_reason: str | None = "There is no modified what-if scenario."

    scenario_rows: tuple[dict[str, Any], ...] = ()
    scenario_rows_reason: str | None = "Build the Survey Observability Field first."

    comparison: ScenarioComparison | None = None
    comparison_reason: str | None = "Choose two design states in Compare."

    # Context for figures, provenance and the report (P3). None = unknown.
    project_description: str | None = None
    environment: Any | None = None
    analysis_domain: Any | None = None
    visibility_configuration: Any | None = None
    build_report: Any | None = None
    scenario_summary: Any | None = None
    snapshots: tuple[Any, ...] = ()
    visibility_store_directory: str | None = None
    invalidation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExportProduct:
    key: str
    group: str
    label: str
    filename: str
    available: bool
    reason: str | None
    kind: str  # "raster" or "table"
    write: Callable[[Path, bool], tuple[Path, ...]] | None = field(
        default=None, repr=False, compare=False
    )

    def outputs(self, directory: Path) -> tuple[Path, ...]:
        path = Path(directory) / self.filename
        return rasters.raster_outputs(path) if self.kind == "raster" else tables.table_outputs(path)


GROUP_SURVEY = "Survey"
GROUP_OBSERVABILITY = "Observability"
GROUP_ANALYSIS = "Analysis"
GROUP_SCENARIO = "Scenarios and comparison"
GROUPS = (GROUP_SURVEY, GROUP_OBSERVABILITY, GROUP_ANALYSIS, GROUP_SCENARIO)

SOF_LABELS = {
    "exposure_count": "Exposure count",
    "normalized_exposure": "Normalized exposure",
    "observability_state": "Observability state (categorical)",
    "blindspot_mask": "Blind-spot mask",
    "observable_mask": "Observable mask",
    "analysis_mask": "AnalysisDomain mask",
    "valid_mask": "Terrain validity mask",
}


def export_catalog(context: ExportContext) -> list[ExportProduct]:
    """All products, in display order, with availability."""

    p = context.prefix
    prov = context.provenance
    products: list[ExportProduct] = []

    def add(key, group, label, filename, kind, available, reason, write):
        products.append(ExportProduct(
            key=key, group=group, label=label, filename=f"{p}_{filename}",
            available=available, reason=None if available else reason,
            kind=kind, write=write if available else None,
        ))

    # -- Survey ---------------------------------------------------------
    configuration = context.configuration
    has_survey = configuration is not None
    add("viewpoints", GROUP_SURVEY, "Viewpoints table", "viewpoints.csv", "table",
        has_survey and configuration.n_viewpoints > 0,
        context.survey_reason or "The Survey has no Viewpoints.",
        lambda path, ow: tables.write_viewpoints(configuration, path, prov, overwrite=ow))
    add("sensors", GROUP_SURVEY, "Sensors table", "sensors.csv", "table",
        bool(context.sensors), "The Survey has no Sensors.",
        lambda path, ow: tables.write_sensors(context.sensors, path, prov, overwrite=ow))
    add("observation_events", GROUP_SURVEY, "ObservationEvents table",
        "observation_events.csv", "table",
        has_survey and configuration.n_events > 0,
        context.survey_reason or "The Survey has no ObservationEvents.",
        lambda path, ow: tables.write_observation_events(configuration, path, prov, overwrite=ow))

    # -- Observability --------------------------------------------------
    sof = context.sof
    for product, label in SOF_LABELS.items():
        add(product, GROUP_OBSERVABILITY, label, f"{product}.tif", "raster",
            sof is not None, context.sof_reason,
            lambda path, ow, product=product: rasters.write_sof_product(
                product, sof, path, prov, overwrite=ow))

    # -- Analysis -------------------------------------------------------
    add("coverage_class", GROUP_ANALYSIS, "Coverage classes (unique / repeated)",
        "coverage_class.tif", "raster", sof is not None, context.sof_reason,
        lambda path, ow: rasters.write_coverage_class(sof, path, prov, overwrite=ow))
    contribution = context.contribution
    add("contributions", GROUP_ANALYSIS, "Sampling-unit contributions",
        "sampling_unit_contributions.csv", "table",
        contribution is not None, context.contribution_reason,
        lambda path, ow: tables.write_contributions(contribution, path, prov, overwrite=ow))

    # -- Scenarios and comparison --------------------------------------
    has_scenario = context.scenario_exposure is not None and sof is not None
    add("scenario_exposure", GROUP_SCENARIO, "Current scenario exposure",
        "scenario_exposure.tif", "raster", has_scenario, context.scenario_reason,
        lambda path, ow: rasters.write_scenario_exposure(
            context.scenario_exposure, sof, path, prov, overwrite=ow, scenario_record=context.scenario_record))
    add("scenario_change", GROUP_SCENARIO, "Baseline → scenario change classes",
        "scenario_change_baseline_to_scenario.tif", "raster",
        has_scenario and context.scenario_change is not None, context.scenario_reason,
        lambda path, ow: rasters.write_scenario_change(
            context.scenario_change, sof, path, prov, overwrite=ow, scenario_record=context.scenario_record))
    rows = list(context.scenario_rows)
    add("scenario_summary", GROUP_SCENARIO, "Scenario summary table",
        "scenario_summary.csv", "table", bool(rows), context.scenario_rows_reason,
        lambda path, ow: tables.write_scenario_summary(rows, path, prov, overwrite=ow))

    comparison = context.comparison
    sides = "comparison"
    if comparison is not None:
        left = safe_filename_part(comparison.left.label, fallback="left")[:24]
        right = safe_filename_part(comparison.right.label, fallback="right")[:24]
        sides = f"comparison_{left}_vs_{right}"
    add("comparison_summary", GROUP_SCENARIO, "Comparison table (right − left)",
        f"{sides}_summary.csv", "table", comparison is not None, context.comparison_reason,
        lambda path, ow: tables.write_comparison(comparison, path, prov, overwrite=ow))
    add("comparison_difference", GROUP_SCENARIO, "Exposure difference raster (right − left)",
        f"{sides}_exposure_difference_right_minus_left.tif", "raster",
        comparison is not None, context.comparison_reason,
        lambda path, ow: rasters.write_comparison_difference(
            comparison, sof, path, prov, overwrite=ow))
    add("comparison_change", GROUP_SCENARIO, "Comparison change classes (left → right)",
        f"{sides}_change_left_to_right.tif", "raster",
        comparison is not None, context.comparison_reason,
        lambda path, ow: rasters.write_comparison_change(
            comparison, sof, path, prov, overwrite=ow))

    return products


# ---------------------------------------------------------------------------
# Running an export
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExportFailure:
    key: str
    label: str
    message: str


@dataclass(frozen=True, slots=True)
class ExportResult:
    directory: Path
    written: tuple[Path, ...]
    failures: tuple[ExportFailure, ...]
    products_written: tuple[str, ...]

    @property
    def successful(self) -> bool:
        return not self.failures


def planned_outputs(products: list[ExportProduct], directory: Path) -> list[Path]:
    return [path for product in products for path in product.outputs(directory)]


def run_export(
    context: ExportContext,
    keys: list[str] | tuple[str, ...],
    directory: Path | str,
    *,
    overwrite: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ExportResult:
    """Write the selected products into ``directory`` (see module docstring)."""

    catalog = {product.key: product for product in export_catalog(context)}
    unknown = [key for key in keys if key not in catalog]
    if unknown:
        raise KeyError(f"Unknown export product(s): {', '.join(unknown)}")
    selected = [catalog[key] for key in dict.fromkeys(keys)]
    if not selected:
        raise ExportUnavailableError("Select at least one product to export.")
    unavailable = [product for product in selected if not product.available]
    if unavailable:
        raise ExportUnavailableError(
            "; ".join(f"{product.label}: {product.reason}" for product in unavailable)
        )

    target = Path(directory).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Not a folder: {target}")
    planned = planned_outputs(selected, target)
    if len(set(planned)) != len(planned):
        raise ExportUnavailableError("Two products would be written to the same file.")
    for path in planned:
        check_path_length(path)
    if not overwrite:
        conflicts = [path for path in planned if path.exists()]
        if conflicts:
            raise ExportConflictError(conflicts)
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    done: list[str] = []
    failures: list[ExportFailure] = []
    total = len(selected)
    for index, product in enumerate(selected):
        if progress_callback is not None:
            progress_callback(index, total, product.label)
        try:
            paths = product.write(target / product.filename, overwrite)
        except Exception as error:  # recorded; remaining products continue
            failures.append(ExportFailure(product.key, product.label, str(error) or type(error).__name__))
            continue
        written.extend(Path(path) for path in paths)
        done.append(product.key)
    if progress_callback is not None:
        progress_callback(total, total, "")

    return ExportResult(
        directory=target,
        written=tuple(written),
        failures=tuple(failures),
        products_written=tuple(done),
    )


def default_prefix(project_name: str | None) -> str:
    return safe_filename_part(project_name or "", fallback="rivelero")
