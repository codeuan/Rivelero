"""ApplicationState <-> project file integration (P1).

Converts the GUI session state to the Qt-free :class:`ProjectData` for
saving, and builds a complete new ApplicationState from a loaded project.
The loaded state is constructed through the canonical ApplicationState API,
so every restored object passes the same validation as an interactive edit
(unique Viewpoint IDs, Sensor and ObservationEvent references, grid/domain
compatibility, SOF identity). Only a fully built state is installed, via
``ApplicationState.adopt``.

Not saved: selection, map extents, tabs, task progress, contribution
analysis (recomputed on demand) and the visibility cache itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import rasterio

from rivelero.analysis.comparison import (
    ScenarioWorkspace,
    SnapshotUnavailableError,
    restore_snapshot,
    snapshot_scenario,
)
from rivelero.gui.application_state import ApplicationState, ProjectState
from rivelero.gui.domain_service import elevation_valid_mask
from rivelero.gui.observability_service import create_visibility_store
from rivelero.project.io import ProjectData, load_project, save_project
from rivelero.project.resources import ResourceStatus
from rivelero.project.schema import PROJECT_SUFFIX

LIVE_SCENARIO_NAME = "Unsaved scenario"


@dataclass(slots=True)
class OpenResult:
    """Outcome of opening a project."""

    state: ApplicationState
    warnings: list[str] = field(default_factory=list)
    field_status: str = "not saved"


def project_data_from_state(state: ApplicationState) -> ProjectData:
    """Collect the scientific content of ``state`` for saving."""

    analysis = state.analysis
    project = state.project
    workspace = analysis.scenario_workspace
    live = analysis.design_scenario
    sof = analysis.survey_observability_field

    live_snapshot = None
    if live is not None and live.baseline is sof and (
        live.deactivated_keys or live.candidates
    ):
        live_snapshot = snapshot_scenario(live, name=LIVE_SCENARIO_NAME)

    store = analysis.visibility_store
    return ProjectData(
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        metadata=dict(project.metadata),
        viewpoint_configuration=state.survey.viewpoint_configuration,
        sensors=dict(state.survey.sensors),
        environment=analysis.environment,
        grid=analysis.analysis_grid,
        domain=analysis.analysis_domain,
        visibility_configuration=analysis.visibility_configuration,
        store_settings=None if store is None else {
            "cache_directory": str(store.cache_directory),
            "max_memory_items": store.max_memory_items,
            "compressed": store.compressed,
        },
        sof=sof,
        build_report=analysis.build_report,
        snapshots=() if workspace is None else workspace.snapshots,
        live_scenario=live_snapshot,
        comparison={} if workspace is None else {
            "left": workspace.left_id,
            "right": workspace.right_id,
        },
    )


def save_state(state: ApplicationState, path: Path | str) -> Path:
    """Save ``state``; on success record the path and mark the project clean."""

    target = Path(path)
    if target.suffix != PROJECT_SUFFIX:
        target = target.with_name(target.name + PROJECT_SUFFIX)
    if state.project.name == ProjectState().name:
        state.rename_project(target.stem)
    saved = save_project(project_data_from_state(state), target)
    state.project.mark_saved(saved)
    return saved


def open_state(path: Path | str) -> OpenResult:
    """Load a project into a new, fully validated ApplicationState."""

    data = load_project(path)
    warnings = list(data.warnings)
    state = ApplicationState()
    state.project = ProjectState(
        name=data.name,
        project_path=Path(path).expanduser().resolve(),
        description=data.description,
        created_at=data.created_at,
        metadata=dict(data.metadata),
    )

    # Survey (source data). Canonical constructors already validated the
    # objects; the state API checks references between them.
    if data.viewpoint_configuration is not None:
        state.set_viewpoint_configuration(data.viewpoint_configuration)
    state.set_sensors(data.sensors)
    _check_references(state)

    # World.
    sof = data.sof
    if data.environment is not None:
        domain = data.domain
        install = True
        if data.elevation is not None and data.elevation.status == ResourceStatus.CHANGED:
            install, domain, message = _refresh_domain_validity(data)
            warnings.append(message)
        if install:
            state.set_environment(data.environment, grid=data.grid)
            if domain is not None:
                state.set_analysis_domain(domain)

    # Assumptions and cache settings.
    state.set_visibility_configuration(data.visibility_configuration)
    state.set_visibility_store(_store(data.store_settings, warnings))

    # Derived observability: installed only if it validates against the
    # restored inputs (the digest was already checked when reading).
    field_status = data.field_status
    if sof is not None:
        try:
            state.set_observability_result(sof, build_report=data.build_report)
        except (ValueError, RuntimeError) as exc:
            field_status = "rejected"
            warnings.append(f"The saved observability field was not restored: {exc}")

    # Design work.
    workspace = ScenarioWorkspace()
    for snapshot in data.snapshots:
        workspace.add_snapshot(snapshot)
    workspace.left_id = data.comparison.get("left") or workspace.left_id
    workspace.right_id = data.comparison.get("right")
    state.analysis.scenario_workspace = workspace
    workspace.prune(state.analysis.survey_observability_field)

    if data.live_scenario is not None:
        current = state.analysis.survey_observability_field
        if current is None or not data.live_scenario.compatible_with(current):
            warnings.append(
                "The unsaved scenario was not restored because its observability "
                "field is not current."
            )
        else:
            try:
                scenario, report = restore_snapshot(
                    data.live_scenario, current, state.analysis.visibility_store
                )
                state.set_design_scenario(scenario)
                if report.candidates_needing_computation:
                    warnings.append(
                        "Candidate visibility will be recomputed for: "
                        + ", ".join(report.candidates_needing_computation) + "."
                    )
            except SnapshotUnavailableError as exc:
                warnings.append(f"The unsaved scenario was not restored: {exc}")

    state.selection.clear()
    state.project.dirty = False
    return OpenResult(state=state, warnings=warnings, field_status=field_status)


def _check_references(state: ApplicationState) -> None:
    configuration = state.survey.viewpoint_configuration
    if configuration is None:
        return
    missing = sorted(
        {v.sensor_id for v in configuration.viewpoints if v.sensor_id is not None}
        - set(state.survey.sensors)
    )
    if missing:
        from rivelero.project.schema import ProjectFormatError

        raise ProjectFormatError(
            "Viewpoints reference undefined Sensors: " + ", ".join(missing) + "."
        )


def _refresh_domain_validity(data: ProjectData):
    """Terrain changed: returns (install terrain?, domain, message).

    If the changed file still has the saved grid, the terrain and the domain
    geometry are kept and the domain's validity (NoData) mask is recomputed
    from the current file. Otherwise the terrain is not installed at all,
    because the saved grid no longer describes it.
    """

    grid = data.grid
    source = data.environment.elevation_model.source
    with rasterio.open(source) as dataset:
        same_grid = (
            dataset.crs == grid.crs
            and dataset.transform == grid.transform
            and dataset.width == grid.width
            and dataset.height == grid.height
        )
    if not same_grid:
        return False, None, (
            "The terrain file changed and no longer matches the saved AnalysisGrid; "
            "terrain and AnalysisDomain were not restored. Load the terrain again in World."
        )
    valid = elevation_valid_mask(source, grid)
    return True, (
        replace(data.domain, valid_mask=valid) if data.domain is not None else None
    ), (
        "The terrain file changed since the project was saved; the AnalysisDomain "
        "validity (NoData) mask was recomputed from the current file."
    )


def _store(settings, warnings):
    if not settings:
        return create_visibility_store()
    try:
        return create_visibility_store(
            settings["cache_directory"],
            max_memory_items=int(settings.get("max_memory_items", 64)),
            compressed=bool(settings.get("compressed", True)),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        warnings.append(
            f"The saved visibility cache location could not be used ({exc}); "
            "the default cache is used instead. Visibility is recomputed when needed."
        )
        return create_visibility_store()
