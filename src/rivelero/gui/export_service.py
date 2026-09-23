"""Bridge between ApplicationState and the Qt-free exporters (P2).

``export_context_from_state`` collects only *current, valid* results:

* the SOF, only if one exists for the current inputs (an invalidated SOF is
  already removed from state; the invalidation reason is reported instead);
* the contribution analysis, only if it belongs to that SOF;
* the live what-if scenario, only if it modifies that SOF;
* the comparison currently selected in Compare, only if both sides can be
  materialised against that SOF (never an out-of-date snapshot).

It reads state and never writes it, so an export cannot change the project
or mark it modified.
"""

from __future__ import annotations

from typing import Any

from rivelero.analysis.comparison import (
    BASELINE_ID,
    LIVE_SCENARIO_ID,
    SnapshotUnavailableError,
    baseline_state,
    compare_states,
)
from rivelero.export.catalog import ExportContext, default_prefix
from rivelero.export.metadata import ExportProvenance
from rivelero.export.tables import scenario_row
from rivelero.gui.application_state import ApplicationState


def export_context_from_state(state: ApplicationState) -> ExportContext:
    project = state.project
    survey = state.survey
    analysis = state.analysis
    configuration = survey.viewpoint_configuration
    sof = analysis.survey_observability_field
    visibility = analysis.visibility_configuration

    provenance = ExportProvenance(
        project_name=project.name,
        project_path=None if project.project_path is None else str(project.project_path),
        project_saved=project.project_path is not None and not project.dirty,
        sof_id=None if sof is None else sof.sof_id,
        viewpoint_configuration_id=(
            None if sof is None else sof.viewpoint_configuration_id
        ) or (None if configuration is None else configuration.configuration_id),
        environment_id=None if sof is None else sof.environment_id,
        analysis_domain_id=(
            sof.analysis_domain_id if sof is not None
            else None if analysis.analysis_domain is None
            else analysis.analysis_domain.domain_id
        ),
        visibility_configuration_id=(
            sof.visibility_configuration_id if sof is not None
            else None if visibility is None else visibility.configuration_id
        ),
        visibility_configuration_name=None if visibility is None else visibility.name,
        sampling_unit=None if sof is None else getattr(sof.sampling_unit, "value", str(sof.sampling_unit)),
    )

    context: dict[str, Any] = {
        "provenance": provenance,
        "prefix": default_prefix(project.name),
        "configuration": configuration,
        "sensors": tuple(survey.sensors.values()),
        "survey_reason": None if configuration is not None else "No Survey has been imported.",
        "project_description": project.description,
        "environment": analysis.environment,
        "analysis_domain": analysis.analysis_domain,
        "visibility_configuration": visibility,
        "snapshots": (
            () if analysis.scenario_workspace is None else analysis.scenario_workspace.snapshots
        ),
        "visibility_store_directory": _store_directory(analysis.visibility_store),
        "invalidation_reason": analysis.invalidation_reason,
    }

    if sof is None:
        reason = analysis.invalidation_reason
        context["sof_reason"] = (
            f"The observability field is out of date ({reason}). Rebuild it first."
            if reason else "Build the Survey Observability Field first."
        )
        for name in ("contribution_reason", "scenario_reason", "scenario_rows_reason",
                     "comparison_reason"):
            context[name] = context["sof_reason"]
        return ExportContext(**context)

    context["sof"] = sof
    context["sof_reason"] = None
    context["build_report"] = analysis.build_report

    contribution = analysis.contribution_analysis
    if contribution is not None and contribution.sof_id == sof.sof_id:
        context["contribution"] = contribution
    else:
        context["contribution_reason"] = "Run the contribution analysis (Analysis › Contribution)."

    live = analysis.design_scenario
    live_current = live is not None and live.baseline is sof
    live_modified = live_current and not live.summary().is_baseline
    if live_modified:
        context["scenario_exposure"] = live.exposure  # read-only; edits make a new array
        context["scenario_change"] = live.change_classes()
        context["scenario_record"] = _scenario_record(live)
        context["scenario_summary"] = live.summary()
    else:
        context["scenario_reason"] = (
            "The current scenario equals the baseline; deactivate units or include "
            "candidates in Analysis › Scenario."
        )

    context["scenario_rows"] = _scenario_rows(sof, live if live_modified else None,
                                              analysis.scenario_workspace)
    context["scenario_rows_reason"] = None

    comparison, reason = _comparison(state, sof, live if live_modified else None)
    context["comparison"] = comparison
    context["comparison_reason"] = reason
    return ExportContext(**context)


def _store_directory(store) -> str | None:
    directory = getattr(store, "cache_directory", None)
    return None if directory is None else str(directory)


def _scenario_record(live) -> dict[str, Any]:
    return {
        "state_id": LIVE_SCENARIO_ID,
        "label": "Current scenario (unsaved)",
        "deactivated_sampling_units": [key.sampling_unit_id for key in live.deactivated_keys],
        "candidates": [
            {
                "viewpoint_id": candidate.candidate_id,
                "x": candidate.viewpoint.x,
                "y": candidate.viewpoint.y,
                "crs": str(candidate.viewpoint.crs),
                "status": candidate.status.value,
                "included": candidate.included,
            }
            for candidate in live.candidates
        ],
        "note": "Only included candidates whose visibility is ready contribute to exposure.",
    }


def _scenario_rows(sof, live, workspace) -> tuple[dict[str, Any], ...]:
    base = baseline_state(sof)
    rows = [
        scenario_row(
            state_id=BASELINE_ID, kind="baseline", label="Baseline", compatible=True,
            baseline_sof_id=sof.sof_id, coverage=base.summary, baseline=None,
            active_existing_units=base.active_existing_units, deactivated_units=0,
            included_candidates=0,
        )
    ]
    if live is not None:
        summary = live.summary()
        rows.append(scenario_row(
            state_id=LIVE_SCENARIO_ID, kind="live_scenario", label="Current scenario (unsaved)",
            compatible=True, baseline_sof_id=summary.baseline_sof_id,
            coverage=summary.scenario, baseline=summary.baseline,
            active_existing_units=summary.active_existing_units,
            deactivated_units=summary.deactivated_units,
            included_candidates=summary.included_candidates,
        ))
    for snapshot in () if workspace is None else workspace.snapshots:
        summary = snapshot.summary
        rows.append(scenario_row(
            state_id=snapshot.snapshot_id, kind="saved_snapshot", label=snapshot.name,
            compatible=snapshot.compatible_with(sof),
            baseline_sof_id=snapshot.baseline_sof_id,
            coverage=summary.scenario, baseline=summary.baseline,
            active_existing_units=summary.active_existing_units,
            deactivated_units=summary.deactivated_units,
            included_candidates=summary.included_candidates,
            snapshot=snapshot,
        ))
    return tuple(rows)


def _comparison(state: ApplicationState, sof, live):
    workspace = state.analysis.scenario_workspace
    if workspace is None or workspace.right_id is None:
        return None, "Choose two design states in Analysis › Compare."
    left_id, right_id = workspace.left_id, workspace.right_id
    for state_id in (left_id, right_id):
        if state_id == LIVE_SCENARIO_ID and live is None:
            return None, "The compared current scenario equals the baseline or no longer exists."
    try:
        store = state.analysis.visibility_store
        left = workspace.state(left_id, sof=sof, store=store, live=live)
        right = workspace.state(right_id, sof=sof, store=store, live=live)
        return compare_states(left, right, sof), None
    except KeyError:
        return None, "A compared scenario no longer exists."
    except (SnapshotUnavailableError, ValueError) as error:
        return None, str(error)
