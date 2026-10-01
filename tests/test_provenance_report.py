"""P3 tests: provenance manifest and HTML report (Qt-free)."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from observability_fixtures import DEM_NODATA, make_ready_state, visibility_configuration
from rivelero.analysis.comparison import BASELINE_ID, ScenarioWorkspace
from rivelero.analysis.contribution import analyse_contributions
from rivelero.analysis.coverage import summarize_coverage
from rivelero.analysis.scenario import SurveyDesignScenario
from rivelero.export import report as report_module
from rivelero.export.catalog import ExportConflictError, ExportContext
from rivelero.export.metadata import ExportProvenance
from rivelero.export.provenance import build_manifest, manifest_markdown
from rivelero.export.report import ReportOptions, generate_report, render_html
from rivelero.gui.export_service import export_context_from_state
from rivelero.gui.observability_service import install_build_result, prepare_build
from rivelero.observability.builder import build_survey_observability_field
from rivelero.project.resources import file_sha256
from rivelero.visibility.configuration import MissingMetadataPolicy
from test_export import _survey

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    """Survey -> World -> SOF -> A2 -> A3 -> saved snapshot -> A4 comparison."""
    root = tmp_path_factory.mktemp("workflow")
    state = make_ready_state(root / "cache", dem=DEM_NODATA, survey_buffer_m=100.0,
                             viewpoint_ids=None,
                             configuration=visibility_configuration(max_distance_m=250.0))
    request = prepare_build(state)
    install_build_result(state, request, build_survey_observability_field(**request.build_kwargs()))
    sof = state.analysis.survey_observability_field
    store = state.analysis.visibility_store
    state.set_contribution_analysis(analyse_contributions(sof, store),
                                    inputs_revision=state.analysis.inputs_revision)
    scenario = SurveyDesignScenario(sof)
    key = sof.active_keys[0]
    scenario.deactivate({key: store.get(key).visibility_mask})
    state.set_design_scenario(scenario)
    workspace = state.analysis.scenario_workspace = ScenarioWorkspace()
    snapshot = workspace.save(scenario, name="Without first unit")
    workspace.left_id, workspace.right_id = BASELINE_ID, snapshot.snapshot_id
    return state


def _survey_only():
    configuration, sensors = _survey()
    return ExportContext(
        provenance=ExportProvenance(project_name="Survey only"), prefix="s",
        configuration=configuration, sensors=sensors, survey_reason=None,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_survey_only_manifest():
    manifest = build_manifest(_survey_only())
    assert manifest["format"] == "rivelero-provenance"
    assert manifest["survey"]["viewpoints"] == 2 and manifest["survey"]["observation_events"] == 2
    assert manifest["world"] is None and manifest["observability"] is None
    assert manifest["assumptions"] is None and manifest["analysis"] == {}
    assert manifest["limitations"] == []
    codes = {w["code"] for w in manifest["warnings"]}
    assert not codes & {"terrain_external", "visibility_cache_external", "dem_nodata_in_domain"}
    assert json.loads(json.dumps(manifest)) == manifest


def test_missing_heading_source_and_policy_are_separate():
    context = replace(_survey_only(), visibility_configuration=visibility_configuration())
    manifest = build_manifest(context)
    # Source: vp_b has no heading (vp_α has 45°); ev_2 inherits vp_b's gap.
    assert manifest["source_metadata"]["viewpoints"]["missing_heading"] == 1
    assert manifest["source_metadata"]["observation_events"]["missing_heading_event_and_viewpoint"] == 1
    # Assumption: the configured policy, and how often it was applied.
    assumptions = manifest["assumptions"]
    assert assumptions["configuration"]["missing_heading_policy"] == "omnidirectional"
    applied = assumptions["applied"]
    assert applied["heading_missing"] == 1
    assert applied["heading_missing_treated_omnidirectional"] == 1
    assert applied["heading_missing_used_default"] == 0
    # The source record is not rewritten as though the assumption were observed.
    assert context.configuration.viewpoints[1].heading_deg is None

    default = visibility_configuration(missing_heading_policy=MissingMetadataPolicy.USE_DEFAULT,
                                       default_heading_deg=90.0)
    applied = build_manifest(replace(context, visibility_configuration=default))["assumptions"]["applied"]
    assert applied["heading_missing_used_default"] == 1
    assert applied["heading_missing_treated_omnidirectional"] == 0
    excluded = visibility_configuration(missing_heading_policy=MissingMetadataPolicy.EXCLUDE)
    applied = build_manifest(replace(context, visibility_configuration=excluded))["assumptions"]["applied"]
    assert applied["excluded_by_policy"] == 1


def test_full_manifest(workflow):
    state = workflow
    context = export_context_from_state(state)
    manifest = build_manifest(context)
    sof = state.analysis.survey_observability_field
    summary = summarize_coverage(sof)

    environment = manifest["world"]["environment"]
    assert environment["environment_id"] == state.analysis.environment.environment_id
    assert environment["elevation"]["sha256"] == file_sha256(DEM_NODATA)
    assert environment["elevation"]["crs"] == "EPSG:32633"
    cells = manifest["world"]["analysis_domain"]["cells"]
    counts = sof.state_counts()
    assert cells == {"outside_domain": counts[0], "invalid": counts[1],
                     "analysable": counts[2] + counts[3]}

    configuration = manifest["assumptions"]["configuration"]
    assert configuration["max_distance_m"] == 250.0
    for name in ("sampling_unit", "default_observer_height_m", "use_direction",
                 "missing_fov_policy", "curvature_coefficient", "backend"):
        assert name in configuration

    observability = manifest["observability"]
    assert observability["status"] == "current" and observability["sof_id"] == sof.sof_id
    assert observability["build_report"]["added_units"] == sof.n_active_units
    coverage = observability["coverage"]
    assert coverage["observable_cells"] == summary.observable_cells
    assert coverage["blind_cells"] == summary.blind_cells
    assert all(check["holds"] for check in observability["checks"])

    analysis = manifest["analysis"]
    assert analysis["contribution"]["sof_id"] == sof.sof_id
    assert analysis["scenario"]["deactivated_units"] == 1
    assert analysis["saved_scenarios"][0]["status"] == "current"
    comparison = analysis["comparison"]
    assert comparison["direction"] == "right_minus_left"
    assert comparison["left"]["state_id"] == BASELINE_ID
    delta = comparison["difference_right_minus_left"]["observable_cells"]
    assert delta == comparison["cells_gained_coverage"] - comparison["cells_lost_coverage"]
    assert comparison["checks"][0]["holds"]

    codes = {w["code"] for w in manifest["warnings"]}
    assert {"terrain_external", "visibility_cache_external", "dem_nodata_in_domain"} <= codes
    assert "snapshots_out_of_date" not in codes
    assert any("not guaranteed detection" in x for x in manifest["limitations"])
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest


def test_stale_snapshot_is_labelled(workflow):
    context = export_context_from_state(workflow)
    stale = replace(context.snapshots[0], baseline_sof_id="sof_earlier",
                    snapshot_id="snapshot_old", name="Old design")
    manifest = build_manifest(replace(context, snapshots=context.snapshots + (stale,)))
    statuses = {s["name"]: s["status"] for s in manifest["analysis"]["saved_scenarios"]}
    assert statuses == {"Without first unit": "current", "Old design": "out_of_date"}
    assert any(w["code"] == "snapshots_out_of_date" for w in manifest["warnings"])


def test_out_of_date_observability_reports_no_results():
    context = replace(_survey_only(), invalidation_reason="the VisibilityConfiguration changed")
    manifest = build_manifest(context)
    assert manifest["observability"]["status"] == "out_of_date"
    assert "coverage" not in manifest["observability"]
    assert any(w["code"] == "observability_out_of_date" for w in manifest["warnings"])


def test_unicode_metadata_round_trips(tmp_path):
    context = replace(_survey_only(), provenance=ExportProvenance(project_name="Sicília 測試 Zoë"),
                      project_description="Transecte núm. 2 — «nord»")
    manifest = build_manifest(context)
    path = tmp_path / "p.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["project"]["name"] == "Sicília 測試 Zoë"
    assert "«nord»" in manifest_markdown(again)


def test_markdown_rendering(workflow):
    text = manifest_markdown(build_manifest(export_context_from_state(workflow)))
    for heading in ("# Rivelero provenance", "## Source metadata (as recorded)",
                    "## Visibility assumptions (configured)", "## Observability",
                    "## Comparison (right − left)", "## Warnings", "## Limitations"):
        assert heading in text
    assert "`observable + blind == analysable`: holds" in text


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _sections(html):
    return re.findall(r"<h2>([^<]+)</h2>", html)


def test_minimal_report_omits_unavailable_sections(tmp_path):
    result = generate_report(_survey_only(), tmp_path / "r", ReportOptions())
    html = (tmp_path / "r" / "report.html").read_text(encoding="utf-8")
    assert _sections(html) == ["1. Project", "2. Survey", "10. Provenance and limitations"]
    assert result.figures == () and set(result.skipped_figures) == set(ReportOptions().figures)
    assert sorted(p.name for p in (tmp_path / "r").iterdir()) == [
        "figures", "provenance.json", "provenance.md", "report.html"]


def test_full_report(workflow, tmp_path):
    context = export_context_from_state(workflow)
    result = generate_report(context, tmp_path / "report")
    html = (tmp_path / "report" / "report.html").read_text(encoding="utf-8")
    assert _sections(html) == [
        "1. Project", "2. Survey", "3. World", "4. Visibility assumptions",
        "5. Observability results", "6. Coverage and exposure", "7. Sampling-unit contribution",
        "8. Design scenario", "9. Scenario comparison", "10. Provenance and limitations"]
    for key in result.figures:
        path = tmp_path / "report" / "figures" / f"{key}.png"
        assert path.stat().st_size > 0 and f'src="figures/{key}.png"' in html
    assert set(result.figures) == set(ReportOptions().figures)
    # Numbers come from the canonical summary.
    summary = summarize_coverage(context.sof)
    assert f"({summary.observable_cells:,} of {summary.analysable_cells:,})" in html
    assert f"{summary.unique_cells:,} cells were observable from exactly one" in html
    assert "right − left" in html and "FAILS" not in html
    manifest = json.loads((tmp_path / "report" / "provenance.json").read_text(encoding="utf-8"))
    assert {o["file"] for o in manifest["outputs"]} >= {f"figures/{k}.png" for k in result.figures}


def test_report_is_offline_and_escaped(tmp_path):
    hostile = "<script>alert('x')</script>"
    configuration, sensors = _survey()
    context = replace(
        _survey_only(),
        provenance=ExportProvenance(project_name=f"P {hostile}"),
        project_description=f'<img src=x onerror="alert(1)"> {hostile}',
        configuration=replace(configuration, name=f"Survey {hostile}"),
    )
    generate_report(context, tmp_path / "r", ReportOptions(title=f"T {hostile}"))
    html = (tmp_path / "r" / "report.html").read_text(encoding="utf-8")
    assert "<script" not in html.lower()
    # The hostile markup survives only as inert, escaped text.
    assert "<img src=x" not in html and 'onerror="' not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html
    assert not re.search(r"https?://", html)
    assert "<link" not in html.lower() and "@import" not in html


def test_existing_report_is_not_replaced_without_permission(tmp_path):
    target = tmp_path / "r"
    target.mkdir()
    (target / "report.html").write_text("previous", encoding="utf-8")
    with pytest.raises(ExportConflictError):
        generate_report(_survey_only(), target)
    assert (target / "report.html").read_text(encoding="utf-8") == "previous"
    generate_report(_survey_only(), target, overwrite=True)
    assert "Rivelero Analysis Report" in (target / "report.html").read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["r"]


def _existing(tmp_path):
    target = tmp_path / "r"
    target.mkdir()
    (target / "report.html").write_text("previous valid report", encoding="utf-8")
    return target


def test_failed_generation_is_atomic(workflow, tmp_path, monkeypatch):
    target = _existing(tmp_path)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(report_module, "save_figure", broken)
    with pytest.raises(OSError, match="disk full"):
        generate_report(export_context_from_state(workflow), target, overwrite=True)
    assert (target / "report.html").read_text(encoding="utf-8") == "previous valid report"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r"]  # no temporary left


def test_cancelled_generation_is_atomic(workflow, tmp_path):
    target = _existing(tmp_path)

    class Cancelled(Exception):
        pass

    def progress(step, total, message):
        if message.startswith("Rendering") and step >= 2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        generate_report(export_context_from_state(workflow), target, overwrite=True,
                        progress_callback=progress)
    assert (target / "report.html").read_text(encoding="utf-8") == "previous valid report"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r"]
    fresh = tmp_path / "fresh"
    with pytest.raises(Cancelled):
        generate_report(export_context_from_state(workflow), fresh, progress_callback=progress)
    assert not fresh.exists()


def test_generation_does_not_change_state(workflow, tmp_path):
    state = workflow
    state.project.dirty = False
    revision = state.analysis.inputs_revision
    sof = state.analysis.survey_observability_field
    generate_report(export_context_from_state(state), tmp_path / "r")
    assert not state.project.dirty
    assert state.analysis.inputs_revision == revision
    assert state.analysis.survey_observability_field is sof


def test_render_html_marks_failed_checks():
    manifest = build_manifest(_survey_only())
    manifest["observability"] = {
        "status": "current", "sof_id": "x", "active_sampling_units": 1,
        "coverage": {"observable_share_of_analysable": 0.5, "observable_cells": 1,
                     "analysable_cells": 2, "blind_cells": 0, "unique_cells": 1,
                     "repeated_cells": 0, "mean_exposure": 0.5, "median_exposure": 0.5,
                     "mean_exposure_observable": 1.0, "maximum_exposure": 1},
        "checks": [{"check": "observable + blind == analysable", "holds": False},
                   {"check": "unique + repeated == observable", "holds": True}],
    }
    html = render_html(manifest, {}, title="t")
    assert "FAILS" in html  # an inconsistency is never hidden
