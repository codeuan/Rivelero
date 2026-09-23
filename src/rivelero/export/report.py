"""Compact scientific report (P3).

``generate_report`` writes a self-contained report directory::

    <report>/
    ├── report.html        offline: inline CSS, no JavaScript, no CDN/fonts
    ├── provenance.json
    ├── provenance.md
    └── figures/
        ├── analysis_state.png
        └── ...

Every number comes from the manifest built by rivelero.export.provenance,
which reads the canonical result objects (never GUI text). Sections for
results that are unavailable are omitted. The language is descriptive:
the report states observability structure and makes no recommendation.

All user/project text is HTML-escaped before insertion.

Atomicity: the report is generated in a temporary sibling directory,
validated, and only then moved into place. A failed or cancelled generation
removes the temporary directory and never replaces an existing report.
"""

from __future__ import annotations

import html
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from rivelero.export.catalog import ExportConflictError, ExportContext
from rivelero.export.figures import (
    FigureOptions,
    figure_catalog,
    figure_record,
    save_figure,
)
from rivelero.export.files import check_path_length
from rivelero.export.provenance import build_manifest, write_json, write_markdown

DEFAULT_REPORT_FIGURES = (
    "analysis_state", "exposure", "coverage_composition", "exposure_distribution",
    "scenario_change", "exposure_difference",
)

# Section in which each figure appears.
FIGURE_SECTIONS = {
    "analysis_state": "observability", "exposure": "observability",
    "normalized_exposure": "observability", "blind_spots": "observability",
    "coverage_class": "coverage", "coverage_composition": "coverage",
    "exposure_distribution": "coverage", "contribution_distribution": "contribution",
    "scenario_change": "scenario", "scenario_exposure": "scenario",
    "comparison_change": "comparison", "exposure_difference": "comparison",
}


class ReportValidationError(RuntimeError):
    """Raised when a generated report is incomplete; nothing is published."""


@dataclass(frozen=True, slots=True)
class ReportOptions:
    title: str = "Rivelero Analysis Report"
    figures: tuple[str, ...] = DEFAULT_REPORT_FIGURES
    figure_format: str = "png"
    dpi: int = 200
    include_viewpoints: bool = True

    def __post_init__(self) -> None:
        if self.figure_format not in ("png", "svg"):
            raise ValueError("Report figures must be PNG or SVG (browser-viewable).")


@dataclass(frozen=True, slots=True)
class ReportResult:
    directory: Path
    files: tuple[Path, ...]
    figures: tuple[str, ...]
    skipped_figures: tuple[str, ...] = field(default=())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_report(
    context: ExportContext,
    directory: Path | str,
    options: ReportOptions = ReportOptions(),
    *,
    overwrite: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ReportResult:
    """Generate the report directory ``directory`` (see module docstring)."""

    target = Path(directory).expanduser().resolve()
    if target.exists() and not overwrite:
        raise ExportConflictError([target])
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Not a folder: {target}")
    catalog = {product.key: product for product in figure_catalog(context)}
    unknown = [key for key in options.figures if key not in catalog]
    if unknown:
        raise KeyError(f"Unknown figure(s): {', '.join(unknown)}")
    figures = [catalog[key] for key in dict.fromkeys(options.figures) if catalog[key].available]
    skipped = tuple(key for key in options.figures if not catalog[key].available)
    for name in ("report.html", "provenance.json", "provenance.md"):
        check_path_length(target / name)
    for product in figures:
        check_path_length(target / "figures" / f"{product.key}.{options.figure_format}")

    total = len(figures) + 3
    step = 0

    def progress(message: str) -> None:
        nonlocal step
        if progress_callback is not None:
            progress_callback(step, total, message)
        step += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{uuid4().hex[:12]}.report-tmp")
    temporary.mkdir()
    try:
        progress("Preparing provenance")
        figure_options = FigureOptions(format=options.figure_format, dpi=options.dpi,
                                       include_viewpoints=options.include_viewpoints)
        (temporary / "figures").mkdir()
        rendered: dict[str, dict[str, Any]] = {}
        for product in figures:
            progress(f"Rendering {product.label}")
            relative = f"figures/{product.key}.{options.figure_format}"
            record = figure_record(context, product, figure_options)
            save_figure(product.build(figure_options), temporary / relative, figure_options, record)
            rendered[product.key] = {"file": relative, "label": product.label,
                                     "meaning": product.meaning}

        progress("Writing report")
        outputs = [{"file": "report.html", "meaning": "Human-readable report."},
                   {"file": "provenance.md", "meaning": "Provenance (Markdown)."},
                   *({"file": f["file"], "meaning": f["meaning"]} for f in rendered.values())]
        manifest = build_manifest(context, outputs=outputs)
        write_json(manifest, temporary / "provenance.json")
        write_markdown(manifest, temporary / "provenance.md")
        (temporary / "report.html").write_text(
            render_html(manifest, rendered, title=options.title), encoding="utf-8"
        )

        progress("Validating report")
        _validate(temporary, rendered)
        _publish(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if progress_callback is not None:
        progress_callback(total, total, "")

    files = tuple(sorted(path for path in target.rglob("*") if path.is_file()))
    return ReportResult(target, files, tuple(rendered), skipped)


def _validate(directory: Path, rendered: dict[str, dict[str, Any]]) -> None:
    for name in ("report.html", "provenance.json", "provenance.md"):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ReportValidationError(f"{name} was not written.")
    text = (directory / "report.html").read_text(encoding="utf-8")
    for figure in rendered.values():
        path = directory / figure["file"]
        if not path.is_file() or path.stat().st_size == 0:
            raise ReportValidationError(f"{figure['file']} was not written.")
        if f'src="{figure["file"]}"' not in text:
            raise ReportValidationError(f"{figure['file']} is not referenced by the report.")


def _publish(temporary: Path, target: Path) -> None:
    """Move the finished report into place; restore the old one on failure."""

    backup = None
    if target.exists():
        backup = target.with_name(f".{uuid4().hex[:12]}.report-old")
        os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except BaseException:
        if backup is not None:
            os.replace(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
body{font:15px/1.55 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
color:#202428;background:#fff;margin:0}
main{max-width:980px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px;
border-bottom:1px solid #E1E5E8;padding-bottom:4px}h3{font-size:16px;margin:20px 0 6px}
.meta,.note{color:#5E676D;font-size:13px}
table{border-collapse:collapse;margin:8px 0 16px;font-size:14px}
th,td{border-bottom:1px solid #E8EBED;padding:4px 12px 4px 0;text-align:left;vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:16px 0}figure img{max-width:100%;height:auto;border:1px solid #E8EBED}
figcaption{color:#5E676D;font-size:13px}
.warn{background:#FFF6E8;border-left:3px solid #C4520F;padding:8px 12px;margin:8px 0}
.ok{color:#1C5CAB}.fail{color:#C4520F;font-weight:600}
code{font-size:13px}
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return _e(value)


def _pct(share: float | None) -> str:
    return "—" if share is None else f"{100.0 * share:.1f}%"


def _pp(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f} percentage points"


def _rows(pairs) -> str:
    body = "".join(
        f"<tr><th scope='row'>{_e(label)}</th><td class='n'>{value}</td></tr>"
        for label, value in pairs
    )
    return f"<table>{body}</table>"


def _figures(rendered: dict[str, dict[str, Any]], section: str) -> str:
    parts = []
    for key, figure in rendered.items():
        if FIGURE_SECTIONS.get(key) != section:
            continue
        parts.append(
            f'<figure><img src="{_e(figure["file"])}" alt="{_e(figure["label"])}">'
            f"<figcaption>{_e(figure['label'])}. {_e(figure['meaning'])}</figcaption></figure>"
        )
    return "".join(parts)


def _checks(checks) -> str:
    items = "".join(
        f"<li><code>{_e(c['check'])}</code>: "
        f"<span class='{'ok' if c['holds'] else 'fail'}'>{'holds' if c['holds'] else 'FAILS'}</span></li>"
        for c in checks
    )
    return f"<ul class='note'>{items}</ul>" if items else ""


def render_html(manifest: dict[str, Any], rendered: dict[str, dict[str, Any]], *,
                title: str) -> str:
    """Render the report HTML from the manifest (escaped, offline)."""

    sections = [
        _section_project(manifest),
        _section_survey(manifest),
        _section_world(manifest),
        _section_assumptions(manifest),
        _section_observability(manifest, rendered),
        _section_coverage(manifest, rendered),
        _section_contribution(manifest, rendered),
        _section_scenario(manifest, rendered),
        _section_comparison(manifest, rendered),
        _section_provenance(manifest),
    ]
    body = "".join(section for section in sections if section)
    software = manifest["software"]
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<meta name='generator' content='Rivelero {_e(software.get('version') or '')} "
        f"{_e(software.get('git_commit') or '')}'>"
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{_e(title)}</h1>"
        f"<p class='meta'>{_e(manifest['project']['name'])} · generated {_e(manifest['generated_at'])}"
        f" · Rivelero {_e(software.get('version') or '(version unavailable)')}"
        f" · commit {_e((software.get('git_commit') or 'unknown')[:12])}</p>"
        f"{body}</main></body></html>\n"
    )


def _section_project(m) -> str:
    project = m["project"]
    description = f"<p>{_e(project['description'])}</p>" if project.get("description") else ""
    return (
        "<section id='project'><h2>1. Project</h2>" + description
        + _rows([("Name", _e(project["name"])),
                 ("Project file", _e(project["path"] or "not saved")),
                 ("Saved without later changes", _num(bool(project["saved"])))])
        + "</section>"
    )


def _section_survey(m) -> str:
    survey = m.get("survey")
    if not survey:
        return ""
    source = m.get("source_metadata") or {}
    vp = source.get("viewpoints", {})
    ev = source.get("observation_events", {})
    sensors = "".join(
        f"<tr><td>{_e(s['sensor_id'])}</td><td>{_e(s['modality'])}</td><td>{_e(s['model'])}</td>"
        f"<td class='n'>{_num(s['horizontal_fov_deg'])}</td></tr>" for s in survey["sensors"]
    )
    sensor_table = (
        "<h3>Sensors</h3><table><tr><th>Sensor</th><th>Modality</th><th>Model</th>"
        f"<th>Horizontal FOV (°)</th></tr>{sensors}</table>" if sensors else ""
    )
    return (
        "<section id='survey'><h2>2. Survey</h2>"
        + _rows([("Configuration", f"{_e(survey['name'])} ({_e(survey['configuration_id'])})"),
                 ("Type", _e(survey["configuration_type"])),
                 ("Viewpoints", _num(survey["viewpoints"])),
                 ("ObservationEvents", _num(survey["observation_events"])),
                 ("Sources", _e(", ".join(survey["sources"]) or "not recorded"))])
        + sensor_table
        + "<h3>Source metadata as recorded</h3>"
        "<p class='note'>Values absent from the source data. These are not corrected here; "
        "how the analysis treated them is listed under Visibility assumptions.</p>"
        + _rows([("Viewpoints with missing heading", _num(vp.get("missing_heading"))),
                 ("Viewpoints with missing observer height", _num(vp.get("missing_observer_height"))),
                 ("Viewpoints with no horizontal FOV (Viewpoint or Sensor)",
                  _num(vp.get("missing_horizontal_fov_viewpoint_and_sensor"))),
                 ("Viewpoints without a Sensor reference", _num(vp.get("missing_sensor_reference"))),
                 ("ObservationEvents with no heading (event or Viewpoint)",
                  _num(ev.get("missing_heading_event_and_viewpoint"))),
                 ("ObservationEvents without timestamp", _num(ev.get("missing_timestamp")))])
        + "</section>"
    )


def _section_world(m) -> str:
    world = m.get("world")
    if not world:
        return ""
    parts = ["<section id='world'><h2>3. World</h2>"]
    environment = world.get("environment")
    if environment:
        dem = environment["elevation"]
        parts.append(_rows([
            ("Environment", f"{_e(environment['name'])} ({_e(environment['environment_id'])})"),
            ("Elevation model", _e(dem["path"])),
            ("SHA-256", f"<code>{_e(dem.get('sha256') or 'unavailable')}</code>"),
            ("CRS", _e(dem["crs"])),
            ("Resolution (m)", _num(dem["resolution_m"])),
            ("NoData value", _num(dem["nodata_value"])),
        ]))
    domain = world.get("analysis_domain")
    if domain:
        cells = domain["cells"]
        parts.append("<h3>AnalysisDomain</h3>" + _rows([
            ("Domain", f"{_e(domain['name'])} ({_e(domain['domain_id'])})"),
            ("Created by", _e(domain.get("creation_method") or "not recorded")),
            ("Grid", f"{domain['grid']['width']:,} × {domain['grid']['height']:,} cells"),
            ("Analysable cells", _num(cells["analysable"])),
            ("Invalid cells (no terrain)", _num(cells["invalid"])),
            ("Outside the domain", _num(cells["outside_domain"])),
        ]))
    parts.append("</section>")
    return "".join(parts)


def _section_assumptions(m) -> str:
    assumptions = m.get("assumptions")
    if not assumptions:
        return ""
    c = assumptions["configuration"]
    applied = assumptions.get("applied") or {}
    rows = [
        ("Configuration", f"{_e(c['name'])} ({_e(c['configuration_id'])})"),
        ("Sampling unit", _e(c["sampling_unit"])),
        ("Maximum distance (m)", _num(c["max_distance_m"])),
        ("Default observer height (m)", _num(c["default_observer_height_m"])),
        ("Target height (m)", _num(c["default_target_height_m"])),
        ("Viewing direction used", _num(c["use_direction"])),
        ("Missing-heading policy", _e(c["missing_heading_policy"])),
        ("Missing-FOV policy", _e(c["missing_fov_policy"])),
        ("Missing-observer-height policy", _e(c["missing_observer_height_policy"])),
        ("Default horizontal FOV (°)", _num(c["default_horizontal_fov_deg"])),
        ("Earth curvature coefficient", _num(c["curvature_coefficient"])),
        ("Backend", _e(c["backend"])),
    ]
    applied_rows = ""
    if applied:
        applied_rows = "<h3>How the assumptions were applied</h3>" + _rows([
            ("Sampling units", _num(applied["sampling_units"])),
            ("Heading missing (source)", _num(applied["heading_missing"])),
            ("… interpreted as omnidirectional (policy)",
             _num(applied["heading_missing_treated_omnidirectional"])),
            ("… given the default heading (policy)", _num(applied["heading_missing_used_default"])),
            ("Default observer height used", _num(applied["used_default_observer_height"])),
            ("Default horizontal FOV used", _num(applied["used_default_horizontal_fov"])),
            ("Excluded by a missing-metadata policy", _num(applied["excluded_by_policy"])),
        ])
    return (
        "<section id='assumptions'><h2>4. Visibility assumptions</h2>"
        f"<p class='note'>{_e(assumptions['note'])}</p>" + _rows(rows) + applied_rows
        + "</section>"
    )


def _section_observability(m, rendered) -> str:
    o = m.get("observability")
    if not o:
        return ""
    if o["status"] != "current":
        return ("<section id='observability'><h2>5. Observability results</h2>"
                f"<p class='warn'>The observability field is out of date ({_e(o['reason'])}); "
                "no results are reported.</p></section>")
    c = o["coverage"]
    build = o.get("build_report") or {}
    return (
        "<section id='observability'><h2>5. Observability results</h2>"
        f"<p>{_pct(c['observable_share_of_analysable'])} of valid analysable cells "
        f"({c['observable_cells']:,} of {c['analysable_cells']:,}) were observable from at least "
        f"one of {o['active_sampling_units']:,} active sampling units under the configured "
        f"visibility model; {c['blind_cells']:,} analysable cells were blind spots.</p>"
        + _rows([("Observability field", f"<code>{_e(o['sof_id'])}</code>"),
                 ("Active sampling units", _num(o["active_sampling_units"])),
                 ("Requested / added units",
                  f"{_num(build.get('requested_units'))} / {_num(build.get('added_units'))}"),
                 ("Excluded / failed units",
                  f"{_num(build.get('excluded_units'))} / {_num(build.get('failed_units'))}"),
                 ("Analysable cells", _num(c["analysable_cells"])),
                 ("Observable cells", _num(c["observable_cells"])),
                 ("Blind cells", _num(c["blind_cells"]))])
        + _checks(o["checks"][:1]) + _figures(rendered, "observability") + "</section>"
    )


def _section_coverage(m, rendered) -> str:
    o = m.get("observability")
    if not o or o["status"] != "current":
        return ""
    c = o["coverage"]
    return (
        "<section id='coverage'><h2>6. Coverage and exposure</h2>"
        f"<p>{c['unique_cells']:,} cells were observable from exactly one active sampling unit "
        f"and {c['repeated_cells']:,} from two or more. Mean exposure over analysable cells was "
        f"{_num(c['mean_exposure'])} sampling units (maximum {c['maximum_exposure']:,}).</p>"
        + _rows([("Unique coverage (exposure 1)", _num(c["unique_cells"])),
                 ("Repeated coverage (exposure ≥ 2)", _num(c["repeated_cells"])),
                 ("Mean exposure", _num(c["mean_exposure"])),
                 ("Median exposure", _num(c["median_exposure"])),
                 ("Mean exposure of observable cells", _num(c["mean_exposure_observable"])),
                 ("Maximum exposure", _num(c["maximum_exposure"]))])
        + _checks(o["checks"][1:2]) + _figures(rendered, "coverage") + "</section>"
    )


def _section_contribution(m, rendered) -> str:
    a = (m.get("analysis") or {}).get("contribution")
    if not a:
        return ""
    return (
        "<section id='contribution'><h2>7. Sampling-unit contribution</h2>"
        f"<p>{a['units_with_unique_cells']:,} of {a['available_units']:,} sampling units observe "
        f"at least one cell no other unit observes; {a['units_without_unique_cells']:,} observe "
        "only cells that other units also observe. Repeated observation can still add "
        "robustness, temporal information or other viewing directions.</p>"
        + _rows([("Sampling units analysed", _num(a["units"])),
                 ("Without readable cached visibility", _num(a["unavailable_units"])),
                 ("Observing no analysable cells", _num(a["units_without_visible_cells"])),
                 ("Unique cells summed over units", _num(a["unique_cells_summed_over_units"])),
                 ("Unique-coverage cells in the field", _num(a["field_unique_cells"])),
                 ("Consistent with the field", _num(a["consistent_with_field"]))])
        + _figures(rendered, "contribution") + "</section>"
    )


def _section_scenario(m, rendered) -> str:
    analysis = m.get("analysis") or {}
    s = analysis.get("scenario")
    saved = analysis.get("saved_scenarios") or []
    if not s and not saved:
        return ""
    parts = ["<section id='scenario'><h2>8. Design scenario</h2>"]
    if s:
        d = s["difference_scenario_minus_baseline"]
        b, sc = s["baseline"], s["scenario"]
        parts.append(
            f"<p>The current what-if scenario deactivates {s['deactivated_units']:,} existing "
            f"sampling unit(s) and includes {s['included_candidates']:,} candidate Viewpoint(s). "
            f"Observable cells change by {d['observable_cells']:+,} "
            f"({_pp(d['coverage_percentage_points'])}) relative to the baseline. The Survey "
            "itself is unchanged.</p>"
            "<table><tr><th></th><th>Baseline</th><th>Scenario</th><th>Scenario − baseline</th></tr>"
            + "".join(
                f"<tr><th scope='row'>{_e(label)}</th><td class='n'>{_num(b[k])}</td>"
                f"<td class='n'>{_num(sc[k])}</td><td class='n'>{_num(d.get(k))}</td></tr>"
                for label, k in (("Observable cells", "observable_cells"),
                                 ("Blind cells", "blind_cells"),
                                 ("Unique cells", "unique_cells"),
                                 ("Repeated cells", "repeated_cells"))
            ) + "</table>" + _checks(s["checks"][:2]) + _figures(rendered, "scenario")
        )
    if saved:
        rows = "".join(
            f"<tr><td>{_e(x['name'])}</td><td>{'current' if x['status'] == 'current' else 'out of date (earlier observability field)'}</td>"
            f"<td class='n'>{_num(x['deactivated_units'])}</td><td class='n'>{_num(x['included_candidates'])}</td>"
            f"<td class='n'>{_num(x['summary']['observable_cells'])}</td></tr>"
            for x in saved
        )
        parts.append(
            "<h3>Saved scenarios</h3><table><tr><th>Name</th><th>Status</th><th>Deactivated</th>"
            f"<th>Candidates</th><th>Observable cells</th></tr>{rows}</table>"
        )
    parts.append("</section>")
    return "".join(parts)


def _section_comparison(m, rendered) -> str:
    c = (m.get("analysis") or {}).get("comparison")
    if not c:
        return ""
    left, right, d = c["left"], c["right"], c["difference_right_minus_left"]
    rows = "".join(
        f"<tr><th scope='row'>{_e(label)}</th><td class='n'>{_num(left.get(k))}</td>"
        f"<td class='n'>{_num(right.get(k))}</td><td class='n'>{_num(d.get(k))}</td></tr>"
        for label, k in (("Observable cells", "observable_cells"), ("Blind cells", "blind_cells"),
                         ("Unique cells", "unique_cells"), ("Repeated cells", "repeated_cells"))
    )
    return (
        "<section id='comparison'><h2>9. Scenario comparison</h2>"
        f"<p>Left: <strong>{_e(left['label'])}</strong>; right: <strong>{_e(right['label'])}</strong>. "
        "Every difference is <strong>right − left</strong>. "
        f"{c['cells_gained_coverage']:,} cells were blind on the left and observable on the right; "
        f"{c['cells_lost_coverage']:,} were observable on the left and blind on the right "
        f"(coverage {_pp(d['coverage_percentage_points'])}).</p>"
        "<table><tr><th></th><th>Left</th><th>Right</th><th>Right − left</th></tr>"
        f"{rows}</table>" + _checks(c["checks"]) + _figures(rendered, "comparison") + "</section>"
    )


def _section_provenance(m) -> str:
    warnings = "".join(f"<p class='warn'>{_e(w['message'])}</p>" for w in m.get("warnings", ()))
    limitations = "".join(f"<li>{_e(x)}</li>" for x in m.get("limitations", ()))
    return (
        "<section id='provenance'><h2>10. Provenance and limitations</h2>"
        "<p>The full machine-readable record is in <a href='provenance.json'>provenance.json</a> "
        "(readable version: <a href='provenance.md'>provenance.md</a>).</p>"
        + warnings + (f"<h3>Limitations</h3><ul>{limitations}</ul>" if limitations else "")
        + "</section>"
    )
