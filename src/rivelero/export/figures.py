"""Standalone scientific figures (P3).

Figures are built from the same ExportContext as the P2 data export, using
the canonical visualization semantics:

* maps use ``rivelero.visualization.layers.map_layer`` - the definition the
  interactive map uses - so colours, whole-field scales, masks, legends and
  the RIGHT - LEFT difference scale are identical to the application;
* A1 charts call ``plot_coverage_composition`` / ``plot_exposure_distribution``
  with ``summarize_coverage(sof)``, exactly as the Analysis page does.

Every figure is drawn on a fresh Matplotlib ``Figure`` (no pyplot state), so
building one never changes a GUI map and is safe on a worker thread. Maps
always show the full grid extent, never a GUI zoom.

Formats: PNG (configurable DPI, default 300), SVG and PDF (vector; text is
kept as text). Lightweight metadata is embedded where the format supports
it and a JSON sidecar records the full description.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from rivelero.analysis.coverage import CoverageSummary, summarize_coverage
from rivelero.export.catalog import ExportContext
from rivelero.export.files import atomic_output, check_path_length
from rivelero.export.metadata import product_metadata, sidecar_path, write_sidecar
from rivelero.visualization import layers
from rivelero.visualization.analysis import (
    plot_coverage_composition,
    plot_exposure_distribution,
)
from rivelero.visualization.maps import project_viewpoints, raster_extent

FORMATS = ("png", "svg", "pdf")
DEFAULT_DPI = 300
MAP_SIZE = (7.5, 6.5)
CHART_SIZE = (8.0, 3.6)

GROUP_MAPS = "Observability maps"
GROUP_ANALYSIS = "Coverage and contribution"
GROUP_DESIGN = "Scenario and comparison"
FIGURE_GROUPS = (GROUP_MAPS, GROUP_ANALYSIS, GROUP_DESIGN)

_TEXT = "#202428"
_MUTED = "#687177"


class FigureUnavailableError(ValueError):
    """Raised when a figure cannot be drawn from the current results."""


@dataclass(frozen=True, slots=True)
class FigureOptions:
    """Deliberately limited customisation."""

    format: str = "png"
    dpi: int = DEFAULT_DPI
    include_viewpoints: bool = True
    title: str | None = None  # overrides the default title
    subtitle: str | None = None  # overrides the default subtitle
    caption: str | None = None

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"Figure format must be one of {', '.join(FORMATS)}.")
        if not 50 <= int(self.dpi) <= 1200:
            raise ValueError("DPI must be between 50 and 1200.")


@dataclass(frozen=True, slots=True)
class FigureProduct:
    key: str
    group: str
    label: str
    meaning: str
    available: bool
    reason: str | None
    build: Callable[[FigureOptions], Figure] | None = field(default=None, repr=False, compare=False)
    sources: dict[str, Any] = field(default_factory=dict, compare=False)

    def filename(self, prefix: str, fmt: str) -> str:
        return f"{prefix}_{self.key}.{fmt}"


# ---------------------------------------------------------------------------
# Map figures
# ---------------------------------------------------------------------------

MAP_FIGURES = {
    "analysis_state": (layers.OBSERVABILITY_STATE, "Analysis state",
                       "Categorical state of every cell: outside the AnalysisDomain, "
                       "invalid terrain, blind spot, observable."),
    "exposure": (layers.EXPOSURE, "Exposure count",
                 "Number of active sampling units from which each analysable cell is "
                 "visible; scale spans the whole field."),
    "normalized_exposure": (layers.NORMALIZED_EXPOSURE, "Normalized exposure",
                            "Exposure count divided by the number of active sampling units."),
    "blind_spots": (layers.BLIND_SPOTS, "Blind spots",
                    "Analysable cells visible from no active sampling unit."),
    "coverage_class": (layers.COVERAGE_CLASS, "Unique and repeated coverage",
                       "Blind spots, cells seen by exactly one sampling unit (unique) and "
                       "by two or more (repeated)."),
}


def plain_text(text: str | None) -> str | None:
    """Escape Matplotlib mathtext so user text (names with "$") prints verbatim."""

    return None if text is None else str(text).replace("$", r"\$")


def _spatial_labels(ax, sof) -> None:
    unit = layers.linear_unit(sof.crs)
    crs = sof.crs.to_string() if sof.crs is not None else "unknown CRS"
    suffix = f"{unit}, {crs}" if unit else crs
    ax.set_xlabel(f"X ({suffix})", color=_MUTED, fontsize=9)
    ax.set_ylabel(f"Y ({suffix})", color=_MUTED, fontsize=9)
    ax.ticklabel_format(style="plain", useOffset=False, axis="both")
    ax.tick_params(labelsize=8, colors=_MUTED)
    ax.set_aspect("equal", adjustable="box")


def _decorate(fig: Figure, ax, options: FigureOptions, title: str, subtitle: str | None) -> None:
    fig.suptitle(plain_text(options.title or title), fontsize=12, color=_TEXT)
    sub = options.subtitle if options.subtitle is not None else subtitle
    if sub:
        ax.set_title(plain_text(sub), fontsize=9, color=_MUTED)
    if options.caption:
        fig.supxlabel(plain_text(textwrap.fill(options.caption, 110)), fontsize=8, color=_MUTED)


def draw_map(
    kind: str,
    context: ExportContext,
    options: FigureOptions,
    *,
    title: str,
    subtitle: str | None = None,
    **arrays: Any,
) -> Figure:
    """Draw one canonical map layer on a new Figure (full grid extent)."""

    sof = context.sof
    if sof is None:
        raise FigureUnavailableError(context.sof_reason or "No observability field.")
    layer = layers.map_layer(kind, sof, **arrays)
    if layer is None:
        raise FigureUnavailableError(f"The {kind} layer has no data.")

    fig = Figure(figsize=MAP_SIZE, layout="constrained")
    ax = fig.add_subplot()
    extent = raster_extent(shape=sof.exposure_count.shape, transform=sof.transform)
    if layer.hatched:
        # Non-analysable cells show the hatched background, never the
        # neutral "no difference" colour.
        ax.patch.set_hatch("////")
        ax.patch.set_edgecolor(layers.NOT_ANALYSABLE_EDGE)
    image = ax.imshow(layer.data, extent=extent, origin="upper", cmap=layer.cmap,
                      norm=layer.norm, interpolation="nearest", zorder=1)
    image.set_gid(f"rivelero_layer_{kind}")

    if options.include_viewpoints and context.configuration is not None:
        ids, xs, ys = project_viewpoints(context.configuration.viewpoints, sof.crs)
        if ids:
            dense = len(ids) > 2000
            points = ax.scatter(xs, ys, s=6 if dense else 16, marker="o", facecolors=_TEXT,
                                edgecolors="white", linewidths=0.4 if dense else 0.6,
                                zorder=5, label="Viewpoints")
            points.set_gid("rivelero_viewpoints")

    handles = list(layer.legend_handles)
    handles += [c for c in ax.collections if c.get_gid() == "rivelero_viewpoints"]
    if options.include_viewpoints and kind in (layers.SCENARIO_CHANGE, layers.SCENARIO_EXPOSURE):
        handles += _draw_scenario_units(ax, context, sof)
    if layer.colorbar_label is not None:
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(layer.colorbar_label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        if layer.integer_ticks:
            colorbar.locator = MaxNLocator(integer=True)
            colorbar.update_ticks()
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=2,
                  fontsize=8, frameon=False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _spatial_labels(ax, sof)
    _decorate(fig, ax, options, title, subtitle)
    return fig


def _draw_scenario_units(ax, context: ExportContext, sof) -> list:
    """Mark deactivated units and included candidates of the scenario."""

    record = context.scenario_record or {}
    artists = []
    deactivated = set(record.get("deactivated_sampling_units", ()))
    if deactivated and context.configuration is not None:
        chosen = [vp for vp in context.configuration.viewpoints if vp.viewpoint_id in deactivated]
        ids, xs, ys = project_viewpoints(chosen, sof.crs)
        if ids:
            artist = ax.scatter(xs, ys, s=70, marker="o", facecolors="none", edgecolors=_TEXT,
                                linewidths=1.2, zorder=6, label="Deactivated sampling unit")
            artist.set_gid("rivelero_deactivated")
            artists.append(artist)
    candidates = [
        _Point(c["viewpoint_id"], c["x"], c["y"], c.get("crs") or sof.crs.to_string())
        for c in record.get("candidates", ()) if c.get("included")
    ]
    ids, xs, ys = project_viewpoints(candidates, sof.crs)
    if ids:
        artist = ax.scatter(xs, ys, s=46, marker="D", facecolors="white", edgecolors=_TEXT,
                            linewidths=1.0, zorder=6, label="Included candidate Viewpoint")
        artist.set_gid("rivelero_candidates")
        artists.append(artist)
    return artists


@dataclass(frozen=True, slots=True)
class _Point:
    viewpoint_id: str
    x: float
    y: float
    crs: str


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def coverage_summary(context: ExportContext) -> CoverageSummary:
    """The Analysis page's A1 summary (same function, same denominators)."""

    if context.sof is None:
        raise FigureUnavailableError(context.sof_reason or "No observability field.")
    return summarize_coverage(context.sof)


def draw_chart(plot, summary: CoverageSummary, options: FigureOptions, *, title: str,
               subtitle: str | None) -> Figure:
    fig = Figure(figsize=CHART_SIZE, layout="constrained")
    ax = fig.add_subplot()
    plot(summary, ax=ax, title="")
    legend = ax.get_legend()
    if legend is not None and plot is plot_coverage_composition:
        # The GUI anchors this legend below the bar by hand; in a standalone
        # figure it goes outside the axes where the layout reserves room.
        handles, labels = ax.get_legend_handles_labels()
        legend.remove()
        fig.legend(handles, labels, loc="outside right upper", fontsize=8, frameon=False)
    _decorate(fig, ax, options, title, subtitle)
    return fig


def draw_contribution_distribution(context: ExportContext, options: FigureOptions) -> Figure:
    """Distribution over sampling units of visible and unique cells.

    Scales to any number of units (histograms, never one bar per unit).
    Units without readable visibility are counted in the subtitle, never
    plotted as zero.
    """

    analysis = context.contribution
    if analysis is None:
        raise FigureUnavailableError(context.contribution_reason or "No contribution analysis.")
    units = analysis.available_units
    without = sum(1 for u in units if u.unique_cells == 0)
    fig = Figure(figsize=(8.0, 3.8), layout="constrained")
    axes = fig.subplots(1, 2)
    panels = (
        (axes[0], [u.visible_cells for u in units], "Visible analysable cells per unit",
         "#2A78D6", f"All {len(units):,} units"),
        # Units without unique cells are stated, not merged into the first bin.
        (axes[1], [u.unique_cells for u in units if u.unique_cells > 0],
         "Unique cells per unit (seen by no other unit)", "#E07B39",
         f"{len(units) - without:,} units with unique cells; {without:,} with none"),
    )
    for ax, values, xlabel, color, panel_title in panels:
        values = np.asarray(values, dtype=float)
        if values.size:
            bins = min(30, max(1, len(np.unique(values))))
            ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        else:
            ax.text(0.5, 0.5, "No units", transform=ax.transAxes, ha="center", color=_MUTED)
        ax.set_title(panel_title, fontsize=9, color=_MUTED)
        ax.set_xlabel(xlabel, fontsize=9, color=_MUTED)
        ax.set_ylabel("Number of sampling units", fontsize=9, color=_MUTED)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=_MUTED)
        ax.grid(axis="y", color="#E8EBED", linewidth=0.8, zorder=0)
    subtitle = (
        f"{len(units):,} sampling units; {without:,} without unique cells "
        "(their coverage is also observed by other units)"
    )
    if analysis.unavailable_units:
        subtitle += f"; {len(analysis.unavailable_units):,} without readable visibility (not shown)"
    fig.suptitle(plain_text(options.title or "Sampling-unit contribution"), fontsize=12,
                 color=_TEXT)
    text = options.subtitle if options.subtitle is not None else subtitle
    if options.caption:
        text = f"{text}. {options.caption}" if text else options.caption
    if text:
        fig.supxlabel(plain_text(textwrap.fill(text, 120)), fontsize=8,
                      color=_MUTED)
    return fig


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def figure_catalog(context: ExportContext) -> list[FigureProduct]:
    sof = context.sof
    products: list[FigureProduct] = []
    project = context.provenance.project_name

    def add(key, group, label, meaning, available, reason, build, sources=None):
        products.append(FigureProduct(
            key=key, group=group, label=label, meaning=meaning, available=available,
            reason=None if available else reason, build=build if available else None,
            sources=sources or {},
        ))

    units = None if sof is None else sof.n_active_units
    base_subtitle = None if sof is None else f"{project} · {units:,} active sampling units"
    for key, (kind, label, meaning) in MAP_FIGURES.items():
        add(key, GROUP_MAPS, f"{label} map", meaning, sof is not None, context.sof_reason,
            lambda options, kind=kind, label=label: draw_map(
                kind, context, options, title=label, subtitle=base_subtitle),
            {"layer": kind, "sof_id": None if sof is None else sof.sof_id})

    add("coverage_composition", GROUP_ANALYSIS, "Coverage composition chart",
        "Shares of analysable cells that are blind, uniquely and repeatedly observed.",
        sof is not None, context.sof_reason,
        lambda options: draw_chart(plot_coverage_composition, coverage_summary(context), options,
                                   title="Composition of analysable space",
                                   subtitle=base_subtitle),
        {"summary": "summarize_coverage(sof)"})
    add("exposure_distribution", GROUP_ANALYSIS, "Exposure distribution chart",
        "Share of analysable cells at each exposure level (same binning as the Analysis page).",
        sof is not None, context.sof_reason,
        lambda options: draw_chart(plot_exposure_distribution, coverage_summary(context), options,
                                   title="Exposure distribution", subtitle=base_subtitle),
        {"summary": "summarize_coverage(sof)"})
    add("contribution_distribution", GROUP_ANALYSIS, "Contribution distribution chart",
        "Distribution over sampling units of visible and unique cells.",
        context.contribution is not None, context.contribution_reason,
        lambda options: draw_contribution_distribution(context, options),
        {"contribution_sof_id": None if context.contribution is None else context.contribution.sof_id})

    has_scenario = sof is not None and context.scenario_exposure is not None
    record = context.scenario_record or {}
    scenario_subtitle = None
    if has_scenario:
        deactivated = len(record.get("deactivated_sampling_units", ()))
        included = sum(1 for c in record.get("candidates", ()) if c.get("included"))
        scenario_subtitle = (f"{project} · {deactivated:,} unit(s) deactivated, "
                             f"{included:,} candidate(s) included")
    add("scenario_change", GROUP_DESIGN, "Baseline → scenario change map",
        "Coverage change of each cell from the baseline to the current what-if scenario.",
        has_scenario and context.scenario_change is not None, context.scenario_reason,
        lambda options: draw_map(layers.SCENARIO_CHANGE, context, options,
                                 title="Baseline → scenario coverage change",
                                 subtitle=scenario_subtitle,
                                 scenario_classes=context.scenario_change),
        {"direction": "baseline_to_scenario", "scenario": record})
    add("scenario_exposure", GROUP_DESIGN, "Scenario exposure map",
        "Exposure count of the current scenario on the baseline scale.",
        has_scenario, context.scenario_reason,
        lambda options: draw_map(layers.SCENARIO_EXPOSURE, context, options,
                                 title="Scenario exposure", subtitle=scenario_subtitle,
                                 scenario_exposure=np.asarray(context.scenario_exposure)),
        {"scenario": record})

    comparison = context.comparison
    sides = {}
    comparison_subtitle = None
    if comparison is not None:
        sides = {"direction": "right_minus_left",
                 "left": {"state_id": comparison.left.state_id, "label": comparison.left.label},
                 "right": {"state_id": comparison.right.state_id, "label": comparison.right.label}}
        comparison_subtitle = f"Left: {comparison.left.label}   ·   Right: {comparison.right.label}"
    add("comparison_change", GROUP_DESIGN, "Comparison coverage-change map (left → right)",
        "Coverage change of each cell from the LEFT to the RIGHT design state.",
        comparison is not None, context.comparison_reason,
        lambda options: draw_map(
            layers.COMPARISON_CHANGE, context, options,
            title=f"Coverage change: {comparison.left.label} → {comparison.right.label}",
            subtitle=comparison_subtitle, comparison_classes=comparison.change_classes()),
        sides)
    add("exposure_difference", GROUP_DESIGN, "Exposure-difference map (right − left)",
        "Exposure of the RIGHT state minus exposure of the LEFT state, symmetric scale "
        "centred on zero.",
        comparison is not None, context.comparison_reason,
        lambda options: draw_map(
            layers.EXPOSURE_DIFFERENCE, context, options,
            title=f"Exposure difference: {comparison.right.label} − {comparison.left.label}",
            subtitle=comparison_subtitle + "   ·   right − left",
            exposure_difference=comparison.exposure_difference()),
        sides)
    return products


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def figure_record(context: ExportContext, product: FigureProduct, options: FigureOptions) -> dict:
    return product_metadata(
        context.provenance,
        product=f"figure_{product.key}",
        meaning=product.meaning,
        extra={
            "figure": {"format": options.format,
                       "dpi": options.dpi if options.format == "png" else None,
                       "include_viewpoints": options.include_viewpoints,
                       "title": options.title, "caption": options.caption},
            "sources": product.sources,
        },
    )


def _embedded_metadata(record: dict, fmt: str, title: str) -> dict[str, str]:
    software = record["software"]
    provenance = record["provenance"]
    creator = f"Rivelero {software.get('version') or ''} ({software.get('git_commit') or 'unknown commit'})"
    summary = json.dumps({
        "product": record["product"], "meaning": record["meaning"],
        "project": provenance.get("project_name"), "sof_id": provenance.get("sof_id"),
        "analysis_domain_id": provenance.get("analysis_domain_id"),
        "visibility_configuration_id": provenance.get("visibility_configuration_id"),
        "exported_at": provenance.get("exported_at"), **record.get("sources", {}),
    }, ensure_ascii=False, default=str)
    if fmt == "png":
        return {"Title": title, "Software": creator, "Description": summary,
                "Creation Time": str(provenance.get("exported_at"))}
    if fmt == "svg":
        return {"Title": title, "Creator": creator, "Description": summary,
                "Date": str(provenance.get("exported_at"))}
    return {"Title": title, "Creator": creator, "Subject": summary}


def save_figure(fig: Figure, path: Path, options: FigureOptions, record: dict, *,
                overwrite: bool = False) -> Path:
    """Save atomically; vector formats keep text as text."""

    title = fig.get_suptitle() or record["product"]
    rc = {"svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "rivelero"}
    with atomic_output(path, overwrite=overwrite) as temporary:
        with matplotlib.rc_context(rc):
            fig.savefig(temporary, format=options.format, dpi=options.dpi,
                        metadata=_embedded_metadata(record, options.format, title),
                        facecolor="white")
    return Path(path).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class FigureExportResult:
    directory: Path
    written: tuple[Path, ...]
    failures: tuple[tuple[str, str], ...]

    @property
    def successful(self) -> bool:
        return not self.failures


def run_figure_export(
    context: ExportContext,
    keys: list[str] | tuple[str, ...],
    directory: Path | str,
    options: FigureOptions = FigureOptions(),
    *,
    overwrite: bool = False,
    sidecars: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> FigureExportResult:
    """Export selected figures (same conflict and failure rules as data export)."""

    from rivelero.export.catalog import ExportConflictError, ExportUnavailableError

    catalog = {product.key: product for product in figure_catalog(context)}
    unknown = [key for key in keys if key not in catalog]
    if unknown:
        raise KeyError(f"Unknown figure(s): {', '.join(unknown)}")
    selected = [catalog[key] for key in dict.fromkeys(keys)]
    if not selected:
        raise ExportUnavailableError("Select at least one figure.")
    unavailable = [p for p in selected if not p.available]
    if unavailable:
        raise ExportUnavailableError("; ".join(f"{p.label}: {p.reason}" for p in unavailable))

    target = Path(directory).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Not a folder: {target}")
    planned = []
    for product in selected:
        path = target / product.filename(context.prefix, options.format)
        planned.append(path)
        if sidecars:
            planned.append(sidecar_path(path))
    for path in planned:
        check_path_length(path)
    if not overwrite:
        conflicts = [path for path in planned if path.exists()]
        if conflicts:
            raise ExportConflictError(conflicts)
    target.mkdir(parents=True, exist_ok=True)

    written, failures = [], []
    for index, product in enumerate(selected):
        if progress_callback is not None:
            progress_callback(index, len(selected), f"Rendering {product.label}")
        path = target / product.filename(context.prefix, options.format)
        try:
            fig = product.build(options)
            record = figure_record(context, product, options)
            existed = path.exists()
            written.append(save_figure(fig, path, options, record, overwrite=overwrite))
            if sidecars:
                try:
                    written.append(write_sidecar(path, record, overwrite=overwrite))
                except BaseException:
                    if not existed:
                        path.unlink(missing_ok=True)
                    written.pop()
                    raise
        except Exception as error:
            failures.append((product.key, str(error) or type(error).__name__))
    if progress_callback is not None:
        progress_callback(len(selected), len(selected), "")
    return FigureExportResult(target, tuple(written), tuple(failures))
