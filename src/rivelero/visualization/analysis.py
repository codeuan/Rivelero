"""Visualization of descriptive survey analysis (Analysis & Design A1).

Charts consume a CoverageSummary computed by rivelero.analysis.coverage.
Like the other visualization modules they perform no scientific calculation,
do not depend on the GUI and return ordinary Matplotlib objects.

Colour semantics extend the shared observability palette: outside domain,
invalid and blind spot keep their observability-state colours, and
observable space is split into two ordered steps of the observable blue
(unique = lighter, repeated = darker). Every pair was checked for separation
under simulated protan, deutan and tritan vision (OKLab ΔE × 100 ≥ 9.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from rivelero.analysis.contribution import UnitContributionClass
from rivelero.analysis.coverage import CoverageClass
from rivelero.analysis.scenario import ScenarioChangeClass
from rivelero.observability.masks import ObservabilityState
from rivelero.visualization.observability import OBSERVABILITY_STATE_COLORS

if TYPE_CHECKING:
    from rivelero.analysis.coverage import CoverageSummary


COVERAGE_CLASS_COLORS: dict[CoverageClass, str] = {
    CoverageClass.OUTSIDE_DOMAIN: OBSERVABILITY_STATE_COLORS[
        ObservabilityState.OUTSIDE_DOMAIN
    ],
    CoverageClass.INVALID: OBSERVABILITY_STATE_COLORS[ObservabilityState.INVALID],
    CoverageClass.BLIND_SPOT: OBSERVABILITY_STATE_COLORS[
        ObservabilityState.BLIND_SPOT
    ],
    CoverageClass.UNIQUE: "#86B6EF",
    CoverageClass.REPEATED: "#1C5CAB",
}

COVERAGE_CLASS_LABELS: dict[CoverageClass, str] = {
    CoverageClass.OUTSIDE_DOMAIN: "Outside domain",
    CoverageClass.INVALID: "Invalid / unanalysable",
    CoverageClass.BLIND_SPOT: "Blind spot (exposure 0)",
    CoverageClass.UNIQUE: "Unique coverage (exposure 1)",
    CoverageClass.REPEATED: "Repeated coverage (exposure ≥ 2)",
}

# Selected-unit contribution map. Unique contribution uses the blind-spot
# orange because those are exactly the cells that would become blind spots
# if the unit were removed; repeated contribution uses the repeated-coverage
# blue. "Not visible" is a mid neutral kept >= 16 OKLab units (x100) from
# every other class under simulated colour-vision deficiency.
UNIT_CONTRIBUTION_COLORS: dict[UnitContributionClass, str] = {
    UnitContributionClass.OUTSIDE_DOMAIN: COVERAGE_CLASS_COLORS[
        CoverageClass.OUTSIDE_DOMAIN
    ],
    UnitContributionClass.INVALID: COVERAGE_CLASS_COLORS[CoverageClass.INVALID],
    UnitContributionClass.NOT_VISIBLE_FROM_UNIT: "#B4BBC1",
    UnitContributionClass.UNIQUE_CONTRIBUTION: COVERAGE_CLASS_COLORS[
        CoverageClass.BLIND_SPOT
    ],
    UnitContributionClass.REPEATED_CONTRIBUTION: COVERAGE_CLASS_COLORS[
        CoverageClass.REPEATED
    ],
}

UNIT_CONTRIBUTION_LABELS: dict[UnitContributionClass, str] = {
    UnitContributionClass.OUTSIDE_DOMAIN: "Outside domain",
    UnitContributionClass.INVALID: "Invalid / unanalysable",
    UnitContributionClass.NOT_VISIBLE_FROM_UNIT: "Not visible from unit",
    UnitContributionClass.UNIQUE_CONTRIBUTION: (
        "Unique contribution (blind if removed)"
    ),
    UnitContributionClass.REPEATED_CONTRIBUTION: (
        "Repeated contribution (also seen by others)"
    ),
}


def unit_contribution_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Fixed categorical colormap and norm for UnitContributionClass rasters."""

    cmap = ListedColormap(
        [UNIT_CONTRIBUTION_COLORS[value] for value in UnitContributionClass],
        name="rivelero_unit_contribution",
    )
    norm = BoundaryNorm(
        np.arange(len(UnitContributionClass) + 1, dtype=float) - 0.5,
        cmap.N,
    )
    return cmap, norm


def unit_contribution_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=UNIT_CONTRIBUTION_COLORS[value],
            edgecolor="#8A9297",
            label=UNIT_CONTRIBUTION_LABELS[value],
        )
        for value in UnitContributionClass
    ]


# Baseline -> scenario change map. Unchanged classes use pale tints and
# changes use saturated orange (lost: became blind) and blue (gained: became
# observable), so lost/gained never rely on red/green. Checked for >= 15
# OKLab units (x100) under normal vision and >= 11 under simulated
# protan/deutan/tritan vision for every pair.
SCENARIO_CHANGE_COLORS: dict[ScenarioChangeClass, str] = {
    ScenarioChangeClass.OUTSIDE_DOMAIN: COVERAGE_CLASS_COLORS[
        CoverageClass.OUTSIDE_DOMAIN
    ],
    ScenarioChangeClass.INVALID: COVERAGE_CLASS_COLORS[CoverageClass.INVALID],
    ScenarioChangeClass.REMAINS_BLIND: "#EDB07F",
    ScenarioChangeClass.REMAINS_OBSERVABLE: "#9EC0EC",
    ScenarioChangeClass.LOST_COVERAGE: "#C4520F",
    ScenarioChangeClass.GAINED_COVERAGE: COVERAGE_CLASS_COLORS[
        CoverageClass.REPEATED
    ],
}

SCENARIO_CHANGE_LABELS: dict[ScenarioChangeClass, str] = {
    ScenarioChangeClass.OUTSIDE_DOMAIN: "Outside domain",
    ScenarioChangeClass.INVALID: "Invalid / unanalysable",
    ScenarioChangeClass.REMAINS_BLIND: "Remains blind",
    ScenarioChangeClass.REMAINS_OBSERVABLE: "Remains observable",
    ScenarioChangeClass.LOST_COVERAGE: "Lost coverage (becomes blind)",
    ScenarioChangeClass.GAINED_COVERAGE: "Gained coverage (becomes observable)",
}


def scenario_change_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(
        [SCENARIO_CHANGE_COLORS[value] for value in ScenarioChangeClass],
        name="rivelero_scenario_change",
    )
    norm = BoundaryNorm(
        np.arange(len(ScenarioChangeClass) + 1, dtype=float) - 0.5,
        cmap.N,
    )
    return cmap, norm


def scenario_change_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=SCENARIO_CHANGE_COLORS[value],
            edgecolor="#8A9297",
            label=SCENARIO_CHANGE_LABELS[value],
        )
        for value in ScenarioChangeClass
    ]


# Left -> right comparison uses the same change colours with neutral wording.
COMPARISON_CHANGE_LABELS: dict[ScenarioChangeClass, str] = {
    ScenarioChangeClass.OUTSIDE_DOMAIN: "Outside domain",
    ScenarioChangeClass.INVALID: "Invalid / unanalysable",
    ScenarioChangeClass.REMAINS_BLIND: "Blind in both",
    ScenarioChangeClass.REMAINS_OBSERVABLE: "Observable in both",
    ScenarioChangeClass.LOST_COVERAGE: "Lost coverage (observable left, blind right)",
    ScenarioChangeClass.GAINED_COVERAGE: "Gained coverage (blind left, observable right)",
}


def comparison_change_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=SCENARIO_CHANGE_COLORS[value],
            edgecolor="#8A9297",
            label=COMPARISON_CHANGE_LABELS[value],
        )
        for value in ScenarioChangeClass
    ]


# Signed exposure difference: orange (fewer sampling units than the left)
# through a neutral grey at zero to blue (more), matching the lost/gained
# language. Always used with a norm symmetric about zero.
EXPOSURE_DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "rivelero_exposure_difference",
    ["#C4520F", "#E8A170", "#F0EFEC", "#8FB3E3", "#1C5CAB"],
)


# Recessive chart ink.
_TEXT = "#202428"
_MUTED = "#687177"
_GRID = "#E8EBED"


def coverage_class_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Fixed categorical colormap and norm for CoverageClass rasters."""

    cmap = ListedColormap(
        [COVERAGE_CLASS_COLORS[value] for value in CoverageClass],
        name="rivelero_coverage_class",
    )
    norm = BoundaryNorm(
        np.arange(len(CoverageClass) + 1, dtype=float) - 0.5,
        cmap.N,
    )
    return cmap, norm


def coverage_class_legend_handles() -> list[Patch]:
    """Legend patches for every coverage class."""

    return [
        Patch(
            facecolor=COVERAGE_CLASS_COLORS[value],
            edgecolor="#8A9297",
            label=COVERAGE_CLASS_LABELS[value],
        )
        for value in CoverageClass
    ]


def plot_exposure_distribution(
    summary: "CoverageSummary",
    *,
    ax: Axes | None = None,
    max_bins: int = 20,
    title: str = "Exposure distribution",
) -> tuple[Figure, Axes]:
    """Bar chart of analysable cells per exposure level.

    Bar heights are shares of analysable cells (the same denominator as
    coverage). Bars are coloured by coverage class so the chart reads with
    the coverage-class map.
    """

    fig, ax = _prepare_axes(ax)
    ax.clear()
    ax.set_axis_on()

    if summary.analysable_cells == 0:
        _empty(ax, "No analysable cells.")
        return fig, ax

    bins = summary.distribution.binned(max_bins=max_bins)
    positions = np.arange(len(bins))
    heights = np.array([100.0 * item.fraction for item in bins])
    colors = [
        COVERAGE_CLASS_COLORS[
            CoverageClass.BLIND_SPOT
            if item.upper == 0
            else CoverageClass.UNIQUE
            if item.upper == 1
            else CoverageClass.REPEATED
        ]
        for item in bins
    ]

    bars = ax.bar(
        positions,
        heights,
        width=0.78,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    bars._rivelero_bins = bins

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [item.label for item in bins],
        fontsize=8,
        rotation=45 if len(bins) > 12 else 0,
    )
    ax.set_xlabel("Exposure (number of sampling units)", color=_MUTED, fontsize=9)
    ax.set_ylabel("Share of analysable cells (%)", color=_MUTED, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, color=_TEXT)
    _recessive_axes(ax)

    # Selective direct labels: the two scientifically distinct levels and
    # the most common repeated level, never every bar.
    labelled = {0, 1} & set(range(len(bins)))
    if len(bins) > 2:
        labelled.add(2 + int(np.argmax(heights[2:])))
    for index in sorted(labelled):
        ax.annotate(
            f"{heights[index]:.1f}%",
            (positions[index], heights[index]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_TEXT,
        )

    ax.set_ylim(0, max(heights.max() * 1.15, 1.0))

    ax.legend(
        handles=[
            Patch(
                facecolor=COVERAGE_CLASS_COLORS[value],
                label=COVERAGE_CLASS_LABELS[value],
            )
            for value in (
                CoverageClass.BLIND_SPOT,
                CoverageClass.UNIQUE,
                CoverageClass.REPEATED,
            )
        ],
        fontsize=8,
        frameon=False,
        loc="upper right",
    )

    return fig, ax


def plot_coverage_composition(
    summary: "CoverageSummary",
    *,
    ax: Axes | None = None,
    title: str = "Composition of analysable space",
) -> tuple[Figure, Axes]:
    """100 % stacked bar: blind, unique and repeated shares of analysable space."""

    fig, ax = _prepare_axes(ax)
    ax.clear()
    ax.set_axis_on()

    if summary.analysable_cells == 0:
        _empty(ax, "No analysable cells.")
        return fig, ax

    parts = [
        (CoverageClass.BLIND_SPOT, summary.blind_fraction),
        (CoverageClass.UNIQUE, summary.unique_fraction),
        (CoverageClass.REPEATED, summary.repeated_fraction),
    ]

    left = 0.0
    for value, fraction in parts:
        width = 100.0 * fraction
        ax.barh(
            0,
            width,
            left=left,
            height=0.6,
            color=COVERAGE_CLASS_COLORS[value],
            edgecolor="white",
            linewidth=2.0,
            label=COVERAGE_CLASS_LABELS[value],
        )
        if width >= 8.0:
            ax.text(
                left + width / 2.0,
                0,
                f"{width:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value == CoverageClass.REPEATED else _TEXT,
            )
        left += width

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Share of analysable cells (%)", color=_MUTED, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, color=_TEXT)
    _recessive_axes(ax, grid_axis="x")
    ax.legend(
        fontsize=8,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.55),
        ncol=1,
    )

    return fig, ax


def _prepare_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is None:
        fig, new_ax = plt.subplots(figsize=(7, 3.5))
        return fig, new_ax
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib.axes.Axes or None.")
    return ax.figure, ax


def _recessive_axes(ax: Axes, *, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(axis=grid_axis, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _empty(ax: Axes, message: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes, ha="center", va="center",
        color=_MUTED, fontsize=10,
    )
