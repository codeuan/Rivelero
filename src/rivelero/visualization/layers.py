"""Canonical map-layer semantics shared by the GUI map and exported figures.

``map_layer(kind, sof, ...)`` returns, for one layer of a Survey
Observability Field, exactly what is drawn: the (masked) array, colormap,
norm, colourbar label and legend handles. The interactive map and the
standalone figures (rivelero.export.figures) both use it, so a colour,
scale, mask or legend can never mean something different in a report than
it does in the application.

Scales are fixed by the whole field, never by the visible extent:

* exposure spans 0 .. whole-field maximum;
* scenario exposure shares the baseline scale;
* the exposure difference is symmetric about zero (RIGHT - LEFT) and
  non-analysable cells are hatched so they cannot be read as "no change".

The module performs no scientific calculation and does not depend on Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.patches import Patch

from rivelero.analysis.coverage import coverage_class_raster
from rivelero.observability.masks import ObservabilityState
from rivelero.observability.survey_field import SurveyObservabilityField
from rivelero.visualization.analysis import (
    EXPOSURE_DIFFERENCE_CMAP,
    comparison_change_legend_handles,
    coverage_class_colormap,
    coverage_class_legend_handles,
    scenario_change_colormap,
    scenario_change_legend_handles,
    unit_contribution_colormap,
    unit_contribution_legend_handles,
)
from rivelero.visualization.observability import (
    CONTEXT_COLOR,
    EXPOSURE_CMAP,
    OBSERVABILITY_STATE_COLORS,
    OBSERVABILITY_STATE_LABELS,
    observability_state_colormap,
    observability_state_legend_handles,
)

# Layer identifiers; identical to the GUI's VisualizationLayer values.
OBSERVABILITY_STATE = "observability_state"
OBSERVABLE_SPACE = "observable_space"
EXPOSURE = "exposure"
NORMALIZED_EXPOSURE = "normalized_exposure"
BLIND_SPOTS = "blind_spots"
INDIVIDUAL_VISIBILITY = "effective_visibility"
COVERAGE_CLASS = "coverage_class"
UNIT_CONTRIBUTION = "unit_contribution"
SCENARIO_CHANGE = "scenario_change"
SCENARIO_EXPOSURE = "scenario_exposure"
COMPARISON_CHANGE = "comparison_change"
EXPOSURE_DIFFERENCE = "exposure_difference"

# Colour of the hatching / outline marking non-analysable cells.
NOT_ANALYSABLE_EDGE = "#C8D0D4"
NOT_ANALYSABLE_LABEL = "Outside domain / invalid (not shown)"
NOT_COMPARED_LABEL = "Outside domain / invalid (not compared)"

# Layers whose colourbar should use integer ticks.
INTEGER_LAYERS = frozenset({EXPOSURE, SCENARIO_EXPOSURE, EXPOSURE_DIFFERENCE})


@dataclass(frozen=True)
class MapLayer:
    """What one layer draws."""

    data: np.ndarray
    cmap: Colormap
    norm: Normalize
    colorbar_label: str | None = None
    legend_handles: list[Patch] = field(default_factory=list)
    # True for the exposure difference: non-analysable cells are hatched.
    hatched: bool = False
    integer_ticks: bool = False


def binary_colormap(off_color: str, on_color: str) -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap([off_color, on_color])
    return cmap, BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)


def not_analysable_handle() -> Patch:
    return Patch(facecolor="none", edgecolor=NOT_ANALYSABLE_EDGE, label=NOT_ANALYSABLE_LABEL)


def map_layer(
    kind: str,
    sof: SurveyObservabilityField,
    *,
    state: np.ndarray | None = None,
    coverage_classes: np.ndarray | None = None,
    individual_mask: np.ndarray | None = None,
    unit_classes: np.ndarray | None = None,
    scenario_exposure: np.ndarray | None = None,
    scenario_classes: np.ndarray | None = None,
    comparison_classes: np.ndarray | None = None,
    exposure_difference: np.ndarray | None = None,
) -> MapLayer | None:
    """Return the layer ``kind`` of ``sof``, or None if its input is missing.

    ``state`` and ``coverage_classes`` may be passed as caches; they are
    derived from ``sof`` otherwise. Layers relative to a unit, scenario or
    comparison need their derived array.
    """

    kind = str(getattr(kind, "value", kind))
    analysable = sof.analysable_mask

    if kind == OBSERVABILITY_STATE:
        cmap, norm = observability_state_colormap()
        return MapLayer(
            sof.observability_state if state is None else state, cmap, norm,
            legend_handles=observability_state_legend_handles(),
        )

    if kind == COMPARISON_CHANGE:
        if comparison_classes is None:
            return None
        cmap, norm = scenario_change_colormap()
        return MapLayer(comparison_classes, cmap, norm,
                        legend_handles=comparison_change_legend_handles())

    if kind == EXPOSURE_DIFFERENCE:
        if exposure_difference is None:
            return None
        data = np.ma.masked_where(
            ~analysable | np.ma.getmaskarray(exposure_difference),
            np.ma.getdata(exposure_difference).astype(np.float32),
        )
        # Symmetric about zero so gains and losses of equal size have equal
        # visual weight.
        limit = float(max(1, int(np.abs(data).max()) if data.count() else 1))
        return MapLayer(
            data, EXPOSURE_DIFFERENCE_CMAP, Normalize(vmin=-limit, vmax=limit),
            "Exposure difference (right − left, sampling units)",
            [Patch(facecolor="white", edgecolor=NOT_ANALYSABLE_EDGE, hatch="////",
                   label=NOT_COMPARED_LABEL)],
            hatched=True, integer_ticks=True,
        )

    if kind == SCENARIO_CHANGE:
        if scenario_classes is None:
            return None
        cmap, norm = scenario_change_colormap()
        return MapLayer(scenario_classes, cmap, norm,
                        legend_handles=scenario_change_legend_handles())

    if kind == SCENARIO_EXPOSURE:
        if scenario_exposure is None:
            return None
        data = np.ma.masked_where(~analysable, scenario_exposure.astype(np.float32))
        # Shared scale with the baseline so colours compare directly.
        maximum = max(1, sof.maximum_exposure,
                      int(scenario_exposure[analysable].max(initial=0)))
        return MapLayer(data, EXPOSURE_CMAP, Normalize(vmin=0.0, vmax=float(maximum)),
                        "Scenario exposure (number of sampling units)", integer_ticks=True)

    if kind == UNIT_CONTRIBUTION:
        if unit_classes is None:
            return None
        cmap, norm = unit_contribution_colormap()
        return MapLayer(unit_classes, cmap, norm,
                        legend_handles=unit_contribution_legend_handles())

    if kind == COVERAGE_CLASS:
        cmap, norm = coverage_class_colormap()
        classes = coverage_classes
        if classes is None:
            classes = coverage_class_raster(
                sof.exposure_count, analysis_mask=sof.analysis_mask, valid_mask=sof.valid_mask
            )
        return MapLayer(classes, cmap, norm, legend_handles=coverage_class_legend_handles())

    if kind == OBSERVABLE_SPACE:
        color = OBSERVABILITY_STATE_COLORS[ObservabilityState.OBSERVABLE]
        cmap, norm = binary_colormap(CONTEXT_COLOR, color)
        return MapLayer(
            np.ma.masked_where(~analysable, sof.observable_mask.astype(np.uint8)), cmap, norm,
            legend_handles=[
                Patch(facecolor=color, label="Observable (exposure ≥ 1)"),
                Patch(facecolor=CONTEXT_COLOR, label="Not observable"),
                not_analysable_handle(),
            ],
        )

    if kind == BLIND_SPOTS:
        color = OBSERVABILITY_STATE_COLORS[ObservabilityState.BLIND_SPOT]
        cmap, norm = binary_colormap(CONTEXT_COLOR, color)
        return MapLayer(
            np.ma.masked_where(~analysable, sof.blindspot_mask.astype(np.uint8)), cmap, norm,
            legend_handles=[
                Patch(facecolor=color,
                      label=OBSERVABILITY_STATE_LABELS[ObservabilityState.BLIND_SPOT]),
                Patch(facecolor=CONTEXT_COLOR, label="Observable"),
                not_analysable_handle(),
            ],
        )

    if kind == EXPOSURE:
        # The scale spans the whole field, so zooming never changes what a
        # colour means.
        return MapLayer(
            np.ma.masked_where(~analysable, sof.exposure_count.astype(np.float32)),
            EXPOSURE_CMAP, Normalize(vmin=0.0, vmax=float(max(1, sof.maximum_exposure))),
            "Exposure (number of sampling units)", integer_ticks=True,
        )

    if kind == NORMALIZED_EXPOSURE:
        return MapLayer(
            np.ma.masked_where(~analysable, sof.normalized_exposure),
            EXPOSURE_CMAP, Normalize(vmin=0.0, vmax=1.0),
            "Normalized exposure (fraction of active sampling units)",
        )

    if kind == INDIVIDUAL_VISIBILITY:
        if individual_mask is None:
            return None
        color = OBSERVABILITY_STATE_COLORS[ObservabilityState.OBSERVABLE]
        cmap, norm = binary_colormap(CONTEXT_COLOR, color)
        return MapLayer(
            np.ma.masked_where(~analysable, individual_mask.astype(np.uint8)), cmap, norm,
            legend_handles=[
                Patch(facecolor=color, label="Visible from unit"),
                Patch(facecolor=CONTEXT_COLOR, label="Not visible from unit"),
                not_analysable_handle(),
            ],
        )

    return None


def linear_unit(crs: Any) -> str | None:
    """Readable linear unit of a projected CRS ("m" for metres)."""

    try:
        if crs is None or not crs.is_projected:
            return None
        unit = str(crs.linear_units or "").strip()
    except AttributeError:
        return None
    if unit.lower() in {"metre", "meter", "metres", "meters", "m"}:
        return "m"
    return unit or None
