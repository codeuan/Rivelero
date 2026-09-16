"""Visualization of survey and viewpoint configurations in Rivelero.

This module visualizes the spatial arrangement of Viewpoints and
ObservationEvents independently of their calculated visibility.

It supports both retrospective survey reconstruction and prospective survey
design because both are represented through the same
ViewpointConfiguration abstraction.

No visibility, exposure, or survey metrics are calculated here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

if TYPE_CHECKING:
    from rivelero.core.configuration import ViewpointConfiguration
    from rivelero.core.domain import AnalysisDomain
    from rivelero.core.viewpoint import Viewpoint


def plot_viewpoints(
    configuration: "ViewpointConfiguration",
    *,
    domain: "AnalysisDomain | None" = None,
    ax: Axes | None = None,
    title: str = "Viewpoint configuration",
    show_ids: bool = False,
    show_orientation: bool = True,
    orientation_length: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot Viewpoints in a ViewpointConfiguration.

    Parameters
    ----------
    configuration
        ViewpointConfiguration to visualize.

    domain
        Optional AnalysisDomain whose geometry is drawn as spatial context.

    ax
        Optional Matplotlib Axes.

    title
        Plot title.

    show_ids
        Whether viewpoint identifiers are displayed.

    show_orientation
        Whether heading arrows are shown for Viewpoints with known heading.

    orientation_length
        Length of heading arrows in CRS units. If omitted, a suitable value
        is estimated from the configuration extent.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.
    """

    _validate_configuration(
        configuration
    )

    fig, ax = _prepare_axes(ax)

    if domain is not None:
        _plot_domain_geometry(
            ax,
            domain,
        )

    viewpoints = list(
        configuration.viewpoints
    )

    if not viewpoints:
        ax.set_title(title)
        return fig, ax

    xs = np.array(
        [
            vp.x
            for vp in viewpoints
        ],
        dtype=float,
    )

    ys = np.array(
        [
            vp.y
            for vp in viewpoints
        ],
        dtype=float,
    )

    ax.scatter(
        xs,
        ys,
        s=38,
        marker="o",
        c="black",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
        label="Viewpoint",
    )

    if orientation_length is None:
        orientation_length = (
            _default_orientation_length(
                xs,
                ys,
                domain=domain,
            )
        )

    if show_orientation:
        for viewpoint in viewpoints:
            if viewpoint.heading_deg is None:
                continue

            _plot_heading(
                ax=ax,
                viewpoint=viewpoint,
                length=orientation_length,
            )

    if show_ids:
        for viewpoint in viewpoints:
            ax.annotate(
                viewpoint.viewpoint_id,
                (
                    viewpoint.x,
                    viewpoint.y,
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                zorder=6,
            )

    ax.set_title(title)
    ax.set_aspect("equal")

    ax.ticklabel_format(
        style="plain",
        useOffset=False,
        axis="both",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax


def plot_observation_sequence(
    configuration: "ViewpointConfiguration",
    *,
    domain: "AnalysisDomain | None" = None,
    ax: Axes | None = None,
    title: str = "Observation sequence",
    show_event_ids: bool = False,
    show_viewpoint_ids: bool = False,
) -> tuple[Figure, Axes]:
    """Plot ordered ObservationEvents as a spatial sequence.

    Repeated ObservationEvents referencing the same Viewpoint are retained
    and therefore appear as repeated sequence positions.

    Sequence order follows the order provided by ViewpointConfiguration.
    """

    _validate_configuration(
        configuration
    )

    events = list(
        configuration.observation_events
    )

    if not events:
        raise ValueError(
            "ViewpointConfiguration contains no ObservationEvents."
        )

    fig, ax = _prepare_axes(ax)

    if domain is not None:
        _plot_domain_geometry(
            ax,
            domain,
        )

    points: list[
        tuple[float, float]
    ] = []

    event_viewpoints = []

    for event in events:

        viewpoint = (
            configuration.get_viewpoint(
                event.viewpoint_id
            )
        )

        points.append(
            (
                viewpoint.x,
                viewpoint.y,
            )
        )

        event_viewpoints.append(
            viewpoint
        )

    xs = np.array(
        [
            point[0]
            for point in points
        ],
        dtype=float,
    )

    ys = np.array(
        [
            point[1]
            for point in points
        ],
        dtype=float,
    )

    # Connect events in their configuration order.
    ax.plot(
        xs,
        ys,
        linewidth=1.2,
        alpha=0.7,
        zorder=3,
    )

    # Event markers.
    ax.scatter(
        xs,
        ys,
        s=45,
        c=np.arange(
            len(events)
        ),
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    # Show sequence index explicitly.
    for index, (
        event,
        viewpoint,
    ) in enumerate(
        zip(
            events,
            event_viewpoints,
        )
    ):
        label = str(index + 1)

        if show_event_ids:
            label += (
                f"\n{event.event_id}"
            )

        if show_viewpoint_ids:
            label += (
                f"\n{viewpoint.viewpoint_id}"
            )

        ax.annotate(
            label,
            (
                viewpoint.x,
                viewpoint.y,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            zorder=7,
        )

    ax.set_title(title)
    ax.set_aspect("equal")

    ax.ticklabel_format(
        style="plain",
        useOffset=False,
        axis="both",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax


def plot_survey_overview(
    configuration: "ViewpointConfiguration",
    *,
    domain: "AnalysisDomain | None" = None,
    ax: Axes | None = None,
    title: str = "Survey configuration",
    show_ids: bool = False,
    show_orientation: bool = True,
    show_sequence: bool = True,
) -> tuple[Figure, Axes]:
    """Plot a combined survey overview.

    This provides a convenient high-level visualization containing:

    - AnalysisDomain, when supplied;
    - Viewpoint locations;
    - heading/orientation;
    - ObservationEvent trajectory, when available.
    """

    _validate_configuration(
        configuration
    )

    fig, ax = plot_viewpoints(
        configuration,
        domain=domain,
        ax=ax,
        title=title,
        show_ids=show_ids,
        show_orientation=show_orientation,
    )

    if (
        show_sequence
        and configuration.observation_events
    ):
        events = list(
            configuration.observation_events
        )

        xs = []
        ys = []

        for event in events:
            viewpoint = (
                configuration.get_viewpoint(
                    event.viewpoint_id
                )
            )

            xs.append(
                viewpoint.x
            )
            ys.append(
                viewpoint.y
            )

        ax.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=1.1,
            alpha=0.6,
            zorder=2,
            label="Observation sequence",
        )

    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            loc="upper right"
        )

    return fig, ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plot_heading(
    *,
    ax: Axes,
    viewpoint: "Viewpoint",
    length: float,
) -> None:
    """Plot one compass-heading arrow."""

    heading_rad = np.deg2rad(
        viewpoint.heading_deg
    )

    # Compass convention:
    # heading 0   -> north
    # heading 90  -> east
    dx = (
        np.sin(heading_rad)
        * length
    )

    dy = (
        np.cos(heading_rad)
        * length
    )

    ax.arrow(
        viewpoint.x,
        viewpoint.y,
        dx,
        dy,
        width=0.0,
        head_width=length * 0.12,
        head_length=length * 0.16,
        length_includes_head=True,
        color="black",
        alpha=0.8,
        zorder=4,
    )


def _plot_domain_geometry(
    ax: Axes,
    domain: "AnalysisDomain",
) -> None:
    """Draw AnalysisDomain polygon boundaries."""

    from rivelero.core.domain import (
        AnalysisDomain,
    )

    if not isinstance(
        domain,
        AnalysisDomain,
    ):
        raise TypeError(
            "domain must be an AnalysisDomain or None."
        )

    geometry = (
        domain.geometry
    )

    if geometry.geom_type == "Polygon":
        polygons = [
            geometry
        ]

    elif geometry.geom_type == "MultiPolygon":
        polygons = list(
            geometry.geoms
        )

    else:
        raise ValueError(
            "AnalysisDomain geometry must be Polygon or MultiPolygon "
            "for survey visualization."
        )

    for polygon in polygons:

        coordinates = np.asarray(
            polygon.exterior.coords
        )

        patch = MplPolygon(
            coordinates,
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            linestyle="-",
            zorder=1,
        )

        ax.add_patch(
            patch
        )


def _default_orientation_length(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    domain: "AnalysisDomain | None",
) -> float:
    """Estimate a useful heading-arrow length."""

    if domain is not None:
        minx, miny, maxx, maxy = (
            domain.geometry.bounds
        )

        span = max(
            maxx - minx,
            maxy - miny,
        )

    elif len(xs) > 1:
        span = max(
            float(
                np.ptp(xs)
            ),
            float(
                np.ptp(ys)
            ),
        )

    else:
        span = 100.0

    if span <= 0:
        span = 100.0

    return span * 0.05


def _prepare_axes(
    ax: Axes | None,
) -> tuple[Figure, Axes]:

    if ax is None:
        fig, new_ax = plt.subplots(
            figsize=(8, 7)
        )

        return fig, new_ax

    if not isinstance(
        ax,
        Axes,
    ):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes or None."
        )

    return ax.figure, ax


def _validate_configuration(
    configuration: "ViewpointConfiguration",
) -> None:

    from rivelero.core.configuration import (
        ViewpointConfiguration,
    )

    if not isinstance(
        configuration,
        ViewpointConfiguration,
    ):
        raise TypeError(
            "configuration must be a ViewpointConfiguration."
        )