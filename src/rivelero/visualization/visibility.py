"""Visualization of individual visibility results in Rivelero.

This module visualizes SingleViewpointVisibility results produced by the
Rivelero visibility engine.

It distinguishes:

- geometric visibility: cells with terrain-based 2.5D line of sight;
- effective visibility: cells remaining after configured camera/viewing
  constraints are applied;
- directional exclusion: geometrically visible cells excluded by camera
  orientation or horizontal field of view.

The module contains no viewshed or observability calculations and does not
depend on the GUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Wedge

if TYPE_CHECKING:
    from rivelero.visibility.engine import SingleViewpointVisibility


def plot_geometric_visibility(
    visibility: "SingleViewpointVisibility",
    *,
    observer_x: float | None = None,
    observer_y: float | None = None,
    ax: Axes | None = None,
    title: str = "Geometric visibility",
    show_observer: bool = True,
) -> tuple[Figure, Axes]:
    """Plot terrain-based geometric visibility for one observation unit."""

    _validate_visibility(visibility)

    fig, ax = _prepare_axes(ax)

    data = np.ma.masked_where(
        ~visibility.valid_mask,
        visibility.geometric_visibility_mask.astype(np.uint8),
    )

    cmap = ListedColormap(
        ["#eeeeee", "#4daf4a"]
    )

    ax.imshow(
        data,
        extent=_raster_extent(visibility),
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    if (
        show_observer
        and observer_x is not None
        and observer_y is not None
    ):
        _plot_observer(
            ax,
            observer_x,
            observer_y,
        )

    ax.set_title(title)
    _format_spatial_axes(ax, visibility)

    handles = [
        Patch(
            facecolor="#eeeeee",
            label="Not geometrically visible",
        ),
        Patch(
            facecolor="#4daf4a",
            label="Geometrically visible",
        ),
    ]

    if (
        show_observer
        and observer_x is not None
        and observer_y is not None
    ):
        handles.append(
            Patch(
                facecolor="black",
                label="Observer",
            )
        )

    ax.legend(
        handles=handles,
        loc="upper right",
    )

    return fig, ax


def plot_effective_visibility(
    visibility: "SingleViewpointVisibility",
    *,
    observer_x: float | None = None,
    observer_y: float | None = None,
    ax: Axes | None = None,
    title: str = "Effective visibility",
    show_observer: bool = True,
    show_view_cone: bool = True,
) -> tuple[Figure, Axes]:
    """Plot final visibility after viewing-direction constraints.

    When observer coordinates, heading, horizontal FOV, and maximum distance
    are available, the configured horizontal viewing cone is overlaid.
    """

    _validate_visibility(visibility)

    fig, ax = _prepare_axes(ax)

    data = np.ma.masked_where(
        ~visibility.valid_mask,
        visibility.visibility_mask.astype(np.uint8),
    )

    cmap = ListedColormap(
        ["#eeeeee", "#4daf4a"]
    )

    ax.imshow(
        data,
        extent=_raster_extent(visibility),
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    if (
        show_observer
        and observer_x is not None
        and observer_y is not None
    ):
        _plot_observer(
            ax,
            observer_x,
            observer_y,
        )

    if (
        show_view_cone
        and observer_x is not None
        and observer_y is not None
    ):
        _plot_view_cone(
            ax=ax,
            observer_x=observer_x,
            observer_y=observer_y,
            heading_deg=(
                visibility.resolved_parameters.heading_deg
            ),
            horizontal_fov_deg=(
                visibility.resolved_parameters.horizontal_fov_deg
            ),
            max_distance_m=(
                visibility.resolved_parameters.max_distance_m
            ),
            omnidirectional=(
                visibility.resolved_parameters.omnidirectional
            ),
        )

    ax.set_title(title)
    _format_spatial_axes(ax, visibility)

    return fig, ax


def plot_visibility_difference(
    visibility: "SingleViewpointVisibility",
    *,
    observer_x: float | None = None,
    observer_y: float | None = None,
    ax: Axes | None = None,
    title: str = "Geometric vs effective visibility",
) -> tuple[Figure, Axes]:
    """Show how directional constraints alter geometric visibility.

    States shown are:

    - not geometrically visible;
    - geometrically visible but excluded by viewing geometry;
    - effectively visible.
    """

    _validate_visibility(visibility)

    fig, ax = _prepare_axes(ax)

    geometric = visibility.geometric_visibility_mask
    effective = visibility.visibility_mask
    valid = visibility.valid_mask

    state = np.zeros(
        geometric.shape,
        dtype=np.uint8,
    )

    # 0 = not geometrically visible
    # 1 = geometric but removed by camera geometry
    # 2 = effective visibility
    removed = (
        geometric
        & ~effective
        & valid
    )

    state[removed] = 1
    state[effective & valid] = 2

    state = np.ma.masked_where(
        ~valid,
        state,
    )

    colors = (
        "#eeeeee",
        "#fdae61",
        "#4daf4a",
    )

    cmap = ListedColormap(colors)

    ax.imshow(
        state,
        extent=_raster_extent(visibility),
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=2,
    )

    if (
        observer_x is not None
        and observer_y is not None
    ):
        _plot_observer(
            ax,
            observer_x,
            observer_y,
        )

        _plot_view_cone(
            ax=ax,
            observer_x=observer_x,
            observer_y=observer_y,
            heading_deg=(
                visibility.resolved_parameters.heading_deg
            ),
            horizontal_fov_deg=(
                visibility.resolved_parameters.horizontal_fov_deg
            ),
            max_distance_m=(
                visibility.resolved_parameters.max_distance_m
            ),
            omnidirectional=(
                visibility.resolved_parameters.omnidirectional
            ),
        )

    ax.set_title(title)
    _format_spatial_axes(ax, visibility)

    handles = [
        Patch(
            facecolor=colors[0],
            label="Not geometrically visible",
        ),
        Patch(
            facecolor=colors[1],
            label="Excluded by viewing geometry",
        ),
        Patch(
            facecolor=colors[2],
            label="Effectively visible",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
    )

    return fig, ax


def plot_visibility_comparison(
    visibility: "SingleViewpointVisibility",
    *,
    observer_x: float | None = None,
    observer_y: float | None = None,
    title: str = "Single-viewpoint visibility",
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create a side-by-side geometric/effective visibility diagnostic.

    This is intended primarily for scientific validation and debugging.
    """

    _validate_visibility(visibility)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    plot_geometric_visibility(
        visibility,
        observer_x=observer_x,
        observer_y=observer_y,
        ax=axes[0],
        title="Geometric visibility",
    )

    plot_effective_visibility(
        visibility,
        observer_x=observer_x,
        observer_y=observer_y,
        ax=axes[1],
        title="Effective visibility",
    )

    fig.suptitle(title)

    return fig, (axes[0], axes[1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plot_observer(
    ax: Axes,
    x: float,
    y: float,
) -> None:
    """Plot observer location."""

    ax.scatter(
        [x],
        [y],
        marker="o",
        s=45,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )


def _plot_view_cone(
    *,
    ax: Axes,
    observer_x: float,
    observer_y: float,
    heading_deg: float | None,
    horizontal_fov_deg: float | None,
    max_distance_m: float | None,
    omnidirectional: bool,
) -> None:
    """Overlay the configured horizontal viewing sector."""

    if (
        omnidirectional
        or heading_deg is None
        or horizontal_fov_deg is None
        or max_distance_m is None
    ):
        return

    if horizontal_fov_deg >= 360.0:
        return

    # Matplotlib Wedge uses mathematical angles:
    # 0 = east, counter-clockwise positive.
    #
    # Rivelero heading:
    # 0 = north, clockwise positive.
    centre_angle = (
        90.0 - heading_deg
    )

    half_fov = (
        horizontal_fov_deg / 2.0
    )

    wedge = Wedge(
        center=(
            observer_x,
            observer_y,
        ),
        r=max_distance_m,
        theta1=centre_angle - half_fov,
        theta2=centre_angle + half_fov,
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        zorder=9,
    )

    ax.add_patch(wedge)


def _prepare_axes(
    ax: Axes | None,
) -> tuple[Figure, Axes]:

    if ax is None:
        fig, new_ax = plt.subplots(
            figsize=(8, 7)
        )
        return fig, new_ax

    if not isinstance(ax, Axes):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes or None."
        )

    return ax.figure, ax


def _raster_extent(
    visibility: "SingleViewpointVisibility",
) -> tuple[float, float, float, float]:

    rows, cols = (
        visibility.visibility_mask.shape
    )

    transform = visibility.transform

    corners = (
        transform * (0, 0),
        transform * (cols, 0),
        transform * (0, rows),
        transform * (cols, rows),
    )

    xs = [
        point[0]
        for point in corners
    ]

    ys = [
        point[1]
        for point in corners
    ]

    return (
        float(min(xs)),
        float(max(xs)),
        float(min(ys)),
        float(max(ys)),
    )


def _format_spatial_axes(
    ax: Axes,
    visibility: "SingleViewpointVisibility",
) -> None:

    ax.set_aspect("equal")

    if visibility.crs.is_projected:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax.ticklabel_format(
        style="plain",
        useOffset=False,
        axis="both",
    )


def _validate_visibility(
    visibility: "SingleViewpointVisibility",
) -> None:

    from rivelero.visibility.engine import (
        SingleViewpointVisibility,
    )

    if not isinstance(
        visibility,
        SingleViewpointVisibility,
    ):
        raise TypeError(
            "visibility must be a SingleViewpointVisibility."
        )

    shape = (
        visibility.visibility_mask.shape
    )

    if (
        visibility.geometric_visibility_mask.shape
        != shape
    ):
        raise ValueError(
            "Geometric and effective visibility shapes differ."
        )

    if visibility.valid_mask.shape != shape:
        raise ValueError(
            "valid_mask shape does not match visibility."
        )