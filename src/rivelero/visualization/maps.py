"""Generic spatial plotting utilities for Rivelero.

This module contains reusable low-level map visualization helpers shared by
Rivelero's scientific visualization modules.

It deliberately contains no scientific concepts such as visibility,
observability, exposure, blind spots, redundancy, or survey quality.

Higher-level modules (visualization.observability, visualization.layers
and the exported figures) use these helpers rather than implementing
duplicate spatial plotting logic.

The functions operate on ordinary Matplotlib Axes and Rivelero spatial
metadata and do not depend on the GUI.
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.crs import CRS


# ---------------------------------------------------------------------------
# Axes creation
# ---------------------------------------------------------------------------


def prepare_axes(
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] = (8.0, 7.0),
) -> tuple[Figure, Axes]:
    """Return a Matplotlib Figure/Axes pair.

    Parameters
    ----------
    ax
        Existing Matplotlib Axes. If None, a new Figure and Axes are
        created.

    figsize
        Figure size used only when creating a new Figure.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes suitable for spatial plotting.
    """

    if ax is None:
        fig, new_ax = plt.subplots(
            figsize=figsize,
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


# ---------------------------------------------------------------------------
# Raster extent
# ---------------------------------------------------------------------------


def raster_extent(
    *,
    shape: tuple[int, int],
    transform: Affine,
) -> tuple[float, float, float, float]:
    """Return Matplotlib extent for a raster grid.

    Parameters
    ----------
    shape
        Raster shape as ``(rows, columns)``.

    transform
        Affine transformation describing the raster grid.

    Returns
    -------
    tuple
        ``(left, right, bottom, top)`` suitable for ``imshow``.

    Notes
    -----
    The four outer raster corners are evaluated explicitly rather than
    assuming a north-up grid.
    """

    _validate_shape(
        shape
    )

    if not isinstance(
        transform,
        Affine,
    ):
        raise TypeError(
            "transform must be an affine.Affine."
        )

    rows, cols = shape

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


def project_viewpoints(
    viewpoints: Iterable[Any],
    target_crs: CRS,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return Viewpoint IDs and coordinates transformed for display.

    Coordinates are transformed into ``target_crs`` for display only; the
    canonical Viewpoints are never modified. Viewpoints with missing or
    non-finite coordinates are skipped.
    """

    ids: list[str] = []
    xs: list[float] = []
    ys: list[float] = []

    transformers: dict[str, Transformer] = {}

    for viewpoint in viewpoints:
        try:
            viewpoint_id = str(viewpoint.viewpoint_id)
            x = float(viewpoint.x)
            y = float(viewpoint.y)
            source_crs = CRS.from_user_input(viewpoint.crs)
        except (AttributeError, TypeError, ValueError):
            continue

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        if source_crs != target_crs:
            key = source_crs.to_string()
            transformer = transformers.get(key)

            if transformer is None:
                transformer = Transformer.from_crs(
                    PyprojCRS.from_user_input(source_crs.to_string()),
                    PyprojCRS.from_user_input(target_crs.to_string()),
                    always_xy=True,
                )
                transformers[key] = transformer

            x, y = transformer.transform(x, y)

        ids.append(viewpoint_id)
        xs.append(float(x))
        ys.append(float(y))

    return ids, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


# ---------------------------------------------------------------------------
# Spatial axes
# ---------------------------------------------------------------------------


def format_spatial_axes(
    ax: Axes,
    *,
    crs: CRS | Any | None = None,
    equal_aspect: bool = True,
) -> None:
    """Apply common spatial-axis formatting.

    Parameters
    ----------
    ax
        Matplotlib Axes.

    crs
        Optional raster/vector CRS.

    equal_aspect
        Whether to use equal spatial scaling on X and Y axes.
    """

    if not isinstance(
        ax,
        Axes,
    ):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes."
        )

    if equal_aspect:
        ax.set_aspect(
            "equal"
        )

    if (
        crs is not None
        and not isinstance(crs, CRS)
    ):
        crs = CRS.from_user_input(
            crs
        )

    if (
        crs is not None
        and crs.is_geographic
    ):
        ax.set_xlabel(
            "Longitude"
        )
        ax.set_ylabel(
            "Latitude"
        )

    elif (
        crs is not None
        and crs.is_projected
    ):
        unit = crs_linear_unit_name(
            crs
        )

        if unit:
            ax.set_xlabel(
                f"X ({unit})"
            )
            ax.set_ylabel(
                f"Y ({unit})"
            )

        else:
            ax.set_xlabel(
                "X"
            )
            ax.set_ylabel(
                "Y"
            )

    else:
        ax.set_xlabel(
            "X"
        )
        ax.set_ylabel(
            "Y"
        )

    # Avoid scientific-offset notation for ordinary projected coordinates.
    try:
        ax.ticklabel_format(
            style="plain",
            useOffset=False,
            axis="both",
        )
    except AttributeError:
        # Some custom Matplotlib axes may not expose ScalarFormatter.
        pass


def crs_linear_unit_name(
    crs: CRS | Any,
) -> str | None:
    """Return a readable projected linear-unit name."""

    if not isinstance(
        crs,
        CRS,
    ):
        crs = CRS.from_user_input(
            crs
        )

    try:
        units = crs.linear_units
    except AttributeError:
        return None

    if units is None:
        return None

    value = str(
        units
    ).strip()

    if not value:
        return None

    if value.lower() in {
        "metre",
        "meter",
        "metres",
        "meters",
        "m",
    }:
        return "m"

    return value


# ---------------------------------------------------------------------------
# Domain geometry
# ---------------------------------------------------------------------------


def plot_domain_geometry(
    ax: Axes,
    geometry: Any,
    *,
    edgecolor: str = "black",
    linewidth: float = 1.2,
    linestyle: str = "-",
    fill: bool = False,
    facecolor: str = "none",
    alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    """Plot Polygon or MultiPolygon geometry on spatial axes.

    Parameters
    ----------
    ax
        Matplotlib Axes.

    geometry
        Shapely Polygon or MultiPolygon.

    Notes
    -----
    Both exterior boundaries and interior rings are drawn.
    """

    if not isinstance(
        ax,
        Axes,
    ):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes."
        )

    if geometry is None:
        raise ValueError(
            "geometry cannot be None."
        )

    geom_type = getattr(
        geometry,
        "geom_type",
        None,
    )

    if geom_type == "Polygon":
        polygons = [
            geometry
        ]

    elif geom_type == "MultiPolygon":
        polygons = list(
            geometry.geoms
        )

    else:
        raise ValueError(
            "geometry must be a Polygon or MultiPolygon."
        )

    for polygon in polygons:

        exterior = np.asarray(
            polygon.exterior.coords
        )

        patch = MplPolygon(
            exterior,
            closed=True,
            fill=fill,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )

        ax.add_patch(
            patch
        )

        # Draw polygon holes separately.
        for interior in polygon.interiors:

            coordinates = np.asarray(
                interior.coords
            )

            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder + 1,
            )


# ---------------------------------------------------------------------------
# Point / observer plotting
# ---------------------------------------------------------------------------


def plot_point(
    ax: Axes,
    x: float,
    y: float,
    *,
    marker: str = "o",
    size: float = 45.0,
    facecolor: str = "black",
    edgecolor: str = "white",
    linewidth: float = 0.8,
    zorder: int = 10,
    label: str | None = None,
) -> None:
    """Plot one spatial point."""

    if not isinstance(
        ax,
        Axes,
    ):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes."
        )

    x = _finite_float(
        "x",
        x,
    )

    y = _finite_float(
        "y",
        y,
    )

    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=size,
        c=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=zorder,
        label=label,
    )


def plot_points(
    ax: Axes,
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    marker: str = "o",
    size: float = 38.0,
    facecolor: str = "black",
    edgecolor: str = "white",
    linewidth: float = 0.7,
    zorder: int = 5,
    label: str | None = None,
) -> None:
    """Plot multiple spatial points."""

    xs = np.asarray(
        x,
        dtype=float,
    )

    ys = np.asarray(
        y,
        dtype=float,
    )

    if (
        xs.ndim != 1
        or ys.ndim != 1
    ):
        raise ValueError(
            "x and y must be one-dimensional."
        )

    if xs.shape != ys.shape:
        raise ValueError(
            "x and y must contain the same number of coordinates."
        )

    if not np.all(
        np.isfinite(xs)
    ):
        raise ValueError(
            "x contains non-finite coordinates."
        )

    if not np.all(
        np.isfinite(ys)
    ):
        raise ValueError(
            "y contains non-finite coordinates."
        )

    ax.scatter(
        xs,
        ys,
        marker=marker,
        s=size,
        c=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=zorder,
        label=label,
    )


# ---------------------------------------------------------------------------
# Heading utilities
# ---------------------------------------------------------------------------


def heading_components(
    heading_deg: float,
    length: float,
) -> tuple[float, float]:
    """Convert compass heading to X/Y displacement.

    Rivelero uses compass convention:

        0 degrees   = north
        90 degrees  = east
        180 degrees = south
        270 degrees = west
    """

    heading = (
        _finite_float(
            "heading_deg",
            heading_deg,
        )
        % 360.0
    )

    length = _finite_float(
        "length",
        length,
    )

    if length < 0:
        raise ValueError(
            "length must be non-negative."
        )

    radians = np.deg2rad(
        heading
    )

    dx = (
        np.sin(radians)
        * length
    )

    dy = (
        np.cos(radians)
        * length
    )

    return (
        float(dx),
        float(dy),
    )


def plot_heading_arrow(
    ax: Axes,
    *,
    x: float,
    y: float,
    heading_deg: float,
    length: float,
    color: str = "black",
    alpha: float = 0.8,
    zorder: int = 6,
) -> None:
    """Plot one compass-heading arrow."""

    dx, dy = heading_components(
        heading_deg,
        length,
    )

    ax.arrow(
        x,
        y,
        dx,
        dy,
        width=0.0,
        head_width=length * 0.12,
        head_length=length * 0.16,
        length_includes_head=True,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )


# ---------------------------------------------------------------------------
# Generic raster plotting
# ---------------------------------------------------------------------------


def plot_raster(
    ax: Axes,
    data: np.ndarray,
    *,
    transform: Affine,
    cmap: str | Any = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "nearest",
    alpha: float = 1.0,
    zorder: int = 1,
):
    """Plot a two-dimensional raster using its Affine transform.

    Returns the Matplotlib image artist so callers may construct colorbars.
    """

    if not isinstance(
        data,
        np.ndarray,
    ):
        raise TypeError(
            "data must be a numpy.ndarray."
        )

    if data.ndim != 2:
        raise ValueError(
            "data must be two-dimensional."
        )

    extent = raster_extent(
        shape=data.shape,
        transform=transform,
    )

    return ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        alpha=alpha,
        zorder=zorder,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_shape(
    shape: tuple[int, int],
) -> None:
    """Validate a two-dimensional raster shape."""

    if (
        not isinstance(
            shape,
            tuple,
        )
        or len(shape) != 2
    ):
        raise TypeError(
            "shape must be a tuple: (rows, columns)."
        )

    if not all(
        isinstance(
            value,
            (int, np.integer),
        )
        for value in shape
    ):
        raise TypeError(
            "Raster dimensions must be integers."
        )

    if (
        shape[0] <= 0
        or shape[1] <= 0
    ):
        raise ValueError(
            "Raster dimensions must be greater than zero."
        )


def _finite_float(
    name: str,
    value: float,
) -> float:
    """Return a validated finite float."""

    if not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    result = float(
        value
    )

    if not np.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result