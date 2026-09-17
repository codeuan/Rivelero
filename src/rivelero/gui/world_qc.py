"""Spatial quality-control services for the Rivelero World workflow.

World QC answers questions such as:

- Are survey and terrain CRSs compatible?
- How many Viewpoints fall inside the terrain?
- How many fall inside the AnalysisDomain?
- Does the AnalysisDomain overlap the terrain?
- Does the domain extend outside the available elevation grid?

The module reports evidence and warnings. It does not silently reproject or
alter canonical scientific objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.crs import CRS
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from rivelero.core.domain import (
    AnalysisDomain,
    AnalysisGrid,
)
from rivelero.gui.domain_service import (
    grid_extent_geometry,
)


class QCSeverity(str, Enum):
    """Severity of one World QC finding."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class WorldQCIssue:
    """One human-readable World QC finding."""

    code: str

    severity: QCSeverity

    title: str

    message: str

    viewpoint_ids: tuple[
        str,
        ...,
    ] = ()


@dataclass(slots=True)
class WorldQCReport:
    """Structured spatial compatibility report."""

    survey_crs_values: tuple[
        str,
        ...,
    ]

    terrain_crs: str

    total_viewpoints: int

    viewpoints_inside_terrain: int

    viewpoints_outside_terrain: int

    viewpoints_inside_domain: int | None = None

    viewpoints_outside_domain: int | None = None

    domain_overlaps_terrain: bool | None = None

    domain_fully_inside_terrain: bool | None = None

    issues: list[
        WorldQCIssue
    ] = field(
        default_factory=list
    )

    @property
    def has_errors(self) -> bool:
        return any(
            issue.severity
            == QCSeverity.ERROR
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        return any(
            issue.severity
            == QCSeverity.WARNING
            for issue in self.issues
        )

    @property
    def compatible(self) -> bool:
        """Whether no blocking spatial error was detected."""

        return not self.has_errors


@dataclass(frozen=True, slots=True)
class ResolvedViewpointPosition:
    """One Viewpoint transformed into terrain/grid coordinates."""

    viewpoint_id: str

    x: float
    y: float


def resolve_viewpoints_to_grid(
    viewpoints: Iterable[Any],
    grid: AnalysisGrid,
) -> tuple[
    ResolvedViewpointPosition,
    ...,
]:
    """Explicitly transform Viewpoints into AnalysisGrid coordinates."""

    if not isinstance(
        grid,
        AnalysisGrid,
    ):
        raise TypeError(
            "grid must be an AnalysisGrid."
        )

    items = list(
        viewpoints
    )

    transformers: dict[
        str,
        Transformer,
    ] = {}

    resolved = []

    for index, viewpoint in enumerate(
        items
    ):

        viewpoint_id = str(
            getattr(
                viewpoint,
                "viewpoint_id",
                f"viewpoint_{index}",
            )
        )

        try:
            x = float(
                viewpoint.x
            )
            y = float(
                viewpoint.y
            )
            source_crs = CRS.from_user_input(
                viewpoint.crs
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Viewpoint {viewpoint_id!r} does not provide "
                "valid x, y and crs."
            ) from exc

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            raise ValueError(
                f"Viewpoint {viewpoint_id!r} has non-finite coordinates."
            )

        if source_crs != grid.crs:

            key = source_crs.to_string()

            transformer = transformers.get(
                key
            )

            if transformer is None:

                transformer = Transformer.from_crs(
                    PyprojCRS.from_user_input(
                        source_crs.to_string()
                    ),
                    PyprojCRS.from_user_input(
                        grid.crs.to_string()
                    ),
                    always_xy=True,
                )

                transformers[
                    key
                ] = transformer

            x, y = transformer.transform(
                x,
                y,
            )

        resolved.append(
            ResolvedViewpointPosition(
                viewpoint_id=viewpoint_id,
                x=float(x),
                y=float(y),
            )
        )

    return tuple(
        resolved
    )


def survey_crs_summary(
    viewpoints: Iterable[Any],
) -> tuple[str, ...]:
    """Return distinct canonical survey CRS representations."""

    values = set()

    for viewpoint in viewpoints:

        try:
            crs = CRS.from_user_input(
                viewpoint.crs
            )

        except (
            AttributeError,
            ValueError,
        ):
            values.add(
                "UNKNOWN"
            )
            continue

        values.add(
            crs.to_string()
        )

    return tuple(
        sorted(
            values
        )
    )


def run_world_qc(
    *,
    viewpoints: Iterable[Any],
    grid: AnalysisGrid,
    domain: AnalysisDomain | None = None,
) -> WorldQCReport:
    """Evaluate Survey ↔ terrain ↔ AnalysisDomain compatibility."""

    if not isinstance(
        grid,
        AnalysisGrid,
    ):
        raise TypeError(
            "grid must be an AnalysisGrid."
        )

    items = list(
        viewpoints
    )

    crs_values = survey_crs_summary(
        items
    )

    report = WorldQCReport(
        survey_crs_values=crs_values,
        terrain_crs=grid.crs.to_string(),
        total_viewpoints=len(items),
        viewpoints_inside_terrain=0,
        viewpoints_outside_terrain=0,
    )

    if not items:

        report.issues.append(
            WorldQCIssue(
                code="survey_empty",
                severity=QCSeverity.WARNING,
                title="Survey contains no Viewpoints",
                message=(
                    "Terrain can be defined, but survey/terrain "
                    "compatibility cannot yet be evaluated."
                ),
            )
        )

        _evaluate_domain(
            report,
            domain=domain,
            grid=grid,
        )

        return report

    if "UNKNOWN" in crs_values:

        report.issues.append(
            WorldQCIssue(
                code="unknown_viewpoint_crs",
                severity=QCSeverity.ERROR,
                title="Unknown Viewpoint CRS",
                message=(
                    "At least one Viewpoint has no usable CRS, so "
                    "Rivelero cannot safely compare it with the terrain."
                ),
            )
        )

        return report

    if len(
        crs_values
    ) > 1:

        report.issues.append(
            WorldQCIssue(
                code="mixed_survey_crs",
                severity=QCSeverity.WARNING,
                title="Survey contains multiple CRSs",
                message=(
                    "Viewpoints use multiple coordinate reference systems. "
                    "Rivelero can transform them explicitly for QC, but a "
                    "single canonical survey CRS is preferable."
                ),
            )
        )

    terrain_crs = (
        grid.crs.to_string()
    )

    if (
        len(crs_values) == 1
        and crs_values[0]
        != terrain_crs
    ):

        report.issues.append(
            WorldQCIssue(
                code="survey_terrain_crs_differ",
                severity=QCSeverity.INFO,
                title="Survey and terrain use different CRSs",
                message=(
                    f"Survey uses {crs_values[0]} while terrain uses "
                    f"{terrain_crs}. QC positions are explicitly transformed "
                    "to the terrain CRS; the canonical survey is not modified."
                ),
            )
        )

    try:
        resolved = resolve_viewpoints_to_grid(
            items,
            grid,
        )

    except ValueError as exc:

        report.issues.append(
            WorldQCIssue(
                code="viewpoint_transform_failed",
                severity=QCSeverity.ERROR,
                title="Viewpoint coordinates could not be resolved",
                message=str(
                    exc
                ),
            )
        )

        return report

    terrain_geometry = (
        grid_extent_geometry(
            grid
        )
    )

    inside_terrain = []
    outside_terrain = []

    for position in resolved:

        point = Point(
            position.x,
            position.y,
        )

        # covers() includes points on the raster boundary.
        if terrain_geometry.covers(
            point
        ):
            inside_terrain.append(
                position.viewpoint_id
            )
        else:
            outside_terrain.append(
                position.viewpoint_id
            )

    report.viewpoints_inside_terrain = (
        len(
            inside_terrain
        )
    )

    report.viewpoints_outside_terrain = (
        len(
            outside_terrain
        )
    )

    if outside_terrain:

        report.issues.append(
            WorldQCIssue(
                code="viewpoints_outside_terrain",
                severity=QCSeverity.WARNING,
                title="Viewpoints outside terrain extent",
                message=(
                    f"{len(outside_terrain):,} of "
                    f"{len(resolved):,} Viewpoints lie outside the "
                    "available elevation grid."
                ),
                viewpoint_ids=tuple(
                    outside_terrain
                ),
            )
        )

    else:

        report.issues.append(
            WorldQCIssue(
                code="all_viewpoints_inside_terrain",
                severity=QCSeverity.SUCCESS,
                title="Survey covered by terrain",
                message=(
                    f"All {len(resolved):,} Viewpoints fall within "
                    "the elevation-grid extent."
                ),
            )
        )

    if domain is not None:

        _evaluate_domain(
            report,
            domain=domain,
            grid=grid,
        )

        if (
            domain.crs
            != grid.crs
        ):

            # Canonical AnalysisDomain already normally prevents this via
            # grid/domain validation, but keep the report defensive.
            report.issues.append(
                WorldQCIssue(
                    code="domain_grid_crs_mismatch",
                    severity=QCSeverity.ERROR,
                    title="Domain/grid CRS mismatch",
                    message=(
                        "AnalysisDomain CRS does not match its terrain grid."
                    ),
                )
            )

        else:

            inside_domain = []
            outside_domain = []

            for position in resolved:

                point = Point(
                    position.x,
                    position.y,
                )

                if domain.geometry.covers(
                    point
                ):
                    inside_domain.append(
                        position.viewpoint_id
                    )
                else:
                    outside_domain.append(
                        position.viewpoint_id
                    )

            report.viewpoints_inside_domain = (
                len(
                    inside_domain
                )
            )

            report.viewpoints_outside_domain = (
                len(
                    outside_domain
                )
            )

            if outside_domain:

                report.issues.append(
                    WorldQCIssue(
                        code="viewpoints_outside_domain",
                        severity=QCSeverity.INFO,
                        title="Viewpoints outside analysis area",
                        message=(
                            f"{len(outside_domain):,} Viewpoints lie "
                            "outside the current AnalysisDomain. This may "
                            "be intentional because observers can exist "
                            "outside the area being evaluated."
                        ),
                        viewpoint_ids=tuple(
                            outside_domain
                        ),
                    )
                )

    return report


def _evaluate_domain(
    report: WorldQCReport,
    *,
    domain: AnalysisDomain | None,
    grid: AnalysisGrid,
) -> None:
    """Evaluate AnalysisDomain against terrain extent."""

    if domain is None:
        return

    terrain = grid_extent_geometry(
        grid
    )

    geometry: BaseGeometry = (
        domain.geometry
    )

    report.domain_overlaps_terrain = (
        terrain.intersects(
            geometry
        )
    )

    report.domain_fully_inside_terrain = (
        terrain.covers(
            geometry
        )
    )

    if not report.domain_overlaps_terrain:

        report.issues.append(
            WorldQCIssue(
                code="domain_outside_terrain",
                severity=QCSeverity.ERROR,
                title="Analysis area does not overlap terrain",
                message=(
                    "The current AnalysisDomain does not overlap the "
                    "available elevation grid."
                ),
            )
        )

    elif not report.domain_fully_inside_terrain:

        report.issues.append(
            WorldQCIssue(
                code="domain_partly_outside_terrain",
                severity=QCSeverity.WARNING,
                title="Analysis area extends beyond terrain",
                message=(
                    "Part of the AnalysisDomain lies outside the available "
                    "elevation grid."
                ),
            )
        )