from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable

from rasterio.crs import CRS

from rivelero.core.configuration import ViewpointConfiguration


@dataclass(slots=True)
class SurveySpatialQC:
    """Data-only survey quality-control summary for a ViewpointConfiguration.

    The class intentionally reports evidence without interpreting whether a
    point is scientifically invalid. It can therefore identify missing metadata,
    repeated locations, and unusual positions while keeping the language cautious
    and deterministic.
    """

    total_viewpoints: int = 0
    unique_coordinate_locations: int = 0
    repeated_location_count: int = 0
    repeated_coordinate_locations: set[tuple[float, float]] = field(default_factory=set)
    bounds: tuple[float, float, float, float] | None = None
    crs: CRS | None = None
    has_consistent_crs: bool = True
    missing_heading_count: int = 0
    missing_fov_count: int = 0
    missing_observer_height_count: int = 0
    potential_outlier_viewpoints: tuple[str, ...] = ()
    viewpoint_by_id: dict[str, object] = field(default_factory=dict)
    coordinate_lookup: dict[tuple[float, float], list[str]] = field(default_factory=dict)

    @classmethod
    def from_configuration(
        cls,
        configuration: ViewpointConfiguration | None,
    ) -> "SurveySpatialQC":
        if configuration is None:
            return cls()

        viewpoints = list(configuration.viewpoints)

        if not viewpoints:
            return cls(
                total_viewpoints=0,
                unique_coordinate_locations=0,
                repeated_location_count=0,
                repeated_coordinate_locations=set(),
                bounds=None,
                crs=None,
                has_consistent_crs=True,
                viewpoint_by_id={},
                coordinate_lookup={},
            )

        crs_values = {
            getattr(viewpoint, "crs", None)
            for viewpoint in viewpoints
        }
        crs_values = {value for value in crs_values if value is not None}
        crs = None
        if len(crs_values) == 1:
            crs = next(iter(crs_values))
        has_consistent_crs = len(crs_values) <= 1

        coordinate_lookup: dict[tuple[float, float], list[str]] = defaultdict(list)
        viewpoint_by_id: dict[str, object] = {}
        xs: list[float] = []
        ys: list[float] = []
        missing_heading = 0
        missing_fov = 0
        missing_height = 0

        for viewpoint in viewpoints:
            viewpoint_by_id[viewpoint.viewpoint_id] = viewpoint
            coordinate_lookup[(float(viewpoint.x), float(viewpoint.y))].append(
                viewpoint.viewpoint_id
            )
            xs.append(float(viewpoint.x))
            ys.append(float(viewpoint.y))

            if viewpoint.heading_deg is None:
                missing_heading += 1
            if viewpoint.horizontal_fov_deg is None:
                missing_fov += 1
            if viewpoint.observer_height_m is None:
                missing_height += 1

        coordinate_choices = {
            key: sorted(values)
            for key, values in coordinate_lookup.items()
        }

        repeated_coordinate_locations = {
            key for key, values in coordinate_choices.items() if len(values) > 1
        }

        bounds = (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

        unique_coordinate_locations = len(coordinate_choices)
        repeated_location_count = len(repeated_coordinate_locations)

        potential_outliers = cls._potential_outlier_viewpoints(viewpoints)

        return cls(
            total_viewpoints=len(viewpoints),
            unique_coordinate_locations=unique_coordinate_locations,
            repeated_location_count=repeated_location_count,
            repeated_coordinate_locations=repeated_coordinate_locations,
            bounds=bounds,
            crs=crs,
            has_consistent_crs=has_consistent_crs,
            missing_heading_count=missing_heading,
            missing_fov_count=missing_fov,
            missing_observer_height_count=missing_height,
            potential_outlier_viewpoints=potential_outliers,
            viewpoint_by_id=viewpoint_by_id,
            coordinate_lookup={key: values for key, values in coordinate_choices.items()},
        )

    @staticmethod
    def _potential_outlier_viewpoints(viewpoints: Iterable[object]) -> tuple[str, ...]:
        views = list(viewpoints)
        if len(views) < 5:
            return ()

        xs = [float(viewpoint.x) for viewpoint in views]
        ys = [float(viewpoint.y) for viewpoint in views]

        x_median = median(xs)
        y_median = median(ys)
        x_mad = median([abs(value - x_median) for value in xs])
        y_mad = median([abs(value - y_median) for value in ys])

        x_threshold = 6.0 * max(x_mad, 1e-9)
        y_threshold = 6.0 * max(y_mad, 1e-9)

        outliers: list[str] = []
        for viewpoint in views:
            if abs(float(viewpoint.x) - x_median) > x_threshold:
                outliers.append(viewpoint.viewpoint_id)
                continue
            if abs(float(viewpoint.y) - y_median) > y_threshold:
                outliers.append(viewpoint.viewpoint_id)

        return tuple(sorted(set(outliers)))

    def lookup_viewpoint_id(self, viewpoint_id: str | None) -> str | None:
        if viewpoint_id is None:
            return None
        return viewpoint_id if viewpoint_id in self.viewpoint_by_id else None

    def lookup_viewpoints_at_point(self, x: float, y: float) -> list[str]:
        return list(self.coordinate_lookup.get((float(x), float(y)), ()))

    def lookup_viewpoint_by_point(self, x: float, y: float) -> str | None:
        matches = self.lookup_viewpoints_at_point(x, y)
        if not matches:
            return None
        return matches[0]

    @property
    def repeated_location_ids(self) -> tuple[tuple[float, float], ...]:
        return tuple(sorted(self.repeated_coordinate_locations))
