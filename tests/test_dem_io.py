"""Tests for canonical DEM acquisition helpers."""

from dataclasses import dataclass

import pytest
from rasterio.crs import CRS

from rivelero.io.dem import (
    buffered_wgs84_bbox,
    wgs84_bbox_from_viewpoints,
)


@dataclass
class DummyViewpoint:
    x: float
    y: float
    crs: CRS


def test_buffered_wgs84_bbox():

    south, north, west, east = (
        buffered_wgs84_bbox(
            [
                (
                    12.0,
                    38.0,
                )
            ],
            buffer_m=100.0,
        )
    )

    assert south < 38.0
    assert north > 38.0

    assert west < 12.0
    assert east > 12.0


def test_viewpoint_bbox_reprojects_projected_coordinates():

    viewpoints = [
        DummyViewpoint(
            x=500000.0,
            y=4100000.0,
            crs=CRS.from_epsg(
                32633
            ),
        )
    ]

    south, north, west, east = (
        wgs84_bbox_from_viewpoints(
            viewpoints,
            buffer_m=100.0,
        )
    )

    assert -90.0 <= south < north <= 90.0
    assert -180.0 <= west < east <= 180.0


def test_empty_viewpoint_bbox_rejected():

    with pytest.raises(
        ValueError
    ):
        wgs84_bbox_from_viewpoints(
            [],
            buffer_m=100.0,
        )