"""OpenStreetMap background: tile geometry, mosaicking and reprojection.

No test touches the network; tiles come from an in-memory fake fetcher.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from matplotlib import image as mpl_image

from rivelero.visualization.basemap import (
    MAX_TILES,
    TILE_SIZE,
    BasemapUnavailableError,
    CachedTileFetcher,
    choose_zoom,
    render_basemap,
    tile_range,
    tile_resolution,
)

# A 4 km square around Oxford, in UTM zone 30N.
OXFORD_UTM = (618000.0, 622000.0, 5733000.0, 5737000.0)


def solid_tile(rgb=(200, 100, 50)) -> bytes:
    buffer = io.BytesIO()
    array = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    array[:] = rgb
    mpl_image.imsave(buffer, array, format="png")
    return buffer.getvalue()


class FakeFetcher:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.content = solid_tile()

    def __call__(self, z, x, y):
        self.calls.append((z, x, y))
        if self.fail:
            raise OSError("offline")
        return self.content


def test_tile_resolution_halves_per_zoom_level():
    assert tile_resolution(0) == pytest.approx(156543.03392804097)
    assert tile_resolution(1) == pytest.approx(tile_resolution(0) / 2)


def test_tile_range_of_whole_world():
    half = 20037508.342789244
    assert tile_range((-half, -half, half, half), 0) == (0, 0, 0, 0)
    assert tile_range((-half, -half, half, half), 2) == (0, 3, 0, 3)
    # North-east quadrant is x = 1, y = 0 at zoom 1.
    assert tile_range((1.0, 1.0, half, half), 1) == (1, 1, 0, 0)


def test_choose_zoom_matches_resolution_and_caps_tiles():
    bounds = (-50000.0, 6700000.0, 50000.0, 6800000.0)
    zoom = choose_zoom(bounds, 800)
    assert tile_resolution(zoom) <= 100000.0 / 800 < tile_resolution(zoom - 1)
    x0, x1, y0, y1 = tile_range(bounds, choose_zoom(bounds, 100000))
    assert (x1 - x0 + 1) * (y1 - y0 + 1) <= MAX_TILES


def test_render_basemap_returns_rgba_in_map_crs():
    fetcher = FakeFetcher()
    image = render_basemap(OXFORD_UTM, "EPSG:32630", (300, 200), fetch_tile=fetcher)

    assert image.rgba.shape == (200, 300, 4)
    assert image.rgba.dtype == np.uint8
    assert image.extent == OXFORD_UTM
    assert image.crs.to_epsg() == 32630
    assert fetcher.calls and len(fetcher.calls) <= MAX_TILES
    # Fully covered by opaque tiles of the fake colour.
    centre = image.rgba[100, 150]
    np.testing.assert_allclose(centre, (200, 100, 50, 255), atol=2)


def test_render_basemap_supports_geographic_crs():
    image = render_basemap((-1.3, -1.2, 51.7, 51.8), "EPSG:4326", (64, 64),
                           fetch_tile=FakeFetcher())
    assert image.rgba[..., 3].min() == 255


def test_render_basemap_reports_unavailable_tiles():
    with pytest.raises(BasemapUnavailableError, match="offline"):
        render_basemap(OXFORD_UTM, "EPSG:32630", (100, 100), fetch_tile=FakeFetcher(fail=True))


def test_render_basemap_rejects_invalid_extent():
    with pytest.raises(ValueError):
        render_basemap((1.0, 0.0, 0.0, 1.0), "EPSG:32630", (10, 10), fetch_tile=FakeFetcher())


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return FakeResponse(b"tile-bytes")


def test_cached_fetcher_reuses_disk_cache(tmp_path):
    session = FakeSession()
    fetcher = CachedTileFetcher(tmp_path, session=session)

    assert fetcher(3, 4, 5) == b"tile-bytes"
    assert fetcher(3, 4, 5) == b"tile-bytes"

    assert session.urls == ["https://tile.openstreetmap.org/3/4/5.png"]
    assert (tmp_path / "3" / "4" / "5.png").read_bytes() == b"tile-bytes"
