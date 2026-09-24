"""OpenTopography download: API keys, resolution, reprojection, errors.

No test touches the network: a fake HTTP session serves a small geographic
GeoTIFF shaped like an OpenTopography response.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from rivelero.io.dem import OpenTopographyError, normalize_opentopo_api_key
from rivelero.io.opentopography import (
    OPENTOPO_DATASETS,
    bbox_area_km2,
    check_opentopo_api_key,
    default_output_crs,
    download_projected_dem,
    estimated_grid_shape,
    native_cell_size_m,
    raw_download_path,
    reproject_dem,
    utm_crs_for,
    validate_projected_output_crs,
)

KEY = "0123456789abcdef0123456789abcdef"
# 0.05° × 0.05° box in Sicily (UTM zone 33N).
BBOX = (37.00, 37.05, 14.00, 14.05)


def write_geographic_dem(path: Path, bbox=BBOX, arcsec: float = 1.0) -> Path:
    """A 1-arc-second WGS84 DEM with a west-east slope, like OpenTopography."""

    south, north, west, east = bbox
    step = arcsec / 3600.0
    width = int(round((east - west) / step))
    height = int(round((north - south) / step))
    data = np.tile(np.linspace(100.0, 200.0, width, dtype=np.float32), (height, 1))
    data[0, 0] = -32767.0
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="float32", crs="EPSG:4326", nodata=-32767.0,
        transform=from_origin(west, north, step, step),
    ) as dataset:
        dataset.write(data, 1)
    return path


@dataclass
class FakeResponse:
    status_code: int
    content: bytes
    headers: dict
    text: str = ""


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, dict(params or {})))
        return self.response


def tiff_response(tmp_path) -> FakeResponse:
    content = write_geographic_dem(tmp_path / "served.tif").read_bytes()
    return FakeResponse(200, content, {"Content-Type": "image/tiff"})


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pasted", [
    KEY, f"  {KEY}\n", f"OPENTOPO_API_KEY={KEY}", f'export OPENTOPO_API_KEY="{KEY}"',
    f"API_Key={KEY}", f"'{KEY}'",
])
def test_pasted_key_is_normalized(pasted):
    assert normalize_opentopo_api_key(pasted) == KEY
    check = check_opentopo_api_key(pasted)
    assert check.status == "valid" and check.usable and check.key == KEY


def test_key_format_checks():
    assert check_opentopo_api_key("").status == "empty"
    assert not check_opentopo_api_key("   ").usable
    assert check_opentopo_api_key(KEY.upper()).status == "valid"
    # Unusual but plausible keys are tried, never blocked.
    unusual = check_opentopo_api_key("demoapikeyot2022")
    assert unusual.status == "unusual" and unusual.usable
    assert check_opentopo_api_key("abc def").status == "invalid"
    assert check_opentopo_api_key("key/with?symbols!").status == "invalid"


# ---------------------------------------------------------------------------
# Datasets, area and resolution
# ---------------------------------------------------------------------------


def test_datasets_state_native_resolution_explicitly():
    for dataset in OPENTOPO_DATASETS.values():
        assert dataset.native_arcsec in (1.0, 3.0)
        assert f"{dataset.nominal_resolution_m:g} m" in dataset.native_resolution_text
        assert "arc-second" in dataset.native_resolution_text
    assert OPENTOPO_DATASETS["COP30"].native_resolution_text == "1 arc-second (≈ 30 m)"
    assert OPENTOPO_DATASETS["COP90"].native_resolution_text == "3 arc-seconds (≈ 90 m)"
    assert OPENTOPO_DATASETS["EU_DTM"].model_type == "dtm"


def test_native_cell_size_shrinks_east_west_with_latitude():
    east_west, north_south = native_cell_size_m("COP30", 0.0)
    assert north_south == pytest.approx(30.92, abs=0.01)
    assert east_west == pytest.approx(north_south)
    east_west_60, _ = native_cell_size_m("COP30", 60.0)
    assert east_west_60 == pytest.approx(north_south / 2, rel=1e-6)


def test_area_and_grid_estimates():
    area = bbox_area_km2(BBOX)
    assert 20 < area < 25  # ≈ 4.44 km × 5.57 km
    rows, columns = estimated_grid_shape(BBOX, 30.0)
    assert rows == pytest.approx(5566 / 30, abs=2)
    assert columns == pytest.approx(4445 / 30, abs=2)


def test_output_crs_defaults():
    @dataclass
    class VP:
        crs: str

    assert default_output_crs([VP("EPSG:32633")] * 2, BBOX).to_epsg() == 32633
    # Geographic or mixed survey CRSs fall back to the area's UTM zone.
    assert default_output_crs([VP("EPSG:4326")], BBOX).to_epsg() == 32633
    assert default_output_crs([VP("EPSG:32633"), VP("EPSG:32632")], BBOX).to_epsg() == 32633
    assert default_output_crs([], (-34.0, -33.9, 18.4, 18.5)).to_epsg() == 32734
    assert utm_crs_for(-3.7, 40.4).to_epsg() == 32630
    with pytest.raises(ValueError, match="projected in metres"):
        validate_projected_output_crs("EPSG:4326")
    with pytest.raises(ValueError, match="Unknown CRS"):
        validate_projected_output_crs("not a crs")


# ---------------------------------------------------------------------------
# Reprojection and download
# ---------------------------------------------------------------------------


def test_reproject_dem_gives_square_metre_cells(tmp_path):
    source = write_geographic_dem(tmp_path / "raw.tif")
    output = reproject_dem(source, tmp_path / "terrain.tif", crs="EPSG:32633", resolution_m=25.0)

    with rasterio.open(output) as dataset:
        assert dataset.crs.to_epsg() == 32633
        assert dataset.res == (25.0, 25.0)
        assert dataset.nodata == -32767.0
        data = dataset.read(1, masked=True)
        assert 100.0 <= data.min() and data.max() <= 200.0
        assert dataset.tags()["RIVELERO_RESOLUTION_M"] == "25"


def test_download_projected_dem_is_usable_by_the_engine(tmp_path):
    session = FakeSession(tiff_response(tmp_path))
    result = download_projected_dem(
        BBOX, dataset="COP30", api_key=f"OPENTOPO_API_KEY={KEY}",
        output_path=tmp_path / "out" / "terrain.tif", crs="EPSG:32633",
        resolution_m=30.0, session=session,
    )

    (url, params), = session.requests
    assert params["demtype"] == "COP30"
    assert params["API_Key"] == KEY
    assert (params["south"], params["north"], params["west"], params["east"]) == BBOX

    assert result.raw_path == raw_download_path(tmp_path / "out" / "terrain.tif", "COP30")
    assert result.raw_path.name == "terrain_COP30_wgs84.tif"
    assert result.raw_cell_size_deg[0] == pytest.approx(1 / 3600)
    with rasterio.open(result.output_path) as dataset:
        assert dataset.crs.is_projected and dataset.res == (30.0, 30.0)
        assert (dataset.height, dataset.width) == result.shape

    provenance = result.provenance()
    assert provenance["native_resolution"] == "1 arc-second (≈ 30 m)"
    assert provenance["output_resolution_m"] == 30.0
    assert provenance["output_crs"] == "EPSG:32633"
    assert KEY not in repr(provenance)


def test_rejected_key_error_never_contains_the_key(tmp_path):
    session = FakeSession(FakeResponse(
        401, b"", {"Content-Type": "text/html"}, text="<error>Invalid API key</error>",
    ))
    with pytest.raises(OpenTopographyError) as error:
        download_projected_dem(
            BBOX, dataset="COP30", api_key=KEY, output_path=tmp_path / "t.tif",
            crs="EPSG:32633", resolution_m=30.0, session=session,
        )
    message = str(error.value)
    assert "rejected the API key" in message and "Invalid API key" in message
    assert KEY not in message


def test_rate_limit_is_not_reported_as_a_bad_key(tmp_path):
    # OpenTopography's real answer when a key's daily quota is used up.
    text = "<error>Error: API maximum rate limit reached. (50 API calls/24hrs)</error>"
    session = FakeSession(FakeResponse(401, b"", {"Content-Type": "text/html"}, text=text))
    with pytest.raises(OpenTopographyError) as error:
        download_projected_dem(
            BBOX, dataset="COP30", api_key=KEY, output_path=tmp_path / "t.tif",
            crs="EPSG:32633", resolution_m=30.0, session=session,
        )
    assert "request limit" in str(error.value)
    assert "rejected the API key" not in str(error.value)


def test_non_geotiff_response_is_reported(tmp_path):
    session = FakeSession(FakeResponse(
        200, b"<error>Bad dataset</error>", {"Content-Type": "text/plain"},
        text="<error>Bad dataset</error>",
    ))
    with pytest.raises(OpenTopographyError, match="Bad dataset"):
        download_projected_dem(
            BBOX, dataset="COP30", api_key=KEY, output_path=tmp_path / "t.tif",
            crs="EPSG:32633", resolution_m=30.0, session=session,
        )


def test_download_validates_inputs_before_requesting(tmp_path):
    session = FakeSession(tiff_response(tmp_path))
    kwargs = dict(output_path=tmp_path / "t.tif", resolution_m=30.0, session=session)
    with pytest.raises(ValueError, match="exceeds OpenTopography's limit"):
        download_projected_dem((30.0, 40.0, 0.0, 10.0), dataset="COP30", api_key=KEY,
                               crs="EPSG:32631", **kwargs)
    with pytest.raises(OpenTopographyError, match="spaces"):
        download_projected_dem(BBOX, dataset="COP30", api_key="a b", crs="EPSG:32633", **kwargs)
    with pytest.raises(ValueError, match="projected in metres"):
        download_projected_dem(BBOX, dataset="COP30", api_key=KEY, crs="EPSG:4326", **kwargs)
    with pytest.raises(ValueError, match="Unknown OpenTopography dataset"):
        download_projected_dem(BBOX, dataset="XYZ", api_key=KEY, crs="EPSG:32633", **kwargs)
    assert session.requests == []
