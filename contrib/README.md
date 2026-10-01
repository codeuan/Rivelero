# Contributed, unsupported code

Code here is **not part of the Rivelero package**: it is not installed, not
imported by the application and not covered by the test suite. It is kept
because it may be useful when the corresponding integration is designed
against Rivelero's canonical models.

## `data_sources/`

| Module | What it does | Status |
|---|---|---|
| `gsv.py` | Google Street View metadata lookup and image download for sample points (needs `GOOGLE_MAPS_API_KEY`). | From the earlier candidate-sampling workflow. A future street-level-imagery importer should produce canonical `Viewpoint` / `ObservationEvent` objects (position, heading, capture date) instead of image files. |
| `sentinel.py` | Sentinel-2 L2A NDVI retrieval from the Copernicus Data Space Ecosystem (needs `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`). | Its only consumer (the removed botanical-suitability field) is gone. Could supply a vegetation `EnvironmentLayer` once environmental layers are modelled. |

Both need only `requests`, `pyproj`, `rasterio` and `numpy`. Use them from a
checkout (`python -c "import sys; sys.path.insert(0, 'contrib/data_sources'); import gsv"`).
