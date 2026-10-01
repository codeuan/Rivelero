"""GeoTIFF exports.

Every raster is written on the exact grid of the SurveyObservabilityField it
derives from (same CRS, affine transform and shape), atomically, with
``RIVELERO_*`` GeoTIFF tags and a JSON sidecar describing its meaning.

NoData semantics
----------------
* Continuous products (exposure, normalized exposure, exposure difference)
  write NoData **only** for cells outside analysable space (outside the
  AnalysisDomain or invalid). A blind spot is a measured zero, never NoData.
* Categorical products (observability state, coverage class, change classes)
  have no NoData: every code, including "outside domain" and "invalid", is a
  meaningful category, so GIS software must not hide any of them.
* Binary masks restricted to analysable space (blind-spot, observable) use
  255 as NoData outside it.

SOF products reuse ``rivelero.io.observability`` so there is exactly one
implementation of their semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from rivelero.analysis.comparison import ScenarioComparison
from rivelero.analysis.coverage import CoverageClass, coverage_class_raster
from rivelero.analysis.scenario import ScenarioChangeClass
from rivelero.export.metadata import (
    ExportProvenance,
    geotiff_tags,
    product_metadata,
    sidecar_path,
    write_sidecar,
)
from rivelero.io import observability as sof_io
from rivelero.observability.survey_field import SurveyObservabilityField

COMPRESS = "deflate"

OBSERVABILITY_STATE_CODES = {
    0: "outside_analysis_domain",
    1: "invalid_unanalysable",
    2: "blind_spot",
    3: "observable",
}

COVERAGE_CLASS_CODES = {
    int(CoverageClass.OUTSIDE_DOMAIN): "outside_analysis_domain",
    int(CoverageClass.INVALID): "invalid_unanalysable",
    int(CoverageClass.BLIND_SPOT): "blind_spot",
    int(CoverageClass.UNIQUE): "observable_unique_exposure_1",
    int(CoverageClass.REPEATED): "observable_repeated_exposure_2_or_more",
}


def change_class_codes(before: str, after: str) -> dict[int, str]:
    """Codes of ScenarioChangeClass, worded for a ``before -> after`` change."""

    return {
        int(ScenarioChangeClass.OUTSIDE_DOMAIN): "outside_analysis_domain",
        int(ScenarioChangeClass.INVALID): "invalid_unanalysable",
        int(ScenarioChangeClass.REMAINS_BLIND): f"blind_in_{before}_and_{after}",
        int(ScenarioChangeClass.REMAINS_OBSERVABLE): f"observable_in_{before}_and_{after}",
        int(ScenarioChangeClass.LOST_COVERAGE): f"observable_in_{before}_blind_in_{after}",
        int(ScenarioChangeClass.GAINED_COVERAGE): f"blind_in_{before}_observable_in_{after}",
    }


def _exposure_nodata(dtype) -> str:
    return (
        f"{np.iinfo(np.dtype(dtype)).max} = not analysable (outside the AnalysisDomain "
        "or invalid terrain). Blind spots are 0, never NoData."
    )


NOT_ANALYSABLE_NAN = (
    "NaN = not analysable (outside the AnalysisDomain or invalid terrain). "
    "Blind spots are 0, never NoData."
)
MASK_NODATA = (
    "255 = not analysable (outside the AnalysisDomain or invalid terrain)."
)
NO_NODATA = "none - every code is a meaningful category"


# ---------------------------------------------------------------------------
# Writing one product
# ---------------------------------------------------------------------------


def _write_with_sidecar(
    path: Path,
    record: dict[str, Any],
    write: Callable[[Path, dict[str, str]], Path],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Write a raster then its sidecar; never leave a raster without one.

    If the sidecar cannot be written, a raster created by this call is
    removed again (a raster that replaced an older file under ``overwrite``
    is kept: the old file is already gone and the new one is valid).
    """

    existed = path.exists()
    raster = write(path, geotiff_tags(record))
    try:
        sidecar = write_sidecar(raster, record, overwrite=overwrite)
    except BaseException:
        if not existed:
            raster.unlink(missing_ok=True)
        raise
    return raster, sidecar


def raster_outputs(path: Path) -> tuple[Path, Path]:
    """All files one raster product creates (for conflict checks)."""

    return path, sidecar_path(path)


def _grid_record(sof: SurveyObservabilityField) -> dict[str, Any]:
    return {
        "grid": {
            "crs": sof.crs.to_string() if sof.crs is not None else None,
            "transform": list(sof.transform)[:6],
            "height": int(sof.exposure_count.shape[0]),
            "width": int(sof.exposure_count.shape[1]),
        }
    }


# ---------------------------------------------------------------------------
# Survey Observability Field products
# ---------------------------------------------------------------------------

SOF_PRODUCTS: dict[str, dict[str, Any]] = {
    "exposure_count": {
        "writer": sof_io.write_exposure_count,
        "meaning": (
            "Number of active sampling units from which each analysable cell "
            "is visible (0 = blind spot)."
        ),
        "units": "count of sampling units",
    },
    "normalized_exposure": {
        "writer": sof_io.write_normalized_exposure,
        "meaning": (
            "exposure_count divided by the number of active sampling units "
            "(global normalization, 0-1)."
        ),
        "units": "fraction of active sampling units",
        "nodata": NOT_ANALYSABLE_NAN,
    },
    "observability_state": {
        "writer": sof_io.write_observability_state,
        "meaning": "Categorical observability state of every grid cell.",
        "codes": OBSERVABILITY_STATE_CODES,
        "nodata": NO_NODATA,
    },
    "blindspot_mask": {
        "writer": sof_io.write_blindspot_mask,
        "meaning": "Analysable cells visible from no active sampling unit.",
        "codes": {0: "observable", 1: "blind_spot"},
        "nodata": MASK_NODATA,
    },
    "observable_mask": {
        "writer": sof_io.write_observable_mask,
        "meaning": "Analysable cells visible from at least one active sampling unit.",
        "codes": {0: "blind_spot", 1: "observable"},
        "nodata": MASK_NODATA,
    },
    "analysis_mask": {
        "writer": sof_io.write_analysis_mask,
        "meaning": "Membership of the AnalysisDomain.",
        "codes": {0: "outside_analysis_domain", 1: "inside_analysis_domain"},
        "nodata": NO_NODATA,
    },
    "valid_mask": {
        "writer": sof_io.write_valid_mask,
        "meaning": "Terrain validity (cells with usable elevation).",
        "codes": {0: "invalid", 1: "valid"},
        "nodata": NO_NODATA,
    },
}


def write_sof_product(
    product: str,
    sof: SurveyObservabilityField,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    spec = SOF_PRODUCTS[product]
    record = product_metadata(
        provenance,
        product=product,
        meaning=spec["meaning"],
        units=spec.get("units"),
        nodata=spec.get("nodata") or _exposure_nodata(sof.exposure_count.dtype),
        codes=spec.get("codes"),
        extra=_grid_record(sof),
    )
    writer = spec["writer"]
    return _write_with_sidecar(
        Path(path),
        record,
        lambda target, tags: writer(
            sof, target, compress=COMPRESS, overwrite=overwrite, extra_tags=tags
        ),
        overwrite=overwrite,
    )


def write_coverage_class(
    sof: SurveyObservabilityField,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """A1 coverage classes: observable space split into unique/repeated."""

    data = coverage_class_raster(
        sof.exposure_count, analysis_mask=sof.analysis_mask, valid_mask=sof.valid_mask
    )
    record = product_metadata(
        provenance,
        product="coverage_class",
        meaning=(
            "Descriptive coverage class: blind spot, observable from exactly one "
            "sampling unit (unique) or from two or more (repeated)."
        ),
        nodata=NO_NODATA,
        codes=COVERAGE_CLASS_CODES,
        extra=_grid_record(sof),
    )
    return _write_array(data, sof, Path(path), record, nodata=None, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Scenario and comparison products
# ---------------------------------------------------------------------------


def write_scenario_exposure(
    exposure: np.ndarray,
    sof: SurveyObservabilityField,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
    scenario_record: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Exposure of the live what-if scenario (same encoding as exposure_count)."""

    exposure = np.asarray(exposure)
    record = product_metadata(
        provenance,
        product="scenario_exposure",
        meaning=(
            "Exposure count of the current what-if scenario: baseline exposure "
            "minus deactivated sampling units plus included candidate Viewpoints."
        ),
        units="count of sampling units",
        nodata=_exposure_nodata(exposure.dtype),
        extra={**_grid_record(sof), **({"scenario": scenario_record} if scenario_record else {})},
    )
    return _write_array(
        exposure.copy(),
        sof,
        Path(path),
        record,
        nodata=int(np.iinfo(exposure.dtype).max),
        valid_data_mask=sof.analysable_mask,
        overwrite=overwrite,
    )


def write_scenario_change(
    change_classes: np.ndarray,
    sof: SurveyObservabilityField,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
    scenario_record: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Baseline -> scenario coverage change classes (A3 classification)."""

    record = product_metadata(
        provenance,
        product="scenario_change_classes",
        meaning="Coverage change of each cell from the baseline to the current scenario.",
        nodata=NO_NODATA,
        codes=change_class_codes("baseline", "scenario"),
        extra={
            **_grid_record(sof),
            "direction": "baseline_to_scenario",
            **({"scenario": scenario_record} if scenario_record else {}),
        },
    )
    return _write_array(
        np.asarray(change_classes).astype(np.uint8), sof, Path(path), record,
        nodata=None, overwrite=overwrite,
    )


def comparison_sides(comparison: ScenarioComparison) -> dict[str, Any]:
    return {
        "direction": "right_minus_left",
        "left": {"state_id": comparison.left.state_id, "label": comparison.left.label},
        "right": {"state_id": comparison.right.state_id, "label": comparison.right.label},
    }


def write_comparison_difference(
    comparison: ScenarioComparison,
    sof: SurveyObservabilityField,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Signed exposure difference RIGHT - LEFT (int32)."""

    nodata = int(np.iinfo(np.int32).min)
    difference = comparison.exposure_difference()
    data = np.ma.filled(difference, 0).astype(np.int32)
    record = product_metadata(
        provenance,
        product="comparison_exposure_difference_right_minus_left",
        meaning=(
            "Exposure count of the RIGHT design state minus exposure count of the "
            "LEFT design state. Positive = more sampling units see the cell on the "
            "right; 0 includes cells blind in both."
        ),
        units="count of sampling units (signed)",
        nodata=(
            f"{nodata} = not analysable (outside the AnalysisDomain or invalid "
            "terrain). 0 is a real value (no change), never NoData."
        ),
        extra={**_grid_record(sof), **comparison_sides(comparison)},
    )
    return _write_array(
        data, sof, Path(path), record, nodata=nodata,
        valid_data_mask=comparison.analysable_mask, overwrite=overwrite,
    )


def write_comparison_change(
    comparison: ScenarioComparison,
    sof: SurveyObservabilityField,
    path: Path,
    provenance: ExportProvenance,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """LEFT -> RIGHT coverage change classes."""

    record = product_metadata(
        provenance,
        product="comparison_change_classes",
        meaning="Coverage change of each cell from the LEFT to the RIGHT design state.",
        nodata=NO_NODATA,
        codes=change_class_codes("left", "right"),
        extra={**_grid_record(sof), **comparison_sides(comparison)},
    )
    return _write_array(
        comparison.change_classes().astype(np.uint8), sof, Path(path), record,
        nodata=None, overwrite=overwrite,
    )


def _write_array(
    data: np.ndarray,
    sof: SurveyObservabilityField,
    path: Path,
    record: dict[str, Any],
    *,
    nodata: int | float | None,
    valid_data_mask: np.ndarray | None = None,
    overwrite: bool,
) -> tuple[Path, Path]:
    if data.shape != sof.exposure_count.shape:
        raise ValueError("Raster does not match the observability field grid.")
    return _write_with_sidecar(
        path,
        record,
        lambda target, tags: sof_io.write_raster(
            path=target,
            data=data,
            transform=sof.transform,
            crs=sof.crs,
            valid_data_mask=valid_data_mask,
            nodata=nodata,
            compress=COMPRESS,
            overwrite=overwrite,
            tags={"rivelero_product": record["product"], "sof_id": sof.sof_id},
            extra_tags=tags,
        ),
        overwrite=overwrite,
    )
