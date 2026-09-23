"""Export SurveyObservabilityField products from Rivelero.

This module writes scientific spatial products derived from a
SurveyObservabilityField (SOF) to interoperable GIS formats.

The module is intentionally limited to input/output responsibilities. It does
not calculate exposure, visibility, observability states, metrics, or
visualizations.

The principal export format is GeoTIFF. A JSON manifest can additionally be
written to preserve analysis provenance and describe the exported products.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from rivelero.export.files import atomic_output
from rivelero.observability.masks import ObservabilityState
from rivelero.observability.survey_field import SurveyObservabilityField


# ---------------------------------------------------------------------------
# Public individual-product writers
# ---------------------------------------------------------------------------


def write_exposure_count(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    compress: str = "deflate",
    overwrite: bool = False,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write cumulative exposure count as a GeoTIFF.

    Pixel values represent the number of active sampling units from which
    each analysable cell was visible.

    Cells outside analysable space are written as nodata.
    """

    _validate_sof(sof)

    data = sof.exposure_count.copy()

    return _write_raster(
        path=path,
        data=data,
        transform=sof.transform,
        crs=sof.crs,
        valid_data_mask=sof.analysable_mask,
        nodata=_integer_nodata(data.dtype),
        compress=compress,
        overwrite=overwrite,
        extra_tags=extra_tags,
        tags={
            "rivelero_product": "exposure_count",
            "description": (
                "Number of active sampling units from which "
                "each analysable cell was visible."
            ),
            "sof_id": sof.sof_id,
        },
    )


def write_normalized_exposure(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    compress: str = "deflate",
    overwrite: bool = False,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write globally normalized exposure as a GeoTIFF.

    Current normalization:

        exposure_count / number_of_active_sampling_units

    Cells outside analysable space are written as nodata.

    Notes
    -----
    This is Rivelero's current global normalization. It should not be
    interpreted as future eligibility-aware normalized exposure.
    """

    _validate_sof(sof)

    data = sof.normalized_exposure.astype(
        np.float32,
        copy=False,
    )

    return _write_raster(
        path=path,
        data=data,
        transform=sof.transform,
        crs=sof.crs,
        valid_data_mask=sof.analysable_mask,
        nodata=np.nan,
        compress=compress,
        overwrite=overwrite,
        extra_tags=extra_tags,
        tags={
            "rivelero_product": "normalized_exposure",
            "normalization": (
                "exposure_count / active_sampling_units"
            ),
            "sof_id": sof.sof_id,
        },
    )


def write_observability_state(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    compress: str = "deflate",
    overwrite: bool = False,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write categorical observability state as a GeoTIFF.

    State values are:

        0 = outside analysis domain
        1 = invalid / unanalysable
        2 = blind spot
        3 = observable

    Unlike most SOF products, all four values are meaningful categories.
    Therefore this raster does not use nodata.
    """

    _validate_sof(sof)

    data = sof.observability_state.astype(
        np.uint8,
        copy=False,
    )

    return _write_raster(
        path=path,
        data=data,
        transform=sof.transform,
        crs=sof.crs,
        valid_data_mask=None,
        nodata=None,
        compress=compress,
        overwrite=overwrite,
        extra_tags=extra_tags,
        tags={
            "rivelero_product": "observability_state",
            "state_0": "outside_domain",
            "state_1": "invalid_unanalysable",
            "state_2": "blind_spot",
            "state_3": "observable",
            "sof_id": sof.sof_id,
        },
    )


def write_blindspot_mask(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    compress: str = "deflate",
    overwrite: bool = False,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write the SOF blind-spot mask as a GeoTIFF.

    Values within analysable space are:

        0 = not a blind spot
        1 = blind spot

    Cells outside analysable space are written as nodata.
    """

    _validate_sof(sof)

    data = sof.blindspot_mask.astype(
        np.uint8,
    )

    return _write_raster(
        path=path,
        data=data,
        transform=sof.transform,
        crs=sof.crs,
        valid_data_mask=sof.analysable_mask,
        nodata=255,
        compress=compress,
        overwrite=overwrite,
        extra_tags=extra_tags,
        tags={
            "rivelero_product": "blindspot_mask",
            "value_0": "not_blind_spot",
            "value_1": "blind_spot",
            "sof_id": sof.sof_id,
        },
    )


def write_observable_mask(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    compress: str = "deflate",
    overwrite: bool = False,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write the observable-space mask as a GeoTIFF."""

    _validate_sof(sof)

    data = sof.observable_mask.astype(
        np.uint8,
    )

    return _write_raster(
        path=path,
        data=data,
        transform=sof.transform,
        crs=sof.crs,
        valid_data_mask=sof.analysable_mask,
        nodata=255,
        compress=compress,
        overwrite=overwrite,
        extra_tags=extra_tags,
        tags={
            "rivelero_product": "observable_mask",
            "value_0": "not_observable",
            "value_1": "observable",
            "sof_id": sof.sof_id,
        },
    )


def write_analysis_mask(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    compress: str = "deflate",
    overwrite: bool = False,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write the AnalysisDomain membership mask."""

    _validate_sof(sof)

    data = sof.analysis_mask.astype(
        np.uint8,
    )

    return _write_raster(
        path=path,
        data=data,
        transform=sof.transform,
        crs=sof.crs,
        valid_data_mask=None,
        nodata=None,
        compress=compress,
        overwrite=overwrite,
        extra_tags=extra_tags,
        tags={
            "rivelero_product": "analysis_mask",
            "value_0": "outside_analysis_domain",
            "value_1": "inside_analysis_domain",
            "sof_id": sof.sof_id,
        },
    )


def write_valid_mask(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    compress: str = "deflate",
    overwrite: bool = False,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write the environmental/spatial validity mask."""

    _validate_sof(sof)

    data = sof.valid_mask.astype(
        np.uint8,
    )

    return _write_raster(
        path=path,
        data=data,
        transform=sof.transform,
        crs=sof.crs,
        valid_data_mask=None,
        nodata=None,
        compress=compress,
        overwrite=overwrite,
        extra_tags=extra_tags,
        tags={
            "rivelero_product": "valid_mask",
            "value_0": "invalid",
            "value_1": "valid",
            "sof_id": sof.sof_id,
        },
    )


# ---------------------------------------------------------------------------
# Complete SOF export
# ---------------------------------------------------------------------------


def export_sof(
    sof: SurveyObservabilityField,
    output_directory: str | Path,
    *,
    prefix: str | None = None,
    include_normalized_exposure: bool = True,
    include_support_masks: bool = True,
    include_manifest: bool = True,
    compress: str = "deflate",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export the principal products of a SurveyObservabilityField.

    Parameters
    ----------
    sof
        SurveyObservabilityField to export.

    output_directory
        Directory receiving the exported products.

    prefix
        Optional filename prefix. If omitted, ``sof.sof_id`` is used.

    include_normalized_exposure
        Whether to export the current globally normalized exposure product.

    include_support_masks
        Whether to export observable, analysis, and validity masks in
        addition to the core exposure/state/blind-spot products.

    include_manifest
        Whether to write a JSON provenance manifest.

    compress
        Rasterio/GDAL compression method for GeoTIFF products.

    overwrite
        Whether existing files may be replaced.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from product name to exported path.
    """

    _validate_sof(sof)

    directory = Path(
        output_directory
    ).expanduser().resolve()

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename_prefix = (
        sof.sof_id
        if prefix is None
        else _safe_prefix(prefix)
    )

    outputs: dict[
        str,
        Path,
    ] = {}

    outputs["exposure_count"] = (
        write_exposure_count(
            sof,
            directory
            / f"{filename_prefix}_exposure_count.tif",
            compress=compress,
            overwrite=overwrite,
        )
    )

    outputs["observability_state"] = (
        write_observability_state(
            sof,
            directory
            / f"{filename_prefix}_observability_state.tif",
            compress=compress,
            overwrite=overwrite,
        )
    )

    outputs["blindspot_mask"] = (
        write_blindspot_mask(
            sof,
            directory
            / f"{filename_prefix}_blindspot_mask.tif",
            compress=compress,
            overwrite=overwrite,
        )
    )

    if include_normalized_exposure:
        outputs["normalized_exposure"] = (
            write_normalized_exposure(
                sof,
                directory
                / (
                    f"{filename_prefix}"
                    "_normalized_exposure.tif"
                ),
                compress=compress,
                overwrite=overwrite,
            )
        )

    if include_support_masks:

        outputs["observable_mask"] = (
            write_observable_mask(
                sof,
                directory
                / f"{filename_prefix}_observable_mask.tif",
                compress=compress,
                overwrite=overwrite,
            )
        )

        outputs["analysis_mask"] = (
            write_analysis_mask(
                sof,
                directory
                / f"{filename_prefix}_analysis_mask.tif",
                compress=compress,
                overwrite=overwrite,
            )
        )

        outputs["valid_mask"] = (
            write_valid_mask(
                sof,
                directory
                / f"{filename_prefix}_valid_mask.tif",
                compress=compress,
                overwrite=overwrite,
            )
        )

    if include_manifest:

        manifest_path = (
            directory
            / f"{filename_prefix}_manifest.json"
        )

        write_sof_manifest(
            sof,
            manifest_path,
            products=outputs,
            overwrite=overwrite,
        )

        outputs["manifest"] = (
            manifest_path
        )

    return outputs


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_sof_manifest(
    sof: SurveyObservabilityField,
    path: str | Path,
    *,
    products: dict[str, Path] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write SOF provenance and export metadata as JSON."""

    _validate_sof(sof)

    output_path = _prepare_output_path(
        path,
        overwrite=overwrite,
    )

    manifest = {
        "format": "Rivelero SurveyObservabilityField export",
        "format_version": 1,
        "sof": {
            "sof_id": sof.sof_id,
            "viewpoint_configuration_id": (
                sof.viewpoint_configuration_id
            ),
            "environment_id": (
                sof.environment_id
            ),
            "analysis_domain_id": (
                sof.analysis_domain_id
            ),
            "visibility_configuration_id": (
                sof.visibility_configuration_id
            ),
            "sampling_unit": (
                sof.sampling_unit.value
            ),
            "created_at": (
                _datetime_to_iso(
                    sof.created_at
                )
            ),
            "active_sampling_units": (
                sof.n_active_units
            ),
            "analysable_cells": (
                sof.n_analysable_cells
            ),
            "observable_cells": (
                sof.n_observable_cells
            ),
            "blindspot_cells": (
                sof.n_blindspot_cells
            ),
        },
        "spatial": {
            "shape": list(
                sof.exposure_count.shape
            ),
            "transform": list(
                tuple(sof.transform)[:6]
            ),
            "crs": (
                sof.crs.to_string()
            ),
        },
        "observability_states": {
            str(
                ObservabilityState.OUTSIDE_DOMAIN.value
            ): "outside_domain",
            str(
                ObservabilityState.INVALID.value
            ): "invalid_unanalysable",
            str(
                ObservabilityState.BLIND_SPOT.value
            ): "blind_spot",
            str(
                ObservabilityState.OBSERVABLE.value
            ): "observable",
        },
        "active_visibility_keys": [
            {
                "sampling_unit_id": (
                    key.sampling_unit_id
                ),
                "sampling_unit_type": (
                    key.sampling_unit_type
                ),
                "viewpoint_id": (
                    key.viewpoint_id
                ),
                "environment_id": (
                    key.environment_id
                ),
                "analysis_domain_id": (
                    key.analysis_domain_id
                ),
                "visibility_configuration_id": (
                    key.visibility_configuration_id
                ),
                "input_fingerprint": key.input_fingerprint,
            }
            for key in sof.active_keys
        ],
        "products": (
            {}
            if products is None
            else {
                name: str(
                    Path(product_path).name
                )
                for name, product_path
                in products.items()
            }
        ),
        "metadata": _json_safe(
            sof.metadata
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ---------------------------------------------------------------------------
# Internal GeoTIFF writer
# ---------------------------------------------------------------------------


def _write_raster(
    *,
    path: str | Path,
    data: np.ndarray,
    transform,
    crs,
    valid_data_mask: np.ndarray | None,
    nodata: int | float | None,
    compress: str,
    overwrite: bool,
    tags: dict[str, Any] | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> Path:
    """Write one two-dimensional SOF product as GeoTIFF.

    The raster is written to a temporary file next to ``path`` and moved into
    place only after a successful write, so a failure never leaves a
    truncated GeoTIFF or replaces an existing file.
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

    output_path = _prepare_output_path(
        path,
        overwrite=overwrite,
    )

    output_data = data.copy()

    if valid_data_mask is not None:

        if not isinstance(
            valid_data_mask,
            np.ndarray,
        ):
            raise TypeError(
                "valid_data_mask must be a numpy.ndarray."
            )

        if (
            valid_data_mask.shape
            != output_data.shape
        ):
            raise ValueError(
                "valid_data_mask shape does not match data."
            )

        mask = valid_data_mask.astype(
            bool,
            copy=False,
        )

        if nodata is None:
            raise ValueError(
                "A nodata value is required when "
                "valid_data_mask is supplied."
            )

        output_data[
            ~mask
        ] = nodata

    profile = {
        "driver": "GTiff",
        "height": (
            output_data.shape[0]
        ),
        "width": (
            output_data.shape[1]
        ),
        "count": 1,
        "dtype": (
            output_data.dtype
        ),
        "crs": crs,
        "transform": transform,
        "compress": compress,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    all_tags = {**(tags or {}), **(extra_tags or {})}

    with atomic_output(output_path, overwrite=overwrite) as temporary:
        with rasterio.open(
            temporary,
            "w",
            **profile,
        ) as dataset:
            dataset.write(
                output_data,
                1,
            )

            if all_tags:
                dataset.update_tags(
                    **{
                        str(key): str(value)
                        for key, value
                        in all_tags.items()
                    }
                )

    return output_path


# Public name for other exporters writing grid-aligned rasters.
write_raster = _write_raster


# ---------------------------------------------------------------------------
# Validation / helpers
# ---------------------------------------------------------------------------


def _validate_sof(
    sof: SurveyObservabilityField,
) -> None:
    """Validate SOF export input."""

    if not isinstance(
        sof,
        SurveyObservabilityField,
    ):
        raise TypeError(
            "sof must be a SurveyObservabilityField."
        )


def _prepare_output_path(
    path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Validate and prepare an output path."""

    output_path = Path(
        path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    return output_path


def _integer_nodata(
    dtype: np.dtype | type,
) -> int:
    """Return a safe nodata sentinel for an integer raster dtype."""

    resolved = np.dtype(
        dtype
    )

    if not np.issubdtype(
        resolved,
        np.integer,
    ):
        raise TypeError(
            "dtype must be an integer dtype."
        )

    return int(
        np.iinfo(
            resolved
        ).max
    )


def _safe_prefix(
    value: str,
) -> str:
    """Validate an export filename prefix."""

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "prefix must be a string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            "prefix cannot be empty."
        )

    forbidden = (
        "/",
        "\\",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
    )

    if any(
        character in result
        for character in forbidden
    ):
        raise ValueError(
            "prefix contains characters that are "
            "not safe in filenames."
        )

    return result


def _datetime_to_iso(
    value: datetime,
) -> str:
    """Convert datetime to ISO-8601 representation."""

    if not isinstance(
        value,
        datetime,
    ):
        raise TypeError(
            "value must be a datetime."
        )

    return value.isoformat()


def _json_safe(
    value: Any,
) -> Any:
    """Convert common metadata values to JSON-safe structures."""

    if (
        value is None
        or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        )
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    return str(
        value
    )