"""Generic viewpoint-region data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS


@dataclass(frozen=True, slots=True)
class ViewpointRegion:
    """One circular or rectangular local OPF region."""

    identifier: str
    x: float
    y: float
    radius_m: float | None = None
    bounds: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class ViewpointOPFResult:
    """The portion of an OPF available to one viewpoint."""

    viewpoint: ViewpointRegion
    field: np.ndarray
    valid_mask: np.ndarray
    transform: Affine
    crs: CRS

    def as_display_mapping(self) -> dict[str, Any]:
        return {
            "data": self.field,
            "transform": self.transform,
            "crs": self.crs,
            "title": f"OPF around {self.viewpoint.identifier}",
            "colour_map": "viridis",
            "colourbar_label": "Observability potential",
            "vmin": -0.5,
            "vmax": 0.8,
        }
