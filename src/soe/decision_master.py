from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.windows import from_bounds

from .NDVI import NDVI
from .optimiser import (
    OptimiserWeights,
    optimise_candidates,
    scores_to_dataframe,
)


# -----------------------------------------------------------------------------
# Hardcoded DecisionMaster settings for the first GUI version.
# These can become GUI fields later, but for now the GUI only supplies:
# samples, DEM path, max distance, metadata CSV path, and project root.
# -----------------------------------------------------------------------------
DEFAULT_LEVELS = 3 #maximum number of levels.
DEFAULT_CELLS_PER_SIDE = 8 #how many cells in each side of the grid.
DEFAULT_INITIAL_PADDING_M = 250.0
DEFAULT_ZOOM_PADDING_M = 0.0
DEFAULT_MIN_POINTS_BEFORE_OPTIMISER = 1 #if only one candidate point exists, run the optimiser immediately.
DEFAULT_NDVI_PIXEL_SIZE_M = 70 #resolution for NDVI.
DEFAULT_DOWNLOAD_IMAGES = False #

DEFAULT_WEIGHTS = OptimiserWeights(
    ndvi=0.40,
    visibility_strength=0.40,
    unseenness=0.00,
    obstacle_penalty=0.20,
) 


@dataclass(slots=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.xmin, self.ymin, self.xmax, self.ymax


@dataclass(slots=True)
class Cell:
    row: int
    col: int
    bounds: Bounds
    point_indices: list[int]


@dataclass(slots=True)
class Blob:
    level: int
    blob_id: int
    cells: list[Cell]
    point_indices: list[int]
    mean_ndvi: float

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_points(self) -> int:
        return len(self.point_indices)

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            xmin=min(cell.bounds.xmin for cell in self.cells),
            ymin=min(cell.bounds.ymin for cell in self.cells),
            xmax=max(cell.bounds.xmax for cell in self.cells),
            ymax=max(cell.bounds.ymax for cell in self.cells),
        )
    #calculate bounds from cells stored inside blob.

def _default_time_range(
    time_from: str | None,
    time_to: str | None,
    days_back: int = 30,
) -> tuple[str, str]:
    if time_to is None:
        time_to = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if time_from is None:
        dt_to = datetime.strptime(time_to, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        time_from = (dt_to - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

    return time_from, time_to


def _bbox_lonlat_from_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    buffer_m: float,
) -> tuple[float, float, float, float]:
    if not sample_metadata:
        raise ValueError("sample_metadata is empty.")

    lons = np.asarray([float(sample["lon"]) for sample in sample_metadata], dtype=float)
    lats = np.asarray([float(sample["lat"]) for sample in sample_metadata], dtype=float)

    centre_lat = float(np.mean(lats))
    lat_buffer_deg = buffer_m / 111_320.0
    lon_buffer_deg = buffer_m / (111_320.0 * max(0.1, np.cos(np.radians(centre_lat))))

    return (
        float(lons.min() - lon_buffer_deg),
        float(lats.min() - lat_buffer_deg),
        float(lons.max() + lon_buffer_deg),
        float(lats.max() + lat_buffer_deg),
    )


def _project_lonlat_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    target_crs,
) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    lons = [float(sample["lon"]) for sample in sample_metadata]
    lats = [float(sample["lat"]) for sample in sample_metadata]
    xs, ys = transformer.transform(lons, lats)
    return np.column_stack([xs, ys]).astype(float)


def _square_bounds_around_points(
    points_xy: np.ndarray,
    indices: Sequence[int],
    padding_m: float,
) -> Bounds:
    if len(indices) == 0:
        raise ValueError("Cannot make bounds around zero points.")

    pts = points_xy[np.asarray(indices, dtype=int)]
    min_x = float(pts[:, 0].min())
    max_x = float(pts[:, 0].max())
    min_y = float(pts[:, 1].min())
    max_y = float(pts[:, 1].max())

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    size = max(max_x - min_x, max_y - min_y) + (2.0 * padding_m)
    size = max(size, 1.0)

    return Bounds(
        xmin=cx - size / 2.0,
        ymin=cy - size / 2.0,
        xmax=cx + size / 2.0,
        ymax=cy + size / 2.0,
    )


def _square_bounds_around_bounds(bounds: Bounds, padding_m: float = 0.0) -> Bounds:
    cx = bounds.cx
    cy = bounds.cy
    size = max(bounds.width, bounds.height) + (2.0 * padding_m)
    size = max(size, 1.0)
    return Bounds(
        xmin=cx - size / 2.0,
        ymin=cy - size / 2.0,
        xmax=cx + size / 2.0,
        ymax=cy + size / 2.0,
    )


def _cell_bounds(region: Bounds, row: int, col: int, cells_per_side: int) -> Bounds:
    cell_size = region.width / cells_per_side
    xmin = region.xmin + col * cell_size
    ymin = region.ymin + row * cell_size
    return Bounds(
        xmin=xmin,
        ymin=ymin,
        xmax=xmin + cell_size,
        ymax=ymin + cell_size,
    )


def _occupied_cells(
    *,
    points_xy: np.ndarray,
    active_indices: Sequence[int],
    region: Bounds,
    cells_per_side: int,
) -> dict[tuple[int, int], Cell]:
    if cells_per_side <= 0:
        raise ValueError("cells_per_side must be greater than zero.")

    cell_size = region.width / cells_per_side
    occupied: dict[tuple[int, int], Cell] = {}

    for point_index in active_indices:
        x = float(points_xy[point_index, 0])
        y = float(points_xy[point_index, 1])

        inside = (
            region.xmin <= x <= region.xmax
            and region.ymin <= y <= region.ymax
        )
        if not inside:
            continue

        col = int((x - region.xmin) / cell_size)
        row = int((y - region.ymin) / cell_size)

        # Points on the max boundary should go into the final cell, not overflow.
        col = min(max(col, 0), cells_per_side - 1)
        row = min(max(row, 0), cells_per_side - 1)

        key = (row, col)
        if key not in occupied:
            occupied[key] = Cell(
                row=row,
                col=col,
                bounds=_cell_bounds(region, row, col, cells_per_side),
                point_indices=[],
            )
        occupied[key].point_indices.append(int(point_index))

    return occupied


def _connected_components_8(
    occupied_cells: dict[tuple[int, int], Cell]
) -> list[list[tuple[int, int]]]:
    remaining = set(occupied_cells.keys())
    components: list[list[tuple[int, int]]] = []

    neighbour_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while remaining:
        start = remaining.pop()
        queue: deque[tuple[int, int]] = deque([start])
        component = [start]

        while queue:
            row, col = queue.popleft()
            for dr, dc in neighbour_offsets:
                neighbour = (row + dr, col + dc)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)

        components.append(component)

    return components


def _mean_ndvi_inside_cells(
    *,
    ndvi_array: np.ndarray,
    ndvi_transform,
    cells: Sequence[Cell],
) -> float:
    total = 0.0
    count = 0

    height, width = ndvi_array.shape

    for cell in cells:
        b = cell.bounds
        window = from_bounds(
            left=b.xmin,
            bottom=b.ymin,
            right=b.xmax,
            top=b.ymax,
            transform=ndvi_transform,
        )

        row0 = max(0, int(np.floor(window.row_off)))
        col0 = max(0, int(np.floor(window.col_off)))
        row1 = min(height, int(np.ceil(window.row_off + window.height)))
        col1 = min(width, int(np.ceil(window.col_off + window.width)))

        if row1 > row0 and col1 > col0:
            values = ndvi_array[row0:row1, col0:col1]
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                total += float(finite.sum())
                count += int(finite.size)
                continue

        # Fallback for very tiny cells: sample the raster pixel at the cell centre.
        try:
            row, col = rowcol(ndvi_transform, b.cx, b.cy)
        except Exception:
            continue

        if 0 <= row < height and 0 <= col < width:
            value = float(ndvi_array[row, col])
            if np.isfinite(value):
                total += value
                count += 1

    if count == 0:
        return float("nan")

    return total / count


def split_region_into_blobs(
    *,
    level: int,
    points_xy: np.ndarray,
    active_indices: Sequence[int],
    region: Bounds,
    cells_per_side: int,
    ndvi_array: np.ndarray,
    ndvi_transform,
) -> list[Blob]:
    occupied = _occupied_cells(
        points_xy=points_xy,
        active_indices=active_indices,
        region=region,
        cells_per_side=cells_per_side,
    )

    components = _connected_components_8(occupied)
    blobs: list[Blob] = []

    for blob_id, component in enumerate(components):
        cells = [occupied[key] for key in component]
        point_indices = sorted(
            {idx for cell in cells for idx in cell.point_indices}
        )
        mean_ndvi = _mean_ndvi_inside_cells(
            ndvi_array=ndvi_array,
            ndvi_transform=ndvi_transform,
            cells=cells,
        )
        blobs.append(
            Blob(
                level=level,
                blob_id=blob_id,
                cells=cells,
                point_indices=point_indices,
                mean_ndvi=mean_ndvi,
            )
        )

    blobs.sort(
        key=lambda blob: (
            -np.inf if not np.isfinite(blob.mean_ndvi) else blob.mean_ndvi,
            blob.n_points,
            blob.n_cells,
        ),
        reverse=True,
    )

    # Re-number after sorting so blob_id 0 is always the currently selected blob.
    for new_id, blob in enumerate(blobs):
        blob.blob_id = new_id

    return blobs


def _blob_summary_row(blob: Blob) -> dict[str, Any]:
    b = blob.bounds
    return {
        "level": blob.level,
        "blob_id": blob.blob_id,
        "n_cells": blob.n_cells,
        "n_points": blob.n_points,
        "mean_ndvi": blob.mean_ndvi,
        "xmin": b.xmin,
        "ymin": b.ymin,
        "xmax": b.xmax,
        "ymax": b.ymax,
    }


def save_grid_snapshot(
    *,
    output_path: Path,
    level: int,
    region: Bounds,
    cells_per_side: int,
    points_xy: np.ndarray,
    active_indices: Sequence[int],
    blobs: Sequence[Blob],
    selected_blob: Blob | None,
    ndvi_array: np.ndarray,
    ndvi_transform,
) -> None:
    """Save a PNG showing the current grid, blobs, and chosen blob."""
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    left = ndvi_transform.c
    top = ndvi_transform.f
    right = left + ndvi_transform.a * ndvi_array.shape[1]
    bottom = top + ndvi_transform.e * ndvi_array.shape[0]

    ax.imshow(
        ndvi_array,
        extent=(left, right, bottom, top),
        origin="upper",
        alpha=0.65,
    )

    # Draw the whole grid.
    for row in range(cells_per_side):
        for col in range(cells_per_side):
            b = _cell_bounds(region, row, col, cells_per_side)
            ax.add_patch(
                Rectangle(
                    (b.xmin, b.ymin),
                    b.width,
                    b.height,
                    fill=False,
                    linewidth=0.5,
                    alpha=0.35,
                )
            )

    # Draw occupied cells, thicker for the chosen blob.
    selected_keys = set()
    if selected_blob is not None:
        selected_keys = {(cell.row, cell.col) for cell in selected_blob.cells}

    for blob in blobs:
        for cell in blob.cells:
            b = cell.bounds
            is_selected = (cell.row, cell.col) in selected_keys
            ax.add_patch(
                Rectangle(
                    (b.xmin, b.ymin),
                    b.width,
                    b.height,
                    fill=False,
                    linewidth=2.0 if is_selected else 1.0,
                    alpha=1.0 if is_selected else 0.75,
                )
            )

        bb = blob.bounds
        ax.text(
            bb.cx,
            bb.cy,
            f"B{blob.blob_id}\nNDVI={blob.mean_ndvi:.3f}\npts={blob.n_points}",
            ha="center",
            va="center",
            fontsize=7,
        )

    if active_indices:
        pts = points_xy[np.asarray(active_indices, dtype=int)]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, marker="o")

    ax.set_xlim(region.xmin, region.xmax)
    ax.set_ylim(region.ymin, region.ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"DecisionMaster level {level} | "
        f"cells {cells_per_side}x{cells_per_side} | "
        f"blobs={len(blobs)}"
    )
    ax.set_xlabel("Projected X")
    ax.set_ylabel("Projected Y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def run_decision_master(
    *,
    sample_metadata: Sequence[Mapping[str, Any]],
    dem_path: str | Path,
    output_dir: str | Path,
    max_distance_m: float,
    levels: int = DEFAULT_LEVELS,
    cells_per_side: int = DEFAULT_CELLS_PER_SIDE,
    initial_padding_m: float = DEFAULT_INITIAL_PADDING_M,
    zoom_padding_m: float = DEFAULT_ZOOM_PADDING_M,
    min_points_before_optimiser: int = DEFAULT_MIN_POINTS_BEFORE_OPTIMISER,
    time_from: str | None = None,
    time_to: str | None = None,
    ndvi_pixel_size_m: int = DEFAULT_NDVI_PIXEL_SIZE_M,
    download_images: bool = DEFAULT_DOWNLOAD_IMAGES,
    weights: OptimiserWeights | None = None,
) -> pd.DataFrame:
    """
    Traverse candidate viewpoints using occupied-cell blobs and NDVI.

    The optimiser is called only once, after the recursive zoom/prune stage ends.
    """
    if not sample_metadata:
        raise ValueError("sample_metadata is empty.")
    if levels < 1:
        raise ValueError("levels must be at least 1.")
    if cells_per_side < 2:
        raise ValueError("cells_per_side must be at least 2.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve each row's original GUI/CSV identity before pruning.
    metadata = [
        {**dict(sample), "original_index": i}
        for i, sample in enumerate(sample_metadata)
    ]

    time_from, time_to = _default_time_range(time_from, time_to)

    print("[1/5] Fetching NDVI once for the whole candidate area...")
    bbox_lonlat = _bbox_lonlat_from_samples(metadata, buffer_m=max_distance_m)
    ndvi_result = NDVI(
        bbox_lonlat=bbox_lonlat,
        time_from=time_from,
        time_to=time_to,
        pixel_size_m=ndvi_pixel_size_m,
    )
    ndvi_array = ndvi_result["ndvi"]
    ndvi_transform = ndvi_result["transform"]
    ndvi_crs = ndvi_result["crs"]

    print("[2/5] Projecting candidate points into the NDVI raster CRS...")
    points_xy = _project_lonlat_samples(metadata, ndvi_crs)

    active_indices = list(range(len(metadata)))
    region = _square_bounds_around_points(
        points_xy=points_xy,
        indices=active_indices,
        padding_m=initial_padding_m,
    )

    decision_path: list[dict[str, Any]] = []
    all_blob_rows: list[dict[str, Any]] = []

    print("[3/5] Traversing occupied-cell blobs...")
    for level in range(levels):
        blobs = split_region_into_blobs(
            level=level,
            points_xy=points_xy,
            active_indices=active_indices,
            region=region,
            cells_per_side=cells_per_side,
            ndvi_array=ndvi_array,
            ndvi_transform=ndvi_transform,
        )

        if not blobs:
            print(f"No blobs found at level {level}; stopping traversal.")
            break

        selected_blob = blobs[0]

        level_csv = output_dir / f"level_{level:02d}_clusters.csv"
        level_df = pd.DataFrame([_blob_summary_row(blob) for blob in blobs])
        level_df.to_csv(level_csv, index=False)
        all_blob_rows.extend(level_df.to_dict(orient="records"))

        snapshot_path = output_dir / f"level_{level:02d}_grid.png"
        save_grid_snapshot(
            output_path=snapshot_path,
            level=level,
            region=region,
            cells_per_side=cells_per_side,
            points_xy=points_xy,
            active_indices=active_indices,
            blobs=blobs,
            selected_blob=selected_blob,
            ndvi_array=ndvi_array,
            ndvi_transform=ndvi_transform,
        )

        path_row = _blob_summary_row(selected_blob)
        path_row["snapshot_path"] = str(snapshot_path)
        decision_path.append(path_row)

        print(
            f"Level {level}: chose blob {selected_blob.blob_id} "
            f"with mean NDVI={selected_blob.mean_ndvi:.4f}, "
            f"points={selected_blob.n_points}, cells={selected_blob.n_cells}."
        )

        active_indices = selected_blob.point_indices

        if len(active_indices) <= min_points_before_optimiser:
            print(
                "Reached min_points_before_optimiser "
                f"({min_points_before_optimiser}); stopping traversal."
            )
            break

        region = _square_bounds_around_bounds(
            selected_blob.bounds,
            padding_m=zoom_padding_m,
        )

    pd.DataFrame(decision_path).to_csv(output_dir / "decision_path.csv", index=False)
    pd.DataFrame(all_blob_rows).to_csv(output_dir / "all_blob_summaries.csv", index=False)

    selected_metadata = [metadata[i] for i in active_indices]
    pd.DataFrame(selected_metadata).to_csv(output_dir / "selected_candidates.csv", index=False)

    print(
        "[4/5] Traversal finished. "
        f"Running optimiser once on {len(selected_metadata)} selected candidates..."
    )
    ranked_scores = optimise_candidates(
        sample_metadata=selected_metadata,
        dem_path=dem_path,
        max_distance_m=max_distance_m,
        weights=weights or DEFAULT_WEIGHTS,
        time_from=time_from,
        time_to=time_to,
        download_images=download_images,
    )

    print("[5/5] Saving final optimiser results...")
    final_df = scores_to_dataframe(ranked_scores)

    # Keep the old GUI output shape: one chunk, then original CSV/GUI row identity.
    final_df.insert(0, "chunk_id", 1)
    final_df.insert(
        1,
        "original_index",
        final_df["index"].apply(
            lambda local_i: selected_metadata[int(local_i)]["original_index"]
        ),
    )

    final_output = output_dir / "final_optimiser_results.csv"
    final_df.to_csv(final_output, index=False)
    final_df.attrs["output_dir"] = str(output_dir)
    final_df.attrs["final_output"] = str(final_output)

    print(f"Saved final optimiser results to: {final_output}")
    return final_df


def _gui_output_dir(
    *,
    metadata_csv_path: str | Path | None,
    vista_root: str | Path | None,
) -> Path:
    """Choose an output folder using GUI information, with a project fallback."""
    if metadata_csv_path:
        return Path(metadata_csv_path).with_name("decision_master_results")

    if vista_root is not None:
        return Path(vista_root) / "Results" / "decision_master"

    return Path.cwd() / "decision_master_results"


def run_decision_master_from_gui(
    *,
    sample_metadata: Sequence[Mapping[str, Any]],
    dem_path: str | Path,
    max_distance_m: float,
    metadata_csv_path: str | Path | None = None,
    vista_root: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    GUI entry point.

    Extracted from GUI:
    - sample_metadata from validate_inputs()
    - dem_path from get_dem_path()
    - max_distance_m from validate_max_distance()
    - metadata_csv_path / vista_root for output location

    Hardcoded for now:
    - levels, grid size, padding, NDVI pixel size, optimiser weights, image download flag
    """
    output_dir = _gui_output_dir(
        metadata_csv_path=metadata_csv_path,
        vista_root=vista_root,
    )

    final_df = run_decision_master(
        sample_metadata=sample_metadata,
        dem_path=dem_path,
        output_dir=output_dir,
        max_distance_m=max_distance_m,
        levels=DEFAULT_LEVELS,
        cells_per_side=DEFAULT_CELLS_PER_SIDE,
        initial_padding_m=DEFAULT_INITIAL_PADDING_M,
        zoom_padding_m=DEFAULT_ZOOM_PADDING_M,
        min_points_before_optimiser=DEFAULT_MIN_POINTS_BEFORE_OPTIMISER,
        time_from=None,
        time_to=None,
        ndvi_pixel_size_m=DEFAULT_NDVI_PIXEL_SIZE_M,
        download_images=DEFAULT_DOWNLOAD_IMAGES,
        weights=DEFAULT_WEIGHTS,
    )

    return final_df, output_dir






















































