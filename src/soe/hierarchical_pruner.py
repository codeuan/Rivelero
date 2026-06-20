

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from rasterio.warp import transform as transform_coords
from random import Random

from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.plot import plotting_extent

@dataclass(frozen=True)
class PlotWindow:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.left, self.right, self.bottom, self.top

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.bottom + self.top) / 2

@dataclass
class Square:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    n_points: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2

    @property
    def size(self) -> float:
        return self.xmax - self.xmin


def load_candidate_points_csv(
    csv_path: str | Path,
    x_col: str,
    y_col: str,
) -> np.ndarray:
    """
    Load candidate viewpoint coordinates from a CSV.

    Returns:
        An Nx2 NumPy array:

        [
            [x1, y1],
            [x2, y2],
            ...
        ]
    """

    points = []
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("Candidate CSV has no header row.")

        if x_col not in reader.fieldnames:
            raise ValueError(
                f"Column {x_col!r} not found in CSV. "
                f"Available columns: {reader.fieldnames}"
            )

        if y_col not in reader.fieldnames:
            raise ValueError(
                f"Column {y_col!r} not found in CSV. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            x = float(row[x_col])
            y = float(row[y_col])
            points.append((x, y))

    if not points:
        raise ValueError(f"No candidate points found in CSV: {csv_path}")

    return np.asarray(points, dtype=float)


def reproject_points_if_needed(
    points: np.ndarray,
    candidate_crs: str | None,
    dem_crs,
) -> np.ndarray:
    """
    Reproject candidate points into the DEM CRS.

    If candidate_crs is None, this assumes the candidate points are already
    in the same CRS as the DEM.
    """

    if candidate_crs is None:
        return points

    if dem_crs is None:
        raise ValueError("DEM has no CRS, so candidate points cannot be reprojected.")

    xs = points[:, 0]
    ys = points[:, 1]

    new_xs, new_ys = transform_coords(
        candidate_crs,
        dem_crs,
        xs.tolist(),
        ys.tolist(),
    )

    return np.column_stack([new_xs, new_ys])


def read_square_dem_crop_for_plotting(
    dem_path: str | Path,
    points: np.ndarray,
    padding: float = 0.0,
    max_pixels_per_side: int = 1600,
):
    """
    Read only a square DEM crop around all candidate points.

    Assumes points are already in the DEM CRS.
    Returns:
        dem, extent, crs, crop_bounds
    """

    dem_path = Path(dem_path) 

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    if points.size == 0:
        raise ValueError("No points provided for DEM crop.")

    xs = points[:, 0]
    ys = points[:, 1]

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    width = max_x - min_x
    height = max_y - min_y

    square_size = max(width, height) + (2 * padding)

    left = cx - square_size / 2
    right = cx + square_size / 2
    bottom = cy - square_size / 2
    top = cy + square_size / 2

    with rasterio.open(dem_path) as src:
        if src.count < 1:
            raise ValueError("DEM has no raster bands.")

        dem_left, dem_bottom, dem_right, dem_top = src.bounds

        left = max(left, dem_left)
        right = min(right, dem_right)
        bottom = max(bottom, dem_bottom)
        top = min(top, dem_top)

        window = rasterio.windows.from_bounds(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            transform=src.transform,
        ).round_offsets().round_lengths() #convert crop into raster pixel window and round so offsets and sizees are whole pixel.

        scale = max(
            window.width / max_pixels_per_side,
            window.height / max_pixels_per_side,
            1,
        )

        out_width = int(window.width / scale)
        out_height = int(window.height / scale)

        dem = src.read(
            1,
            window=window,
            out_shape=(out_height, out_width),
            masked=True,
        ) #read only the window of the crop, and scale to a suitable plotting size.

        crop_transform = src.window_transform(window) #create new transform for the DEM.

        extent = rasterio.plot.plotting_extent(
            dem,
            transform=crop_transform,
        ) #work out plotting bounds for new DEM.


    return dem, extent 


def choose_next_square_random(
    squares: list[Square],
    rng: Random,
) -> Square:
    """
    Randomly choose the next non-empty square to zoom into.

    This is a temporary placeholder for a real heuristic later.
    """
    if not squares:
        raise ValueError("Cannot choose from an empty list of squares.")

    return rng.choice(squares) #randomly select a blob.


def count_points_in_square(
    points: np.ndarray,
    xmin: float,
    ymin: float,
    size: float,
) -> Square:
    """
    Create one square and count how many candidate viewpoints are inside it.
    """

    xmax = xmin + size
    ymax = ymin + size

    xs = points[:, 0]
    ys = points[:, 1]

    inside = (
        (xs >= xmin)
        & (xs < xmax)
        & (ys >= ymin)
        & (ys < ymax)
    )

    n_points = int(np.count_nonzero(inside))

    return Square(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        n_points=n_points,
    )


def make_initial_squares(
    points: np.ndarray,
    square_size: float,
) -> list[Square]:
    """
    Make the largest starting squares around the candidate viewpoint area.

    Empty squares are ignored.
    """

    xs = points[:, 0]
    ys = points[:, 1]

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    start_x = np.floor(min_x / square_size) * square_size
    start_y = np.floor(min_y / square_size) * square_size

    squares = []

    x = start_x
    while x <= max_x:
        y = start_y

        while y <= max_y:
            square = count_points_in_square(
                points=points,
                xmin=x,
                ymin=y,
                size=square_size,
            )

            if square.n_points > 0:
                squares.append(square)

            y += square_size

        x += square_size

    return squares



def split_plot_window_into_grid(
    *,
    points: np.ndarray,
    window: PlotWindow,
    subdivisions_per_side: int,
) -> list[Square]:
    """
    Split the current plotting window into a grid.

    The cell size is calculated from the current plotted window, not from
    a fixed initial_size.
    """
    if subdivisions_per_side <= 1:
        raise ValueError("subdivisions_per_side must be greater than 1.")

    if window.width <= 0 or window.height <= 0:
        raise ValueError("Plot window must have positive width and height.")

    # Use square cells so the hierarchy stays spatially consistent.
    # The grid is based on the smaller visible dimension so all cells fit inside the window.
    cell_size = min(window.width, window.height) / subdivisions_per_side

    children = []

    for row in range(subdivisions_per_side):
        for col in range(subdivisions_per_side):
            xmin = window.left + col * cell_size
            ymin = window.bottom + row * cell_size

            child = count_points_in_square(
                points=points,
                xmin=xmin,
                ymin=ymin,
                size=cell_size,
            )

            if child.n_points > 0:
                children.append(child)

    return children

def make_root_plot_window_for_points(
    points: np.ndarray,
    padding_fraction: float = 0.05,
    min_padding: float = 100.0,
) -> PlotWindow:
    """
    Create the first plotting window around all candidate points.
    """
    if points.size == 0:
        raise ValueError("No points provided.")

    xs = points[:, 0]
    ys = points[:, 1]

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    width = max_x - min_x
    height = max_y - min_y

    base_size = max(width, height)

    if base_size <= 0:
        base_size = min_padding

    padding = max(base_size * padding_fraction, min_padding)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    window_size = base_size + 2 * padding

    return PlotWindow(
        left=cx - window_size / 2,
        right=cx + window_size / 2,
        bottom=cy - window_size / 2,
        top=cy + window_size / 2,
    )

def read_dem_crop_for_plot_window(
    *,
    dem_path: str | Path,
    window: PlotWindow,
    max_pixels_per_side: int = 1600,
):
    """
    Read a DEM crop for the current plotting window.

    This means each level gets a genuinely new DEM crop/window.
    """
    dem_path = Path(dem_path)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        dem_left, dem_bottom, dem_right, dem_top = src.bounds

        left = max(window.left, dem_left)
        right = min(window.right, dem_right)
        bottom = max(window.bottom, dem_bottom)
        top = min(window.top, dem_top)

        if left >= right or bottom >= top:
            raise ValueError("Requested plot window does not overlap the DEM.")

        raster_window = from_bounds(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            transform=src.transform,
        ).round_offsets().round_lengths()

        raster_window = raster_window.crop(
            height=src.height,
            width=src.width,
        )

        scale = max(
            raster_window.width / max_pixels_per_side,
            raster_window.height / max_pixels_per_side,
            1,
        )

        out_width = max(1, int(raster_window.width / scale))
        out_height = max(1, int(raster_window.height / scale))

        dem = src.read(
            1,
            window=raster_window,
            out_shape=(out_height, out_width),
            masked=True,
        )

        crop_transform = src.window_transform(raster_window)

    extent = plotting_extent(dem, transform=crop_transform)

    return dem, extent

def build_square_levels(
    *,
    points: np.ndarray,
    min_size: float,
    subdivisions_per_side: int,
    random_seed: int | None = None,
) -> list[tuple[PlotWindow, list[Square]]]:
    """
    Build a random zoom path.

    Each level:
    - splits the current plotting window into a grid
    - randomly chooses one non-empty child cell
    - makes that child cell the next plotting window
    """
    if min_size <= 0:
        raise ValueError("min_size must be greater than 0.")

    if subdivisions_per_side <= 1:
        raise ValueError("subdivisions_per_side must be greater than 1.")

    rng = Random(random_seed)

    current_window = make_root_plot_window_for_points(points)

    levels = []

    while True:
        children = split_plot_window_into_grid(
            points=points,
            window=current_window,
            subdivisions_per_side=subdivisions_per_side,
        )

        if not children:
            break

        selected_square = choose_next_square_random(
            squares=children,
            rng=rng,
        )

        levels.append((current_window, [selected_square]))

        if selected_square.size < min_size:
            break

        xmin, ymin, xmax, ymax = selected_square.bounds

        current_window = PlotWindow(
            left=xmin,
            right=xmax,
            bottom=ymin,
            top=ymax,
        )

    return levels

def draw_square(
    ax,
    square: Square,
    linewidth: float = 1.5,
) -> None:
    """
    Draw one square and write the candidate viewpoint count inside it.
    """

    xmin, ymin, xmax, ymax = square.bounds

    rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        linewidth=linewidth,
    )

    ax.add_patch(rect)

    ax.text(
        square.cx,
        square.cy,
        str(square.n_points),
        ha="center",
        va="center",
        fontsize=7,
    )


def plot_level(
    *,
    dem,
    extent,
    points: np.ndarray,
    squares: list[Square],
    level_index: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(
        dem,
        extent=extent,
        origin="upper",
    )

    left, right, bottom, top = extent

    xs = points[:, 0]
    ys = points[:, 1]

    visible_points_mask = (
        (xs >= left)
        & (xs <= right)
        & (ys >= bottom)
        & (ys <= top)
    )

    visible_points = points[visible_points_mask]

    if len(visible_points) > 0:
        ax.scatter(
            visible_points[:, 0],
            visible_points[:, 1],
            s=10,
            marker="o",
            label="Candidate viewpoints",
        )

    for square in squares:
        draw_square(ax, square)

    selected_square = squares[0]
    square_size = selected_square.size

    ax.set_title(
        f"Subdivision level {level_index} | "
        f"cell size = {square_size:.2f} | "
        f"selected squares = {len(squares)}"
    )

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal", adjustable="box")

    if len(visible_points) > 0:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def read_dem_crs(dem_path: str | Path):
    """
    Read only the DEM CRS without reading the DEM data.
    """

    dem_path = Path(dem_path)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        return src.crs

def run_hierarchical_pruner_from_gui(
    *,
    sample_metadata: list[dict],
    dem_path: str | Path,
    output_dir: str | Path,
    min_size: float,
    subdivisions_per_side: int,
    random_seed: int | None = 42,
) -> tuple[Path, list[tuple[PlotWindow, list[Square]]]]:
    """
    Run the hierarchical pruner from GUI sample metadata.

    The GUI already has validated lon/lat samples in memory, so this avoids
    forcing the GUI to go back through a CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sample_metadata:
        raise ValueError("No sample metadata was provided to the hierarchical pruner.")

    points = np.asarray(
        [
            (sample["lon"], sample["lat"])
            for sample in sample_metadata
        ],
        dtype=float,
    )

    print("[1/4] Reading DEM CRS...")
    dem_crs = read_dem_crs(dem_path)
    print(f"DEM CRS: {dem_crs}")

    print("[2/4] Reprojecting GUI candidate viewpoints...")
    points = reproject_points_if_needed(
        points=points,
        candidate_crs="EPSG:4326",
        dem_crs=dem_crs,
    )

    if dem_crs is not None and dem_crs.is_geographic:
        print(
            "WARNING: DEM CRS appears to be geographic, so min_size "
            "is probably in degrees, not metres. A projected DEM is better."
        )

    print("[3/4] Building random subdivision path...")
    levels = build_square_levels(
        points=points,
        min_size=min_size,
        subdivisions_per_side=subdivisions_per_side,
        random_seed=random_seed,
    )

    print(f"Generated {len(levels)} subdivision levels.")

    print("[4/4] Reading cropped DEM windows and saving plots...")
    for level_index, (plot_window, squares) in enumerate(levels):
        dem, extent = read_dem_crop_for_plot_window(
            dem_path=dem_path,
            window=plot_window,
            max_pixels_per_side=1600,
        )

        output_path = output_dir / f"level_{level_index:02d}.png"

        plot_level(
            dem=dem,
            extent=extent,
            points=points,
            squares=squares,
            level_index=level_index,
            output_path=output_path,
        )

        print(f"Saved {output_path}")
        
    print("Hierarchical pruner done.")

    return output_dir, levels


def main() -> None:

    base_dir = Path(__file__).resolve().parent.parent.parent

    dem_path = base_dir / "GeoTIFF" / "sicily_cop30_utm33.tif"
    candidates_path = base_dir / "GSV" / "Libro4.csv"

    x_col = "lon"
    y_col = "lat"

    # Use "EPSG:4326" if the CSV points are longitude/latitude.
    # Use None if the CSV points are already in the same CRS as the DEM.
    candidate_crs = "EPSG:4326"

    output_dir = base_dir / "Results" / "hierarchical_subdivision_demo"

    initial_size = 1000.0
    min_size = 50.0
    subdivisions_per_side = 3

    random_seed = 42 #seed so 'random' behaviour is repeatable for debugging.

    run_hierarchical_pruner_from_gui(
        dem_path=dem_path,
        candidates_path=candidates_path,
        x_col=x_col,
        y_col=y_col,
        candidate_crs=candidate_crs,
        output_dir=output_dir,
        initial_size=initial_size,
        min_size=min_size,
        subdivisions_per_side=subdivisions_per_side,
        random_seed=random_seed,
    )

