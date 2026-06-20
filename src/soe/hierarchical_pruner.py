

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


def subdivide_square(
    square: Square,
    points: np.ndarray,
    subdivisions_per_side: int,
) -> list[Square]:
    """
    Split one square into smaller child squares.

    Example:
        subdivisions_per_side = 3

    means:
        1 parent square becomes 3 x 3 = 9 child squares.

    Empty child squares are ignored.
    """

    if subdivisions_per_side <= 1:
        raise ValueError("subdivisions_per_side must be greater than 1.") #avoids infinite loop.

    child_size = square.size / subdivisions_per_side

    children = []

    for row in range(subdivisions_per_side):
        for col in range(subdivisions_per_side):
            xmin = square.xmin + col * child_size
            ymin = square.ymin + row * child_size

            child = count_points_in_square(
                points=points,
                xmin=xmin,
                ymin=ymin,
                size=child_size,
            )

            if child.n_points > 0:
                children.append(child)

    return children


def build_square_levels(
    points: np.ndarray,
    initial_size: float,
    min_size: float,
    subdivisions_per_side: int,
    random_seed: int | None = None,
) -> list[list[Square]]:
    """
    Build a single random zoom path through the hierarchy.

    Level 0: randomly chosen non-empty starting square.
    Level 1: randomly chosen non-empty child of level 0.
    Level 2: randomly chosen non-empty child of level 1.
    And so on until the next square size would be smaller than min_size.

    Later, choose_next_square_random() can be replaced with a real heuristic.
    """
    if initial_size <= 0:
        raise ValueError("initial_size must be greater than 0.")

    if min_size <= 0:
        raise ValueError("min_size must be greater than 0.")

    if initial_size < min_size:
        raise ValueError("initial_size must be greater than or equal to min_size.")

    if subdivisions_per_side <= 1:
        raise ValueError("subdivisions_per_side must be greater than 1.") #prevent an infinite loop.

    rng = Random(random_seed)

    initial_squares = make_initial_squares(
        points=points,
        square_size=initial_size,
    )

    if not initial_squares:
        raise ValueError("No non-empty initial squares were created.")

    selected_square = choose_next_square_random(
        squares=initial_squares,
        rng=rng,
    ) #randomly select a blob to zoom into.

    levels = [[selected_square]] #a list of lists.

    while True:
        next_size = selected_square.size / subdivisions_per_side

        if next_size < min_size:
            print("Stopping because selected square produced no children.")
            break

        children = subdivide_square(
            square=selected_square,
            points=points,
            subdivisions_per_side=subdivisions_per_side,
        ) #find children of square.

        if not children:
            break

        selected_square = choose_next_square_random(
            squares=children,
            rng=rng,
        )

        levels.append([selected_square]) #append to square.

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
    """
    Save one PNG showing one subdivision level.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(dem, extent=extent, origin="upper")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        marker="o",
        label="Candidate viewpoints",
    )

    for square in squares:
        draw_square(ax, square)

    square_size = squares[0].size if squares else 0

    ax.set_title(
        f"Subdivision level {level_index} | "
        f"square size = {square_size:.2f} | "
        f"non-empty squares = {len(squares)}"
    )

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")

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
    initial_size: float,
    min_size: float,
    subdivisions_per_side: int,
    random_seed: int | None = 42,
) -> tuple[Path, list[list[Square]]]:
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

    print("[1/5] Reading DEM CRS...")
    dem_crs = read_dem_crs(dem_path)
    print(f"DEM CRS: {dem_crs}")

    print("[2/5] Reprojecting GUI candidate viewpoints...")
    points = reproject_points_if_needed(
        points=points,
        candidate_crs="EPSG:4326",
        dem_crs=dem_crs,
    )

    print("[3/5] Reading cropped DEM...")
    dem, extent = read_square_dem_crop_for_plotting(
        dem_path=dem_path,
        points=points,
        padding=1000.0,
    )

    if dem_crs is not None and dem_crs.is_geographic:
        print(
            "WARNING: DEM CRS appears to be geographic, so initial_size/min_size "
            "are probably in degrees, not metres. A projected DEM is better."
        )

    print("[4/5] Building random subdivision path...")
    levels = build_square_levels(
        points=points,
        initial_size=initial_size,
        min_size=min_size,
        subdivisions_per_side=subdivisions_per_side,
        random_seed=random_seed,
    )

    print(f"Generated {len(levels)} subdivision levels.")

    print("[5/5] Saving level PNGs...")
    for level_index, squares in enumerate(levels):
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

if __name__ == "__main__":
    main()
