

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Allows saving PNGs from terminal without opening a window.

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from rasterio.warp import transform as transform_coords


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


def read_dem_for_plotting(
    dem_path: str | Path,
    max_pixels_per_side: int = 1600,
):
    """
    Read a DEM band for background plotting.

    Large DEMs are downsampled for display so PNG generation stays fast.
    """

    dem_path = Path(dem_path)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        if src.count < 1:
            raise ValueError("DEM has no raster bands.")

        scale = max(
            src.width / max_pixels_per_side,
            src.height / max_pixels_per_side,
            1,
        )

        out_width = int(src.width / scale)
        out_height = int(src.height / scale)

        dem = src.read(
            1,
            out_shape=(out_height, out_width),
            masked=True,
        )

        # Real-world map coordinates for plotting the DEM with imshow().
        extent = plotting_extent(src)
        crs = src.crs

    return dem, extent, crs


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
        raise ValueError("subdivisions_per_side must be greater than 1.")

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
) -> list[list[Square]]:
    """
    Build every subdivision level.

    Level 0:
        largest non-empty squares

    Level 1:
        subdivisions of level 0 squares

    Level 2:
        subdivisions of level 1 squares

    and so on until the next square size would be smaller than min_size.
    """

    if initial_size <= 0:
        raise ValueError("initial_size must be greater than 0.")

    if min_size <= 0:
        raise ValueError("min_size must be greater than 0.")

    if initial_size < min_size:
        raise ValueError("initial_size must be greater than or equal to min_size.")

    if subdivisions_per_side <= 1:
        raise ValueError("subdivisions_per_side must be greater than 1.")

    levels = []

    current_level = make_initial_squares(
        points=points,
        square_size=initial_size,
    )

    if not current_level:
        raise ValueError("No non-empty initial squares were created.")

    levels.append(current_level)

    while current_level:
        current_size = current_level[0].size
        next_size = current_size / subdivisions_per_side

        if next_size < min_size:
            break

        next_level = []

        for square in current_level:
            children = subdivide_square(
                square=square,
                points=points,
                subdivisions_per_side=subdivisions_per_side,
            )

            next_level.extend(children)

        if not next_level:
            break

        levels.append(next_level)
        current_level = next_level

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


def run_demo(
    *,
    dem_path: str | Path,
    candidates_path: str | Path,
    x_col: str,
    y_col: str,
    candidate_crs: str | None,
    output_dir: str | Path,
    initial_size: float,
    min_size: float,
    subdivisions_per_side: int,
) -> None:
    """
    Run the simple hierarchical subdivision demo.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Reading DEM...")
    dem, extent, dem_crs = read_dem_for_plotting(dem_path)

    print("[2/4] Reading candidate viewpoints...")
    points = load_candidate_points_csv(
        csv_path=candidates_path,
        x_col=x_col,
        y_col=y_col,
    )

    print(f"Loaded {len(points)} candidate viewpoints.")

    print("[3/4] Reprojecting candidate viewpoints if needed...")
    points = reproject_points_if_needed(
        points=points,
        candidate_crs=candidate_crs,
        dem_crs=dem_crs,
    )

    if dem_crs is not None and dem_crs.is_geographic:
        print(
            "WARNING: DEM CRS appears to be geographic, so initial_size/min_size "
            "are probably in degrees, not metres. A projected DEM is better."
        )

    print("[4/4] Building subdivision levels...")
    levels = build_square_levels(
        points=points,
        initial_size=initial_size,
        min_size=min_size,
        subdivisions_per_side=subdivisions_per_side,
    )

    print(f"Generated {len(levels)} subdivision levels.")

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

    print("Done.")


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent.parent

    dem_path = base_dir / "GeoTIFF" / "sicily_cop30_utm33.tif"
    candidates_path = base_dir / "GSV" / "Libro1.csv"

    x_col = "lon"
    y_col = "lat"

    # Use "EPSG:4326" if the CSV points are longitude/latitude.
    # Use None if the CSV points are already in the same CRS as the DEM.
    candidate_crs = "EPSG:4326"

    output_dir = base_dir / "Results" / "hierarchical_subdivision_demo"

    initial_size = 1000.0
    min_size = 50.0
    subdivisions_per_side = 3

    run_demo(
        dem_path=dem_path,
        candidates_path=candidates_path,
        x_col=x_col,
        y_col=y_col,
        candidate_crs=candidate_crs,
        output_dir=output_dir,
        initial_size=initial_size,
        min_size=min_size,
        subdivisions_per_side=subdivisions_per_side,
    )


if __name__ == "__main__":
    main()
