"""Prepare and score fixed-camera layouts against independent Unreal traces."""
import argparse
import csv
import json
import math
import uuid
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def select(points, count, mode, rng):
    if mode == "random":
        return rng.choice(len(points), count, replace=False)
    first = int(rng.integers(len(points)))
    distance = np.sum((points - points[first]) ** 2, axis=1)
    if mode == "clustered":
        return np.argsort(distance, kind="stable")[:count]
    chosen = [first]
    while len(chosen) < count:
        distance[chosen] = -1
        candidates = np.flatnonzero(distance == distance.max())
        nxt = int(rng.choice(candidates))
        chosen.append(nxt)
        distance = np.minimum(distance, np.sum((points - points[nxt]) ** 2, axis=1))
    return chosen


def prepare(args, root, layout_builder=None, checkpoint=lambda: None):
    from affine import Affine
    from rivelero.visibility.directional import apply_horizontal_fov
    from rivelero.visibility.viewshed import open_viewshed_dem, run_viewshed, accumulate_viewshed

    dem = open_viewshed_dem(args.dem)
    band = dem.GetRasterBand(1)
    transform = Affine.from_gdal(*dem.GetGeoTransform())
    crs = dem.GetSpatialRef()
    if crs and (crs.IsGeographic() or not math.isclose(crs.GetLinearUnits(), 1)):
        raise ValueError("Provide a metre-based projected or local DEM.")
    if transform.b or transform.d or transform.a <= 0 or transform.e >= 0:
        raise ValueError("Provide a north-up DEM with positive eastward spacing.")
    heights = band.ReadAsArray().astype("<f4")
    if not np.isfinite(heights).all() or not band.GetMaskBand().ReadAsArray().all():
        raise ValueError("The experiment requires a complete DEM without missing cells.")
    h, w = heights.shape
    if min(h, w) < 20 or not 1 <= args.cameras <= 100 or args.replicates < 1:
        raise ValueError("Need at least 20x20 cells, 1–100 cameras and positive replicates.")
    if not math.isfinite(args.camera_height_m) or args.camera_height_m <= 0:
        raise ValueError("Camera height must be finite and positive.")
    if not math.isfinite(args.distance) or args.distance <= 0:
        raise ValueError("Distance must be positive and finite.")
    # Ten stratified rows and columns over the central half of each dimension.
    rows = ((0.25 + (np.arange(10) + 0.5) / 20) * h).astype(int)
    cols = ((0.25 + (np.arange(10) + 0.5) / 20) * w).astype(int)
    centre_x, centre_y = transform * (w / 2, h / 2)
    viewpoints = []
    (root / "predicted").mkdir()
    for row in rows:
        for col in cols:
            checkpoint()
            x, y = transform * (col + 0.5, row + 0.5)
            if (x, y) == (centre_x, centre_y):
                raise ValueError("Viewpoint at exact centre has undefined heading.")
            yaw = math.degrees(math.atan2(centre_x - x, centre_y - y)) % 360
            vid = f"v{len(viewpoints):03}"
            viewpoints.append(dict(viewpoint=vid, row=int(row), col=int(col),
                                   east_m=x, north_m=y, elevation_m=heights[row, col].item(),
                                   heading_deg=yaw))
            view = run_viewshed(band=band, observer_x=x, observer_y=y,
                                observer_height_m=args.camera_height_m, target_height_m=0.5,
                                max_distance_m=args.distance, curvature_coefficient=0.0)
            full = np.zeros((h, w), dtype=np.uint8)
            accumulate_viewshed(accumulator=full, viewshed_dataset=view, dem_transform=transform)
            mask = apply_horizontal_fov(full, observer_x=x, observer_y=y, transform=transform,
                                        heading_deg=yaw, horizontal_fov_deg=90)
            np.save(root / "predicted" / vid, mask)
    write_csv(root / "viewpoints.csv", viewpoints)
    heights.tofile(root / "terrain.f32")
    rng = np.random.default_rng(args.seed)
    points = np.asarray([(v["east_m"], v["north_m"]) for v in viewpoints])
    layouts = []
    if layout_builder is not None:
        layouts = layout_builder(root, viewpoints, points, rng)
    else:
        for mode in ("random", "dispersed", "clustered"):
            for rep in range(1, args.replicates + 1):
                for i in select(points, args.cameras, mode, rng):
                    layouts.append(dict(run=f"{mode}-{rep:03}", viewpoint=viewpoints[i]["viewpoint"]))
    write_csv(root / "layouts.csv", layouts)
    config = dict(batch=uuid.uuid4().hex, dem=str(args.dem.resolve()), shape=[h, w],
                  west_edge=transform.c, north_edge=transform.f,
                  east_step=transform.a, south_step=-transform.e, distance=args.distance,
                  cameras=args.cameras, replicates=args.replicates, seed=args.seed,
                  observer_m=args.camera_height_m, target_m=0.5, fov_deg=90)
    (root / "batch.json").write_text(json.dumps(config, indent=2))


def compare(root, threshold, checkpoint=lambda: None):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one.")
    config = json.loads((root / "batch.json").read_text())
    with (root / "layouts.csv").open() as stream:
        layouts = {}
        for row in csv.DictReader(stream):
            layouts.setdefault(row["run"], []).append(row["viewpoint"])
    shape = tuple(config["shape"])
    observed = root / "unreal" / config["batch"]
    scores, summaries = [], []
    for run, ids in layouts.items():
        checkpoint()
        unions = [np.zeros(shape, bool), np.zeros(shape, bool)]
        passed = True
        for vid in ids:
            pred = np.load(root / "predicted" / f"{vid}.npy", allow_pickle=False)
            if pred.shape != shape or pred.dtype != bool:
                raise ValueError(f"Invalid prediction for {vid}")
            path = observed / run / f"{vid}.z"
            if not path.exists():
                raise FileNotFoundError(f"Unfinished Unreal batch: {path}")
            mask = np.frombuffer(zlib.decompress(path.read_bytes()), np.uint8).reshape(shape)
            if mask.max() > 1:
                raise ValueError(f"Nonbinary Unreal mask: {path}")
            hit = mask.astype(bool)
            intersection = np.count_nonzero(pred & hit)
            union = np.count_nonzero(pred | hit)
            iou = intersection / union if union else 0.0
            passed &= iou > threshold
            scores.append(dict(run=run, viewpoint=vid, iou=iou,
                               rivelero_only=np.count_nonzero(pred & ~hit),
                               unreal_only=np.count_nonzero(hit & ~pred)))
            unions[0] |= pred
            unions[1] |= hit
        a, b = unions
        u = np.count_nonzero(a | b)
        combined = np.count_nonzero(a & b) / u if u else 0.0
        summaries.append(dict(run=run, cameras=len(ids), union_iou=combined,
                              threshold=threshold, accepted=bool(passed and combined > threshold)))
    write_csv(root / "camera_scores.csv", scores)
    write_csv(root / "run_scores.csv", summaries)
    rejected = sum(not row["accepted"] for row in summaries)
    print(f"{len(summaries) - rejected} accepted, {rejected} rejected")
    return int(bool(rejected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("dem", type=Path)
    prep.add_argument("out", type=Path)
    prep.add_argument("--camera-height-m", type=float, default=1.75)
    prep.add_argument("--distance", required=True, type=float)
    prep.add_argument("--cameras", type=int, default=20)
    prep.add_argument("--replicates", type=int, default=100)
    prep.add_argument("--seed", type=int, default=0)
    score = commands.add_parser("compare")
    score.add_argument("out", type=Path)
    score.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        return compare(args.out, args.threshold)
    if args.out.exists():
        parser.error("Choose a new batch directory.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "batch"
        stage.mkdir()
        prepare(args, stage)
        stage.rename(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
