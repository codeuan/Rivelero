"""Scientific designs and orchestration for the three desktop experiment tabs."""
import csv
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .plants import sample_plants
from .visibility_validation import compare, prepare, select, write_csv


@dataclass(frozen=True)
class Settings:
    experiment: int
    dem: Path
    cameras: int
    replicates: int
    seed: int
    distance: float
    threshold: float
    plants: int = 100
    scale_m: float = 100
    contrast: float = 5
    layout: str = "random"
    camera_height_m: float = 1.75


def domain(shape):
    """Cell centres in the central half of each physical dimension."""
    bounds = []
    for n in shape:
        indices = np.flatnonzero((np.arange(n)+0.5 >= n/4) & (np.arange(n)+0.5 < 3*n/4))
        bounds.append(slice(indices[0], indices[-1]+1))
    return tuple(bounds)


def design(settings, root, viewpoints, points, rng):
    def entries(name, indices):
        return [dict(run=name, viewpoint=viewpoints[i]["viewpoint"]) for i in indices]

    n = settings.cameras
    if settings.experiment == 2:
        return entries("fixed", select(points, n, settings.layout, rng))
    rows = []
    if settings.experiment == 1:
        masks = [np.load(root / "predicted" / (v["viewpoint"]+".npy")) for v in viewpoints]
        region = domain(masks[0].shape)
        masks = np.stack([mask[region] for mask in masks])
        order = np.argsort(-masks.sum(axis=(1, 2)), kind="stable")
        rows.extend(entries("individual", order[:n]))
        seen = np.zeros(masks.shape[1:], bool)
        chosen = []
        for _ in range(n):
            gains = (masks & ~seen).sum(axis=(1, 2))
            gains[chosen] = -1
            best = int(gains.argmax())
            chosen.append(best)
            seen |= masks[best]
        rows.extend(entries("greedy", chosen))
        modes = ["random"]
    else:
        modes = ["random", "dispersed", "clustered"]
    for mode in modes:
        for rep in range(1, settings.replicates+1):
            rows.extend(entries(f"{mode}-{rep:03}", select(points, n, mode, rng)))
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_planting_list(root, settings, cfg, checkpoint):
    ys, xs = domain(cfg["shape"])
    shape = (ys.stop-ys.start, xs.stop-xs.start)
    terrain = np.memmap(root / "terrain.f32", dtype="<f4", mode="r", shape=tuple(cfg["shape"]))
    seeds = np.random.SeedSequence([settings.seed, 2]).spawn(settings.replicates)
    with (root / "planting_list.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["replicate", "plant", "row", "col", "east_m", "north_m", "ground_m"])
        for rep, seed in enumerate(seeds, 1):
            checkpoint()
            cells = sample_plants(shape, settings.plants, (cfg["south_step"], cfg["east_step"]),
                                  settings.scale_m, settings.contrast, np.random.default_rng(seed))
            for plant, (r, c) in enumerate(cells, 1):
                row, col = r+ys.start, c+xs.start
                east = cfg["west_edge"] + (col+0.5)*cfg["east_step"]
                north = cfg["north_edge"] - (row+0.5)*cfg["south_step"]
                writer.writerow([rep, plant, row, col, east, north, float(terrain[row, col])])


def score_plants(root, settings, seen, checkpoint):
    audit_path = root / "plant_validation.csv"
    if not audit_path.is_file():
        raise RuntimeError("Missing Unreal plant validation. Restart the updated UE experiment server.")
    expected = read_csv(root / "planting_list.csv")
    audit = read_csv(audit_path)
    def key(row):
        return row["replicate"], row["plant"]
    keys = {key(row) for row in expected}
    if (len(keys) != settings.replicates * settings.plants or len(audit) != len(keys)
            or {key(row) for row in audit if row["verified"] == "True"} != keys):
        raise RuntimeError("Unreal did not verify every expected Othonna exactly once.")
    totals = [0] * settings.replicates
    for row in expected:
        if row["plant"] == "1":
            checkpoint()
        row["detected"] = int(seen[int(row["row"]), int(row["col"])] > 0)
        totals[int(row["replicate"])-1] += row["detected"]
    write_csv(root / "plants.csv", expected)
    return [dict(replicate=i, plants=settings.plants, detected=n, missed=settings.plants-n)
            for i, n in enumerate(totals, 1)]


def analyse(root, settings, checkpoint):
    cfg = json.loads((root / "batch.json").read_text())
    region = domain(cfg["shape"])
    layouts = {}
    for item in read_csv(root / "layouts.csv"):
        layouts.setdefault(item["run"], []).append(item["viewpoint"])
    scores = read_csv(root / "run_scores.csv")
    results = []
    for score in scores:
        checkpoint()
        if score["accepted"] != "True":
            continue
        name = score["run"]
        exposure = np.zeros(cfg["shape"], dtype=np.uint16)
        central = exposure[region]
        for k, vid in enumerate(layouts[name], 1):
            exposure += np.load(root / "predicted" / f"{vid}.npy")
            if settings.experiment == 1:
                inside = np.count_nonzero(central)
                results.append(dict(layout=name, cameras=k, added=vid, visible_inside=inside,
                                    visible_outside=np.count_nonzero(exposure)-inside,
                                    fraction_inside=inside/central.size))
        if settings.experiment == 2:
            results = score_plants(root, settings, exposure, checkpoint)
        if settings.experiment == 3:
            results.append(dict(layout=name, blind=np.count_nonzero(central == 0),
                                unique=np.count_nonzero(central == 1), repeated=np.count_nonzero(central >= 2),
                                mean_exposure=central.mean(), max_exposure=central.max()))
    if results:
        write_csv(root / "results.csv", results)
    return sum(row["accepted"] == "True" for row in scores), len(scores)


def run_experiment(settings, shared_root, endpoint, *, task_context):
    checkpoint = task_context.cancellation_token.raise_if_cancelled
    job = uuid.uuid4().hex
    root = Path(shared_root) / job
    root.mkdir(parents=True)
    previous = None

    def status(message):
        nonlocal previous
        if message != previous:
            task_context.progress(0, None, None, message)
            previous = message

    try:
        status("Computing Rivelero masks and selecting cameras")
        prepare(settings, root,
                lambda *args: design(settings, *args), checkpoint)
        cfg = json.loads((root / "batch.json").read_text())
        ry, rx = domain(cfg["shape"])
        if settings.experiment == 2 and settings.plants > (ry.stop-ry.start)*(rx.stop-rx.start):
            raise ValueError("Plant count exceeds the number of distinct analysis cells.")
        if settings.experiment == 2:
            write_planting_list(root, settings, cfg, checkpoint)
        (root / "request.json").write_text(json.dumps(asdict(settings), default=str, indent=2))
        request = Request(f"{endpoint.rstrip('/')}/jobs/{job}", data=b"", method="POST")
        while True:
            checkpoint()
            try:
                with urlopen(request, timeout=3) as response:
                    state = json.load(response)["status"]
            except HTTPError as err:
                raise RuntimeError(f"UE server rejected batch ({err.code}: {err.reason}). Check shared folder.") from err
            except (URLError, TimeoutError, ConnectionError):
                state = f"Waiting for Unreal server at {endpoint}"
            if state == "Complete":
                break
            if state.startswith("Error:") or state == "Cancelled":
                raise RuntimeError(state)
            status(state)
            time.sleep(0.5)
        status("Comparing masks and computing experiment results")
        compare(root, settings.threshold, checkpoint)
        accepted, total = analyse(root, settings, checkpoint)
        return f"{accepted}/{total} layouts accepted. Results: {root}"
    except BaseException:
        (root / "cancelled").touch()
        raise
