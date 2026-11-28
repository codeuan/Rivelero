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


def plant_replicates(root, settings, cfg, region, seen, checkpoint):
    ys, xs = region
    counts = []
    seeds = np.random.SeedSequence([settings.seed, 2]).spawn(settings.replicates)
    with (root / "plants.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["replicate", "plant", "row", "col", "east_m", "north_m", "detected"])
        for rep, seed in enumerate(seeds, 1):
            checkpoint()
            cells = sample_plants(seen.shape, settings.plants, (cfg["south_step"], cfg["east_step"]),
                                  settings.scale_m, settings.contrast, np.random.default_rng(seed))
            detected = int(seen[cells[:, 0], cells[:, 1]].sum())
            for plant, (r, c) in enumerate(cells, 1):
                row, col = r+ys.start, c+xs.start
                east = cfg["west_edge"] + (col+0.5)*cfg["east_step"]
                north = cfg["north_edge"] - (row+0.5)*cfg["south_step"]
                writer.writerow([rep, plant, row, col, east, north, int(seen[r, c])])
            counts.append(dict(replicate=rep, plants=settings.plants, detected=detected,
                               missed=settings.plants-detected))
    return counts


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
            results = plant_replicates(root, settings, cfg, region, central > 0, checkpoint)
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
