"""Run inside the UE5 editor with the DEM landscape level open. See visibility_validation.md."""
import array
import csv
import json
import math
import sys
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import unreal


def trace(camera, start, end):
    # Stock UE Visibility channel. Only the terrain should block this channel.
    return unreal.SystemLibrary.line_trace_single(
        camera, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True,
        [camera], unreal.DrawDebugTrace.NONE, True)


def run(batch_dir, northwest_cm, datum_z_cm=0):
    """NW is the XY of the north-west CELL CENTRE; world +X north, +Y east."""
    root = Path(batch_dir)
    cfg = json.loads((root / "batch.json").read_text())
    h, w = cfg["shape"]
    heights = array.array("f")
    heights.frombytes((root / "terrain.f32").read_bytes())
    if sys.byteorder != "little":
        heights.byteswap()
    if len(heights) != h * w:
        raise ValueError("Terrain dimensions do not match batch.")
    with (root / "viewpoints.csv").open() as stream:
        viewpoints = {row["viewpoint"]: row for row in csv.DictReader(stream)}
    with (root / "layouts.csv").open() as stream:
        layouts = {}
        for row in csv.DictReader(stream):
            layouts.setdefault(row["run"], []).append(row["viewpoint"])
    result = root / "unreal" / cfg["batch"]
    result.mkdir(parents=True, exist_ok=True)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    step_x, step_y = cfg["south_step"] * 100, cfg["east_step"] * 100
    radius = cfg["distance"] * 100
    north, west = northwest_cm
    for name, ids in layouts.items():
        if (result / name).is_dir():
            continue
        cameras = {}
        try:
            for vid in ids:
                v = viewpoints[vid]
                r, c = int(v["row"]), int(v["col"])
                ground = datum_z_cm + float(v["elevation_m"]) * 100
                origin = unreal.Vector(north - r * step_x, west + c * step_y,
                                       ground + cfg["observer_m"] * 100)
                yaw = float(v["heading_deg"])
                camera = actors.spawn_actor_from_class(
                    unreal.CameraActor, origin, unreal.Rotator(yaw=yaw), transient=True)
                if camera is None:
                    raise RuntimeError(f"Could not spawn {vid}")
                cameras[vid] = camera
                camera.set_actor_label(vid)
                lens = camera.get_component_by_class(unreal.CameraComponent)
                lens.set_field_of_view(cfg["fov_deg"])
                eye = lens.get_world_location()
                forward = lens.get_forward_vector()
                actual = math.degrees(math.atan2(forward.y, forward.x))
                if (math.dist((eye.x, eye.y, eye.z), (origin.x, origin.y, origin.z)) > 0.1
                        or abs((actual - yaw + 180) % 360 - 180) > 0.01
                        or abs(forward.z) > 0.0002
                        or abs(lens.field_of_view - cfg["fov_deg"]) > 0.01):
                    raise RuntimeError(f"Camera pose/FOV mismatch: {vid}")
                above = unreal.Vector(eye.x, eye.y, ground + 5)
                below = unreal.Vector(eye.x, eye.y, ground - 5)
                if trace(camera, above, below) is None:
                    raise RuntimeError(f"No terrain within 5 cm of expected ground at {vid}")
            with TemporaryDirectory(dir=result) as tmp:
                stage = Path(tmp) / name
                stage.mkdir()
                for vid, camera in cameras.items():
                    lens = camera.get_component_by_class(unreal.CameraComponent)
                    eye = lens.get_world_location()
                    forward = lens.get_forward_vector()
                    cosine = math.cos(math.radians(lens.field_of_view / 2))
                    r, c = int(viewpoints[vid]["row"]), int(viewpoints[vid]["col"])
                    dr, dc = math.ceil(radius / step_x), math.ceil(radius / step_y)
                    mask = bytearray(h * w)
                    for row in range(max(0, r-dr), min(h, r+dr+1)):
                        x = north - row * step_x
                        for col in range(max(0, c-dc), min(w, c+dc+1)):
                            y = west + col * step_y
                            dx, dy = x - eye.x, y - eye.y
                            distance = math.hypot(dx, dy)
                            if distance > radius or dx*forward.x + dy*forward.y < distance*(cosine-1e-7):
                                continue
                            index = row*w + col
                            z = datum_z_cm + (heights[index] + cfg["target_m"]) * 100
                            end = unreal.Vector(x, y, z)
                            mask[index] = trace(camera, eye, end) is None
                    (stage / f"{vid}.z").write_bytes(zlib.compress(mask))
                stage.rename(result / name)
            unreal.log(f"Visibility masks complete: {name}")
        finally:
            for camera in cameras.values():
                actors.destroy_actor(camera)
