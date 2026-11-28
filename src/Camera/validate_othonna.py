"""Validate transient Othonna static meshes in the registered UE landscape."""
import csv
import json
import math
from itertools import groupby

import unreal


def validate_plants(root, northwest_cm, datum_z_cm, asset_path, request):
    cfg = json.loads((root / "batch.json").read_text())
    mesh = unreal.load_asset(asset_path) if asset_path else None
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError("Configure othonna_mesh with the project's Othonna Static Mesh asset path.")
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    tolerance_cm = 5.0
    with (root / "planting_list.csv").open(newline="") as source, (
            root / "plant_validation.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["replicate", "plant", "expected_x_cm", "expected_y_cm", "expected_z_cm",
                         "actual_x_cm", "actual_y_cm", "actual_z_cm", "error_cm", "verified"])
        completed = 0
        for rep, group in groupby(csv.DictReader(source), key=lambda r: int(r["replicate"])):
            rows = list(group)
            if rep != completed + 1 or len(rows) != request["plants"]:
                raise ValueError(f"Incomplete planting list for replicate {rep}")
            actors = []
            expected = {}
            try:
                for row in rows:
                    plant = int(row["plant"])
                    label = f"Rivelero_{cfg['batch']}_{rep}_Othonna_{plant:03d}"
                    if label in expected or not 1 <= plant <= request["plants"]:
                        raise ValueError(f"Invalid plant ID: {label}")
                    x = northwest_cm[0] - (cfg["north_edge"] - float(row["north_m"])
                                            - cfg["south_step"]/2)*100
                    y = northwest_cm[1] + (float(row["east_m"]) - cfg["west_edge"]
                                            - cfg["east_step"]/2)*100
                    z = datum_z_cm + float(row["ground_m"])*100
                    expected[label] = (plant, (x, y, z))
                    # Spawn from raster indices; verify independently against exported map coordinates.
                    spawn_x = northwest_cm[0] - int(row["row"])*cfg["south_step"]*100
                    spawn_y = northwest_cm[1] + int(row["col"])*cfg["east_step"]*100
                    actor = subsystem.spawn_actor_from_object(
                        mesh, unreal.Vector(spawn_x, spawn_y, z), transient=True)
                    if actor is None:
                        raise RuntimeError(f"Could not spawn {label}")
                    actors.append(actor)
                    actor.set_actor_label(label)
                # Read back from the level after the entire replicate has been spawned.
                found = set()
                for actor in subsystem.get_all_level_actors():
                    label = actor.get_actor_label()
                    if label not in expected:
                        continue
                    if label in found:
                        raise RuntimeError(f"Duplicate Othonna actor: {label}")
                    found.add(label)
                    plant, intended = expected[label]
                    component = actor.get_component_by_class(unreal.StaticMeshComponent)
                    if component is None or component.get_editor_property("static_mesh") != mesh:
                        raise RuntimeError(f"Incorrect Othonna mesh: {label}")
                    pos = component.get_world_location()
                    actual = (pos.x, pos.y, pos.z)
                    error = math.dist(actual, intended)
                    ground = unreal.SystemLibrary.line_trace_single(
                        actor, unreal.Vector(pos.x, pos.y, pos.z+tolerance_cm),
                        unreal.Vector(pos.x, pos.y, pos.z-tolerance_cm),
                        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, actors,
                        unreal.DrawDebugTrace.NONE, True)
                    verified = error <= tolerance_cm and ground is not None
                    writer.writerow([rep, plant, *intended, *actual, error, verified])
                    if not verified:
                        raise RuntimeError(f"{label}: placement or ground check failed ({error:.2f} cm XYZ error).")
                if found != expected.keys():
                    raise RuntimeError(f"Missing Othonna actors: {sorted(expected.keys()-found)}")
            finally:
                for actor in actors:
                    subsystem.destroy_actor(actor)
            output.flush()
            completed += 1
            yield f"Verified Othonna replicate {rep}", f"{len(rows)} plants"
        if completed != request["replicates"]:
            raise RuntimeError("Missing Othonna replicates in planting list.")
