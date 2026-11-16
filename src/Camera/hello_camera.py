"""Rivelero case study: switch between three fixed Project AirSim cameras.

Run from your existing Project AirSim client environment after starting Unreal.
The scene must contain Camera1, Camera2 and Camera3, each with a DownCamera
sensor capturing segmentation (3) and custom Rivelero X-ray (6).

Controls (focus the Windows CMD / PowerShell terminal):
    Left arrow:  previous camera
    Right arrow: next camera
    Q:           quit

Camera locations are now loaded from cameralocations.csv, so the script can
create and switch between any number of fixed cameras in CSV row order.
"""

import csv
import msvcrt
import time
from pathlib import Path

import cv2
import numpy as np
import commentjson

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.types import ImageType
from projectairsim.utils import unpack_image
from projectairsim.image_utils import SEGMENTATION_PALLETE

import threading
from queue import SimpleQueue, Empty

SENSOR_NAME = "DownCamera"
XRAY_IMAGE_TYPE = 6  # Rivelero's modified Project AirSim build
TARGET_CSV = Path(__file__).resolve().parent / "flightpath.csv"
WINDOW_NAME = "Rivelero | Segmentation + X-ray + RGB"

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "sim_config"

CSV_PATH = BASE_DIR / "cameralocations.csv"
SOURCE_SCENE = CONFIG_DIR / "scene_basic_drone.jsonc"
GENERATED_SCENE = CONFIG_DIR / "scene_cameras_generated.jsonc"
SCENE_CONFIG = GENERATED_SCENE.name
MAX_TELEMETRY_ROWS = 15
TELEMETRY_ROW_HEIGHT = 24

# The Project AirSim palette is RGB; as_bgr() gives images ready for OpenCV's BGR display.
SEGMENTATION_BGR_PALETTE = np.asarray(SEGMENTATION_PALLETE, dtype=np.uint8)[:, ::-1]


def generate_camera_scene():
    """Replace the template scene's actors with cameras from the CSV."""
    with SOURCE_SCENE.open("r", encoding="utf-8") as file:
        scene = commentjson.load(file)

    actors = []
    camera_names = []
    camera_positions = {}

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"x_m", "y_m", "z_m"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            camera_name = f"Camera{len(actors) + 1}"

            try:
                x = float(row["x_m"])
                y = float(row["y_m"])
                z = float(row["z_m"])
                yaw = float(row.get("yaw_deg") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid camera coordinates at CSV line {line_number}: {row}"
                ) from exc

            actor = {
                "type": "robot",
                "name": camera_name,
                "origin": {
                    "xyz": f"{x} {y} {z}",
                    "rpy-deg": f"0 0 {yaw}",
                },
                "robot-config": "robot_camera_only.jsonc",
            }

            actors.append(actor)
            camera_names.append(camera_name)
            camera_positions[camera_name] = np.array([x, y, z], dtype=float)

    if not actors:
        raise ValueError("No camera positions found in CSV")

    scene["actors"] = actors

    with GENERATED_SCENE.open("w", encoding="utf-8") as file:
        commentjson.dump(scene, file, indent=2)

    print(f"Configured {len(actors)} cameras from {CSV_PATH.name}")
    return camera_names, camera_positions


def configure_plant_segmentation(world):
    """Retain the original script's per-Othonna segmentation assignments."""
    if not TARGET_CSV.exists():
        print(f"No {TARGET_CSV.name} found; using the scene's current segmentation IDs.")
        return {}

    with TARGET_CSV.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"{TARGET_CSV.name} must contain an 'id' column")
        plant_ids = [row["id"].strip() for row in reader if row["id"].strip()]

    plant_segmentation_ids = {}
    used_ids = set()

    # Same initialization approach as the old flying-drone case study.
    world.set_segmentation_id_by_name(r"[\w]*", 0, True, True)

    for plant_id in plant_ids:
        try:
            segmentation_id = int(plant_id.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Expected a plant ID such as Othonna_017: {plant_id}") from exc
        if not 1 <= segmentation_id <= 255:
            raise ValueError(f"Segmentation ID outside range 1..255: {plant_id}")

        if segmentation_id in used_ids:
            raise ValueError(f"Duplicate segmentation ID {segmentation_id} in {TARGET_CSV.name}")
        used_ids.add(segmentation_id)
        plant_segmentation_ids[plant_id] = segmentation_id

        found = world.set_segmentation_id_by_name(
            plant_id, segmentation_id, False, True
        )
        if not found:
            print(f"WARNING: Plant actor {plant_id!r} was not found in Unreal")

    print(f"Configured segmentation IDs for {len(plant_ids)} plant targets")
    return plant_segmentation_ids


def as_bgr(image):
    """Normalize Project AirSim images for OpenCV display."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) #convert single-channel grayscale to 3-channel BGR
    if image.shape[2] == 4:
        return image[:, :, :3].copy()
    if image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"Unsupported camera image shape: {image.shape}")


def keyboard_listener(commands):
    while True:
        key = msvcrt.getwch()

        if key in ("\x00", "\xe0"): #special keys.
            arrow = msvcrt.getwch() #arrow keys.

            if arrow == "K":
                commands.put(-1)

            elif arrow == "M":
                commands.put(1)

            elif arrow == "H":
                commands.put("scroll-up")

            elif arrow == "P":
                commands.put("scroll-down")

        elif key.lower() == "q" or key == "\x1b":
            commands.put("quit")
            return


def pack_bgr(image):
    """Encode exact BGR segmentation colours as integer keys for efficient counts."""
    if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 colour segmentation image, got {image.shape}/{image.dtype}")
    pixels = image[:, :, :3].astype(np.uint32)
    return (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]


def segmentation_color_key(segmentation_id):
    """Use Project AirSim's official segmentation-ID colour palette."""
    b, g, r = (int(c) for c in SEGMENTATION_BGR_PALETTE[segmentation_id])
    return (b << 16) | (g << 8) | r


def count_plant_pixels(image, plant_segmentation_ids):
    """Return plant pixel counts without resizing/interpolating segmentation colours."""
    values, counts = np.unique(pack_bgr(image), return_counts=True)
    frequency = dict(zip(values.tolist(), counts.tolist()))
    return {
        name: frequency.get(segmentation_color_key(seg_id), 0)
        for name, seg_id in plant_segmentation_ids.items()
    }


def get_plant_position(world, plant_name, cache):
    """Read plant origin once, caching failures as well as successful lookups."""
    if plant_name not in cache:
        try:
            pose = world.get_object_pose(plant_name)
            pos = pose["translation"]
            cache[plant_name] = np.array(
                [float(pos["x"]), float(pos["y"]), float(pos["z"])], dtype=float
            )
        except Exception as exc:
            print(f"WARNING: Could not get position for {plant_name!r}: {exc}")
            cache[plant_name] = None
    return cache[plant_name]


def make_telemetry_panel(camera_name, camera_position, plant_segmentation_ids,
                         visible_counts, possible_counts, world, plant_position_cache,
                         scroll_offset, image_width):
    """Draw live plant telemetry under the RGB / segmentation / X-ray images."""
    rows = [
        (name, visible_counts[name], possible_counts[name])
        for name in plant_segmentation_ids
        if visible_counts[name] or possible_counts[name]
    ]
    rows.sort(key=lambda row: (-row[2], -row[1], row[0]))

    shown_count = min(MAX_TELEMETRY_ROWS, len(rows))
    max_scroll = max(0, len(rows) - MAX_TELEMETRY_ROWS)
    scroll_offset = min(scroll_offset, max_scroll)

    panel_height = 144 + max(3, shown_count) * TELEMETRY_ROW_HEIGHT
    if len(rows) > MAX_TELEMETRY_ROWS:
        panel_height += 28
    panel = np.full((panel_height, image_width, 3), (25, 28, 31), dtype=np.uint8)

    def write(txt, x, y, scale=0.56, color=(240, 240, 240), bold=False):
        cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 2 if bold else 1, cv2.LINE_AA)

    write(f"LIVE PLANT TELEMETRY  |  {camera_name}", 20, 28, 0.74, bold=True)
    write(f"Plants in flightpath.csv: {len(plant_segmentation_ids)}   |   In image: {len(rows)}",
          20, 55, 0.51, (180, 190, 200))
    write("Distance = camera origin to Unreal plant actor origin (metres)",
          20, 76, 0.45, (180, 190, 200))

    invalid_xray = any(visible > possible for _, visible, possible in rows)
    if invalid_xray:
        status = "CHECK X-RAY: some visible plant pixels exceed theoretical pixels; occlusion unavailable for those plants"
        status_color = (80, 180, 255)
    else:
        status = "Occlusion = (theoretical - visible) / theoretical  |  -- = undefined"
        status_color = (165, 185, 205)
    write(status, 20, 98, 0.43, status_color)

    col_x = (20, 315, 475, 650, 850)
    for x, heading in zip(col_x, ("PLANT", "VISIBLE PX", "X-RAY PX", "OCCLUDED", "DISTANCE")):
        write(heading, x, 125, 0.51, (160, 235, 190), bold=True)
    cv2.line(panel, (18, 135), (image_width - 18, 135), (95, 95, 95), 1)

    for idx, (name, visible, possible) in enumerate(rows[scroll_offset:scroll_offset + MAX_TELEMETRY_ROWS]):
        y = 158 + idx * TELEMETRY_ROW_HEIGHT
        if possible == 0 or visible > possible:
            occluded = "--" if possible == 0 else "CHECK"
        else:
            occluded = f"{100 * (possible - visible) / possible:.1f}%"

        plant_position = get_plant_position(world, name, plant_position_cache)
        distance = (f"{np.linalg.norm(camera_position - plant_position):.1f} m"
                    if plant_position is not None else "--")
        for x, value in zip(col_x, (name, str(visible), str(possible), occluded, distance)):
            write(value, x, y, 0.55)

    if not plant_segmentation_ids:
        write("No plant IDs configured. Restore flightpath.csv to enable per-plant telemetry.",
              20, 162, 0.56, (80, 180, 255))
    elif not rows:
        write("No listed plants detected in either image for this camera.",
              20, 162, 0.56, (80, 180, 255))
    if len(rows) > MAX_TELEMETRY_ROWS:
        write(f"UP/DOWN: scroll plants  |  Showing {scroll_offset + 1}-{scroll_offset + shown_count} of {len(rows)}",
              20, panel_height - 12, 0.47, (175, 185, 200))

    return panel, scroll_offset


def main():
    camera_names, camera_positions = generate_camera_scene()
    client = ProjectAirSimClient()
    try:
        client.connect()
        world = World(client, SCENE_CONFIG, delay_after_load_sec=2)

        # These are stationary non-physics actors: no arming or flight commands.
        cameras = {
            name: Drone(client, world, name)
            for name in camera_names
        }
        plant_segmentation_ids = configure_plant_segmentation(world)
        plant_position_cache = {}

        current_index = 0
        scroll_offset = 0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 650)
        print("LEFT / RIGHT arrow: switch cameras | UP / DOWN: scroll plants | Q: quit")
        print("Keep the Windows terminal focused to use its arrow keys.")
        print(f"Now viewing: {camera_names[current_index]}")

        commands = SimpleQueue()

        keyboard_thread = threading.Thread(
            target=keyboard_listener,
            args=(commands,),
            daemon=True
        )

        keyboard_thread.start()

        while True:
            should_quit = False

            try:
                while True:
                    command = commands.get_nowait()

                    if command == "quit":
                        should_quit = True
                        break

                    if command == "scroll-up":
                        scroll_offset = max(0, scroll_offset - 1)
                        continue
                    if command == "scroll-down":
                        scroll_offset += 1
                        continue

                    current_index = (
                        current_index + command
                    ) % len(camera_names)
                    scroll_offset = 0

                    print(f"Now viewing: {camera_names[current_index]}")

            except Empty:
                pass

            if should_quit:
                break

            # YOUR IMAGE CAPTURE CODE GOES HERE
            camera_name = camera_names[current_index]

            responses = cameras[camera_name].get_images(
                SENSOR_NAME,
                [ImageType.SCENE, ImageType.SEGMENTATION, XRAY_IMAGE_TYPE],
            )

            scene = as_bgr(unpack_image(responses[ImageType.SCENE]))
            segmentation = as_bgr(unpack_image(responses[ImageType.SEGMENTATION]))
            xray = as_bgr(unpack_image(responses[XRAY_IMAGE_TYPE]))

            height, width = segmentation.shape[:2]

            if scene.shape[:2] != (height, width):
                scene = cv2.resize(scene, (width, height))

            if xray.shape[:2] != (height, width):
                xray = cv2.resize(xray, (width, height), interpolation=cv2.INTER_NEAREST)

            # Count pixels at camera resolution; never interpolate segmentation colours.
            visible_counts = count_plant_pixels(segmentation, plant_segmentation_ids)
            possible_counts = count_plant_pixels(xray, plant_segmentation_ids)

            combined = np.hstack((scene, segmentation, xray))
            panel, scroll_offset = make_telemetry_panel(
                camera_name,
                camera_positions[camera_name],
                plant_segmentation_ids,
                visible_counts,
                possible_counts,
                world,
                plant_position_cache,
                scroll_offset,
                combined.shape[1],
            )
            combined = np.vstack((combined, panel))

            cv2.imshow(WINDOW_NAME, combined)
            cv2.waitKey(1)

            time.sleep(0.03)

    finally:
        cv2.destroyAllWindows()
        client.disconnect()


if __name__ == "__main__":
    main()
