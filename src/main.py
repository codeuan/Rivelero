#main.py

print("Launching GUI...")

import shutil

from .GUI import start_gui
from .visibility_frequency import visibility_frequency as run_program


if not shutil.which("gdal") and not shutil.which("gdal_raster_viewshed"):
    print("WARNING: GDAL CLI not found. Viewshed may not work.")


if __name__ == "__main__":
    start_gui(run_program)