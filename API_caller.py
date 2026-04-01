import os
import tempfile
from pathlib import Path
import requests


OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY") #retrieve the API key from the local system environment variable (cannot be compromised).

def download_dem_from_opentopo(
    south: float,
    north: float,
    west: float,
    east: float,
    demtype: str = "COP30",
) -> str:
    if not OPENTOPO_API_KEY:
        raise RuntimeError("OPENTOPO_API_KEY is not set.") #if no API key is found, throw an error.

    url = "https://portal.opentopography.org/API/globaldem" #API that is used to obtain GeoTIFF data.

    params = {
        "demtype": demtype, #which dataset will be accessed.
        "south": south, #southern boundary.
        "north": north, #northern boundary.
        "west": west, #western boundary.
        "east": east, #eastern boundary.
        "outputFormat": "GTiff", #result should be a GeoTIFF file.
        "API_Key": OPENTOPO_API_KEY, #API key.
    }

    response = requests.get(url, params=params, timeout=120) #send HTTP GET request with timeout of 120 seconds.
    response.raise_for_status() #if an error code is returned, raise an error.

    content_type = response.headers.get("Content-Type", "").lower() #store HTTP header to determine what kind of data was returned.
    if "html" in content_type: #if HTML returned there is likely an error, so raise one.
        raise RuntimeError("OpenTopography returned HTML instead of a GeoTIFF. Check your parameters and API key.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif") #allocate space on the hard disk for a tif file.
    tmp.write(response.content) #store data on the tif file.
    tmp.close() #close file to free space.

    return tmp.name #return path to tif file.
