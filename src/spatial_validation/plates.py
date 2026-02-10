# === Distance (centroid -> nearest plate boundary) + boxplot (FIXED) ===

import numpy as np
import pandas as pd
import requests

from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points
from pyproj import Transformer

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# INPUT esperado:
#   tmp : DataFrame com ["clat","clon","cluster"] (já time-gated se você quiser)
# -----------------------

# 1) Load plate boundaries GeoJSON (USGS ArcGIS)
PLATES_URL = (
    "https://earthquake.usgs.gov/arcgis/rest/services/eq/map_plateboundaries/MapServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson"
)
plates_geojson = requests.get(PLATES_URL, timeout=60).json()

# 2) Build valid LineStrings (WGS84) and project to EPSG:3857 (meters)
to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

valid_lines_m = []

def _coords_to_lines(coords):
    """Yield list(s) of point sequences for LineString or MultiLineString coords."""
    # coords format:
    # LineString: [[lon,lat], ...]
    # MultiLineString: [[[lon,lat], ...], [[lon,lat], ...], ...]
    if not coords:
        return
    # detect nesting depth
    first = coords[0]
    if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (int, float)):
        # LineString
        yield coords
    else:
        # MultiLineString
        for part in coords:
            if part:
                yield part

for feat in plates_geojson.get("features", []):
    geom = feat.get("geometry", {})
    if not geom:
        continue
    coords = geom.get("coordinates", None)
    if coords is None:
        continue

    for seq in _coords_to_lines(coords):
        # seq is list of [lon,lat,(z?)]
        if seq is None or len(seq) < 2:
            continue
        # project points (lon,lat) -> (x,y)
        proj_pts = []
        for xy in seq:
            if xy is None or len(xy) < 2:
                continue
            lon, lat = xy[0], xy[1]
            if lon is None or lat is None:
                continue
            proj_pts.append(to_m(lon, lat))
        if len(proj_pts) < 2:
            continue
        ls = LineString(proj_pts)
        if not ls.is_empty and ls.length > 0:
            valid_lines_m.append(ls)

if len(valid_lines_m) == 0:
    raise RuntimeError("No valid plate boundary lines after filtering. Check PLATES_URL response.")

plate_m = MultiLineString(valid_lines_m)

# 3) Distance function (meters -> km)
def centroid_to_plate_distance_km(lat, lon):
    x, y = to_m(lon, lat)
    p = Point(x, y)
    p_near, l_near = nearest_points(p, plate_m)
    return p.distance(l_near) / 1000.0

# 4) Compute distances and boxplot
tmp_dist = tmp.dropna(subset=["clat","clon","cluster"]).copy()
tmp_dist["dist_plate_km"] = tmp_dist.apply(
    lambda r: centroid_to_plate_distance_km(r["clat"], r["clon"]),
    axis=1
)

display(
    tmp_dist.groupby("cluster")["dist_plate_km"]
    .agg(count="count", mean_km="mean", median_km="median", std_km="std",
         q25_km=lambda s: s.quantile(0.25), q75_km=lambda s: s.quantile(0.75),
         p90_km=lambda s: s.quantile(0.90))
    .sort_index()
)

plt.figure(figsize=(7,4))
sns.boxplot(data=tmp_dist, x="cluster", y="dist_plate_km")
plt.title("Centroid distance to nearest plate boundary (km)")
plt.tight_layout()
plt.show()
