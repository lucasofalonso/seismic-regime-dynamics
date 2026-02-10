# src/spatial_validation/distance.py
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points
from pyproj import Transformer

import matplotlib.pyplot as plt


def _coords_to_lines(coords):
    if not coords:
        return
    first = coords[0]
    if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (int, float)):
        # LineString: [[lon,lat], ...]
        yield coords
    else:
        # MultiLineString: [[[lon,lat], ...], ...]
        for part in coords:
            if part:
                yield part


def build_plate_multiline_3857(plates_geojson: Dict[str, Any]) -> Tuple[MultiLineString, Transformer]:
    """
    Constrói MultiLineString em EPSG:3857 (metros) para cálculo de distâncias (km).
    """
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    tf = to_m.transform

    valid_lines_m = []

    for feat in plates_geojson.get("features", []):
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", None)
        if coords is None:
            continue

        for seq in _coords_to_lines(coords):
            if seq is None or len(seq) < 2:
                continue

            proj_pts = []
            for xy in seq:
                if xy is None or len(xy) < 2:
                    continue
                lon, lat = xy[0], xy[1]
                if lon is None or lat is None:
                    continue
                proj_pts.append(tf(lon, lat))

            if len(proj_pts) < 2:
                continue

            ls = LineString(proj_pts)
            if not ls.is_empty and ls.length > 0:
                valid_lines_m.append(ls)

    if len(valid_lines_m) == 0:
        raise RuntimeError("No valid plate boundary lines after filtering. Check plates_geojson content.")

    return MultiLineString(valid_lines_m), to_m


def centroid_to_plate_distance_km(lat: float, lon: float, plate_m: MultiLineString, to_m: Transformer) -> float:
    x, y = to_m.transform(lon, lat)
    p = Point(x, y)
    _, l_near = nearest_points(p, plate_m)
    return p.distance(l_near) / 1000.0


def add_plate_distance_column(
    centroids_df: pd.DataFrame,
    plates_geojson: Dict[str, Any],
    *,
    lat_col: str = "clat",
    lon_col: str = "clon",
    out_col: str = "dist_plate_km",
) -> pd.DataFrame:
    """
    Retorna uma cópia do df com a coluna dist_plate_km.
    Espera que centroids_df tenha lat/lon e cluster (para uso imediato em summary/boxplot).
    """
    if "cluster" not in centroids_df.columns:
        raise KeyError("centroids_df precisa ter a coluna 'cluster' para summary/boxplot.")

    df = centroids_df.dropna(subset=[lat_col, lon_col, "cluster"]).copy()

    # cast seguro (evita object/strings quebrando transform)
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    # garante cluster limpo (opcional, mas ajuda)
    try:
        df["cluster"] = df["cluster"].astype(int)
    except Exception:
        pass

    # se ficou vazio, devolve vazio com a coluna esperada
    if df.empty:
        df[out_col] = pd.Series(dtype=float)
        return df

    plate_m, to_m = build_plate_multiline_3857(plates_geojson)

    df[out_col] = df.apply(
        lambda r: centroid_to_plate_distance_km(float(r[lat_col]), float(r[lon_col]), plate_m, to_m),
        axis=1,
    )
    return df



def summarize_plate_distance_by_cluster(
    df_with_dist: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    dist_col: str = "dist_plate_km",
) -> pd.DataFrame:
    """
    Stats por cluster: count, mean, median, std, q25, q75, p90.
    """
    g = df_with_dist.groupby(cluster_col)[dist_col]
    out = g.agg(
        count="count",
        mean_km="mean",
        median_km="median",
        std_km="std",
        q25_km=lambda s: s.quantile(0.25),
        q75_km=lambda s: s.quantile(0.75),
        p90_km=lambda s: s.quantile(0.90),
    ).sort_index()
    return out


def plot_distance_boxplot(
    df_with_dist: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    dist_col: str = "dist_plate_km",
    figsize: tuple[int, int] = (7, 4),
    title: str = "Centroid distance to nearest plate boundary (km)",
    show_fliers: bool = True,
    ax: Optional[plt.Axes] = None,
):
    """
    Boxplot (matplotlib puro) de dist_plate_km por cluster.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ordena clusters
    clusters = sorted(df_with_dist[cluster_col].dropna().unique().tolist())
    data = []
    labels = []
    for c in clusters:
        vals = df_with_dist.loc[df_with_dist[cluster_col] == c, dist_col].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(str(c))

    if len(data) == 0:
        raise ValueError("Sem dados suficientes para boxplot (distâncias vazias por cluster).")


    ax.boxplot(
        data,
        labels=labels,
        showfliers=show_fliers,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Distance to plate (km)")
    ax.set_title(title)
    fig.tight_layout()

    return fig, ax
