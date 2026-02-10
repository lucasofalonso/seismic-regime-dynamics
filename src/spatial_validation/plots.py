# src/spatial_validation/plots.py
from __future__ import annotations

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLATES_URL = (
    "https://earthquake.usgs.gov/arcgis/rest/services/eq/map_plateboundaries/MapServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson"
)

def load_plates_geojson(url: str = PLATES_URL, timeout: int = 60) -> dict:
    return requests.get(url, timeout=timeout).json()

def make_time_gate(
    *,
    year: int,
    month: int,
    n_months: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year, month, 1, tz="UTC")
    end = start + pd.DateOffset(months=n_months)
    return start, end

def prepare_spatial_frames(
    *,
    features_df: pd.DataFrame,
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    minmag: float = 4.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # 1. Preparar Centróides (Janelas)
    tmp = features_df.dropna(subset=["clat", "clon", "cluster"]).copy()
    tmp = tmp[(tmp["window_start"] >= start) & (tmp["window_start"] < end)]
    tmp["cluster"] = tmp["cluster"].astype(str)

    # 2. Preparar Raw Events (Terremotos)
    ev = df.dropna(subset=["latitude", "longitude"]).copy()
    time_col = "time" if "time" in ev.columns else "time_utc"
    ev[time_col] = pd.to_datetime(ev[time_col], utc=True, errors="coerce")
    ev = ev.dropna(subset=[time_col])
    
    ev = ev[(ev[time_col] >= start) & (ev[time_col] < end)]
    if "mag" in ev.columns:
        ev = ev[ev["mag"].notna() & (ev["mag"] >= minmag)]

    # Forçar precisão idêntica (nanossegundos) em ambos os dataframes
    ev[time_col] = ev[time_col].astype("datetime64[ns, UTC]")
    
    feats_sorted = features_df.sort_values(by="window_start").copy()
    feats_sorted["window_start"] = feats_sorted["window_start"].astype("datetime64[ns, UTC]")
    feats_sorted["window_end"] = feats_sorted["window_end"].astype("datetime64[ns, UTC]")
    
    # 3. Merge AsOf
    ev = ev.sort_values(by=time_col)
    
    ev_merged = pd.merge_asof(
        ev, 
        feats_sorted[["window_start", "window_end", "cluster"]], 
        left_on=time_col, 
        right_on="window_start", 
        direction="backward"
    )
    
    # Filtragem de segurança
    ev_merged = ev_merged[ev_merged[time_col] < ev_merged["window_end"]]
    ev_merged["cluster"] = ev_merged["cluster"].astype(str)

    return tmp, ev_merged

def plot_clusters_with_plates(
    *,
    tmp: pd.DataFrame,   # Centróides
    ev: pd.DataFrame,    # Raw Events (agora com coluna 'cluster')
    plates_geojson: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    minmag: float = 4.5,
) -> go.Figure:
    
    # --- PLOT 1: Raw Events Coloridos (realidade) ---
    fig = px.scatter_geo(
        ev, 
        lat="latitude", 
        lon="longitude",
        color="cluster", # Os eventos respeitam a cor do regime!
        size="mag",      # Tamanho proporcional à magnitude real
        size_max=10,    
        opacity=0.6,
        hover_data=["time", "mag", "depth", "cluster"],
        projection="natural earth",
        title=f"Eventos Reais (Coloridos por Regime) + Centróides | {start.date()} → {end.date()}"
    )

    # --- PLOT 2: Centróides das Janelas (abstração) ---
    # Os pontos coloridos são a realidade, o quadrado grande é o estado macroscópico"
    clusters = sorted(tmp['cluster'].unique())
    
    fig.add_trace(go.Scattergeo(
        lat=tmp["clat"],
        lon=tmp["clon"],
        mode="markers",
        marker=dict(
            size=12, 
            symbol="square-open", 
            color="black",        
            line=dict(width=1)
        ),
        name="Centróides (Janela)",
        text=tmp["cluster"],
        hoverinfo="text+lat+lon"
    ))

    # --- PLOT 3: Placas Tectônicas (contexto) ---
    for feat in plates_geojson.get("features", []):
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", [])
        gtype = geom.get("type")

        def add_line(line_coords):
            lons = [p[0] for p in line_coords]
            lats = [p[1] for p in line_coords]
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode="lines",
                line=dict(width=1, color="black"), 
                opacity=0.3,
                hoverinfo="skip",
                showlegend=False
            ))

        if gtype == "LineString":
            add_line(coords)
        elif gtype == "MultiLineString":
            for line in coords:
                add_line(line)

    fig.update_layout(
        width=1300, 
        height=750, 
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title="Cluster (Regime)"
    )
    
    return fig