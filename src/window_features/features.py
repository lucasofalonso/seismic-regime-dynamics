# src/window_features/features.py
from __future__ import annotations

from typing import Any, Dict
from math import radians, sin, cos, atan2, sqrt

import numpy as np
import pandas as pd


def energy_sum_log10(mag: pd.Series) -> float:
    """
    log10(sum(10^(1.5*mag))) em magnitudes válidas.
    """
    mag = pd.to_numeric(mag, errors="coerce").dropna()
    return float(np.log10((10 ** (1.5 * mag)).sum())) if len(mag) > 0 else np.nan


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  # Earth radius (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def spherical_centroid_latlon(lat: pd.Series, lon: pd.Series) -> tuple[float, float]:
    """
    Centróide esférico (lat/lon) usando média de vetores no R^3.
    """
    valid = lat.notna() & lon.notna()
    lat_r = np.radians(pd.to_numeric(lat[valid], errors="coerce").values.astype(float))
    lon_r = np.radians(pd.to_numeric(lon[valid], errors="coerce").values.astype(float))

    if len(lat_r) == 0:
        return np.nan, np.nan

    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)

    x_m, y_m, z_m = x.mean(), y.mean(), z.mean()
    norm = np.sqrt(x_m**2 + y_m**2 + z_m**2)
    if norm == 0:
        return np.nan, np.nan

    x_m, y_m, z_m = x_m / norm, y_m / norm, z_m / norm
    lat_c = np.degrees(np.arcsin(z_m))
    lon_c = np.degrees(np.arctan2(y_m, x_m))
    return float(lat_c), float(lon_c)


def spread_geodesic_km(lat: pd.Series, lon: pd.Series) -> Dict[str, float]:
    """
    Dispersão geodésica (distâncias haversine ao centróide esférico).
    Retorna as chaves:
      spread_km_mean, spread_km_median, spread_km_max, spread_km_std, spread_km_p95
    """
    valid = lat.notna() & lon.notna()
    lat = pd.to_numeric(lat[valid], errors="coerce").dropna()
    lon = pd.to_numeric(lon[valid], errors="coerce").dropna()

    if len(lat) == 0 or len(lon) == 0:
        return dict(
            spread_km_mean=np.nan,
            spread_km_median=np.nan,
            spread_km_max=np.nan,
            spread_km_std=np.nan,
            spread_km_p95=np.nan,
        )

    clat, clon = spherical_centroid_latlon(lat, lon)
    if not (np.isfinite(clat) and np.isfinite(clon)):
        return dict(
            spread_km_mean=np.nan,
            spread_km_median=np.nan,
            spread_km_max=np.nan,
            spread_km_std=np.nan,
            spread_km_p95=np.nan,
        )

    dists = [haversine_km(clat, clon, float(a), float(b)) for a, b in zip(lat.values, lon.values)]
    dists_arr = np.asarray(dists, dtype=float)

    return dict(
        spread_km_mean=float(np.mean(dists_arr)),
        spread_km_median=float(np.median(dists_arr)),
        spread_km_max=float(np.max(dists_arr)),
        spread_km_std=float(np.std(dists_arr)),
        spread_km_p95=float(np.quantile(dists_arr, 0.95)),
    )


def safe_quantile(x: pd.Series, q: float, default: float = np.nan) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(x.quantile(q)) if (len(x) > 0 and x.notna().any()) else default


def inter_event_stats(ts: pd.Series) -> Dict[str, float]:
    """
    Estatísticas de delta-t (em segundos) entre eventos consecutivos.
    """
    ts = pd.to_datetime(ts, errors="coerce").dropna().sort_values().values
    if len(ts) < 2:
        return {"dt_mean_s": np.nan, "dt_median_s": np.nan, "dt_std_s": np.nan, "dt_p95_s": np.nan}

    dts = np.diff(ts).astype("timedelta64[s]").astype(float)
    return {
        "dt_mean_s": float(np.mean(dts)),
        "dt_median_s": float(np.median(dts)),
        "dt_std_s": float(np.std(dts)),
        "dt_p95_s": float(np.quantile(dts, 0.95)),
    }


def extract_features(
    win: pd.DataFrame,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    *,
    time_col: str = "time",
) -> Dict[str, Any]:
    """
    Extrai features de uma janela (win) no intervalo [win_start, win_end) (ou o que você usou no windowing).

    Mantém nomes e lógica do notebook:
      spread_*,
      energy_sum_log10,
      dt_*,
      x_geo,y_geo,z_geo, etc.
    """
    n = len(win)
    days = (win_end - win_start) / pd.Timedelta("1d")
    rate_per_day = n / days if days > 0 else np.nan

    # ---- séries "seguras"
    mag = win["mag"] if "mag" in win.columns else pd.Series(dtype=float)
    depth = win["depth"] if "depth" in win.columns else pd.Series(dtype=float)
    lat = win["latitude"] if "latitude" in win.columns else pd.Series(dtype=float)
    lon = win["longitude"] if "longitude" in win.columns else pd.Series(dtype=float)

    # ---- labels / composição
    n_eq = int((win["type"] == "earthquake").sum()) if "type" in win.columns else np.nan
    frac_eq = n_eq / n if n > 0 else np.nan

    # ---- centroid + unit sphere (x_geo,y_geo,z_geo) usando média simples de lat/lon (igual seu notebook)
    clat, clon = spherical_centroid_latlon(lat, lon)

    lat_mean = float(pd.to_numeric(lat, errors="coerce").mean()) if len(lat) else np.nan
    lon_mean = float(pd.to_numeric(lon, errors="coerce").mean()) if len(lon) else np.nan

    if np.isfinite(lat_mean) and np.isfinite(lon_mean):
        lat_rad = np.radians(lat_mean)
        lon_rad = np.radians(lon_mean)
        x_geo = float(np.cos(lat_rad) * np.cos(lon_rad))
        y_geo = float(np.cos(lat_rad) * np.sin(lon_rad))
        z_geo = float(np.sin(lat_rad))
    else:
        x_geo = y_geo = z_geo = np.nan

    # ---- energy
    energy_sum = energy_sum_log10(mag)

    # ---- spread geodésico
    geo = spread_geodesic_km(lat, lon)

    # ---- inter-event timing
    if time_col in win.columns:
        time_stats = inter_event_stats(win[time_col])
    else:
        time_stats = {"dt_mean_s": np.nan, "dt_median_s": np.nan, "dt_std_s": np.nan, "dt_p95_s": np.nan}

    window_seconds = (win_end - win_start).total_seconds()
    if np.isfinite(window_seconds) and window_seconds > 0 and n >= 2:
        dt_mean_norm = time_stats["dt_mean_s"] / window_seconds
        dt_std_norm = time_stats.get("dt_std_s", np.nan) / window_seconds
        dt_p95_norm = time_stats.get("dt_p95_s", np.nan) / window_seconds
    else:
        dt_mean_norm = dt_std_norm = dt_p95_norm = np.nan

    # ---- features finais (mesmos nomes)
    feats: Dict[str, Any] = {
        # Window metadata
        "window_start": win_start.tz_convert("UTC") if win_start.tzinfo else win_start,
        "window_end": win_end.tz_convert("UTC") if win_end.tzinfo else win_end,

        # Counts / rates
        "n_events": n,
        "rate_per_day": rate_per_day,

        # Labels / composition
        "n_earthquakes": n_eq,
        "frac_earthquake": frac_eq,

        # Magnitude stats
        "mag_mean": pd.to_numeric(mag, errors="coerce").mean(),
        "mag_std": pd.to_numeric(mag, errors="coerce").std(),
        "mag_min": pd.to_numeric(mag, errors="coerce").min(),
        "mag_q25": safe_quantile(mag, 0.25),
        "mag_median": safe_quantile(mag, 0.50),
        "mag_q75": safe_quantile(mag, 0.75),
        "mag_max": pd.to_numeric(mag, errors="coerce").max(),

        # Depth stats
        "depth_mean": pd.to_numeric(depth, errors="coerce").mean(),
        "depth_std": pd.to_numeric(depth, errors="coerce").std(),
        "depth_max": pd.to_numeric(depth, errors="coerce").max(),

        # Spatial dispersion
        "clat": clat,
        "clon": clon,
        "lat_mean": lat_mean,
        "lon_mean": lon_mean,
        "lat_std": pd.to_numeric(lat, errors="coerce").std(),
        "lon_std": pd.to_numeric(lon, errors="coerce").std(),
        "x_geo": x_geo,
        "y_geo": y_geo,
        "z_geo": z_geo,
        **geo,

        # Catalog diversity
        "unique_magType": win["magType"].nunique() if "magType" in win.columns else np.nan,
        "unique_net": win["net"].nunique() if "net" in win.columns else np.nan,

        # Energy
        "energy_sum_log10": energy_sum,

        # Inter-event timing (normalized)
        "dt_mean_norm": dt_mean_norm,
        "dt_std_norm": dt_std_norm,
        "dt_p95_norm": dt_p95_norm,

        # Inter-event timing (not normalized)
        **time_stats,
    }

    return feats
