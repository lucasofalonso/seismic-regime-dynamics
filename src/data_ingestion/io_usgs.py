from __future__ import annotations

import calendar
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict
from pathlib import Path



BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


@dataclass(frozen=True)
class USGCQuery:
    """
    Parâmetros fixos do endpoint (query-params) que controlam a resposta do servidor. 
    """
    minmag: float = 0.0
    orderby: str = "time-asc"
    limit: int = 20000
    fmt: str = "geojson"

    def to_params(self, start_date: str, end_date: str) -> Dict[str, object]:
        return {
            "starttime": start_date,
            "endtime": end_date,
            "minmagnitude": self.minmag,
            "orderby": self.orderby,
            "limit": self.limit,
            "format": self.fmt
        }


def fetch_usgs_month(start_date: str, end_date: str, query: USGCQuery) -> pd.DataFrame:
    """
    Baixa eventos no intervalo [start_date, end_date) (como strings yyyy-mm-dd)
    e retorna um DataFrame com as colunas referentes aos campos do payload de resposta.
    """
    params = query.to_params(start_date, end_date)
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    rows = []
    for f in js.get("features", []):
        prop = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = (geom.get("coordinates") or [None, None, None])

        rows.append(
            {
                # --- Identificação / tempo
                "id": f.get("id"),
                "time": pd.to_datetime(prop.get("time"), unit="ms", utc=True),
                "updated": (
                    pd.to_datetime(prop.get("updated"), unit="ms", utc=True)
                    if prop.get("updated") is not None
                    else pd.NaT
                ),
                

                # --- Magnitude
                "mag": prop.get("mag"),
                "magType": prop.get("magType"),
                
                
                # --- Localização (GeoJSON: [lon, lat, depth])
                "longitude": coords[0],
                "latitude": coords[1],
                "depth": coords[2],
                
                
                # --- Qualidade / instrumentação
                "nst": prop.get("nst"),
                "gap": prop.get("gap"),
                "dmin": prop.get("dmin"),
                "rms": prop.get("rms"),
                "net": prop.get("net"),
                
                
                # --- Metadados
                "place": prop.get("place"),
                "type": prop.get("type"),
                "status": prop.get("status"),
            }
        )

    return pd.DataFrame(rows)


def month_ranges(years_back):
    """
    Gera pares (start, end) mensais em formato 'YYYY-MM-DD', cobrindo years_back anos até hoje.
    - start = primeiro dia do mês
    - end   = primeiro dia do próximo mês (END EXCLUSIVO)
    """
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end.replace(year=end.year - int(years_back))

    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first_month_day = datetime(y, m, 1, tzinfo=timezone.utc)
        last_month_day = calendar.monthrange(y, m)[1]

        next_month = first_month_day + timedelta(days=last_month_day)
        # Ajuste para não passar do dia atual
        if next_month > end + timedelta(days=1):
            next_month = end + timedelta(days=1)

        yield first_month_day.date().isoformat(), next_month.date().isoformat()

        # avança mês
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def build_catalog(years_back, query: USGCQuery) -> pd.DataFrame:
    """
    Baixa mês-a-mês, concatena, remove duplicatas por id e ordena por time.
    """
    dfs = []
    for start, end in month_ranges(years_back):
        df = fetch_usgs_month(start, end, query)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["id"]).sort_values("time")
    return out.reset_index(drop=True)


def save_catalog_parquet(df: pd.DataFrame, path: str, compression: str = "snappy") -> None:
    """
    Salva catálogo USGS em parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=False)


def load_catalog_parquet(path: str) -> pd.DataFrame:
    """
    Carrega catálogo USGS do arquivo parquet.
    """
    return pd.read_parquet(path)