import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "seismic.db"
PARQUET_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "usgs_3y_minmag4p5.parquet"

def ingest_parquet_to_sqlite():
    df = pd.read_parquet(PARQUET_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("earthquakes", conn, if_exists="replace", index=False)
    
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lat ON earthquakes(latitude)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lon ON earthquakes(longitude)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mag ON earthquakes(mag)")
    conn.commit()
    conn.close()
    print(f"Ingestão completa: {len(df)} eventos → {DB_PATH}")

if __name__ == "__main__":
    ingest_parquet_to_sqlite()