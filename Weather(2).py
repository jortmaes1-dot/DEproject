import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text

# ============================================================
# INSTELLINGEN & SQL CONFIGURATIE
# ============================================================
DB_NAME = "deproject.db"
DB_URL = f"sqlite:///{DB_NAME}"
engine = create_engine(DB_URL)

BASE_URL = "https://opendata.meteo.be/service/ows"
SPOTIFY_DAILY_FILE = "daily_valence_summary.csv"
WEATHER_OUTPUT_FILE = "weather.csv"
FINAL_OUTPUT_FILE = "final_dataset.csv"
THRESHOLDS_OUTPUT_FILE = "weather_thresholds.csv"

TIMEOUT_SECONDS = 180
MIN_PRCP_STRICT_RAINY = 1.0
MAX_PRCP_STRICT_SUNNY = 0.2

# ============================================================
# SQL HULPFUNCTIES
# ============================================================

def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_cache (
                date TEXT PRIMARY KEY,
                prcp REAL,
                tsun REAL
            )
        """))
        conn.commit()

def get_cached_weather(start_date, end_date):
    query = "SELECT * FROM weather_cache WHERE date BETWEEN :start AND :end"
    return pd.read_sql(query, engine, params={"start": start_date.strftime('%Y-%m-%d'), 
                                              "end": end_date.strftime('%Y-%m-%d')})

def save_to_cache(df):
    if df.empty: return
    df_to_save = df.copy()
    df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d')
    try:
        df_to_save.to_sql("weather_cache", engine, if_exists="append", index=False)
    except:
        pass

# ============================================================
# DATA VERWERKING LOGICA
# ============================================================

def zscore(series):
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean(skipna=True)) / sd

def pick_column(df, candidates):
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"Kolom niet gevonden: {candidates}")

def fetch_chunk(start_date, end_date):
    cql = f"timestamp >= '{start_date.strftime('%Y-%m-%dT00:00:00')}' AND timestamp <= '{end_date.strftime('%Y-%m-%dT23:59:59')}'"
    params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
              "typenames": "aws:aws_1day", "outputFormat": "csv", "CQL_FILTER": cql}
    
    print(f"  -> KMI API: {start_date.date()} tot {end_date.date()}")
    r = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text.strip())) if r.text.strip() else pd.DataFrame()

def add_weather_columns_full(df):
    df = df.copy()
    df["prcp"] = pd.to_numeric(df["prcp"], errors="coerce")
    df["tsun"] = pd.to_numeric(df["tsun"], errors="coerce")
    
    # 1. Basis booleans
    df["rainy_day"] = df["prcp"] > 0
    df["heavy_rain"] = df["prcp"] >= 5
    df["dry_day"] = df["prcp"] <= 0
    
    # 2. Thresholds berekenen
    tsun_low = df["tsun"].quantile(0.40)
    tsun_high = df["tsun"].quantile(0.60)
    
    # 3. Weather scores (Z-scores)
    df["weather_score_sunny"] = zscore(df["tsun"]) - zscore(df["prcp"])
    df["weather_score_dreary"] = zscore(df["prcp"]) - zscore(df["tsun"])
    
    # 4. Classificaties
    df["rain_dry_type"] = np.where(df["prcp"] > 0, "Regenachtig", "Droog")
    
    cond = [
        (df["prcp"] >= MIN_PRCP_STRICT_RAINY) & (df["tsun"] <= tsun_low),
        (df["prcp"] <= MAX_PRCP_STRICT_SUNNY) & (df["tsun"] >= tsun_high)
    ]
    choices = ["Strict regenachtig", "Strict zonnig"]
    df["strict_weather_type"] = np.select(cond, choices, default="Gemengd/overig")
    
    # 5. Weather type key
    df["weather_type_key"] = df["strict_weather_type"].map({
        "Strict regenachtig": "rainy",
        "Strict zonnig": "sunny",
        "Gemengd/overig": "mixed",
    })
    
    return df, tsun_low, tsun_high

# ============================================================
# MAIN
# ============================================================

def main():
    init_db()
    
    if not Path(SPOTIFY_DAILY_FILE).exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {SPOTIFY_DAILY_FILE}")

    spotify_daily = pd.read_csv(SPOTIFY_DAILY_FILE)
    spotify_daily["date"] = pd.to_datetime(spotify_daily["date"]).dt.normalize()
    start, end = spotify_daily["date"].min(), spotify_daily["date"].max()

    # SQL Cache Check
    cached = get_cached_weather(start, end)
    
    if len(cached) < ((end - start).days * 0.9):
        print("Data ontbreekt in SQL. API wordt geraadpleegd...")
        all_data = []
        for year in range(start.year, end.year + 1):
            y_start = max(start, pd.Timestamp(year, 1, 1))
            y_end = min(end, pd.Timestamp(year, 12, 31))
            chunk = fetch_chunk(y_start, y_end)
            if not chunk.empty: all_data.append(chunk)
        
        raw = pd.concat(all_data)
        d_col = pick_column(raw, ["timestamp", "date"])
        p_col = pick_column(raw, ["precip_quantity", "prcp"])
        t_col = pick_column(raw, ["sun_duration", "tsun"])
        
        raw[d_col] = pd.to_datetime(raw[d_col]).dt.normalize()
        daily = raw.groupby(d_col).agg({p_col: "mean", t_col: "mean"}).reset_index()
        daily.columns = ["date", "prcp", "tsun"]
        
        save_to_cache(daily)
        weather_data = daily
    else:
        print("Data geladen uit SQL-cache.")
        weather_data = cached
        weather_data['date'] = pd.to_datetime(weather_data['date'])

    # Bereken alle kolommen
    full_range = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    weather_merged = full_range.merge(weather_data, on="date", how="left")
    
    # Verkrijg de data én de thresholds
    weather, t_low, t_high = add_weather_columns_full(weather_merged)

    # OUTPUT 1: weather_thresholds.csv (Nu expliciet in de main opgeslagen)
    thresholds_df = pd.DataFrame([
        {"threshold": "min_prcp_strict_rainy", "value": MIN_PRCP_STRICT_RAINY},
        {"threshold": "max_prcp_strict_sunny", "value": MAX_PRCP_STRICT_SUNNY},
        {"threshold": "tsun_low_quantile_40", "value": t_low},
        {"threshold": "tsun_high_quantile_60", "value": t_high},
    ])
    thresholds_df.to_csv(THRESHOLDS_OUTPUT_FILE, index=False)

    # OUTPUT 2: weather.csv
    weather_out = weather.copy()
    weather_out["date"] = weather_out["date"].dt.strftime("%Y-%m-%d")
    weather_out.to_csv(WEATHER_OUTPUT_FILE, index=False)

    # OUTPUT 3: final_dataset.csv
    final = spotify_daily.merge(weather, on="date", how="left")
    final["date"] = final["date"].dt.strftime("%Y-%m-%d")
    final.to_csv(FINAL_OUTPUT_FILE, index=False)

    print(f"\nKlaar! Alle bestanden zijn aangemaakt:")
    print(f"- {THRESHOLDS_OUTPUT_FILE}")
    print(f"- {WEATHER_OUTPUT_FILE}")
    print(f"- {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()