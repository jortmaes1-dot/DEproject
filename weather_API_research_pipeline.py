# ============================================================
# WEATHER_API.PY
# ============================================================
# DOEL:
# 1. KMI/RMI daily weather data ophalen voor de periode uit
#    daily_valence_summary.csv.
# 2. Per dag een Belgisch gemiddelde maken over stations.
# 3. Weather types maken die bruikbaar zijn voor de paper:
#    - Strict regenachtig
#    - Strict zonnig
#    - Gemengd/overig
# 4. Weatherdata mergen met Spotify dagdata.
#
# INPUT:
# - daily_valence_summary.csv  output van main.py
#
# OUTPUT:
# - weather.csv
# - final_dataset.csv
# - weather_thresholds.csv
#
# BELANGRIJK:
# - prcp = gemiddelde dagelijkse neerslag
# - tsun = gemiddelde dagelijkse zonneschijnduur
# - strict_weather_type gebruikt zowel prcp als tsun.
# ============================================================

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# INSTELLINGEN
# ============================================================

BASE_URL = "https://opendata.meteo.be/service/ows"
SPOTIFY_DAILY_FILE = "daily_valence_summary.csv"
WEATHER_OUTPUT_FILE = "weather.csv"
FINAL_OUTPUT_FILE = "final_dataset.csv"
THRESHOLDS_OUTPUT_FILE = "weather_thresholds.csv"

TIMEOUT_SECONDS = 180

# Deze drempels zijn bewust conservatief:
# - Strict regenachtig: merkbare regen én weinig zon.
# - Strict zonnig: weinig/geen regen én veel zon t.o.v. de dataset.
MIN_PRCP_STRICT_RAINY = 1.0
MAX_PRCP_STRICT_SUNNY = 0.2


# ============================================================
# HULPFUNCTIES
# ============================================================

def pick_column(df, candidates, required=True):
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        key = str(candidate).lower()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise KeyError(
            f"Geen passende kolom gevonden. Gezocht: {candidates}. "
            f"Beschikbare kolommen: {df.columns.tolist()}"
        )
    return None


def year_ranges(start_date, end_date):
    ranges = []
    for year in range(start_date.year, end_date.year + 1):
        range_start = max(start_date, pd.Timestamp(year=year, month=1, day=1))
        range_end = min(end_date, pd.Timestamp(year=year, month=12, day=31))
        ranges.append((range_start, range_end))
    return ranges


def build_cql_filter(start_date, end_date):
    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%dT00:00:00")
    end_str = pd.Timestamp(end_date).strftime("%Y-%m-%dT23:59:59")
    return f"timestamp >= '{start_str}' AND timestamp <= '{end_str}'"


def fetch_chunk(start_date, end_date):
    cql_filter = build_cql_filter(start_date, end_date)
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typenames": "aws:aws_1day",
        "outputFormat": "csv",
        "CQL_FILTER": cql_filter,
    }

    print(f"\nKMI chunk ophalen: {start_date.date()} tot {end_date.date()}")
    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        print("  -> Leeg antwoord")
        return pd.DataFrame()

    chunk = pd.read_csv(io.StringIO(text))
    print("  -> Rijen ontvangen:", len(chunk))
    return chunk


def zscore(series):
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean(skipna=True)) / sd


def add_weather_types(df):
    """
    Maakt weerclassificaties.

    Keuze:
    - prcp blijft nuttig als regenmaat.
    - tsun blijft nuttig als zonnemaat.
    - Voor de duidelijke boxplots gebruiken we strict_weather_type:
        Strict regenachtig = prcp >= 1.0 mm én tsun behoort tot de laagste 40%.
        Strict zonnig      = prcp <= 0.2 mm én tsun behoort tot de hoogste 60%.

    Waarom percentielen?
    Omdat tsun in Belgische data seizoensgebonden is en niet elke periode dezelfde
    absolute maximumzon heeft. Percentielen zijn daardoor robuuster.
    """
    df = df.copy()
    df["prcp"] = pd.to_numeric(df["prcp"], errors="coerce")
    df["tsun"] = pd.to_numeric(df["tsun"], errors="coerce")

    tsun_low = df["tsun"].quantile(0.40)
    tsun_high = df["tsun"].quantile(0.60)

    # Fallback indien tsun bijna volledig ontbreekt
    if pd.isna(tsun_low):
        tsun_low = np.inf
    if pd.isna(tsun_high):
        tsun_high = -np.inf

    df["rainy_day"] = df["prcp"] > 0
    df["heavy_rain"] = df["prcp"] >= 5
    df["dry_day"] = df["prcp"] <= MAX_PRCP_STRICT_SUNNY

    df["weather_score_sunny"] = zscore(df["tsun"]) - zscore(df["prcp"])
    df["weather_score_dreary"] = zscore(df["prcp"]) - zscore(df["tsun"])

    conditions = [
        (df["prcp"] >= MIN_PRCP_STRICT_RAINY) & (df["tsun"] <= tsun_low),
        (df["prcp"] <= MAX_PRCP_STRICT_SUNNY) & (df["tsun"] >= tsun_high),
    ]
    choices = ["Strict regenachtig", "Strict zonnig"]
    df["strict_weather_type"] = np.select(conditions, choices, default="Gemengd/overig")

    # Voor PowerBI: korte Engelse labels zonder spaties kunnen handig zijn.
    df["weather_type_key"] = df["strict_weather_type"].map({
        "Strict regenachtig": "rainy",
        "Strict zonnig": "sunny",
        "Gemengd/overig": "mixed",
    })

    thresholds = pd.DataFrame([
        {"threshold": "min_prcp_strict_rainy", "value": MIN_PRCP_STRICT_RAINY},
        {"threshold": "max_prcp_strict_sunny", "value": MAX_PRCP_STRICT_SUNNY},
        {"threshold": "tsun_low_quantile_40", "value": tsun_low},
        {"threshold": "tsun_high_quantile_60", "value": tsun_high},
    ])
    thresholds.to_csv(THRESHOLDS_OUTPUT_FILE, index=False)

    return df


# ============================================================
# STAP 1: SPOTIFY DAILY INLADEN
# ============================================================

def main():
    if not Path(SPOTIFY_DAILY_FILE).exists():
        raise FileNotFoundError(
            f"Bestand niet gevonden: {SPOTIFY_DAILY_FILE}. Run eerst main.py."
        )

    spotify_daily = pd.read_csv(SPOTIFY_DAILY_FILE)
    if "date" not in spotify_daily.columns:
        raise ValueError(f"Kolom 'date' niet gevonden in {SPOTIFY_DAILY_FILE}")

    spotify_daily["date"] = pd.to_datetime(spotify_daily["date"], errors="coerce").dt.normalize()
    spotify_daily = spotify_daily.dropna(subset=["date"]).copy()

    start_date = spotify_daily["date"].min().normalize()
    end_date = spotify_daily["date"].max().normalize()

    print("Analyseperiode afgeleid uit daily_valence_summary.csv:")
    print(start_date.date(), "tot", end_date.date())

    # ============================================================
    # STAP 2: KMI-DATA OPHALEN
    # ============================================================

    all_chunks = []
    for range_start, range_end in year_ranges(start_date, end_date):
        chunk = fetch_chunk(range_start, range_end)
        if not chunk.empty:
            all_chunks.append(chunk)

    if not all_chunks:
        raise RuntimeError("Geen weatherdata ontvangen van KMI.")

    raw = pd.concat(all_chunks, ignore_index=True)
    print("\nTotaal ruwe weer-rijen:", len(raw))
    print("Kolommen:", raw.columns.tolist())

    # ============================================================
    # STAP 3: KOLOMMEN KIEZEN EN CLEANEN
    # ============================================================

    date_col = pick_column(raw, ["timestamp", "dateTime", "datetime", "date", "DATE"])
    prcp_col = pick_column(raw, ["precip_quantity", "PRECIP_QUANTITY", "prcp", "precipitation"])
    tsun_col = pick_column(raw, ["sun_duration", "SUN_DURATION", "tsun", "sunshine_duration"])
    station_col = pick_column(raw, ["code", "station_code", "name", "station", "station_name"], required=False)

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
    raw[prcp_col] = pd.to_numeric(raw[prcp_col], errors="coerce")
    raw[tsun_col] = pd.to_numeric(raw[tsun_col], errors="coerce")

    raw = raw[(raw[date_col] >= start_date) & (raw[date_col] <= end_date)].copy()

    if station_col is not None:
        raw[station_col] = raw[station_col].astype(str).str.strip()
        raw = raw.drop_duplicates(subset=[station_col, date_col])
    else:
        raw = raw.drop_duplicates(subset=[date_col, prcp_col, tsun_col])

    # ============================================================
    # STAP 4: DAGGEMIDDELDE OVER STATIONS
    # ============================================================

    weather_daily = (
        raw.groupby(date_col, as_index=False)
        .agg({prcp_col: "mean", tsun_col: "mean"})
        .rename(columns={date_col: "date", prcp_col: "prcp", tsun_col: "tsun"})
    )

    full_calendar = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})
    weather = full_calendar.merge(weather_daily, on="date", how="left")
    weather = add_weather_types(weather)

    print("\nWeather daily klaar:")
    print("Aantal dagen totaal:", len(weather))
    print("Dagen met prcp:", weather["prcp"].notna().sum())
    print("Dagen met tsun:", weather["tsun"].notna().sum())
    print("\nVerdeling strict_weather_type:")
    print(weather["strict_weather_type"].value_counts(dropna=False))

    weather_out = weather.copy()
    weather_out["date"] = weather_out["date"].dt.strftime("%Y-%m-%d")
    weather_out.to_csv(WEATHER_OUTPUT_FILE, index=False)

    # ============================================================
    # STAP 5: MERGE MET SPOTIFY DAGDATA
    # ============================================================

    final = spotify_daily.merge(weather, on="date", how="left")
    final = final.sort_values("date").reset_index(drop=True)
    final["date"] = final["date"].dt.strftime("%Y-%m-%d")
    final.to_csv(FINAL_OUTPUT_FILE, index=False)

    print("\nBestanden opgeslagen:")
    print("-", WEATHER_OUTPUT_FILE)
    print("-", FINAL_OUTPUT_FILE)
    print("-", THRESHOLDS_OUTPUT_FILE)
    print("\nKlaar. Run nu final_analysis.py.")


if __name__ == "__main__":
    main()
