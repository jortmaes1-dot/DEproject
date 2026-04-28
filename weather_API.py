import io
import requests
import pandas as pd
from pathlib import Path

# ============================================================
# WEATHER_API.PY
# ============================================================
# DOEL:
# - KMI / RMI open weather data ophalen
# - Analyseperiode automatisch afleiden uit daily_top200_valence_summary.csv
# - Weather ophalen in jaarblokken om grote calls te vermijden
# - Per dag Belgisch gemiddelde maken over alle stations
#
# INPUT:
# - daily_top200_valence_summary.csv
#
# OUTPUT:
# - weather.csv met:
#   date, prcp, tsun
# ============================================================

BASE_URL = "https://opendata.meteo.be/service/ows"

SPOTIFY_DAILY_FILE = "daily_top200_valence_summary.csv"
OUTPUT_FILE = "weather.csv"

TIMEOUT_SECONDS = 180


def pick_column(df, candidates, required=True):
    lower_map = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        key = str(candidate).lower()

        if key in lower_map:
            return lower_map[key]

    for col in df.columns:
        col_lower = str(col).lower()

        for candidate in candidates:
            candidate_lower = str(candidate).lower()

            if candidate_lower in col_lower:
                return col

    if required:
        raise KeyError(
            f"Geen passende kolom gevonden.\n"
            f"Gezochte opties: {candidates}\n"
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
        "CQL_FILTER": cql_filter
    }

    print(f"\nChunk ophalen: {start_date.date()} tot {end_date.date()}")

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=TIMEOUT_SECONDS
    )

    response.raise_for_status()

    text = response.text.strip()

    if not text:
        print("  -> Leeg antwoord")
        return pd.DataFrame()

    chunk = pd.read_csv(io.StringIO(text))

    print("  -> Rijen ontvangen:", len(chunk))

    return chunk


# ============================================================
# STAP 1: PERIODE BEPALEN
# ============================================================

if not Path(SPOTIFY_DAILY_FILE).exists():
    raise FileNotFoundError(
        f"Bestand niet gevonden: {SPOTIFY_DAILY_FILE}\n"
        "Run eerst main.py."
    )

spotify_daily = pd.read_csv(SPOTIFY_DAILY_FILE)

if "date" not in spotify_daily.columns:
    raise ValueError(f"Kolom 'date' niet gevonden in {SPOTIFY_DAILY_FILE}.")

spotify_daily["date"] = pd.to_datetime(
    spotify_daily["date"],
    errors="coerce"
)

spotify_daily = spotify_daily.dropna(subset=["date"]).copy()

if spotify_daily.empty:
    raise ValueError(f"{SPOTIFY_DAILY_FILE} bevat geen geldige datums.")

start_date = spotify_daily["date"].min().normalize()
end_date = spotify_daily["date"].max().normalize()

print("Analyseperiode afgeleid uit daily_top200_valence_summary.csv:")
print(start_date.date(), "tot", end_date.date())


# ============================================================
# STAP 2: KMI DATA OPHALEN
# ============================================================

all_chunks = []

for range_start, range_end in year_ranges(start_date, end_date):
    try:
        chunk = fetch_chunk(range_start, range_end)

        if not chunk.empty:
            all_chunks.append(chunk)

    except Exception as e:
        print(f"Fout bij chunk {range_start.date()} tot {range_end.date()}:")
        print(repr(e))

if not all_chunks:
    raise RuntimeError(
        "Geen weatherdata ontvangen van KMI.\n"
        "Controleer internetverbinding, KMI-service of analyseperiode."
    )

raw = pd.concat(all_chunks, ignore_index=True)

print("\nTotaal aantal ruwe rijen:", len(raw))
print("Kolommen:")
print(raw.columns.tolist())


# ============================================================
# STAP 3: KOLOMMEN HERKENNEN
# ============================================================

date_col = pick_column(
    raw,
    ["timestamp", "dateTime", "datetime", "date", "DATE"]
)

prcp_col = pick_column(
    raw,
    ["precip_quantity", "PRECIP_QUANTITY", "prcp", "precipitation"]
)

tsun_col = pick_column(
    raw,
    ["sun_duration", "SUN_DURATION", "tsun", "sunshine_duration"]
)

station_col = pick_column(
    raw,
    ["code", "station_code", "name", "station", "station_name"],
    required=False
)

print("\nGeselecteerde kolommen:")
print("date_col:", date_col)
print("prcp_col:", prcp_col)
print("tsun_col:", tsun_col)
print("station_col:", station_col)


# ============================================================
# STAP 4: TYPES CLEANEN
# ============================================================

raw[date_col] = pd.to_datetime(
    raw[date_col],
    errors="coerce"
).dt.normalize()

raw[prcp_col] = pd.to_numeric(raw[prcp_col], errors="coerce")
raw[tsun_col] = pd.to_numeric(raw[tsun_col], errors="coerce")

if station_col is not None:
    raw[station_col] = raw[station_col].astype(str).str.strip()

print("\nRuwe datums:")
print(raw[date_col].min(), "tot", raw[date_col].max())

print("\nMissende waarden vóór cleaning:")
print(raw[[date_col, prcp_col, tsun_col]].isna().sum())


# ============================================================
# STAP 5: FILTEREN EN DEDUPLICEREN
# ============================================================

raw = raw[
    (raw[date_col] >= start_date) &
    (raw[date_col] <= end_date)
].copy()

raw = raw.dropna(subset=[date_col]).copy()

if station_col is not None:
    raw = raw.drop_duplicates(subset=[station_col, date_col])
else:
    raw = raw.drop_duplicates(subset=[date_col, prcp_col, tsun_col])

if raw.empty:
    raise ValueError("Na filtering op de Spotify-periode is de weatherdataset leeg.")

print("\nNa filtering:")
print("Shape:", raw.shape)
print("Datum:", raw[date_col].min(), "tot", raw[date_col].max())


# ============================================================
# STAP 6: GEMIDDELDE PER DAG
# ============================================================

weather_daily = (
    raw
    .groupby(date_col, as_index=False)
    .agg({
        prcp_col: "mean",
        tsun_col: "mean"
    })
    .copy()
)

weather_daily = weather_daily.rename(
    columns={
        date_col: "date",
        prcp_col: "prcp",
        tsun_col: "tsun"
    }
)

weather_daily["prcp"] = pd.to_numeric(weather_daily["prcp"], errors="coerce")
weather_daily["tsun"] = pd.to_numeric(weather_daily["tsun"], errors="coerce")

print("\nNa groeperen per dag:")
print(weather_daily.head().to_string(index=False))


# ============================================================
# STAP 7: VOLLEDIGE DATUMREEKS MAKEN
# ============================================================

full_calendar = pd.DataFrame({
    "date": pd.date_range(start=start_date, end=end_date, freq="D")
})

weather = pd.merge(
    full_calendar,
    weather_daily,
    on="date",
    how="left"
)

weather["date"] = weather["date"].dt.strftime("%Y-%m-%d")

print("\nVolledige weather dataset:")
print("Aantal dagen:", len(weather))
print("Dagen met prcp:", weather["prcp"].notna().sum())
print("Dagen met tsun:", weather["tsun"].notna().sum())

print("\nEerste 10 rijen:")
print(weather.head(10).to_string(index=False))

print("\nLaatste 10 rijen:")
print(weather.tail(10).to_string(index=False))


# ============================================================
# STAP 8: OPSLAAN
# ============================================================

weather.to_csv(OUTPUT_FILE, index=False)

print(f"\nKlaar. Bestand opgeslagen als {OUTPUT_FILE}")