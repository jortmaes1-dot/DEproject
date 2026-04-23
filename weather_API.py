import time
import requests
import pandas as pd

# ============================================================
# WEATHER_API.PY
# ============================================================
# DOEL:
# We halen dagelijkse weerdata op via Meteostat / RapidAPI.
#
# Input:
# - daily_valence_summary.csv
#
# Output:
# - weather.csv
# - station_candidate_coverage.csv
#
# Belangrijkste weerkolommen:
# - prcp = neerslag in mm
# - tsun = zonneschijnduur in minuten
# - tavg = gemiddelde temperatuur
# - tmin = minimumtemperatuur
# - tmax = maximumtemperatuur
# - wspd = gemiddelde windsnelheid
#
# We zoeken automatisch weerstations rond Brussel.
# Daarna testen we enkele stations en kiezen we het station met
# de beste dekking voor prcp.
# ============================================================


# ============================================================
# INSTELLINGEN
# ============================================================

SPOTIFY_DAILY_FILE = "daily_valence_summary.csv"
OUTPUT_WEATHER_FILE = "weather.csv"
OUTPUT_STATION_CHECK_FILE = "station_candidate_coverage.csv"

RAPIDAPI_HOST = "meteostat.p.rapidapi.com"

# Plak hier je eigen RapidAPI-key.
# Voorbeeld:
# RAPIDAPI_KEY = "abc123..."
RAPIDAPI_KEY = "6c94cf6ca3msh149ba21fd5b0570p1bb21ejsn41d2728e6626"

# Brussel als centrale Belgische locatie
LATITUDE = 50.8503
LONGITUDE = 4.3517

# Aantal weerstations rond Brussel dat we willen testen
STATION_LIMIT = 5

# Even wachten tussen API-calls om problemen met rate limits te vermijden
REQUEST_SLEEP_SECONDS = 1


# ============================================================
# CONTROLE API KEY
# ============================================================

if RAPIDAPI_KEY == "PLAK_HIER_JE_RAPIDAPI_KEY" or not RAPIDAPI_KEY.strip():
    raise ValueError(
        "Je hebt je RapidAPI-key nog niet ingevuld.\n"
        "Vervang deze regel:\n"
        'RAPIDAPI_KEY = "PLAK_HIER_JE_RAPIDAPI_KEY"\n'
        "door je echte key."
    )

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST
}


# ============================================================
# HULPFUNCTIES
# ============================================================

def get_station_name(station):
    """
    Meteostat geeft station name soms als string en soms als dictionary.
    Deze functie maakt daar een leesbare naam van.
    """
    name = station.get("name", "onbekend")

    if isinstance(name, dict):
        return (
            name.get("en")
            or name.get("nl")
            or name.get("fr")
            or next(iter(name.values()), "onbekend")
        )

    return str(name)


def split_date_range_by_year(start_date, end_date):
    """
    Splitst het datumbereik per jaar.
    Dat is overzichtelijker en veiliger dan één zeer grote call.
    """
    date_ranges = []

    for year in range(start_date.year, end_date.year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1).date()
        year_end = pd.Timestamp(year=year, month=12, day=31).date()

        range_start = max(start_date, year_start)
        range_end = min(end_date, year_end)

        date_ranges.append((range_start, range_end))

    return date_ranges


def fetch_nearby_stations():
    """
    Zoekt weerstations rond Brussel.
    """
    url = f"https://{RAPIDAPI_HOST}/stations/nearby"

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "limit": STATION_LIMIT
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Fout bij ophalen van nabijgelegen weerstations.\n"
            f"Status code: {response.status_code}\n"
            f"Antwoord: {response.text}"
        )

    stations = response.json().get("data", [])

    if not stations:
        raise RuntimeError("Geen weerstations gevonden rond Brussel.")

    return stations


def fetch_daily_weather_for_station(station_id, date_ranges):
    """
    Haalt dagelijkse weerdata op voor één station.
    """
    url = f"https://{RAPIDAPI_HOST}/stations/daily"

    all_rows = []

    for range_start, range_end in date_ranges:
        print(f"  Weather ophalen voor station {station_id}: {range_start} tot {range_end}")

        params = {
            "station": station_id,
            "start": str(range_start),
            "end": str(range_end),
            "units": "metric"
        }

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=60
        )

        if response.status_code != 200:
            print(
                f"  Fout voor station {station_id}, periode {range_start} tot {range_end}.\n"
                f"  Status code: {response.status_code}\n"
                f"  Antwoord: {response.text}"
            )
            continue

        data = response.json().get("data", [])
        all_rows.extend(data)

        time.sleep(REQUEST_SLEEP_SECONDS)

    weather = pd.DataFrame(all_rows)

    if weather.empty:
        return pd.DataFrame()

    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather = weather.dropna(subset=["date"]).copy()
    weather["date"] = weather["date"].dt.normalize()

    wanted_cols = [
        "date",
        "prcp",
        "tsun",
        "tavg",
        "tmin",
        "tmax",
        "wspd",
        "pres"
    ]

    available_cols = [col for col in wanted_cols if col in weather.columns]
    weather = weather[available_cols].copy()

    for col in weather.columns:
        if col != "date":
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather.drop_duplicates(subset=["date"])
    weather = weather.sort_values("date").reset_index(drop=True)

    return weather


# ============================================================
# STAP 1: SPOTIFY DAGDATA INLADEN
# ============================================================

print("Spotify dagdata wordt ingeladen...")

spotify_daily = pd.read_csv(SPOTIFY_DAILY_FILE)

if "date" not in spotify_daily.columns:
    raise ValueError(
        f"Kolom 'date' niet gevonden in {SPOTIFY_DAILY_FILE}."
    )

spotify_daily["date"] = pd.to_datetime(spotify_daily["date"], errors="coerce")
spotify_daily = spotify_daily.dropna(subset=["date"]).copy()
spotify_daily["date"] = spotify_daily["date"].dt.normalize()

start_date = spotify_daily["date"].min().date()
end_date = spotify_daily["date"].max().date()

print("\nDatumbereik uit Spotify-dataset:")
print(start_date, "tot", end_date)

date_ranges = split_date_range_by_year(start_date, end_date)

print("\nPeriodes die via de API worden opgehaald:")
for s, e in date_ranges:
    print(s, "tot", e)


# ============================================================
# STAP 2: WEERSTATIONS ROND BRUSSEL ZOEKEN
# ============================================================

print("\nWeerstations rond Brussel zoeken...")

stations = fetch_nearby_stations()

stations_info = []

print("\nGevonden stations:")

for station in stations:
    station_id = str(station.get("id"))
    station_name = get_station_name(station)
    distance = station.get("distance", None)

    print(f"- {station_id} | {station_name} | afstand: {distance} meter")

    stations_info.append({
        "station_id": station_id,
        "station_name": station_name,
        "distance_m": distance
    })


# ============================================================
# STAP 3: STATIONS TESTEN OP DATADEKKING
# ============================================================

print("\nStations worden getest op weather-dekking...")

station_results = []
weather_by_station = {}

expected_days = spotify_daily["date"].nunique()

for station in stations_info:
    station_id = station["station_id"]

    print(f"\nStation testen: {station_id} - {station['station_name']}")

    weather = fetch_daily_weather_for_station(station_id, date_ranges)

    if weather.empty:
        station_results.append({
            "station_id": station_id,
            "station_name": station["station_name"],
            "distance_m": station["distance_m"],
            "received_days": 0,
            "expected_days": expected_days,
            "prcp_non_missing": 0,
            "tsun_non_missing": 0,
            "prcp_coverage": 0,
            "tsun_coverage": 0
        })
        continue

    received_days = weather["date"].nunique()

    prcp_non_missing = weather["prcp"].notna().sum() if "prcp" in weather.columns else 0
    tsun_non_missing = weather["tsun"].notna().sum() if "tsun" in weather.columns else 0

    prcp_coverage = prcp_non_missing / expected_days
    tsun_coverage = tsun_non_missing / expected_days

    station_results.append({
        "station_id": station_id,
        "station_name": station["station_name"],
        "distance_m": station["distance_m"],
        "received_days": received_days,
        "expected_days": expected_days,
        "prcp_non_missing": prcp_non_missing,
        "tsun_non_missing": tsun_non_missing,
        "prcp_coverage": prcp_coverage,
        "tsun_coverage": tsun_coverage
    })

    weather_by_station[station_id] = weather

station_check = pd.DataFrame(station_results)

station_check = station_check.sort_values(
    by=["prcp_coverage", "received_days", "tsun_coverage"],
    ascending=False
).reset_index(drop=True)

print("\nStation coverage controle:")
print(station_check.to_string(index=False))

station_check.to_csv(OUTPUT_STATION_CHECK_FILE, index=False)

print(f"\nStation coverage opgeslagen als: {OUTPUT_STATION_CHECK_FILE}")


# ============================================================
# STAP 4: BESTE STATION KIEZEN
# ============================================================

if station_check.empty or station_check.iloc[0]["received_days"] == 0:
    raise RuntimeError("Geen enkel station gaf bruikbare weatherdata terug.")

best_station_id = str(station_check.iloc[0]["station_id"])
best_station_name = station_check.iloc[0]["station_name"]

print("\nBeste station gekozen:")
print("Station ID:", best_station_id)
print("Naam:", best_station_name)

weather = weather_by_station[best_station_id].copy()


# ============================================================
# STAP 5: WEATHERDATA CONTROLEREN EN OPSLAAN
# ============================================================

print("\nWeatherdata van gekozen station:")
print("Aantal dagen:", len(weather))
print("Datumbereik:")
print(weather["date"].min(), "tot", weather["date"].max())

print("\nAantal missende waarden:")
print(weather.isna().sum())

print("\nEerste 10 rijen:")
print(weather.head(10).to_string(index=False))

weather.to_csv(OUTPUT_WEATHER_FILE, index=False)

print(f"\nWeatherdata opgeslagen als: {OUTPUT_WEATHER_FILE}")
print("\nKlaar.")