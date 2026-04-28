import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# FINAL_ANALYSIS.PY
# ============================================================
# DOEL:
# - daily_valence_summary.csv en weather.csv samenbrengen
# - alleen 2 extreme weer-groepen vergelijken:
#     1. strict_rainy
#     2. strict_sunny
# - correlaties berekenen
# - grafieken tonen
#
# DEFINITIES:
# - sad_song in main.py: valence <= 0.40
#
# EXTREME WEERGROEPEN:
# - strict_rainy_day = prcp >= 5.0 EN tsun <= q25
# - strict_sunny_day = prcp < 1.0 EN tsun >= q75
#
# BELANGRIJK:
# - dagen tussenin worden NIET meegenomen in de extreme vergelijking
# ============================================================

SPOTIFY_DAILY_FILE = "daily_valence_summary.csv"
WEATHER_FILE = "weather.csv"

OUTPUT_FINAL_DATASET = "final_dataset.csv"
OUTPUT_EXTREME_DATASET = "extreme_weather_dataset.csv"
OUTPUT_SUMMARY_EXTREME = "summary_strict_rainy_vs_strict_sunny.csv"

STRICT_RAIN_THRESHOLD = 5.0
DRY_FOR_SUNNY_THRESHOLD = 1.0
COVERAGE_THRESHOLD = 0.70


# ============================================================
# STAP 1: BESTANDEN CONTROLEREN
# ============================================================

if not Path(SPOTIFY_DAILY_FILE).exists():
    raise FileNotFoundError(
        f"Bestand niet gevonden: {SPOTIFY_DAILY_FILE}\n"
        "Run eerst main.py."
    )

if not Path(WEATHER_FILE).exists():
    raise FileNotFoundError(
        f"Bestand niet gevonden: {WEATHER_FILE}\n"
        "Run eerst weather_API.py."
    )


# ============================================================
# STAP 2: DATA INLADEN
# ============================================================

spotify = pd.read_csv(SPOTIFY_DAILY_FILE)
weather = pd.read_csv(WEATHER_FILE)

if "date" not in spotify.columns:
    raise ValueError(f"Kolom 'date' niet gevonden in {SPOTIFY_DAILY_FILE}")

if "date" not in weather.columns:
    raise ValueError(f"Kolom 'date' niet gevonden in {WEATHER_FILE}")

spotify["date"] = pd.to_datetime(spotify["date"], errors="coerce")
weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

numeric_candidates = [
    "top200_rows",
    "tracks_with_valence",
    "valence_coverage",
    "avg_valence",
    "median_valence",
    "std_valence",
    "min_valence",
    "max_valence",
    "sad_songs_count",
    "share_sad_songs",
    "total_streams",
    "sad_streams",
    "weighted_avg_valence",
    "share_sad_streams",
    "prcp",
    "tsun"
]

for col in numeric_candidates:
    if col in spotify.columns:
        spotify[col] = pd.to_numeric(spotify[col], errors="coerce")
    if col in weather.columns:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

print("Spotify daily shape:", spotify.shape)
print("Weather shape:", weather.shape)

print("\nSpotify datumbereik:")
print(spotify["date"].min(), "tot", spotify["date"].max())

print("\nWeather datumbereik:")
print(weather["date"].min(), "tot", weather["date"].max())


# ============================================================
# STAP 3: MERGEN
# ============================================================

weather_small = weather[["date", "prcp", "tsun"]].copy()

df = pd.merge(
    spotify,
    weather_small,
    on="date",
    how="inner"
)

print("\nShape na merge:", df.shape)
print("\nEerste 5 rijen:")
print(df.head().to_string(index=False))


# ============================================================
# STAP 4: HOOFDVARIABELEN KIEZEN
# ============================================================
# We kiezen weighted metrics als die bestaan en bruikbaar zijn.
# Anders vallen we terug op song-share metrics.
# ============================================================

main_valence_metric = "avg_valence"
main_sad_metric = "share_sad_songs"

if "weighted_avg_valence" in df.columns and df["weighted_avg_valence"].notna().sum() > 0:
    main_valence_metric = "weighted_avg_valence"

if "share_sad_streams" in df.columns and df["share_sad_streams"].notna().sum() > 0:
    main_sad_metric = "share_sad_streams"

print("\nGebruikte hoofdvariabelen:")
print("Valence metric:", main_valence_metric)
print("Sad metric:", main_sad_metric)


# ============================================================
# STAP 5: ANALYSEBASIS OPSCHONEN
# ============================================================

needed_cols = [
    "date",
    "valence_coverage",
    main_valence_metric,
    main_sad_metric,
    "prcp",
    "tsun"
]

df = df.dropna(subset=needed_cols).copy()
df = df[df["valence_coverage"] >= COVERAGE_THRESHOLD].copy()

print("\nShape na cleaning + coveragefilter:", df.shape)
print(f"Coveragefilter: valence_coverage >= {COVERAGE_THRESHOLD}")


# ============================================================
# STAP 6: STRENGE DEFINITIES VOOR 2 EXTREME GROEPEN
# ============================================================

low_sun_threshold = df["tsun"].quantile(0.25)
high_sun_threshold = df["tsun"].quantile(0.75)

df["strict_rainy_day"] = (
    (df["prcp"] >= STRICT_RAIN_THRESHOLD) &
    (df["tsun"] <= low_sun_threshold)
)

df["strict_sunny_day"] = (
    (df["prcp"] < DRY_FOR_SUNNY_THRESHOLD) &
    (df["tsun"] >= high_sun_threshold)
)

df["extreme_group"] = pd.NA
df.loc[df["strict_rainy_day"], "extreme_group"] = "strict_rainy"
df.loc[df["strict_sunny_day"], "extreme_group"] = "strict_sunny"

extreme_df = df[df["extreme_group"].notna()].copy()

print("\nGebruikte strenge definities:")
print(f"strict_rainy_day = prcp >= {STRICT_RAIN_THRESHOLD} EN tsun <= q25 ({round(low_sun_threshold, 4)})")
print(f"strict_sunny_day = prcp < {DRY_FOR_SUNNY_THRESHOLD} EN tsun >= q75 ({round(high_sun_threshold, 4)})")

print("\nAantal strict rainy days:")
print(df["strict_rainy_day"].value_counts())

print("\nAantal strict sunny days:")
print(df["strict_sunny_day"].value_counts())

print("\nAantal dagen in extreme vergelijking:")
print(extreme_df["extreme_group"].value_counts())

if extreme_df.empty:
    raise ValueError(
        "Geen extreme dagen gevonden met deze definities. "
        "Maak de drempels iets minder streng."
    )

if extreme_df["extreme_group"].nunique() < 2:
    raise ValueError(
        "Slechts één extreme groep gevonden. "
        "Maak de definities iets minder streng."
    )


# ============================================================
# STAP 7: CORRELATIES OP VOLLEDIGE GECLEANDE DATASET
# ============================================================
# Correlaties berekenen we op de volledige bruikbare dataset,
# niet alleen op de extreme subset.
# ============================================================

corr_cols = [main_valence_metric, main_sad_metric, "prcp", "tsun"]
corr_matrix = df[corr_cols].corr()

print("\nCorrelatiematrix op volledige cleaned dataset:")
print(corr_matrix.to_string())


# ============================================================
# STAP 8: GEMIDDELDEN OP EXTREME GROEPEN
# ============================================================

print("\nGemiddelden strict rainy vs strict sunny:")
print(
    extreme_df.groupby("extreme_group")[[main_valence_metric, main_sad_metric, "prcp", "tsun"]]
    .mean()
    .to_string()
)

mean_valence_rainy = extreme_df.loc[
    extreme_df["extreme_group"] == "strict_rainy",
    main_valence_metric
].mean()

mean_valence_sunny = extreme_df.loc[
    extreme_df["extreme_group"] == "strict_sunny",
    main_valence_metric
].mean()

mean_sad_rainy = extreme_df.loc[
    extreme_df["extreme_group"] == "strict_rainy",
    main_sad_metric
].mean()

mean_sad_sunny = extreme_df.loc[
    extreme_df["extreme_group"] == "strict_sunny",
    main_sad_metric
].mean()

print("\nInterpretatie strict rainy vs strict sunny:")
print(f"Gemiddelde {main_valence_metric} op strict rainy days:", round(mean_valence_rainy, 4))
print(f"Gemiddelde {main_valence_metric} op strict sunny days:", round(mean_valence_sunny, 4))
print(f"Verschil {main_valence_metric} (strict rainy - strict sunny):", round(mean_valence_rainy - mean_valence_sunny, 4))

print(f"Gemiddelde {main_sad_metric} op strict rainy days:", round(mean_sad_rainy, 4))
print(f"Gemiddelde {main_sad_metric} op strict sunny days:", round(mean_sad_sunny, 4))
print(f"Verschil {main_sad_metric} (strict rainy - strict sunny):", round(mean_sad_rainy - mean_sad_sunny, 4))

if mean_valence_rainy < mean_valence_sunny:
    print("\nEerste indicatie: op strikte regendagen ligt de valence lager dan op strikte zonnige dagen.")
else:
    print("\nEerste indicatie: op strikte regendagen ligt de valence niet lager dan op strikte zonnige dagen.")

if mean_sad_rainy > mean_sad_sunny:
    print("Eerste indicatie: op strikte regendagen ligt het aandeel sad songs hoger dan op strikte zonnige dagen.")
else:
    print("Eerste indicatie: op strikte regendagen ligt het aandeel sad songs niet hoger dan op strikte zonnige dagen.")


# ============================================================
# STAP 9: BESTANDEN OPSLAAN
# ============================================================

df.to_csv(OUTPUT_FINAL_DATASET, index=False)
extreme_df.to_csv(OUTPUT_EXTREME_DATASET, index=False)

summary_extreme = extreme_df.groupby("extreme_group").agg(
    aantal_dagen=("date", "count"),
    gemiddelde_valence=(main_valence_metric, "mean"),
    gemiddelde_sad_share=(main_sad_metric, "mean"),
    gemiddelde_prcp=("prcp", "mean"),
    gemiddelde_tsun=("tsun", "mean")
).reset_index()

summary_extreme.to_csv(OUTPUT_SUMMARY_EXTREME, index=False)

print(f"\nBestand opgeslagen als {OUTPUT_FINAL_DATASET}")
print(f"Bestand opgeslagen als {OUTPUT_EXTREME_DATASET}")
print(f"Bestand opgeslagen als {OUTPUT_SUMMARY_EXTREME}")


# ============================================================
# STAP 10: GRAFIEKEN
# ============================================================

# Tijdsgrafiek volledige cleaned dataset
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df[main_valence_metric])
plt.title(f"{main_valence_metric} doorheen de tijd")
plt.xlabel("Datum")
plt.ylabel(main_valence_metric)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafiek_valence_tijd.png")
plt.show()

# Scatter regen vs valence op volledige cleaned dataset
plt.figure(figsize=(8, 5))
plt.scatter(df["prcp"], df[main_valence_metric])
plt.title(f"Regen (prcp) vs {main_valence_metric}")
plt.xlabel("Neerslag (mm)")
plt.ylabel(main_valence_metric)
plt.tight_layout()
plt.savefig("grafiek_regen_vs_valence.png")
plt.show()

# Scatter zon vs sad share op volledige cleaned dataset
plt.figure(figsize=(8, 5))
plt.scatter(df["tsun"], df[main_sad_metric])
plt.title(f"Zon (tsun) vs {main_sad_metric}")
plt.xlabel("Zonneschijnduur")
plt.ylabel(main_sad_metric)
plt.tight_layout()
plt.savefig("grafiek_zon_vs_sadshare.png")
plt.show()

# Boxplot extreme groepen: sad metric
plt.figure(figsize=(8, 5))
extreme_df.boxplot(column=main_sad_metric, by="extreme_group")
plt.title(f"{main_sad_metric} op strict rainy vs strict sunny days")
plt.suptitle("")
plt.xlabel("Extreme group")
plt.ylabel(main_sad_metric)
plt.tight_layout()
plt.savefig("grafiek_boxplot_strict_rainy_vs_strict_sunny_sad.png")
plt.show()

# Boxplot extreme groepen: valence metric
plt.figure(figsize=(8, 5))
extreme_df.boxplot(column=main_valence_metric, by="extreme_group")
plt.title(f"{main_valence_metric} op strict rainy vs strict sunny days")
plt.suptitle("")
plt.xlabel("Extreme group")
plt.ylabel(main_valence_metric)
plt.tight_layout()
plt.savefig("grafiek_boxplot_strict_rainy_vs_strict_sunny_valence.png")
plt.show()

print("\nKlaar.")