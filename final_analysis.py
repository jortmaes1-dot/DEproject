import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
<<<<<<< HEAD
from pathlib import Path

try:
    from scipy.stats import ttest_ind, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
=======
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
>>>>>>> e3f4d82540d7fff0eb431d53d14af90df3fd9450

# ============================================================
# FINAL_ANALYSIS.PY
# ============================================================
# DOEL:
# - daily_valence_summary.csv en weather.csv samenbrengen
# - 2 extreme weergroepen definiëren:
#     1. strict_rainy
#     2. strict_sunny
# - Vergelijken via BOXPLOTS voor:
#     A. avg_valence         (gemiddelde positiviteit per dag)
#     B. avg_energy          (gemiddelde energie per dag)
#     C. share_sad_songs     (aandeel lage valence songs)
#     D. share_low_energy    (aandeel lage energy songs)
#     E. share_depressief    (aandeel lage valence + lage energy)
# - Significantietest (Mann-Whitney U) per metric
# - Samenvattende statistieken opslaan
#
# EXTREME WEERGROEPEN:
# - strict_rainy_day = prcp >= 5.0 EN tsun <= q25
# - strict_sunny_day = prcp < 1.0 EN tsun >= q75
#
# DREMPELWAARDEN (overeenkomen met main.py):
# - SAD_THRESHOLD        = 0.40
# - LOW_ENERGY_THRESHOLD = 0.50
# DOEL:
<<<<<<< HEAD
# - daily_top200_valence_summary.csv combineren met weather.csv
# - Belgische Spotify Top 200 analyseren op dagniveau
# - Onderzoeken of valence samenhangt met:
#   1. seizoenen
#   2. regen
#   3. zon
#   4. druilerige versus zonnige dagen
#
# INPUT:
# - daily_top200_valence_summary.csv
# - weather.csv
#
# OUTPUT:
# - final_daily_valence_weather_dataset.csv
# - season_summary.csv
# - weather_summary.csv
# - season_weather_summary.csv
# - drizzly_vs_sunny_summary.csv
# - rainy_vs_dry_summary.csv
# - correlation_valence_weather.csv
# - significance_tests_valence_weather.csv
# - grafieken als PNG
# ============================================================

SPOTIFY_DAILY_FILE = "daily_valence_summary.csv"
WEATHER_FILE = "weather.csv"

OUTPUT_FINAL_DATASET = "final_dataset.csv"
OUTPUT_EXTREME_DATASET = "extreme_weather_dataset.csv"
OUTPUT_SUMMARY_EXTREME = "summary_strict_rainy_vs_strict_sunny.csv"

STRICT_RAIN_THRESHOLD = 5.0
DRY_FOR_SUNNY_THRESHOLD = 1.0
COVERAGE_THRESHOLD = 0.70

COLOR_RAINY = "#4C72B0"
COLOR_SUNNY = "#DD8452"

SPOTIFY_DAILY_FILE = "daily_top200_valence_summary.csv"
WEATHER_FILE = "weather.csv"

OUTPUT_FINAL_DATASET = "final_daily_valence_weather_dataset.csv"

COVERAGE_THRESHOLD = 0.70

DRIZZLY_RAIN_THRESHOLD = 0.5
SUNNY_RAIN_MAX = 0.5


# ============================================================
# STAP 1: BESTANDEN CONTROLEREN
# HULPFUNCTIES
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
def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def assign_season(date_value):
    if pd.isna(date_value):
        return pd.NA

    month = date_value.month

    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    if month in [9, 10, 11]:
        return "autumn"

    return pd.NA


def choose_valence_metric(df):
    if (
        "weighted_avg_valence_streams" in df.columns and
        safe_to_numeric(df["weighted_avg_valence_streams"]).notna().sum() > 0
    ):
        return "weighted_avg_valence_streams", "streamgewogen gemiddelde valence"

    if "weighted_avg_valence_rank" in df.columns:
        return "weighted_avg_valence_rank", "rankgewogen gemiddelde valence"

    if "avg_valence" in df.columns:
        return "avg_valence", "ongewogen gemiddelde valence"

    raise ValueError("Geen bruikbare valence metric gevonden.")


def choose_sad_metric(df):
    if (
        "weighted_sad_share_streams" in df.columns and
        safe_to_numeric(df["weighted_sad_share_streams"]).notna().sum() > 0
    ):
        return "weighted_sad_share_streams", "streamgewogen aandeel sad songs"

    if "weighted_sad_share_rank" in df.columns:
        return "weighted_sad_share_rank", "rankgewogen aandeel sad songs"

    if "share_sad_songs" in df.columns:
        return "share_sad_songs", "ongewogen aandeel sad songs"

    return None, None


def correlation_safe(df, col_a, col_b):
    temp = df[[col_a, col_b]].dropna().copy()

    if len(temp) < 3:
        return pd.NA

    if temp[col_a].nunique() < 2 or temp[col_b].nunique() < 2:
        return pd.NA

    return temp[col_a].corr(temp[col_b])


def save_bar_plot(data, x_col, y_col, title, xlabel, ylabel, filename):
    if data.empty:
        print(f"Geen data voor grafiek: {filename}")
        return

    plt.figure(figsize=(9, 5))
    plt.bar(data[x_col].astype(str), data[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print("Grafiek opgeslagen:", filename)


def save_boxplot(df, column, by, title, xlabel, ylabel, filename):
    if df.empty:
        print(f"Geen data voor grafiek: {filename}")
        return

    if df[by].dropna().nunique() < 2:
        print(f"Te weinig groepen voor boxplot: {filename}")
        return

    plt.figure(figsize=(8, 5))
    df.boxplot(column=column, by=by)
    plt.title(title)
    plt.suptitle("")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print("Grafiek opgeslagen:", filename)


# ============================================================
# STAP 1: BESTANDEN CONTROLEREN
# ============================================================

print_section("STAP 1 - Bestanden controleren")

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

print("Bestanden gevonden:")
print("-", SPOTIFY_DAILY_FILE)
print("-", WEATHER_FILE)


# ============================================================
# STAP 2: DATA INLADEN
# ============================================================

print_section("STAP 2 - Data inladen")

spotify = pd.read_csv(SPOTIFY_DAILY_FILE)
weather = pd.read_csv(WEATHER_FILE)

if "date" not in spotify.columns:
    raise ValueError(f"Kolom 'date' ontbreekt in {SPOTIFY_DAILY_FILE}.")

if "date" not in weather.columns:
    raise ValueError(f"Kolom 'date' ontbreekt in {WEATHER_FILE}.")

spotify["date"] = pd.to_datetime(spotify["date"], errors="coerce").dt.normalize()
weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()

spotify = spotify.dropna(subset=["date"]).copy()
weather = weather.dropna(subset=["date"]).copy()

for col in ["prcp", "tsun"]:
    if col not in weather.columns:
        raise ValueError(f"Kolom '{col}' ontbreekt in weather.csv.")

    weather[col] = safe_to_numeric(weather[col])

numeric_cols = [
    "top200_rows",
    "tracks_with_valence",
    "valence_coverage",
    "matched_rows",
    "avg_valence",
    "median_valence",
    "std_valence",
    "min_valence",
    "max_valence",
    "sad_songs_count",
    "share_sad_songs",
    "weighted_avg_valence_rank",
    "weighted_sad_share_rank",
    "weighted_avg_valence_streams",
    "weighted_sad_share_streams",
    "total_rank_weight_all",
    "total_streams_all"
]

for col in numeric_cols:
    if col in spotify.columns:
        spotify[col] = safe_to_numeric(spotify[col])

print("Spotify daily shape:", spotify.shape)
print("Weather shape:", weather.shape)

print("\nSpotify datumbereik:")
print(spotify["date"].min(), "tot", spotify["date"].max())

print("\nWeather datumbereik:")
print(weather["date"].min(), "tot", weather["date"].max())


# ============================================================
# STAP 3: METRICS KIEZEN
# ============================================================

print_section("STAP 3 - Metrics kiezen")

valence_metric, valence_description = choose_valence_metric(spotify)
sad_metric, sad_description = choose_sad_metric(spotify)

print("Valence metric:", valence_metric, "-", valence_description)

if sad_metric is not None:
    print("Sad metric:", sad_metric, "-", sad_description)


# ============================================================
# STAP 4: MERGEN MET WEATHER
# ============================================================

print_section("STAP 4 - Mergen met weather")

weather_small = weather[["date", "prcp", "tsun"]].drop_duplicates(subset=["date"]).copy()

df = pd.merge(
    spotify,
    weather_small,
    on="date",
    how="left"
)

print("Shape na merge:", df.shape)

print("\nMissende weatherwaarden:")
print(df[["prcp", "tsun"]].isna().sum())


=======
# - daily_valence_summary.csv en weather.csv samenbrengen
# - 2 extreme weergroepen definiëren:
#     1. strict_rainy
#     2. strict_sunny
# - Vergelijken via BOXPLOTS voor:
#     A. avg_valence         (gemiddelde positiviteit per dag)
#     B. avg_energy          (gemiddelde energie per dag)
#     C. share_sad_songs     (aandeel lage valence songs)
#     D. share_low_energy    (aandeel lage energy songs)
#     E. share_depressief    (aandeel lage valence + lage energy)
# - Significantietest (Mann-Whitney U) per metric
# - Samenvattende statistieken opslaan
#
# EXTREME WEERGROEPEN:
# - strict_rainy_day = prcp >= 5.0 EN tsun <= q25
# - strict_sunny_day = prcp < 1.0 EN tsun >= q75
#
# DREMPELWAARDEN (overeenkomen met main.py):
# - SAD_THRESHOLD        = 0.40
# - LOW_ENERGY_THRESHOLD = 0.50
# ============================================================

SPOTIFY_DAILY_FILE = "daily_valence_summary.csv"
WEATHER_FILE = "weather.csv"

OUTPUT_FINAL_DATASET = "final_dataset.csv"
OUTPUT_EXTREME_DATASET = "extreme_weather_dataset.csv"
OUTPUT_SUMMARY_EXTREME = "summary_strict_rainy_vs_strict_sunny.csv"

STRICT_RAIN_THRESHOLD = 5.0
DRY_FOR_SUNNY_THRESHOLD = 1.0
COVERAGE_THRESHOLD = 0.70

COLOR_RAINY = "#4C72B0"
COLOR_SUNNY = "#DD8452"


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
    "top200_rows", "tracks_with_valence", "valence_coverage",
    "avg_valence", "median_valence", "std_valence", "min_valence", "max_valence",
    "sad_songs_count", "share_sad_songs",
    "avg_energy", "median_energy", "std_energy",
    "tracks_with_energy", "low_energy_count", "share_low_energy",
    "tracks_with_both", "depressief_count", "share_depressief",
    "total_streams", "sad_streams", "weighted_avg_valence", "share_sad_streams",
    "weighted_avg_energy", "share_low_energy_streams", "share_depressief_streams",
    "prcp", "tsun"
]

for col in numeric_candidates:
    if col in spotify.columns:
        spotify[col] = pd.to_numeric(spotify[col], errors="coerce")
    if col in weather.columns:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

print("Spotify daily shape:", spotify.shape)
print("Weather shape:", weather.shape)
print("\nSpotify datumbereik:", spotify["date"].min().date(), "tot", spotify["date"].max().date())
print("Weather datumbereik:", weather["date"].min().date(), "tot", weather["date"].max().date())

has_energy          = "avg_energy"        in spotify.columns and spotify["avg_energy"].notna().any()
has_share_low_energy = "share_low_energy" in spotify.columns and spotify["share_low_energy"].notna().any()
has_share_depressief = "share_depressief" in spotify.columns and spotify["share_depressief"].notna().any()

print(f"\nBeschikbare metrics:")
print(f"  avg_valence       : altijd")
print(f"  avg_energy        : {has_energy}")
print(f"  share_sad_songs   : altijd")
print(f"  share_low_energy  : {has_share_low_energy}")
print(f"  share_depressief  : {has_share_depressief}")


# ============================================================
# STAP 3: MERGEN
# ============================================================

weather_small = weather[["date", "prcp", "tsun"]].copy()
df = pd.merge(spotify, weather_small, on="date", how="inner")
weather_small = weather[["date", "prcp", "tsun"]].copy()
df = pd.merge(spotify, weather_small, on="date", how="inner")

print("\nShape na merge:", df.shape)
print("\nShape na merge:", df.shape)


# ============================================================
# STAP 4: HOOFDVARIABELEN KIEZEN
>>>>>>> e3f4d82540d7fff0eb431d53d14af90df3fd9450
# ============================================================
# STAP 5: CLEANING + SEIZOENEN + WEERTYPES
# ============================================================

print_section("STAP 5 - Cleaning, seizoenen en weertypes")

if "valence_coverage" not in df.columns:
    raise ValueError("Kolom 'valence_coverage' ontbreekt in daily_top200_valence_summary.csv.")

needed_cols = ["date", "valence_coverage", valence_metric, "prcp", "tsun"]

df = df.dropna(subset=needed_cols).copy()
df = df[df["valence_coverage"] >= COVERAGE_THRESHOLD].copy()

if df.empty:
    raise ValueError(
        "Geen data over na cleaning.\n"
        "Mogelijke oorzaken:\n"
        "- te lage valence coverage\n"
        "- weather.csv overlapt niet met daily_top200_valence_summary.csv\n"
        "- prcp of tsun bevat te veel missende waarden"
    )

df["season"] = df["date"].apply(assign_season)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

median_tsun = df["tsun"].median()

<<<<<<< HEAD
df["rainy_day"] = df["prcp"] > 0
df["dry_day"] = df["prcp"] == 0

df["drizzly_day"] = (
    (df["prcp"] >= DRIZZLY_RAIN_THRESHOLD) &
    (df["tsun"] <= median_tsun)
)

df["sunny_day"] = (
    (df["prcp"] < SUNNY_RAIN_MAX) &
    (df["tsun"] >= median_tsun)
)

df["weather_group"] = pd.NA
df.loc[df["drizzly_day"], "weather_group"] = "drizzly"
df.loc[df["sunny_day"], "weather_group"] = "sunny"

def classify_weather(row):
    if row["drizzly_day"]:
        return "drizzly"
    if row["sunny_day"]:
        return "sunny"
    if row["prcp"] > 0 and row["tsun"] > median_tsun:
        return "rainy_but_bright"
    if row["prcp"] == 0 and row["tsun"] < median_tsun:
        return "dry_cloudy"
    return "other"

df["weather_category"] = df.apply(classify_weather, axis=1)

df.to_csv(OUTPUT_FINAL_DATASET, index=False)

print(f"Bestand opgeslagen als {OUTPUT_FINAL_DATASET}")

print("\nAantal dagen na cleaning:", len(df))
print("Valence coverage threshold:", COVERAGE_THRESHOLD)
print("Mediaan tsun:", round(median_tsun, 4))

print("\nAantal dagen per seizoen:")
print(df["season"].value_counts())

print("\nAantal dagen per weather_category:")
print(df["weather_category"].value_counts())

print("\nAantal drizzly/sunny dagen:")
print(df["weather_group"].value_counts(dropna=False))


# ============================================================
# STAP 6: SEIZOENSSAMENVATTING
# ============================================================

print_section("STAP 6 - Seizoenssamenvatting")

agg_dict = {
    "aantal_dagen": ("date", "count"),
    "gemiddelde_valence": (valence_metric, "mean"),
    "mediaan_valence": (valence_metric, "median"),
    "std_valence": (valence_metric, "std"),
    "gemiddelde_prcp": ("prcp", "mean"),
    "gemiddelde_tsun": ("tsun", "mean"),
    "gemiddelde_valence_coverage": ("valence_coverage", "mean")
}

if sad_metric is not None:
    agg_dict["gemiddelde_sad_share"] = (sad_metric, "mean")

season_summary = (
    df
    .groupby("season")
    .agg(**agg_dict)
    .reset_index()
)

season_order = ["winter", "spring", "summer", "autumn"]

season_summary["season"] = pd.Categorical(
    season_summary["season"],
    categories=season_order,
    ordered=True
)

season_summary = season_summary.sort_values("season").reset_index(drop=True)

season_summary.to_csv("season_summary.csv", index=False)

print("Bestand opgeslagen als season_summary.csv")
print(season_summary.to_string(index=False))


# ============================================================
# STAP 7: WEERSAMENVATTING
# ============================================================

print_section("STAP 7 - Weersamenvatting")

weather_summary = (
    df
    .groupby("weather_category")
    .agg(**agg_dict)
    .reset_index()
    .sort_values("gemiddelde_valence", ascending=False)
)

weather_summary.to_csv("weather_summary.csv", index=False)

print("Bestand opgeslagen als weather_summary.csv")
print(weather_summary.to_string(index=False))


# ============================================================
# STAP 8: SEIZOEN + WEER
# ============================================================

print_section("STAP 8 - Seizoen + weer")

season_weather_summary = (
    df
    .groupby(["season", "weather_category"])
    .agg(**agg_dict)
    .reset_index()
)

season_weather_summary["season"] = pd.Categorical(
    season_weather_summary["season"],
    categories=season_order,
    ordered=True
)

season_weather_summary = season_weather_summary.sort_values(
    ["season", "weather_category"]
).reset_index(drop=True)

season_weather_summary.to_csv("season_weather_summary.csv", index=False)

print("Bestand opgeslagen als season_weather_summary.csv")
print(season_weather_summary.to_string(index=False))


# ============================================================
# STAP 9: DRIZZLY VS SUNNY
# ============================================================

print_section("STAP 9 - Drizzly vs sunny")

comparison_df = df[df["weather_group"].isin(["drizzly", "sunny"])].copy()

if not comparison_df.empty:
    drizzly_vs_sunny_summary = (
        comparison_df
        .groupby("weather_group")
        .agg(**agg_dict)
        .reset_index()
    )

    drizzly_vs_sunny_summary.to_csv("drizzly_vs_sunny_summary.csv", index=False)

    print("Bestand opgeslagen als drizzly_vs_sunny_summary.csv")
    print(drizzly_vs_sunny_summary.to_string(index=False))

    if comparison_df["weather_group"].nunique() == 2:
        drizzly_mean = comparison_df.loc[
            comparison_df["weather_group"] == "drizzly",
            valence_metric
        ].mean()

        sunny_mean = comparison_df.loc[
            comparison_df["weather_group"] == "sunny",
            valence_metric
        ].mean()

        print("\nGemiddelde valence drizzly:", round(drizzly_mean, 4))
        print("Gemiddelde valence sunny:", round(sunny_mean, 4))
        print("Verschil drizzly - sunny:", round(drizzly_mean - sunny_mean, 4))
else:
    print("Geen drizzly/sunny dagen gevonden.")


# ============================================================
# STAP 10: RAINY VS DRY
# ============================================================

print_section("STAP 10 - Rainy vs dry")

df["rain_group"] = pd.NA
df.loc[df["rainy_day"], "rain_group"] = "rainy"
df.loc[df["dry_day"], "rain_group"] = "dry"

rainy_vs_dry_summary = (
    df
    .dropna(subset=["rain_group"])
    .groupby("rain_group")
    .agg(**agg_dict)
    .reset_index()
)

rainy_vs_dry_summary.to_csv("rainy_vs_dry_summary.csv", index=False)

print("Bestand opgeslagen als rainy_vs_dry_summary.csv")
print(rainy_vs_dry_summary.to_string(index=False))


# ============================================================
# STAP 11: CORRELATIES
# ============================================================

print_section("STAP 11 - Correlaties")

corr_rows = [
    {
        "metric": valence_metric,
        "description": valence_description,
        "corr_with_prcp": correlation_safe(df, valence_metric, "prcp"),
        "corr_with_tsun": correlation_safe(df, valence_metric, "tsun")
    }
]

if sad_metric is not None:
    corr_rows.append({
        "metric": sad_metric,
        "description": sad_description,
        "corr_with_prcp": correlation_safe(df, sad_metric, "prcp"),
        "corr_with_tsun": correlation_safe(df, sad_metric, "tsun")
    })

correlation_df = pd.DataFrame(corr_rows)

correlation_df.to_csv("correlation_valence_weather.csv", index=False)

print("Bestand opgeslagen als correlation_valence_weather.csv")
print(correlation_df.to_string(index=False))


# ============================================================
# STAP 12: SIGNIFICANTIETESTEN
# ============================================================

print_section("STAP 12 - Significantietesten")

test_rows = []

if SCIPY_AVAILABLE:
    if comparison_df["weather_group"].nunique() == 2:
        drizzly_values = comparison_df.loc[
            comparison_df["weather_group"] == "drizzly",
            valence_metric
        ].dropna()

        sunny_values = comparison_df.loc[
            comparison_df["weather_group"] == "sunny",
            valence_metric
        ].dropna()

        if len(drizzly_values) >= 2 and len(sunny_values) >= 2:
            t_stat, t_p = ttest_ind(
                drizzly_values,
                sunny_values,
                equal_var=False,
                nan_policy="omit"
            )

            u_stat, u_p = mannwhitneyu(
                drizzly_values,
                sunny_values,
                alternative="two-sided"
            )

            test_rows.append({
                "comparison": "drizzly_vs_sunny",
                "metric": valence_metric,
                "n_group_1": len(drizzly_values),
                "n_group_2": len(sunny_values),
                "mean_group_1": drizzly_values.mean(),
                "mean_group_2": sunny_values.mean(),
                "difference_group_1_minus_group_2": drizzly_values.mean() - sunny_values.mean(),
                "welch_t_test_p_value": t_p,
                "mann_whitney_p_value": u_p
            })

    rainy_values = df.loc[df["rain_group"] == "rainy", valence_metric].dropna()
    dry_values = df.loc[df["rain_group"] == "dry", valence_metric].dropna()

    if len(rainy_values) >= 2 and len(dry_values) >= 2:
        t_stat, t_p = ttest_ind(
            rainy_values,
            dry_values,
            equal_var=False,
            nan_policy="omit"
        )

        u_stat, u_p = mannwhitneyu(
            rainy_values,
            dry_values,
            alternative="two-sided"
        )

        test_rows.append({
            "comparison": "rainy_vs_dry",
            "metric": valence_metric,
            "n_group_1": len(rainy_values),
            "n_group_2": len(dry_values),
            "mean_group_1": rainy_values.mean(),
            "mean_group_2": dry_values.mean(),
            "difference_group_1_minus_group_2": rainy_values.mean() - dry_values.mean(),
            "welch_t_test_p_value": t_p,
            "mann_whitney_p_value": u_p
        })

    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv("significance_tests_valence_weather.csv", index=False)

    print("Bestand opgeslagen als significance_tests_valence_weather.csv")

    if not tests_df.empty:
        print(tests_df.to_string(index=False))
    else:
        print("Niet genoeg observaties voor significantietesten.")
else:
    print("SciPy niet beschikbaar. Geen significantietesten uitgevoerd.")


# ============================================================
# STAP 13: GRAFIEKEN
# ============================================================

print_section("STAP 13 - Grafieken maken")

save_bar_plot(
    data=season_summary,
    x_col="season",
    y_col="gemiddelde_valence",
    title="Gemiddelde Top 200 valence per seizoen",
    xlabel="Seizoen",
    ylabel=valence_description,
    filename="grafiek_valence_per_seizoen.png"
)

if sad_metric is not None and "gemiddelde_sad_share" in season_summary.columns:
    save_bar_plot(
        data=season_summary,
        x_col="season",
        y_col="gemiddelde_sad_share",
        title="Gemiddeld aandeel sad songs per seizoen",
        xlabel="Seizoen",
        ylabel=sad_description,
        filename="grafiek_sad_share_per_seizoen.png"
    )

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df[valence_metric])
plt.title("Top 200 valence doorheen de tijd")
plt.xlabel("Datum")
plt.ylabel(valence_description)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafiek_valence_doorheen_tijd.png")
plt.close()

print("Grafiek opgeslagen: grafiek_valence_doorheen_tijd.png")

plt.figure(figsize=(8, 5))
plt.scatter(df["prcp"], df[valence_metric])
plt.title("Neerslag vs Top 200 valence")
plt.xlabel("Neerslag / prcp")
plt.ylabel(valence_description)
plt.tight_layout()
plt.savefig("grafiek_prcp_vs_valence.png")
plt.close()

print("Grafiek opgeslagen: grafiek_prcp_vs_valence.png")

plt.figure(figsize=(8, 5))
plt.scatter(df["tsun"], df[valence_metric])
plt.title("Zonneschijnduur vs Top 200 valence")
plt.xlabel("Zonneschijnduur / tsun")
plt.ylabel(valence_description)
plt.tight_layout()
plt.savefig("grafiek_tsun_vs_valence.png")
plt.close()

print("Grafiek opgeslagen: grafiek_tsun_vs_valence.png")

save_boxplot(
    df=comparison_df,
    column=valence_metric,
    by="weather_group",
    title="Top 200 valence: drizzly vs sunny",
    xlabel="Weather group",
    ylabel=valence_description,
    filename="grafiek_boxplot_drizzly_vs_sunny_valence.png"
)

save_boxplot(
    df=df.dropna(subset=["rain_group"]),
    column=valence_metric,
    by="rain_group",
    title="Top 200 valence: rainy vs dry",
    xlabel="Rain group",
    ylabel=valence_description,
    filename="grafiek_boxplot_rainy_vs_dry_valence.png"
)

save_bar_plot(
    data=weather_summary,
    x_col="weather_category",
    y_col="gemiddelde_valence",
    title="Gemiddelde valence per weertype",
    xlabel="Weertype",
    ylabel=valence_description,
    filename="grafiek_valence_per_weertype.png"
)


# ============================================================
# STAP 14: INTERPRETATIE
# ============================================================

print_section("STAP 14 - Korte interpretatie")

print("Gebruikte valence metric:")
print("-", valence_metric, "=", valence_description)

if sad_metric is not None:
    print("\nGebruikte sad metric:")
    print("-", sad_metric, "=", sad_description)

best_season = season_summary.sort_values("gemiddelde_valence", ascending=False).iloc[0]
worst_season = season_summary.sort_values("gemiddelde_valence", ascending=True).iloc[0]

print(
    f"\nSeizoen met hoogste gemiddelde Top 200 valence: "
    f"{best_season['season']} ({round(best_season['gemiddelde_valence'], 4)})"
)

print(
    f"Seizoen met laagste gemiddelde Top 200 valence: "
    f"{worst_season['season']} ({round(worst_season['gemiddelde_valence'], 4)})"
)

if comparison_df["weather_group"].nunique() == 2:
    drizzly_mean = comparison_df.loc[
        comparison_df["weather_group"] == "drizzly",
        valence_metric
    ].mean()

    sunny_mean = comparison_df.loc[
        comparison_df["weather_group"] == "sunny",
        valence_metric
    ].mean()

    print("\nGemiddelde valence op druilerige dagen:", round(drizzly_mean, 4))
    print("Gemiddelde valence op zonnige dagen:", round(sunny_mean, 4))
    print("Verschil druilerig - zonnig:", round(drizzly_mean - sunny_mean, 4))

corr_prcp = correlation_safe(df, valence_metric, "prcp")
corr_tsun = correlation_safe(df, valence_metric, "tsun")

print(
    "\nCorrelatie tussen neerslag en valence:",
    round(float(corr_prcp), 4) if pd.notna(corr_prcp) else "niet beschikbaar"
)

print(
    "Correlatie tussen zonneschijnduur en valence:",
    round(float(corr_tsun), 4) if pd.notna(corr_tsun) else "niet beschikbaar"
)

print(
    "\nBelangrijk: deze analyse meet patronen binnen de Belgische Spotify Top 200, "
    "niet binnen alle Spotify-streams in België."
)

=======
main_valence_metric      = "avg_valence"
main_sad_metric          = "share_sad_songs"
main_energy_metric       = "avg_energy"       if has_energy          else None
main_low_energy_metric   = "share_low_energy" if has_share_low_energy else None
main_depressief_metric   = "share_depressief" if has_share_depressief else None

if "weighted_avg_valence" in df.columns and df["weighted_avg_valence"].notna().sum() > 0:
    main_valence_metric = "weighted_avg_valence"

if "share_sad_streams" in df.columns and df["share_sad_streams"].notna().sum() > 0:
    main_sad_metric = "share_sad_streams"

if has_energy and "weighted_avg_energy" in df.columns and df["weighted_avg_energy"].notna().sum() > 0:
    main_energy_metric = "weighted_avg_energy"

if has_share_low_energy and "share_low_energy_streams" in df.columns and df["share_low_energy_streams"].notna().sum() > 0:
    main_low_energy_metric = "share_low_energy_streams"

if has_share_depressief and "share_depressief_streams" in df.columns and df["share_depressief_streams"].notna().sum() > 0:
    main_depressief_metric = "share_depressief_streams"

print("\nGebruikte hoofdvariabelen:")
print(f"  Valence metric    : {main_valence_metric}")
print(f"  Sad metric        : {main_sad_metric}")
print(f"  Energy metric     : {main_energy_metric}")
print(f"  Low energy metric : {main_low_energy_metric}")
print(f"  Depressief metric : {main_depressief_metric}")


# ============================================================
# STAP 5: ANALYSEBASIS OPSCHONEN
# ============================================================

needed_cols = ["date", "valence_coverage", main_valence_metric, main_sad_metric, "prcp", "tsun"]
df = df.dropna(subset=needed_cols).copy()
df = df[df["valence_coverage"] >= COVERAGE_THRESHOLD].copy()

print(f"\nShape na cleaning + coveragefilter (>= {COVERAGE_THRESHOLD}):", df.shape)


# ============================================================
# STAP 6: EXTREME WEERGROEPEN DEFINIËREN
# ============================================================

low_sun_threshold  = df["tsun"].quantile(0.25)
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
rainy = extreme_df[extreme_df["extreme_group"] == "strict_rainy"]
sunny = extreme_df[extreme_df["extreme_group"] == "strict_sunny"]

print(f"\nStrenge definities:")
print(f"  strict_rainy_day = prcp >= {STRICT_RAIN_THRESHOLD} EN tsun <= q25 ({round(low_sun_threshold, 1)})")
print(f"  strict_sunny_day = prcp < {DRY_FOR_SUNNY_THRESHOLD} EN tsun >= q75 ({round(high_sun_threshold, 1)})")
print(f"\nAantal strict rainy days : {len(rainy)}")
print(f"Aantal strict sunny days : {len(sunny)}")

if extreme_df.empty:
    raise ValueError("Geen extreme dagen gevonden. Maak de drempels iets minder streng.")
if extreme_df["extreme_group"].nunique() < 2:
    raise ValueError("Slechts één extreme groep gevonden. Maak de definities iets minder streng.")


# ============================================================
# STAP 7: SIGNIFICANTIETESTS (Mann-Whitney U)
# ============================================================

def mann_whitney_p(serie_a, serie_b):
    a = serie_a.dropna()
    b = serie_b.dropna()
    if len(a) < 3 or len(b) < 3:
        return None
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p

def p_naar_sterren(p):
    if p is None:
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def print_mw(label, col, rainy_df, sunny_df):
    p = mann_whitney_p(rainy_df[col], sunny_df[col])
    sig = p_naar_sterren(p)
    p_str = f"{round(p, 4)}" if p is not None else "n/a"
    print(f"  {label:<35} p = {p_str:<8} {sig}")

print("\nSignificantietests (Mann-Whitney U, tweezijdig):")
print("  (* p<0.05  ** p<0.01  *** p<0.001  ns = niet significant)\n")

print_mw(main_valence_metric, main_valence_metric, rainy, sunny)
print_mw(main_sad_metric, main_sad_metric, rainy, sunny)

if main_energy_metric and main_energy_metric in extreme_df.columns:
    print_mw(main_energy_metric, main_energy_metric, rainy, sunny)

if main_low_energy_metric and main_low_energy_metric in extreme_df.columns:
    print_mw(main_low_energy_metric, main_low_energy_metric, rainy, sunny)

if main_depressief_metric and main_depressief_metric in extreme_df.columns:
    print_mw(main_depressief_metric, main_depressief_metric, rainy, sunny)


# ============================================================
# STAP 8: GEMIDDELDEN PRINTEN
# ============================================================

def print_vergelijking(label, col, rainy_df, sunny_df):
    r = rainy_df[col].dropna()
    s = sunny_df[col].dropna()
    if r.empty or s.empty:
        print(f"\n{label}: geen data")
        return
    diff = r.mean() - s.mean()
    richting = "LAGER" if diff < 0 else "HOGER"
    print(f"\n{label}:")
    print(f"  strict_rainy  gem={round(r.mean(),4)}  med={round(r.median(),4)}")
    print(f"  strict_sunny  gem={round(s.mean(),4)}  med={round(s.median(),4)}")
    print(f"  Verschil (rainy - sunny) = {round(diff,4)}  -> rainy ligt {richting} dan sunny")

print("\n" + "="*60)
print("GEMIDDELDEN STRICT RAINY VS STRICT SUNNY")
print("="*60)

print_vergelijking("Valence", main_valence_metric, rainy, sunny)
print_vergelijking("Sad share", main_sad_metric, rainy, sunny)

if main_energy_metric and main_energy_metric in extreme_df.columns:
    print_vergelijking("Energy", main_energy_metric, rainy, sunny)

if main_low_energy_metric and main_low_energy_metric in extreme_df.columns:
    print_vergelijking("Low energy share", main_low_energy_metric, rainy, sunny)

if main_depressief_metric and main_depressief_metric in extreme_df.columns:
    print_vergelijking("Depressief share", main_depressief_metric, rainy, sunny)


# ============================================================
# STAP 9: BESTANDEN OPSLAAN
# ============================================================

df.to_csv(OUTPUT_FINAL_DATASET, index=False)
extreme_df.to_csv(OUTPUT_EXTREME_DATASET, index=False)

agg_dict = {
    "aantal_dagen":  ("date", "count"),
    "gem_valence":   (main_valence_metric, "mean"),
    "med_valence":   (main_valence_metric, "median"),
    "gem_sad_share": (main_sad_metric, "mean"),
    "gem_prcp":      ("prcp", "mean"),
    "gem_tsun":      ("tsun", "mean"),
}

if main_energy_metric and main_energy_metric in extreme_df.columns:
    agg_dict["gem_energy"] = (main_energy_metric, "mean")
    agg_dict["med_energy"] = (main_energy_metric, "median")

if main_low_energy_metric and main_low_energy_metric in extreme_df.columns:
    agg_dict["gem_low_energy_share"] = (main_low_energy_metric, "mean")

if main_depressief_metric and main_depressief_metric in extreme_df.columns:
    agg_dict["gem_depressief_share"] = (main_depressief_metric, "mean")

summary_extreme = extreme_df.groupby("extreme_group").agg(**agg_dict).reset_index()
summary_extreme.to_csv(OUTPUT_SUMMARY_EXTREME, index=False)

print(f"\nBestanden opgeslagen:")
print(f"  {OUTPUT_FINAL_DATASET}")
print(f"  {OUTPUT_EXTREME_DATASET}")
print(f"  {OUTPUT_SUMMARY_EXTREME}")


# ============================================================
# STAP 10: BOXPLOTS
# ============================================================
# Één gecombineerde figuur met alle beschikbare metrics.
# Elke subplot = één metric, rainy (blauw) vs sunny (oranje).
# Significantie (Mann-Whitney U) boven elke boxplot.

def teken_boxplot(ax, rainy_data, sunny_data, ylabel, titel, p_waarde=None):
    r = rainy_data.dropna().values
    s = sunny_data.dropna().values

    bp = ax.boxplot(
        [r, s],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.4, linestyle="none")
    )

    bp["boxes"][0].set_facecolor(COLOR_RAINY)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(COLOR_SUNNY)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Regenachtig\n(n={len(r)})", f"Zonnig\n(n={len(s)})"],
        fontsize=9
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(titel, fontsize=10, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Significantie-annotatie
    if p_waarde is not None:
        sterren = p_naar_sterren(p_waarde)
        all_vals = list(r) + list(s)
        if all_vals:
            y_max = max(all_vals)
            y_range = y_max - min(all_vals)
            y_pos = y_max + y_range * 0.08
            ax.annotate(
                sterren,
                xy=(1.5, y_pos),
                ha="center",
                fontsize=13,
                fontweight="bold",
                color="black"
            )


# --- Metrics samenstellen ---
plot_metrics = []

plot_metrics.append({
    "rainy": rainy[main_valence_metric],
    "sunny": sunny[main_valence_metric],
    "ylabel": "Gemiddelde valence per dag",
    "titel": "Valence\n(positiviteit van songs)",
    "p": mann_whitney_p(rainy[main_valence_metric], sunny[main_valence_metric])
})

if main_energy_metric and main_energy_metric in extreme_df.columns:
    plot_metrics.append({
        "rainy": rainy[main_energy_metric],
        "sunny": sunny[main_energy_metric],
        "ylabel": "Gemiddelde energy per dag",
        "titel": "Energy\n(intensiteit van songs)",
        "p": mann_whitney_p(rainy[main_energy_metric], sunny[main_energy_metric])
    })

plot_metrics.append({
    "rainy": rainy[main_sad_metric],
    "sunny": sunny[main_sad_metric],
    "ylabel": "Aandeel songs",
    "titel": "Aandeel lage valence\n(valence ≤ 0.40)",
    "p": mann_whitney_p(rainy[main_sad_metric], sunny[main_sad_metric])
})

if main_low_energy_metric and main_low_energy_metric in extreme_df.columns:
    plot_metrics.append({
        "rainy": rainy[main_low_energy_metric],
        "sunny": sunny[main_low_energy_metric],
        "ylabel": "Aandeel songs",
        "titel": "Aandeel lage energy\n(energy ≤ 0.50)",
        "p": mann_whitney_p(rainy[main_low_energy_metric], sunny[main_low_energy_metric])
    })

if main_depressief_metric and main_depressief_metric in extreme_df.columns:
    plot_metrics.append({
        "rainy": rainy[main_depressief_metric],
        "sunny": sunny[main_depressief_metric],
        "ylabel": "Aandeel songs",
        "titel": "Aandeel depressieve songs\n(valence ≤ 0.40 én energy ≤ 0.50)",
        "p": mann_whitney_p(rainy[main_depressief_metric], sunny[main_depressief_metric])
    })

# --- Layout berekenen ---
n_plots = len(plot_metrics)
ncols = min(n_plots, 3)
nrows = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5 * ncols, 5.5 * nrows))
axes = [axes] if n_plots == 1 else axes.flatten()

for i, metric in enumerate(plot_metrics):
    teken_boxplot(
        axes[i],
        metric["rainy"],
        metric["sunny"],
        metric["ylabel"],
        metric["titel"],
        p_waarde=metric["p"]
    )

for j in range(n_plots, len(axes)):
    axes[j].set_visible(False)

# Legenda
rainy_patch = mpatches.Patch(color=COLOR_RAINY, alpha=0.75, label="Strict regenachtig")
sunny_patch = mpatches.Patch(color=COLOR_SUNNY, alpha=0.75, label="Strict zonnig")
fig.legend(
    handles=[rainy_patch, sunny_patch],
    loc="lower center",
    ncol=2,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, 0.0)
)

fig.suptitle(
    "Spotify muziekkenmerken op regenachtige vs zonnige dagen in België",
    fontsize=13,
    fontweight="bold",
    y=1.02
)
fig.text(
    0.5, -0.01,
    "* p<0.05   ** p<0.01   *** p<0.001   ns = niet significant  |  Mann-Whitney U test",
    ha="center", fontsize=8, color="gray"
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("boxplots_regen_vs_zon.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGrafiek opgeslagen als boxplots_regen_vs_zon.png")
>>>>>>> e3f4d82540d7fff0eb431d53d14af90df3fd9450
print("\nKlaar.")