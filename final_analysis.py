import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

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

print("\nShape na merge:", df.shape)


# ============================================================
# STAP 4: HOOFDVARIABELEN KIEZEN
# ============================================================

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
print("\nKlaar.")