import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ============================================================
# FINAL_ANALYSIS_WEATHER_SPOTIFY.PY
# ============================================================
# DOEL:
# Sterkere analyse van de vraag:
#
# "Wordt er op regenachtige / slechte weerdagen meer naar droevige
# muziek geluisterd dan op droge dagen?"
#
# Belangrijkste methodologische verbeteringen:
#
# 1. Niet gewone gemiddelde valence als hoofdvariabele.
#    Want dan telt plaats 1 even zwaar als plaats 200.
#
# 2. Wel streamgewogen variabelen gebruiken:
#    - share_sad_streams_matched
#    - weighted_avg_valence
#
# 3. Regen sterker definiëren:
#    - droge dag: prcp < 1 mm
#    - regendag: prcp >= 1 mm
#    - zware regen: prcp >= 5 mm
#
# 4. Regenintensiteit gebruiken:
#    - droog
#    - lichte regen
#    - zware regen
#
# 5. Seizoen en weekday toevoegen als controlevariabelen.
#    Zo vermijden we dat seizoenseffecten het verband vertekenen.
#
# 6. Regressies draaien:
#    - met log_prcp
#    - met rainy_day
#    - gecontroleerd voor seizoen en weekdag
# ============================================================


# ============================================================
# INSTELLINGEN
# ============================================================

SPOTIFY_FILE = "spotify_belgium_top200_with_valence.csv"
WEATHER_FILE = "weather.csv"

OUTPUT_ANALYSIS_DATASET = "analysis_dataset_stronger.csv"
OUTPUT_FINAL_MERGED_DATASET = "final_spotify_weather_dataset_stronger.csv"

SAD_THRESHOLD = 0.35

# Kwaliteitsfilters
MIN_TOP200_ROWS = 180
MIN_VALENCE_COVERAGE = 0.50

# Regen-definities
RAIN_THRESHOLD_MM = 1.0
HEAVY_RAIN_THRESHOLD_MM = 5.0


# ============================================================
# HULPFUNCTIES
# ============================================================

def check_file_exists(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Bestand niet gevonden: {file_path}\n"
            f"Zorg dat dit bestand in dezelfde map staat als dit script."
        )


def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "lente"
    elif month in [6, 7, 8]:
        return "zomer"
    else:
        return "herfst"


def cohen_d(group_a, group_b):
    """
    Effect size.
    Positief betekent: gemiddelde groep A > gemiddelde groep B.
    """
    a = group_a.dropna()
    b = group_b.dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_std = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )

    if pooled_std == 0 or pd.isna(pooled_std):
        return np.nan

    return (a.mean() - b.mean()) / pooled_std


def welch_ttest(group_a, group_b):
    """
    Welch t-test.
    Geeft p-value terug.
    """
    a = group_a.dropna()
    b = group_b.dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    if not SCIPY_AVAILABLE:
        return np.nan

    test = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return test.pvalue


def safe_weighted_average(values, weights):
    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return np.average(values[valid], weights=weights[valid])


def calculate_daily_spotify_metrics(group):
    """
    Zet song-level Spotify-data om naar dagniveau.

    Belangrijk:
    - We berekenen de hoofdvariabelen enkel op nummers met valence.
    - Streams worden gebruikt als gewicht.
    - Zo meten we beter effectief luistergedrag.
    """

    top200_rows = len(group)

    valid_valence = group[group["valence"].notna()].copy()
    tracks_with_valence = len(valid_valence)

    if top200_rows > 0:
        valence_coverage = tracks_with_valence / top200_rows
    else:
        valence_coverage = np.nan

    if tracks_with_valence > 0:
        avg_valence = valid_valence["valence"].mean()
        sad_songs_count = (valid_valence["valence"] < SAD_THRESHOLD).sum()
        share_sad_songs_matched = sad_songs_count / tracks_with_valence
    else:
        avg_valence = np.nan
        sad_songs_count = np.nan
        share_sad_songs_matched = np.nan

    # Stream-based analyse
    valid_streams = valid_valence[
        valid_valence["streams"].notna() &
        (valid_valence["streams"] > 0)
    ].copy()

    total_streams_all = group["streams"].sum(skipna=True)
    total_streams_matched = valid_streams["streams"].sum(skipna=True)

    sad_streams = valid_streams.loc[
        valid_streams["valence"] < SAD_THRESHOLD,
        "streams"
    ].sum(skipna=True)

    if total_streams_matched > 0:
        share_sad_streams_matched = sad_streams / total_streams_matched
        weighted_avg_valence = safe_weighted_average(
            valid_streams["valence"],
            valid_streams["streams"]
        )
    else:
        share_sad_streams_matched = np.nan
        weighted_avg_valence = np.nan

    if total_streams_all > 0:
        share_sad_streams_all = sad_streams / total_streams_all
    else:
        share_sad_streams_all = np.nan

    result = {
        "top200_rows": top200_rows,
        "unique_tracks": group["track_id"].nunique() if "track_id" in group.columns else np.nan,
        "tracks_with_valence": tracks_with_valence,
        "valence_coverage": valence_coverage,

        # Niet-streamgewogen variabelen
        "avg_valence": avg_valence,
        "sad_songs_count": sad_songs_count,
        "share_sad_songs_matched": share_sad_songs_matched,

        # Streamgewogen hoofdvariabelen
        "total_streams_all": total_streams_all,
        "total_streams_matched": total_streams_matched,
        "sad_streams": sad_streams,
        "share_sad_streams_matched": share_sad_streams_matched,
        "share_sad_streams_all": share_sad_streams_all,
        "weighted_avg_valence": weighted_avg_valence
    }

    return pd.Series(result)


def make_group_summary(df, group_col, main_sad_metric, main_valence_metric):
    """
    Maakt samenvatting per groep.
    Bijvoorbeeld:
    - rainy_day
    - rain_intensity
    - season
    """

    agg_dict = {
        "aantal_dagen": ("date", "count"),
        "gemiddelde_prcp": ("prcp", "mean"),
        "mediaan_prcp": ("prcp", "median"),
        "gemiddelde_sad_stream_share": (main_sad_metric, "mean"),
        "mediaan_sad_stream_share": (main_sad_metric, "median"),
        "gemiddelde_weighted_valence": (main_valence_metric, "mean"),
        "mediaan_weighted_valence": (main_valence_metric, "median"),
        "gemiddelde_valence_coverage": ("valence_coverage", "mean"),
        "gemiddelde_top200_rows": ("top200_rows", "mean")
    }

    if "tsun" in df.columns and df["tsun"].notna().sum() > 0:
        agg_dict["gemiddelde_tsun"] = ("tsun", "mean")

    summary = df.groupby(group_col).agg(**agg_dict).reset_index()

    return summary


def compare_two_groups(df, group_col, group_a_value, group_b_value, metrics):
    """
    Vergelijkt twee groepen.
    Bijvoorbeeld:
    - regendagen vs droge dagen
    - zware regen vs droog
    """

    results = []

    group_a = df[df[group_col] == group_a_value].copy()
    group_b = df[df[group_col] == group_b_value].copy()

    for metric in metrics:
        a = group_a[metric].dropna()
        b = group_b[metric].dropna()

        results.append({
            "comparison": f"{group_a_value}_vs_{group_b_value}",
            "group_col": group_col,
            "metric": metric,
            "group_a": group_a_value,
            "group_b": group_b_value,
            "n_a": len(a),
            "n_b": len(b),
            "mean_a": a.mean() if len(a) > 0 else np.nan,
            "mean_b": b.mean() if len(b) > 0 else np.nan,
            "difference_a_minus_b": (
                a.mean() - b.mean()
                if len(a) > 0 and len(b) > 0
                else np.nan
            ),
            "cohen_d": cohen_d(a, b),
            "p_value_welch": welch_ttest(a, b)
        })

    return pd.DataFrame(results)


def run_regression(df, dependent_variable, independent_variable):
    """
    Regressie met controle voor seizoen en weekdag.
    We gebruiken robuuste standaardfouten.
    """

    if not STATSMODELS_AVAILABLE:
        return {
            "dependent_variable": dependent_variable,
            "independent_variable": independent_variable,
            "coefficient": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "n_obs": np.nan,
            "note": "statsmodels niet geïnstalleerd"
        }

    needed_cols = [
        dependent_variable,
        independent_variable,
        "season",
        "weekday"
    ]

    reg_df = df[needed_cols].dropna().copy()

    if len(reg_df) < 20:
        return {
            "dependent_variable": dependent_variable,
            "independent_variable": independent_variable,
            "coefficient": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "n_obs": len(reg_df),
            "note": "te weinig observaties"
        }

    formula = f"{dependent_variable} ~ {independent_variable} + C(season) + C(weekday)"

    try:
        model = smf.ols(formula=formula, data=reg_df).fit(cov_type="HC3")

        return {
            "dependent_variable": dependent_variable,
            "independent_variable": independent_variable,
            "coefficient": model.params.get(independent_variable, np.nan),
            "p_value": model.pvalues.get(independent_variable, np.nan),
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
            "note": "OLS met seizoen en weekday controls, robuuste standaardfouten HC3"
        }

    except Exception as e:
        return {
            "dependent_variable": dependent_variable,
            "independent_variable": independent_variable,
            "coefficient": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "n_obs": len(reg_df),
            "note": f"regressiefout: {repr(e)}"
        }


# ============================================================
# STAP 1: BESTANDEN CONTROLEREN EN INLADEN
# ============================================================

check_file_exists(SPOTIFY_FILE)
check_file_exists(WEATHER_FILE)

spotify = pd.read_csv(SPOTIFY_FILE)
weather = pd.read_csv(WEATHER_FILE)

print("Spotify-data ingeladen:")
print(spotify.shape)

print("\nWeather-data ingeladen:")
print(weather.shape)


# ============================================================
# STAP 2: BASISCONTROLE KOLOMMEN
# ============================================================

required_spotify_cols = ["date", "valence", "streams"]
required_weather_cols = ["date", "prcp"]

for col in required_spotify_cols:
    if col not in spotify.columns:
        raise ValueError(
            f"Kolom '{col}' ontbreekt in {SPOTIFY_FILE}.\n"
            f"Voor deze sterkere analyse zijn streams verplicht."
        )

for col in required_weather_cols:
    if col not in weather.columns:
        raise ValueError(
            f"Kolom '{col}' ontbreekt in {WEATHER_FILE}."
        )

print("\nVereiste kolommen gevonden.")
print("Streams zijn aanwezig, dus we kunnen effectief luistergedrag analyseren.")


# ============================================================
# STAP 3: DATUMS EN TYPES GOED ZETTEN
# ============================================================

spotify["date"] = pd.to_datetime(spotify["date"], errors="coerce").dt.normalize()
weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()

spotify = spotify.dropna(subset=["date"]).copy()
weather = weather.dropna(subset=["date"]).copy()

spotify["valence"] = pd.to_numeric(spotify["valence"], errors="coerce")
spotify["streams"] = pd.to_numeric(spotify["streams"], errors="coerce")

if "rank" in spotify.columns:
    spotify["rank"] = pd.to_numeric(spotify["rank"], errors="coerce")

for col in ["prcp", "tsun", "tavg", "tmin", "tmax", "wspd", "pres"]:
    if col in weather.columns:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

print("\nDatumbereik Spotify:")
print(spotify["date"].min(), "tot", spotify["date"].max())

print("\nDatumbereik Weather:")
print(weather["date"].min(), "tot", weather["date"].max())

print("\nAantal Spotify-rijen zonder valence:")
print(spotify["valence"].isna().sum())

print("\nAantal Spotify-rijen zonder streams:")
print(spotify["streams"].isna().sum())


# ============================================================
# STAP 4: DAGELIJKSE SPOTIFY-METRICS MAKEN
# ============================================================

print("\nDagelijkse Spotify-metrics worden berekend...")

daily_spotify = spotify.groupby("date").apply(
    calculate_daily_spotify_metrics
).reset_index()

print("\nDagelijkse Spotify-data:")
print(daily_spotify.shape)

print("\nEerste 5 rijen:")
print(daily_spotify.head().to_string(index=False))


# ============================================================
# STAP 5: MERGEN MET WEATHER
# ============================================================

weather = weather.drop_duplicates(subset=["date"]).copy()

final = pd.merge(
    daily_spotify,
    weather,
    on="date",
    how="inner"
)

print("\nFinale dataset na merge:")
print(final.shape)

print("\nDatumbereik finale dataset:")
print(final["date"].min(), "tot", final["date"].max())

final.to_csv(OUTPUT_FINAL_MERGED_DATASET, index=False)


# ============================================================
# STAP 6: KWALITEITSFILTER
# ============================================================

analysis_df = final[
    (final["top200_rows"] >= MIN_TOP200_ROWS) &
    (final["valence_coverage"] >= MIN_VALENCE_COVERAGE) &
    (final["prcp"].notna()) &
    (final["share_sad_streams_matched"].notna()) &
    (final["weighted_avg_valence"].notna())
].copy()

print("\nDataset na kwaliteitsfilter:")
print(analysis_df.shape)

print("\nVerwijderde dagen door kwaliteitsfilter:")
print(len(final) - len(analysis_df))

if analysis_df.empty:
    raise ValueError(
        "Na filtering blijven geen dagen over.\n"
        "Verlaag eventueel MIN_VALENCE_COVERAGE naar 0.40 of MIN_TOP200_ROWS naar 150."
    )


# ============================================================
# STAP 7: WEER- EN TIJDVARIABELEN MAKEN
# ============================================================

analysis_df["rainy_day"] = analysis_df["prcp"] >= RAIN_THRESHOLD_MM
analysis_df["rainy_day_int"] = analysis_df["rainy_day"].astype(int)

analysis_df["dry_day"] = analysis_df["prcp"] < RAIN_THRESHOLD_MM
analysis_df["heavy_rain"] = analysis_df["prcp"] >= HEAVY_RAIN_THRESHOLD_MM
analysis_df["heavy_rain_int"] = analysis_df["heavy_rain"].astype(int)

analysis_df["log_prcp"] = np.log1p(analysis_df["prcp"])

analysis_df["rain_intensity"] = np.select(
    [
        analysis_df["prcp"] < RAIN_THRESHOLD_MM,
        (analysis_df["prcp"] >= RAIN_THRESHOLD_MM) &
        (analysis_df["prcp"] < HEAVY_RAIN_THRESHOLD_MM),
        analysis_df["prcp"] >= HEAVY_RAIN_THRESHOLD_MM
    ],
    [
        "droog",
        "lichte_regen",
        "zware_regen"
    ],
    default="onbekend"
)

analysis_df["month"] = analysis_df["date"].dt.month
analysis_df["weekday"] = analysis_df["date"].dt.day_name()
analysis_df["season"] = analysis_df["month"].apply(get_season)

# Optioneel: zon gebruiken als tsun bruikbaar is
tsun_usable = "tsun" in analysis_df.columns and analysis_df["tsun"].notna().sum() > 10

if tsun_usable:
    tsun_q25 = analysis_df["tsun"].quantile(0.25)
    tsun_q75 = analysis_df["tsun"].quantile(0.75)

    analysis_df["low_sun"] = analysis_df["tsun"] <= tsun_q25
    analysis_df["high_sun"] = analysis_df["tsun"] >= tsun_q75
    analysis_df["bad_weather"] = analysis_df["rainy_day"] | analysis_df["low_sun"]
    analysis_df["sunny_dry_day"] = analysis_df["dry_day"] & analysis_df["high_sun"]

    print("\nTsun is bruikbaar.")
    print("Onderste 25% zon:", round(tsun_q25, 2))
    print("Bovenste 25% zon:", round(tsun_q75, 2))
else:
    analysis_df["low_sun"] = False
    analysis_df["high_sun"] = False
    analysis_df["bad_weather"] = analysis_df["rainy_day"]
    analysis_df["sunny_dry_day"] = analysis_df["dry_day"]

    print("\nTsun is niet bruikbaar of grotendeels leeg.")
    print("Slecht weer wordt daarom vooral op regen gebaseerd.")


# ============================================================
# STAP 8: HOOFDVARIABELEN KIEZEN
# ============================================================

main_sad_metric = "share_sad_streams_matched"
main_valence_metric = "weighted_avg_valence"

robustness_metrics = [
    "share_sad_songs_matched",
    "avg_valence"
]

print("\nHoofdvariabelen:")
print("Sad-share:", main_sad_metric)
print("Valence:", main_valence_metric)

print("\nInterpretatie:")
print("- Hogere share_sad_streams_matched = groter streamaandeel naar droevige nummers.")
print("- Lagere weighted_avg_valence = gemiddeld droevigere beluisterde muziek.")


# ============================================================
# STAP 9: BESCHRIJVENDE STATISTIEKEN
# ============================================================

print("\nAantal dagen per rain_intensity:")
print(analysis_df["rain_intensity"].value_counts())

print("\nAantal regendagen vs droge dagen:")
print(analysis_df["rainy_day"].value_counts())

print("\nAantal dagen per seizoen:")
print(analysis_df["season"].value_counts())


summary_rain_vs_dry = make_group_summary(
    analysis_df,
    "rainy_day",
    main_sad_metric,
    main_valence_metric
)

summary_rain_intensity = make_group_summary(
    analysis_df,
    "rain_intensity",
    main_sad_metric,
    main_valence_metric
)

summary_by_season = make_group_summary(
    analysis_df,
    "season",
    main_sad_metric,
    main_valence_metric
)

summary_by_season_and_rain = analysis_df.groupby(
    ["season", "rainy_day"]
).agg(
    aantal_dagen=("date", "count"),
    gemiddelde_prcp=("prcp", "mean"),
    gemiddelde_sad_stream_share=(main_sad_metric, "mean"),
    gemiddelde_weighted_valence=(main_valence_metric, "mean"),
    gemiddelde_valence_coverage=("valence_coverage", "mean")
).reset_index()

print("\nSamenvatting regen vs droog:")
print(summary_rain_vs_dry.to_string(index=False))

print("\nSamenvatting regenintensiteit:")
print(summary_rain_intensity.to_string(index=False))

print("\nSamenvatting per seizoen:")
print(summary_by_season.to_string(index=False))

print("\nSamenvatting per seizoen en regen/droog:")
print(summary_by_season_and_rain.to_string(index=False))


# ============================================================
# STAP 10: GROEPSVERGELIJKINGEN
# ============================================================

metrics_to_compare = [
    main_sad_metric,
    main_valence_metric,
    "share_sad_songs_matched",
    "avg_valence"
]

comparison_results = []

# Regen vs droog
rain_vs_dry = compare_two_groups(
    df=analysis_df,
    group_col="rainy_day",
    group_a_value=True,
    group_b_value=False,
    metrics=metrics_to_compare
)
comparison_results.append(rain_vs_dry)

# Zware regen vs droog
heavy_vs_dry = compare_two_groups(
    df=analysis_df,
    group_col="rain_intensity",
    group_a_value="zware_regen",
    group_b_value="droog",
    metrics=metrics_to_compare
)
comparison_results.append(heavy_vs_dry)

# Lichte regen vs droog
light_vs_dry = compare_two_groups(
    df=analysis_df,
    group_col="rain_intensity",
    group_a_value="lichte_regen",
    group_b_value="droog",
    metrics=metrics_to_compare
)
comparison_results.append(light_vs_dry)

comparison_results_df = pd.concat(comparison_results, ignore_index=True)

print("\nGroepsvergelijkingen:")
print(comparison_results_df.to_string(index=False))


# ============================================================
# STAP 11: CORRELATIES
# ============================================================

corr_cols = [
    "prcp",
    "log_prcp",
    main_sad_metric,
    main_valence_metric,
    "share_sad_songs_matched",
    "avg_valence"
]

if tsun_usable:
    corr_cols.append("tsun")

corr_cols = [
    col for col in corr_cols
    if col in analysis_df.columns and analysis_df[col].notna().sum() > 1
]

correlations = analysis_df[corr_cols].corr()

print("\nCorrelatiematrix:")
print(correlations)


# ============================================================
# STAP 12: REGRESSIES MET CONTROLES
# ============================================================
# Belangrijk:
# Dit is methodologisch sterker dan gewone correlatie,
# omdat we controleren voor:
# - seizoen
# - weekdag
#
# We draaien regressies voor:
# 1. share_sad_streams_matched
# 2. weighted_avg_valence
#
# Met als verklarende variabelen:
# - log_prcp
# - rainy_day_int
# - heavy_rain_int
# ============================================================

regression_results = []

dependent_variables = [
    main_sad_metric,
    main_valence_metric
]

independent_variables = [
    "log_prcp",
    "rainy_day_int",
    "heavy_rain_int"
]

for dep_var in dependent_variables:
    for indep_var in independent_variables:
        result = run_regression(
            df=analysis_df,
            dependent_variable=dep_var,
            independent_variable=indep_var
        )
        regression_results.append(result)

regression_results_df = pd.DataFrame(regression_results)

print("\nRegressieresultaten met seizoen en weekday controls:")
print(regression_results_df.to_string(index=False))


# ============================================================
# STAP 13: BELANGRIJKSTE INTERPRETATIE PRINTEN
# ============================================================

rainy = analysis_df[analysis_df["rainy_day"]].copy()
dry = analysis_df[analysis_df["dry_day"]].copy()

mean_sad_rain = rainy[main_sad_metric].mean()
mean_sad_dry = dry[main_sad_metric].mean()
diff_sad = mean_sad_rain - mean_sad_dry

mean_valence_rain = rainy[main_valence_metric].mean()
mean_valence_dry = dry[main_valence_metric].mean()
diff_valence = mean_valence_rain - mean_valence_dry

print("\n============================================================")
print("HOOFDINTERPRETATIE")
print("============================================================")

print("\n1. Streamaandeel droevige muziek")
print("Gemiddelde share sad streams op regendagen:", round(mean_sad_rain, 4))
print("Gemiddelde share sad streams op droge dagen:", round(mean_sad_dry, 4))
print("Verschil regen - droog:", round(diff_sad, 4))

if diff_sad > 0:
    print("Interpretatie: op regendagen gaat een groter streamaandeel naar droevige nummers.")
elif diff_sad < 0:
    print("Interpretatie: op regendagen gaat een kleiner streamaandeel naar droevige nummers.")
else:
    print("Interpretatie: geen verschil in streamaandeel droevige nummers.")

print("\n2. Streamgewogen gemiddelde valence")
print("Gemiddelde weighted valence op regendagen:", round(mean_valence_rain, 4))
print("Gemiddelde weighted valence op droge dagen:", round(mean_valence_dry, 4))
print("Verschil regen - droog:", round(diff_valence, 4))

if diff_valence < 0:
    print("Interpretatie: op regendagen klinkt de beluisterde muziek gemiddeld droeviger.")
elif diff_valence > 0:
    print("Interpretatie: op regendagen klinkt de beluisterde muziek gemiddeld vrolijker.")
else:
    print("Interpretatie: geen verschil in streamgewogen valence.")

print("\nLet op:")
print("Deze analyse toont samenhang, geen causaliteit.")


# ============================================================
# STAP 14: GRAFIEKEN
# ============================================================

# Grafiek 1: sad stream share regen vs droog
plt.figure(figsize=(8, 5))
analysis_df.boxplot(column=main_sad_metric, by="rainy_day")
plt.title("Share sad streams: regendagen vs droge dagen")
plt.suptitle("")
plt.xlabel("Regendag")
plt.ylabel(main_sad_metric)
plt.tight_layout()
plt.savefig("grafiek_share_sad_streams_regen_vs_droog.png")
plt.close()

# Grafiek 2: weighted valence regen vs droog
plt.figure(figsize=(8, 5))
analysis_df.boxplot(column=main_valence_metric, by="rainy_day")
plt.title("Weighted valence: regendagen vs droge dagen")
plt.suptitle("")
plt.xlabel("Regendag")
plt.ylabel(main_valence_metric)
plt.tight_layout()
plt.savefig("grafiek_weighted_valence_regen_vs_droog.png")
plt.close()

# Grafiek 3: regenintensiteit vs sad share
rain_order = ["droog", "lichte_regen", "zware_regen"]
rain_means = analysis_df.groupby("rain_intensity")[main_sad_metric].mean()
rain_means = rain_means.reindex([x for x in rain_order if x in rain_means.index])

plt.figure(figsize=(8, 5))
rain_means.plot(kind="bar")
plt.title("Gemiddelde share sad streams per regenintensiteit")
plt.xlabel("Regenintensiteit")
plt.ylabel(main_sad_metric)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("grafiek_share_sad_streams_per_regenintensiteit.png")
plt.close()

# Grafiek 4: regenintensiteit vs weighted valence
rain_valence_means = analysis_df.groupby("rain_intensity")[main_valence_metric].mean()
rain_valence_means = rain_valence_means.reindex([x for x in rain_order if x in rain_valence_means.index])

plt.figure(figsize=(8, 5))
rain_valence_means.plot(kind="bar")
plt.title("Gemiddelde weighted valence per regenintensiteit")
plt.xlabel("Regenintensiteit")
plt.ylabel(main_valence_metric)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("grafiek_weighted_valence_per_regenintensiteit.png")
plt.close()

# Grafiek 5: prcp vs share sad streams
plt.figure(figsize=(8, 5))
plt.scatter(analysis_df["prcp"], analysis_df[main_sad_metric])
plt.title("Neerslag vs share sad streams")
plt.xlabel("Neerslag, prcp in mm")
plt.ylabel(main_sad_metric)
plt.tight_layout()
plt.savefig("grafiek_prcp_vs_share_sad_streams.png")
plt.close()

# Grafiek 6: log_prcp vs share sad streams
plt.figure(figsize=(8, 5))
plt.scatter(analysis_df["log_prcp"], analysis_df[main_sad_metric])
plt.title("Log neerslag vs share sad streams")
plt.xlabel("log(1 + prcp)")
plt.ylabel(main_sad_metric)
plt.tight_layout()
plt.savefig("grafiek_log_prcp_vs_share_sad_streams.png")
plt.close()

# Grafiek 7: seizoen en regen
season_rain_summary = analysis_df.groupby(
    ["season", "rainy_day"]
)[main_sad_metric].mean().reset_index()

season_rain_summary["group"] = (
    season_rain_summary["season"].astype(str)
    + "_rain_"
    + season_rain_summary["rainy_day"].astype(str)
)

plt.figure(figsize=(10, 5))
plt.bar(
    season_rain_summary["group"],
    season_rain_summary[main_sad_metric]
)
plt.title("Share sad streams per seizoen en regen/droog")
plt.xlabel("Seizoen en regenstatus")
plt.ylabel(main_sad_metric)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafiek_share_sad_streams_seizoen_regen.png")
plt.close()


# ============================================================
# STAP 15: OUTPUTS OPSLAAN
# ============================================================

analysis_df.to_csv(OUTPUT_ANALYSIS_DATASET, index=False)

summary_rain_vs_dry.to_csv("summary_rain_vs_dry_stronger.csv", index=False)
summary_rain_intensity.to_csv("summary_rain_intensity_stronger.csv", index=False)
summary_by_season.to_csv("summary_by_season_stronger.csv", index=False)
summary_by_season_and_rain.to_csv("summary_by_season_and_rain_stronger.csv", index=False)

comparison_results_df.to_csv("group_comparisons_stronger.csv", index=False)
correlations.to_csv("correlations_weather_music_stronger.csv")
regression_results_df.to_csv("regression_results_stronger.csv", index=False)

print("\nBestanden opgeslagen:")
print(f"- {OUTPUT_FINAL_MERGED_DATASET}")
print(f"- {OUTPUT_ANALYSIS_DATASET}")
print("- summary_rain_vs_dry_stronger.csv")
print("- summary_rain_intensity_stronger.csv")
print("- summary_by_season_stronger.csv")
print("- summary_by_season_and_rain_stronger.csv")
print("- group_comparisons_stronger.csv")
print("- correlations_weather_music_stronger.csv")
print("- regression_results_stronger.csv")

print("\nGrafieken opgeslagen:")
print("- grafiek_share_sad_streams_regen_vs_droog.png")
print("- grafiek_weighted_valence_regen_vs_droog.png")
print("- grafiek_share_sad_streams_per_regenintensiteit.png")
print("- grafiek_weighted_valence_per_regenintensiteit.png")
print("- grafiek_prcp_vs_share_sad_streams.png")
print("- grafiek_log_prcp_vs_share_sad_streams.png")
print("- grafiek_share_sad_streams_seizoen_regen.png")

print("\nKlaar.")