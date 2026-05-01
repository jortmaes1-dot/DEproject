# ============================================================
# FINAL_ANALYSIS.PY - RESEARCH PIPELINE V4
# ============================================================
# DOEL:
# Alle gekozen onderzoeken voor de paper uitvoeren, maar met de
# aangescherpte onderzoekslijn:
#
# HOOFDONDERZOEKEN
# 1. Gemiddelde valence van de Top 200 op regenachtig/slecht weer vs zonnig/goed weer.
# 2. Aandeel sad songs / sad streams op regenachtig/slecht weer vs zonnig/goed weer.
#
# EXTRA ONDERZOEKEN
# 3. Seizoenen vs valence / sad share.
# 4. Genre vs weer: altijd tegenover prcp en tsun.
# 5. Artiesten vs weer: altijd tegenover prcp en tsun.
# 6. Audiofeatures vs weer: altijd tegenover prcp en tsun.
#    Popularity wordt NIET behandeld als audiofeature.
# 7. Danceability, speechiness en energy vs weer.
#
# Belangrijke keuzes:
# - Geen scatterplots.
# - Geen neutrale/mixed weercategorie in de finale vergelijkingen: tussendagen worden niet als aparte groep getest.
# - Regenachtig/slecht weer = duidelijke neerslag én weinig zon.
# - Zonnig/goed weer = weinig/geen neerslag én veel zon.
# - Alle statistische resultaten krijgen rechtstreeks een conclusie:
#   SIGNIFICANT of NIET SIGNIFICANT op alpha = 0.05.
# - Voor veel vergelijkingen tegelijk gebruiken we Benjamini-Hochberg.
# - Alle resultaten krijgen een PowerBI-ready CSV.
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None


# ============================================================
# INSTELLINGEN
# ============================================================

DAILY_FILE = "final_dataset.csv"
SONG_FILE = "spotify_belgium_top200_with_features.csv"

OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
STATS_DIR = OUTPUT_DIR / "stats"
POWERBI_DIR = OUTPUT_DIR / "powerbi"

ALPHA = 0.05
MIN_DAYS_ENTITY = 20
MIN_ROWS_FEATURE = 30

BAD_WEATHER_LABEL = "Regenachtig/slecht weer"
GOOD_WEATHER_LABEL = "Zonnig/goed weer"
MIN_WEATHER_GROUP_DAYS = 20

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]

# Alleen echte audiofeatures. Popularity staat hier bewust NIET tussen.
AUDIOFEATURES = [
    "valence",
    "energy",
    "danceability",
    "speechiness",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "loudness",
]

DAILY_AUDIOFEATURE_MAP = {
    "valence": "weighted_avg_valence",
    "energy": "weighted_avg_energy",
    "danceability": "weighted_avg_danceability",
    "speechiness": "weighted_avg_speechiness",
    "tempo": "weighted_avg_tempo",
    "acousticness": "weighted_avg_acousticness",
    "instrumentalness": "weighted_avg_instrumentalness",
    "liveness": "weighted_avg_liveness",
    "loudness": "weighted_avg_loudness",
}


# ============================================================
# ALGEMENE HULPFUNCTIES
# ============================================================

def ensure_output_dirs():
    for directory in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, STATS_DIR, POWERBI_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def to_numeric_if_exists(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def season_from_month(month):
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def zscore(series):
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean(skipna=True)) / sd


def add_weather_variables(df):
    """
    Maakt een strikte goed-weer-vs-slecht-weer benchmark.

    Waarom niet gewoon regenachtig vs goed weer?
    - Een droge winterdag met bijna geen zon is niet noodzakelijk "goed weer".
    - Een dag met een klein beetje neerslag maar veel zon is niet noodzakelijk "slecht weer".

    Daarom combineren we prcp en tsun:
    - Zonnig/goed weer: weinig/geen neerslag én relatief veel zon.
    - Regenachtig/slecht weer: duidelijke neerslag én relatief weinig zon.

    Alle tussengevallen krijgen geen aparte categorie. Ze blijven in de dataset
    voor correlaties met prcp/tsun, maar worden uitgesloten uit de directe
    goed-vs-slecht-vergelijkingen.
    """
    df = df.copy()
    df["prcp"] = pd.to_numeric(df.get("prcp"), errors="coerce")
    df["tsun"] = pd.to_numeric(df.get("tsun"), errors="coerce")

    valid_prcp = df["prcp"].dropna()
    valid_tsun = df["tsun"].dropna()

    # Vaste ondergrens: 0.2 mm beschouwen we als praktisch goed weer.
    # Voor slecht weer vragen we minstens 1 mm of de bovenste 33% neerslag,
    # afhankelijk van welke drempel strenger is.
    prcp_good_max = 0.2
    prcp_bad_min = 1.0
    if not valid_prcp.empty:
        prcp_bad_min = max(1.0, float(valid_prcp.quantile(0.67)))

    # Zonneschijnduur werkt dataset-afhankelijk. Daarom gebruiken we percentielen.
    # Top 33% zon = goed weer; laagste 33% zon = slecht weer.
    tsun_good_min = np.nan
    tsun_bad_max = np.nan
    if not valid_tsun.empty and valid_tsun.nunique() > 1:
        tsun_good_min = float(valid_tsun.quantile(0.67))
        tsun_bad_max = float(valid_tsun.quantile(0.33))

    if pd.isna(tsun_good_min) or pd.isna(tsun_bad_max):
        good_mask = df["prcp"] <= prcp_good_max
        bad_mask = df["prcp"] >= prcp_bad_min
        benchmark_version = "prcp_only_fallback"
    else:
        good_mask = (df["prcp"] <= prcp_good_max) & (df["tsun"] >= tsun_good_min)
        bad_mask = (df["prcp"] >= prcp_bad_min) & (df["tsun"] <= tsun_bad_max)
        benchmark_version = "strict_prcp_and_tsun"

        # Als de strikte definitie te weinig observaties geeft, versoepelen we licht.
        if good_mask.sum() < MIN_WEATHER_GROUP_DAYS or bad_mask.sum() < MIN_WEATHER_GROUP_DAYS:
            tsun_good_min = float(valid_tsun.quantile(0.50))
            tsun_bad_max = float(valid_tsun.quantile(0.50))
            prcp_bad_min = max(0.5, float(valid_prcp.quantile(0.50))) if not valid_prcp.empty else 0.5
            good_mask = (df["prcp"] <= prcp_good_max) & (df["tsun"] >= tsun_good_min)
            bad_mask = (df["prcp"] >= prcp_bad_min) & (df["tsun"] <= tsun_bad_max)
            benchmark_version = "relaxed_prcp_and_tsun_due_to_low_n"

    df["good_weather"] = good_mask & ~bad_mask
    df["bad_weather"] = bad_mask & ~good_mask
    df["good_bad_weather_type"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["bad_weather"], "good_bad_weather_type"] = BAD_WEATHER_LABEL
    df.loc[df["good_weather"], "good_bad_weather_type"] = GOOD_WEATHER_LABEL
    df["selected_weather_benchmark_day"] = df["good_bad_weather_type"].notna()

    # Scores blijven nuttig voor PowerBI, maar de statistische groepsvergelijking
    # gebeurt op basis van good_bad_weather_type.
    df["weather_score_good"] = zscore(df["tsun"]) - zscore(df["prcp"])
    df["weather_score_bad"] = zscore(df["prcp"]) - zscore(df["tsun"])

    thresholds = pd.DataFrame([
        {"parameter": "prcp_good_max", "value": prcp_good_max, "meaning": "Maximum neerslag voor zonnig/goed weer"},
        {"parameter": "prcp_bad_min", "value": prcp_bad_min, "meaning": "Minimum neerslag voor regenachtig/slecht weer"},
        {"parameter": "tsun_good_min", "value": tsun_good_min, "meaning": "Minimum zonneschijn voor zonnig/goed weer"},
        {"parameter": "tsun_bad_max", "value": tsun_bad_max, "meaning": "Maximum zonneschijn voor regenachtig/slecht weer"},
        {"parameter": "n_good_weather_days", "value": int(df["good_weather"].sum()), "meaning": "Aantal geselecteerde zonnige/goede dagen"},
        {"parameter": "n_bad_weather_days", "value": int(df["bad_weather"].sum()), "meaning": "Aantal geselecteerde regenachtige/slechte dagen"},
        {"parameter": "benchmark_version", "value": benchmark_version, "meaning": "Gebruikte definitie"},
    ])
    thresholds.to_csv(TABLE_DIR / "table_00_weather_benchmark_thresholds.csv", index=False)
    thresholds.to_csv(POWERBI_DIR / "pbi_00_weather_benchmark_thresholds.csv", index=False)

    return df


def conclusion_from_p(p_value, p_adjusted=np.nan):
    p_to_use = p_adjusted if not pd.isna(p_adjusted) else p_value
    if pd.isna(p_to_use):
        return "GEEN TEST"
    return "SIGNIFICANT" if p_to_use < ALPHA else "NIET SIGNIFICANT"


def p_adjust_bh(p_values):
    """Benjamini-Hochberg FDR-correctie."""
    p = pd.to_numeric(pd.Series(p_values), errors="coerce")
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return adjusted

    order = valid.sort_values().index
    ranked = valid.loc[order]
    n = len(ranked)
    raw_adj = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(raw_adj.iloc[::-1]).iloc[::-1]
    monotone = monotone.clip(upper=1)
    adjusted.loc[order] = monotone.values
    return adjusted


def add_conclusion_columns(df, p_col="p_value", p_adj_col="p_adjusted"):
    df = df.copy()
    if p_adj_col not in df.columns:
        df[p_adj_col] = np.nan
    df["alpha"] = ALPHA
    df["significant_005"] = [
        conclusion_from_p(row[p_col], row[p_adj_col]) == "SIGNIFICANT"
        for _, row in df.iterrows()
    ]
    df["conclusion_005"] = [
        conclusion_from_p(row[p_col], row[p_adj_col])
        for _, row in df.iterrows()
    ]
    return df


def add_significance_row(rows, analysis, test, variable, comparison, p_value, statistic=np.nan, n=np.nan, p_adjusted=np.nan):
    rows.append({
        "analysis": analysis,
        "test": test,
        "variable": variable,
        "comparison": comparison,
        "n": n,
        "statistic": statistic,
        "p_value": p_value,
        "p_adjusted": p_adjusted,
        "alpha": ALPHA,
        "significant_005": conclusion_from_p(p_value, p_adjusted) == "SIGNIFICANT",
        "conclusion_005": conclusion_from_p(p_value, p_adjusted),
    })


def save_csv_dual(df, table_filename=None, powerbi_filename=None, stats_filename=None):
    if table_filename:
        df.to_csv(TABLE_DIR / table_filename, index=False)
    if powerbi_filename:
        df.to_csv(POWERBI_DIR / powerbi_filename, index=False)
    if stats_filename:
        df.to_csv(STATS_DIR / stats_filename, index=False)


def safe_pearson_spearman(data, x_col, y_col, min_n=10):
    result = {
        "x": x_col,
        "y": y_col,
        "n": 0,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }

    if not SCIPY_AVAILABLE:
        return result
    if x_col not in data.columns or y_col not in data.columns:
        return result

    d = data[[x_col, y_col]].dropna().copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna()

    result["n"] = len(d)
    if len(d) < min_n or d[x_col].nunique() < 2 or d[y_col].nunique() < 2:
        return result

    pearson = stats.pearsonr(d[x_col], d[y_col])
    spearman = stats.spearmanr(d[x_col], d[y_col])

    result.update({
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    })
    return result


def welch_and_mannwhitney(data, group_col, value_col, group_a=BAD_WEATHER_LABEL, group_b=GOOD_WEATHER_LABEL):
    result = {
        "group_a": group_a,
        "group_b": group_b,
        "value": value_col,
        "n_a": 0,
        "n_b": 0,
        "mean_a": np.nan,
        "mean_b": np.nan,
        "median_a": np.nan,
        "median_b": np.nan,
        "difference_a_minus_b": np.nan,
        "welch_t": np.nan,
        "welch_p": np.nan,
        "mannwhitney_u": np.nan,
        "mannwhitney_p": np.nan,
        "cohens_d": np.nan,
    }

    if not SCIPY_AVAILABLE or group_col not in data.columns or value_col not in data.columns:
        return result

    a = pd.to_numeric(data.loc[data[group_col] == group_a, value_col], errors="coerce").dropna()
    b = pd.to_numeric(data.loc[data[group_col] == group_b, value_col], errors="coerce").dropna()

    result["n_a"] = len(a)
    result["n_b"] = len(b)
    if len(a) == 0 or len(b) == 0:
        return result

    result["mean_a"] = float(a.mean())
    result["mean_b"] = float(b.mean())
    result["median_a"] = float(a.median())
    result["median_b"] = float(b.median())
    result["difference_a_minus_b"] = float(a.mean() - b.mean())

    if len(a) >= 2 and len(b) >= 2:
        welch = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        result["welch_t"] = float(welch.statistic)
        result["welch_p"] = float(welch.pvalue)

        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        result["mannwhitney_u"] = float(mw.statistic)
        result["mannwhitney_p"] = float(mw.pvalue)

        pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        if pooled_sd > 0:
            result["cohens_d"] = float((a.mean() - b.mean()) / pooled_sd)

    return result


def anova_and_kruskal(data, group_col, value_col):
    result = {
        "group_col": group_col,
        "value": value_col,
        "n_total": 0,
        "groups": np.nan,
        "anova_f": np.nan,
        "anova_p": np.nan,
        "kruskal_h": np.nan,
        "kruskal_p": np.nan,
    }

    if not SCIPY_AVAILABLE or group_col not in data.columns or value_col not in data.columns:
        return result

    d = data[[group_col, value_col]].dropna().copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna()
    grouped = [g[value_col].values for _, g in d.groupby(group_col) if len(g) >= 2]

    result["n_total"] = len(d)
    result["groups"] = d[group_col].nunique()

    if len(grouped) >= 2:
        anova = stats.f_oneway(*grouped)
        kruskal = stats.kruskal(*grouped)
        result["anova_f"] = float(anova.statistic)
        result["anova_p"] = float(anova.pvalue)
        result["kruskal_h"] = float(kruskal.statistic)
        result["kruskal_p"] = float(kruskal.pvalue)

    return result


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


# ============================================================
# DATA INLADEN
# ============================================================

def load_daily_data():
    if not Path(DAILY_FILE).exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {DAILY_FILE}. Run eerst main.py en weather_API.py.")

    daily = pd.read_csv(DAILY_FILE)
    if "date" not in daily.columns:
        raise ValueError(f"Kolom 'date' ontbreekt in {DAILY_FILE}")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).copy()

    numeric_cols = [
        "prcp", "tsun", "avg_valence", "median_valence", "weighted_avg_valence",
        "weighted_share_sad", "weighted_share_low_energy", "weighted_share_depressive",
        "share_sad_songs", "share_sad_streams", "main_sad_share_metric",
        "weighted_avg_energy", "weighted_avg_danceability", "weighted_avg_speechiness",
        "weighted_avg_tempo", "weighted_avg_acousticness", "weighted_avg_instrumentalness",
        "weighted_avg_liveness", "weighted_avg_loudness",
    ]
    daily = to_numeric_if_exists(daily, numeric_cols)
    daily = add_weather_variables(daily)

    # Hoofdmetric 1: gemiddelde valence van de Top 200.
    if "weighted_avg_valence" in daily.columns and daily["weighted_avg_valence"].notna().sum() > 0:
        daily["main_avg_valence_metric"] = daily["weighted_avg_valence"]
        daily["main_avg_valence_label"] = "rank/streamgewogen gemiddelde valence"
    elif "avg_valence" in daily.columns:
        daily["main_avg_valence_metric"] = daily["avg_valence"]
        daily["main_avg_valence_label"] = "gemiddelde valence"
    else:
        daily["main_avg_valence_metric"] = np.nan
        daily["main_avg_valence_label"] = "gemiddelde valence"

    # Hoofdmetric 2: aandeel sad songs/sad streams.
    if "main_sad_share_metric" not in daily.columns or daily["main_sad_share_metric"].notna().sum() == 0:
        if "share_sad_streams" in daily.columns and daily["share_sad_streams"].notna().sum() > 0:
            daily["main_sad_share_metric"] = daily["share_sad_streams"]
            daily["main_sad_share_label"] = "streamgewogen aandeel sad songs"
        elif "weighted_share_sad" in daily.columns and daily["weighted_share_sad"].notna().sum() > 0:
            daily["main_sad_share_metric"] = daily["weighted_share_sad"]
            daily["main_sad_share_label"] = "rankgewogen aandeel sad songs"
        elif "share_sad_songs" in daily.columns:
            daily["main_sad_share_metric"] = daily["share_sad_songs"]
            daily["main_sad_share_label"] = "aandeel sad songs"
        else:
            daily["main_sad_share_metric"] = np.nan
            daily["main_sad_share_label"] = "aandeel sad songs"

    if "main_sad_share_label" not in daily.columns:
        daily["main_sad_share_label"] = "aandeel sad songs"

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["season"] = daily["month"].apply(season_from_month)
    daily["season_order"] = daily["season"].map({s: i for i, s in enumerate(SEASON_ORDER, start=1)})
    daily["date"] = daily["date"].dt.normalize()

    return daily


def load_song_weather(daily):
    if not Path(SONG_FILE).exists():
        print(f"Waarschuwing: {SONG_FILE} niet gevonden. Genre/artiest/audiofeature-analyses worden beperkt.")
        return pd.DataFrame()

    song = pd.read_csv(SONG_FILE, low_memory=False)
    if "date" not in song.columns:
        raise ValueError(f"Kolom 'date' ontbreekt in {SONG_FILE}")

    song["date"] = pd.to_datetime(song["date"], errors="coerce").dt.normalize()
    song = song.dropna(subset=["date"]).copy()

    numeric_cols = [
        "rank", "streams", "rank_weight", "analysis_weight",
        "valence", "energy", "danceability", "speechiness", "tempo",
        "acousticness", "instrumentalness", "liveness", "loudness", "popularity",
    ]
    song = to_numeric_if_exists(song, numeric_cols)

    if "rank_weight" not in song.columns:
        song["rank_weight"] = (201 - pd.to_numeric(song.get("rank"), errors="coerce")).clip(lower=1)

    if "analysis_weight" not in song.columns:
        if "streams" in song.columns and song["streams"].fillna(0).sum() > 0:
            song["analysis_weight"] = song["streams"]
        else:
            song["analysis_weight"] = song["rank_weight"]

    if "artist_first" not in song.columns:
        if "artist" in song.columns:
            song["artist_first"] = song["artist"].astype(str).str.lower().str.split(",").str[0].str.strip()
        elif "artist_clean" in song.columns:
            song["artist_first"] = song["artist_clean"].astype(str)
        else:
            song["artist_first"] = "unknown"

    if "track_genre" not in song.columns:
        song["track_genre"] = "unknown"

    song["artist_first"] = song["artist_first"].astype(str).str.lower().str.strip()
    song["track_genre"] = song["track_genre"].astype(str).str.lower().str.strip()

    weather_cols = [
        "date", "prcp", "tsun", "bad_weather", "good_weather", "good_bad_weather_type", "selected_weather_benchmark_day",
        "season", "year", "month",
    ]
    weather_cols = [c for c in weather_cols if c in daily.columns]
    song_weather = song.merge(daily[weather_cols].drop_duplicates(subset=["date"]), on="date", how="left")

    return song_weather


# ============================================================
# ANALYSE 1: HOOFDONDERZOEK - VALENCE EN SAD SHARE VS WEER
# ============================================================

def analyse_main_weather(daily, significance_rows):
    analysis_name = "01_main_valence_and_sad_share_vs_weather"

    pbi_cols = [
        "date", "year", "month", "season", "prcp", "tsun", "bad_weather", "good_weather", "good_bad_weather_type", "selected_weather_benchmark_day",
        "avg_valence", "weighted_avg_valence", "main_avg_valence_metric", "main_avg_valence_label",
        "share_sad_songs", "weighted_share_sad", "share_sad_streams", "main_sad_share_metric", "main_sad_share_label",
        "weighted_avg_energy", "weighted_avg_danceability", "weighted_avg_speechiness",
        "weighted_share_low_energy", "weighted_share_depressive",
    ]
    pbi_cols = [c for c in pbi_cols if c in daily.columns]
    pbi = daily[pbi_cols].copy()
    pbi["date"] = pbi["date"].dt.strftime("%Y-%m-%d")
    pbi.to_csv(POWERBI_DIR / "pbi_01_daily_weather_valence.csv", index=False)

    # Correlaties met prcp en tsun, geen scatterplots.
    metric_info = {
        "main_avg_valence_metric": "Gemiddelde valence Top 200",
        "main_sad_share_metric": "Aandeel sad songs/sad streams",
    }
    corr_rows = []
    for metric, metric_label in metric_info.items():
        for weather_var in ["prcp", "tsun"]:
            res = safe_pearson_spearman(daily, weather_var, metric, min_n=10)
            corr_rows.append({
                "metric": metric,
                "metric_label": metric_label,
                "weather_var": weather_var,
                "n": res["n"],
                "pearson_r": res["pearson_r"],
                "pearson_p": res["pearson_p"],
                "spearman_rho": res["spearman_rho"],
                "spearman_p": res["spearman_p"],
            })
            add_significance_row(significance_rows, analysis_name, "Pearson correlation", metric_label, f"{metric_label} vs {weather_var}", res["pearson_p"], res["pearson_r"], res["n"])
            add_significance_row(significance_rows, analysis_name, "Spearman correlation", metric_label, f"{metric_label} vs {weather_var}", res["spearman_p"], res["spearman_rho"], res["n"])

    corr_df = pd.DataFrame(corr_rows)
    corr_df["pearson_conclusion_005"] = corr_df["pearson_p"].apply(lambda p: conclusion_from_p(p))
    corr_df["spearman_conclusion_005"] = corr_df["spearman_p"].apply(lambda p: conclusion_from_p(p))
    save_csv_dual(corr_df, stats_filename="stats_01_main_weather_correlations.csv", powerbi_filename="pbi_01_main_weather_correlations.csv")

    # Regenachtig vs goed weer voor gemiddelde valence en sad share.
    t_rows = []
    for metric, metric_label in metric_info.items():
        res = welch_and_mannwhitney(daily, "good_bad_weather_type", metric, BAD_WEATHER_LABEL, GOOD_WEATHER_LABEL)
        res["metric"] = metric
        res["metric_label"] = metric_label
        res["welch_conclusion_005"] = conclusion_from_p(res["welch_p"])
        res["mannwhitney_conclusion_005"] = conclusion_from_p(res["mannwhitney_p"])
        t_rows.append(res)
        add_significance_row(significance_rows, analysis_name, "Welch t-test", metric_label, f"{BAD_WEATHER_LABEL} vs {GOOD_WEATHER_LABEL}", res["welch_p"], res["welch_t"], res["n_a"] + res["n_b"])
        add_significance_row(significance_rows, analysis_name, "Mann-Whitney U", metric_label, f"{BAD_WEATHER_LABEL} vs {GOOD_WEATHER_LABEL}", res["mannwhitney_p"], res["mannwhitney_u"], res["n_a"] + res["n_b"])

    t_df = pd.DataFrame(t_rows)
    save_csv_dual(t_df, stats_filename="stats_01_main_good_vs_bad_weather_tests.csv", powerbi_filename="pbi_01_main_good_vs_bad_weather_tests.csv")

    # PowerBI bar-data: gemiddelden per weertype.
    bar_rows = []
    for metric, metric_label in metric_info.items():
        tmp = daily.groupby("good_bad_weather_type", as_index=False).agg(
            n_days=("date", "count"),
            mean_value=(metric, "mean"),
            median_value=(metric, "median"),
            sd_value=(metric, "std"),
            mean_prcp=("prcp", "mean"),
            mean_tsun=("tsun", "mean"),
        )
        tmp["metric"] = metric
        tmp["metric_label"] = metric_label
        bar_rows.append(tmp)
    bar_df = pd.concat(bar_rows, ignore_index=True)
    save_csv_dual(bar_df, table_filename="table_01_good_vs_bad_weather_summary.csv", powerbi_filename="pbi_01_good_vs_bad_weather_summary.csv")

    # Figuur: barplot zonder scatter.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (metric, metric_label) in zip(axes, metric_info.items()):
        plot_df = bar_df[bar_df["metric"] == metric].copy()
        plot_df["order"] = plot_df["good_bad_weather_type"].map({BAD_WEATHER_LABEL: 1, GOOD_WEATHER_LABEL: 2})
        plot_df = plot_df.sort_values("order")
        ax.bar(plot_df["good_bad_weather_type"], plot_df["mean_value"])
        ax.set_title(metric_label)
        ax.set_xlabel("Weertype")
        ax.set_ylabel("Gemiddelde waarde")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Hoofdonderzoek: regenachtig/slecht weer vs zonnig/goed weer", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig_01_main_good_vs_bad_weather_bars.png", dpi=200)
    plt.close(fig)


# ============================================================
# ANALYSE 2: SEIZOENEN VS VALENCE
# ============================================================

def analyse_seasons(daily, significance_rows):
    analysis_name = "02_seasons_vs_valence"

    season_summary = daily.groupby("season", as_index=False).agg(
        n_days=("date", "count"),
        mean_sad_share=("main_sad_share_metric", "mean"),
        median_sad_share=("main_sad_share_metric", "median"),
        sd_sad_share=("main_sad_share_metric", "std"),
        mean_avg_valence=("main_avg_valence_metric", "mean"),
        median_avg_valence=("main_avg_valence_metric", "median"),
        sd_avg_valence=("main_avg_valence_metric", "std"),
        mean_prcp=("prcp", "mean"),
        mean_tsun=("tsun", "mean"),
    )
    season_summary["season_order"] = season_summary["season"].map({s: i for i, s in enumerate(SEASON_ORDER, start=1)})
    season_summary = season_summary.sort_values("season_order")
    save_csv_dual(season_summary, table_filename="table_02_season_valence.csv", powerbi_filename="pbi_02_season_valence.csv")

    for metric, metric_label in {
        "main_sad_share_metric": "Aandeel sad songs/sad streams",
        "main_avg_valence_metric": "Gemiddelde valence Top 200",
    }.items():
        res = anova_and_kruskal(daily, "season", metric)
        stat_df = pd.DataFrame([res])
        stat_df["anova_conclusion_005"] = stat_df["anova_p"].apply(lambda p: conclusion_from_p(p))
        stat_df["kruskal_conclusion_005"] = stat_df["kruskal_p"].apply(lambda p: conclusion_from_p(p))
        save_csv_dual(stat_df, stats_filename=f"stats_02_season_{metric}.csv", powerbi_filename=f"pbi_02_season_{metric}_stats.csv")
        add_significance_row(significance_rows, analysis_name, "One-way ANOVA", metric_label, "season groups", res["anova_p"], res["anova_f"], res["n_total"])
        add_significance_row(significance_rows, analysis_name, "Kruskal-Wallis", metric_label, "season groups", res["kruskal_p"], res["kruskal_h"], res["n_total"])

    # Grafiek zoals voorbeeld: gemiddeld aandeel sad songs per seizoen.
    plt.figure(figsize=(10, 5))
    plt.bar(season_summary["season"], season_summary["mean_sad_share"])
    plt.title("Gemiddeld aandeel sad songs per seizoen")
    plt.xlabel("Seizoen")
    plt.ylabel("Rank/streamgewogen aandeel sad songs")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_02a_season_sad_share.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(season_summary["season"], season_summary["mean_avg_valence"])
    plt.title("Gemiddelde valence van de Top 200 per seizoen")
    plt.xlabel("Seizoen")
    plt.ylabel("Gemiddelde valence")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_02b_season_avg_valence.png", dpi=200)
    plt.close()


# ============================================================
# ENTITY DAILY SHARES: GENRES EN ARTIESTEN
# ============================================================

def create_entity_daily_shares(song_weather, entity_col):
    required_cols = ["date", entity_col, "analysis_weight", "prcp", "tsun"]
    for col in required_cols:
        if col not in song_weather.columns:
            return pd.DataFrame()

    df = song_weather.dropna(subset=["date", entity_col, "analysis_weight"]).copy()
    df = df[df[entity_col].astype(str).str.strip().ne("")].copy()
    df = df[~df[entity_col].astype(str).str.lower().isin(["nan", "none", "unknown"])]
    df["analysis_weight"] = pd.to_numeric(df["analysis_weight"], errors="coerce")
    df = df.dropna(subset=["analysis_weight"])
    df = df[df["analysis_weight"] > 0]

    if df.empty:
        return pd.DataFrame()

    total_day = df.groupby("date", as_index=False).agg(day_total_weight=("analysis_weight", "sum"))
    entity_day = df.groupby(["date", entity_col], as_index=False).agg(
        entity_weight=("analysis_weight", "sum"),
        entity_tracks=(entity_col, "count"),
    )
    entity_day = entity_day.merge(total_day, on="date", how="left")
    entity_day["entity_share"] = entity_day["entity_weight"] / entity_day["day_total_weight"]

    weather_cols = ["date", "prcp", "tsun", "good_bad_weather_type", "selected_weather_benchmark_day", "season", "year", "month"]
    weather_cols = [c for c in weather_cols if c in song_weather.columns]
    weather_day = song_weather[weather_cols].drop_duplicates(subset=["date"])
    entity_day = entity_day.merge(weather_day, on="date", how="left")

    return entity_day


def entity_weather_correlations(entity_day, entity_col, analysis_name, significance_rows):
    all_rows = []
    for weather_var in ["prcp", "tsun"]:
        rows = []
        for entity, group in entity_day.groupby(entity_col):
            if len(group) < MIN_DAYS_ENTITY:
                continue
            res = safe_pearson_spearman(group, weather_var, "entity_share", min_n=MIN_DAYS_ENTITY)
            rows.append({
                entity_col: entity,
                "weather_var": weather_var,
                "n_days": res["n"],
                "pearson_r": res["pearson_r"],
                "pearson_p": res["pearson_p"],
                "spearman_rho": res["spearman_rho"],
                "spearman_p": res["spearman_p"],
                "mean_entity_share": group["entity_share"].mean(),
            })

        part = pd.DataFrame(rows)
        if part.empty:
            continue

        part["pearson_p_adj_bh"] = p_adjust_bh(part["pearson_p"])
        part["spearman_p_adj_bh"] = p_adjust_bh(part["spearman_p"])
        part["pearson_conclusion_005"] = [conclusion_from_p(p, pa) for p, pa in zip(part["pearson_p"], part["pearson_p_adj_bh"])]
        part["spearman_conclusion_005"] = [conclusion_from_p(p, pa) for p, pa in zip(part["spearman_p"], part["spearman_p_adj_bh"])]
        part["significant_pearson_005"] = part["pearson_conclusion_005"].eq("SIGNIFICANT")
        part["significant_spearman_005"] = part["spearman_conclusion_005"].eq("SIGNIFICANT")

        for _, row in part.iterrows():
            add_significance_row(
                significance_rows,
                analysis=analysis_name,
                test="Pearson correlation + BH correction",
                variable=str(row[entity_col]),
                comparison=f"entity_share vs {weather_var}",
                p_value=row["pearson_p"],
                p_adjusted=row["pearson_p_adj_bh"],
                statistic=row["pearson_r"],
                n=row["n_days"],
            )
            add_significance_row(
                significance_rows,
                analysis=analysis_name,
                test="Spearman correlation + BH correction",
                variable=str(row[entity_col]),
                comparison=f"entity_share vs {weather_var}",
                p_value=row["spearman_p"],
                p_adjusted=row["spearman_p_adj_bh"],
                statistic=row["spearman_rho"],
                n=row["n_days"],
            )

        all_rows.append(part)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def plot_signed_correlations(df, label_col, corr_col, title, xlabel, filename, weather_var, top_n_each_side=5):
    if df.empty or corr_col not in df.columns:
        return

    plot_df = df[df["weather_var"] == weather_var].dropna(subset=[corr_col]).copy()
    if plot_df.empty:
        return

    pos = plot_df.sort_values(corr_col, ascending=False).head(top_n_each_side)
    neg = plot_df.sort_values(corr_col, ascending=True).head(top_n_each_side)
    combined = pd.concat([neg, pos], ignore_index=True).drop_duplicates(subset=[label_col])
    combined = combined.sort_values(corr_col)

    plt.figure(figsize=(10, 6))
    plt.barh(combined[label_col].astype(str), combined[corr_col])
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(label_col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


# ============================================================
# ANALYSE 3: GENRE VS WEER
# ============================================================

def analyse_genres(song_weather, significance_rows):
    analysis_name = "03_genre_vs_weather_prcp_tsun"
    if song_weather.empty or "track_genre" not in song_weather.columns:
        return

    genre_day = create_entity_daily_shares(song_weather, "track_genre")
    if genre_day.empty:
        return

    genre_day_out = genre_day.copy()
    genre_day_out["date"] = pd.to_datetime(genre_day_out["date"]).dt.strftime("%Y-%m-%d")
    genre_day_out.to_csv(POWERBI_DIR / "pbi_03_genre_daily_weather_share.csv", index=False)

    corr = entity_weather_correlations(genre_day, "track_genre", analysis_name, significance_rows)
    if corr.empty:
        return
    corr.to_csv(STATS_DIR / "stats_03_genre_weather_correlations.csv", index=False)
    corr.to_csv(POWERBI_DIR / "pbi_03_genre_weather_correlations.csv", index=False)

    plot_signed_correlations(
        corr,
        label_col="track_genre",
        corr_col="pearson_r",
        title="Genres met sterkste correlatie met neerslag",
        xlabel="Correlatie met prcp",
        filename="fig_03a_genres_correlation_prcp.png",
        weather_var="prcp",
    )
    plot_signed_correlations(
        corr,
        label_col="track_genre",
        corr_col="pearson_r",
        title="Genres met sterkste correlatie met zonneschijnduur",
        xlabel="Correlatie met tsun",
        filename="fig_03b_genres_correlation_tsun.png",
        weather_var="tsun",
    )


# ============================================================
# ANALYSE 4: ARTIESTEN VS WEER
# ============================================================

def analyse_artists(song_weather, significance_rows):
    analysis_name = "04_artists_vs_weather_prcp_tsun"
    if song_weather.empty or "artist_first" not in song_weather.columns:
        return

    artist_day = create_entity_daily_shares(song_weather, "artist_first")
    if artist_day.empty:
        return

    artist_day_out = artist_day.copy()
    artist_day_out["date"] = pd.to_datetime(artist_day_out["date"]).dt.strftime("%Y-%m-%d")
    artist_day_out.to_csv(POWERBI_DIR / "pbi_04_artist_daily_weather_share.csv", index=False)

    corr = entity_weather_correlations(artist_day, "artist_first", analysis_name, significance_rows)
    if corr.empty:
        return
    corr.to_csv(STATS_DIR / "stats_04_artist_weather_correlations.csv", index=False)
    corr.to_csv(POWERBI_DIR / "pbi_04_artist_weather_correlations.csv", index=False)

    plot_signed_correlations(
        corr,
        label_col="artist_first",
        corr_col="pearson_r",
        title="Artiesten met sterkste correlatie met zonneschijnduur",
        xlabel="Correlatie met tsun",
        filename="fig_04a_artists_correlation_tsun.png",
        weather_var="tsun",
    )
    plot_signed_correlations(
        corr,
        label_col="artist_first",
        corr_col="pearson_r",
        title="Artiesten met sterkste correlatie met neerslag",
        xlabel="Correlatie met prcp",
        filename="fig_04b_artists_correlation_prcp.png",
        weather_var="prcp",
    )


# ============================================================
# ANALYSE 5/6/7: AUDIOFEATURES VS WEER
# ============================================================

def analyse_audiofeatures_weather(daily, significance_rows):
    analysis_name = "05_06_07_audiofeatures_vs_weather_prcp_tsun"

    feature_map = {k: v for k, v in DAILY_AUDIOFEATURE_MAP.items() if v in daily.columns and daily[v].notna().sum() > 0}
    if not feature_map:
        return

    # Long PowerBI-dataset per dag x feature.
    long_rows = []
    for feature, col in feature_map.items():
        for _, row in daily[["date", "prcp", "tsun", "good_bad_weather_type", "selected_weather_benchmark_day", col]].dropna(subset=[col]).iterrows():
            long_rows.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "feature": feature,
                "feature_col": col,
                "value": row[col],
                "prcp": row["prcp"],
                "tsun": row["tsun"],
                "good_bad_weather_type": row["good_bad_weather_type"],
            })
    feature_daily_long = pd.DataFrame(long_rows)
    feature_daily_long.to_csv(POWERBI_DIR / "pbi_05_audiofeatures_daily_weather_long.csv", index=False)

    # Correlaties uitsluitend tegenover prcp en tsun.
    rows = []
    for feature, col in feature_map.items():
        for weather_var in ["prcp", "tsun"]:
            res = safe_pearson_spearman(daily, weather_var, col, min_n=10)
            rows.append({
                "feature": feature,
                "feature_col": col,
                "weather_var": weather_var,
                "n": res["n"],
                "pearson_r": res["pearson_r"],
                "pearson_p": res["pearson_p"],
                "spearman_rho": res["spearman_rho"],
                "spearman_p": res["spearman_p"],
            })

    result = pd.DataFrame(rows)
    if result.empty:
        return

    # BH-correctie per testtype over alle feature-weather-combinaties.
    result["pearson_p_adj_bh"] = p_adjust_bh(result["pearson_p"])
    result["spearman_p_adj_bh"] = p_adjust_bh(result["spearman_p"])
    result["pearson_conclusion_005"] = [conclusion_from_p(p, pa) for p, pa in zip(result["pearson_p"], result["pearson_p_adj_bh"])]
    result["spearman_conclusion_005"] = [conclusion_from_p(p, pa) for p, pa in zip(result["spearman_p"], result["spearman_p_adj_bh"])]
    result["significant_pearson_005"] = result["pearson_conclusion_005"].eq("SIGNIFICANT")
    result["significant_spearman_005"] = result["spearman_conclusion_005"].eq("SIGNIFICANT")

    result.to_csv(STATS_DIR / "stats_05_audiofeatures_weather_correlations.csv", index=False)
    result.to_csv(POWERBI_DIR / "pbi_05_audiofeatures_weather_correlations.csv", index=False)

    for _, row in result.iterrows():
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Pearson correlation + BH correction",
            variable=row["feature"],
            comparison=f"{row['feature']} vs {row['weather_var']}",
            p_value=row["pearson_p"],
            p_adjusted=row["pearson_p_adj_bh"],
            statistic=row["pearson_r"],
            n=row["n"],
        )
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Spearman correlation + BH correction",
            variable=row["feature"],
            comparison=f"{row['feature']} vs {row['weather_var']}",
            p_value=row["spearman_p"],
            p_adjusted=row["spearman_p_adj_bh"],
            statistic=row["spearman_rho"],
            n=row["n"],
        )

    # Regenachtig vs goed weer voor audiofeatures als extra, maar nog steeds direct weer-gerelateerd.
    t_rows = []
    for feature, col in feature_map.items():
        res = welch_and_mannwhitney(daily, "good_bad_weather_type", col, BAD_WEATHER_LABEL, GOOD_WEATHER_LABEL)
        res["feature"] = feature
        res["feature_col"] = col
        t_rows.append(res)
    t_df = pd.DataFrame(t_rows)
    if not t_df.empty:
        t_df["welch_p_adj_bh"] = p_adjust_bh(t_df["welch_p"])
        t_df["mannwhitney_p_adj_bh"] = p_adjust_bh(t_df["mannwhitney_p"])
        t_df["welch_conclusion_005"] = [conclusion_from_p(p, pa) for p, pa in zip(t_df["welch_p"], t_df["welch_p_adj_bh"])]
        t_df["mannwhitney_conclusion_005"] = [conclusion_from_p(p, pa) for p, pa in zip(t_df["mannwhitney_p"], t_df["mannwhitney_p_adj_bh"])]
        t_df["significant_welch_005"] = t_df["welch_conclusion_005"].eq("SIGNIFICANT")
        t_df.to_csv(STATS_DIR / "stats_05_audiofeatures_good_vs_bad_weather_tests.csv", index=False)
        t_df.to_csv(POWERBI_DIR / "pbi_05_audiofeatures_good_vs_bad_weather_tests.csv", index=False)

        for _, row in t_df.iterrows():
            add_significance_row(
                significance_rows,
                analysis=analysis_name,
                test="Welch t-test + BH correction",
                variable=row["feature"],
                comparison=f"{BAD_WEATHER_LABEL} vs {GOOD_WEATHER_LABEL}",
                p_value=row["welch_p"],
                p_adjusted=row["welch_p_adj_bh"],
                statistic=row["welch_t"],
                n=row["n_a"] + row["n_b"],
            )

    # Figuur zoals voorbeeld: sterkte van verband per audiofeature, maar nu tegenover prcp/tsun.
    strength = result.copy()
    strength["abs_best_corr"] = strength[["pearson_r", "spearman_rho"]].abs().max(axis=1)
    strength_summary = strength.groupby("feature", as_index=False).agg(
        max_abs_corr_weather=("abs_best_corr", "max")
    ).sort_values("max_abs_corr_weather", ascending=True)
    strength_summary.to_csv(POWERBI_DIR / "pbi_05_audiofeatures_weather_strength.csv", index=False)

    if not strength_summary.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(strength_summary["feature"], strength_summary["max_abs_corr_weather"])
        plt.title("Sterkte van verband per audiofeature met weer")
        plt.xlabel("Grootste absolute correlatie met prcp of tsun")
        plt.ylabel("Spotify-audiofeature")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_05_audiofeatures_weather_strength.png", dpi=200)
        plt.close()

    # Extra focusfiguur voor valence, energy, danceability en speechiness.
    focus_features = ["valence", "energy", "danceability", "speechiness"]
    focus_cols = {f: feature_map[f] for f in focus_features if f in feature_map}
    if focus_cols:
        n_metrics = len(focus_cols)
        n_cols = 2
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.8 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, (feature, col) in zip(axes, focus_cols.items()):
            subset = daily[daily["good_bad_weather_type"].isin([BAD_WEATHER_LABEL, GOOD_WEATHER_LABEL])]
            rainy = subset.loc[subset["good_bad_weather_type"] == BAD_WEATHER_LABEL, col].dropna()
            dry = subset.loc[subset["good_bad_weather_type"] == GOOD_WEATHER_LABEL, col].dropna()
            ax.boxplot([rainy, dry], labels=[f"Regenachtig\n(n={len(rainy)})", f"Goed weer\n(n={len(dry)})"], patch_artist=True)
            ax.set_title(feature, fontsize=10, fontweight="bold")
            ax.set_ylabel("Dagelijks gewogen gemiddelde")
            ax.grid(axis="y", alpha=0.25)
        for j in range(n_metrics, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Audiofeatures op regenachtig/slecht weer vs zonnig/goed weer", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(FIG_DIR / "fig_06_audiofeatures_good_vs_bad_weather_boxplots.png", dpi=200)
        plt.close(fig)


# ============================================================
# EXTRA BOXPLOTS VOOR HOOFDMETRICS
# ============================================================

def analyse_good_bad_weather_boxplots(daily, significance_rows):
    analysis_name = "08_good_vs_bad_weather_boxplots"

    metric_map = {
        "Gemiddelde valence Top 200": "main_avg_valence_metric",
        "Aandeel sad songs/sad streams": "main_sad_share_metric",
        "Energy": "weighted_avg_energy",
        "Danceability": "weighted_avg_danceability",
        "Speechiness": "weighted_avg_speechiness",
        "Aandeel lage energy": "weighted_share_low_energy",
        "Aandeel depressieve songs": "weighted_share_depressive",
    }
    metric_map = {label: col for label, col in metric_map.items() if col in daily.columns and daily[col].notna().sum() > 0}

    long_rows = []
    stat_rows = []
    for label, col in metric_map.items():
        for _, row in daily[["date", "good_bad_weather_type", "selected_weather_benchmark_day", col, "prcp", "tsun"]].dropna(subset=[col, "good_bad_weather_type"]).iterrows():
            long_rows.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "good_bad_weather_type": row["good_bad_weather_type"],
                "metric_label": label,
                "metric_col": col,
                "value": row[col],
                "prcp": row["prcp"],
                "tsun": row["tsun"],
            })
        res = welch_and_mannwhitney(daily, "good_bad_weather_type", col, BAD_WEATHER_LABEL, GOOD_WEATHER_LABEL)
        res["metric_label"] = label
        res["metric_col"] = col
        res["welch_conclusion_005"] = conclusion_from_p(res["welch_p"])
        res["mannwhitney_conclusion_005"] = conclusion_from_p(res["mannwhitney_p"])
        stat_rows.append(res)

        add_significance_row(significance_rows, analysis_name, "Welch t-test", label, f"{BAD_WEATHER_LABEL} vs {GOOD_WEATHER_LABEL}", res["welch_p"], res["welch_t"], res["n_a"] + res["n_b"])

    boxplot_data = pd.DataFrame(long_rows)
    boxplot_data.to_csv(POWERBI_DIR / "pbi_07_good_vs_bad_weather_boxplot_data.csv", index=False)
    boxplot_data.to_csv(TABLE_DIR / "table_07_good_vs_bad_weather_boxplot_data.csv", index=False)

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(STATS_DIR / "stats_07_good_vs_bad_weather_boxplot_tests.csv", index=False)
    stat_df.to_csv(POWERBI_DIR / "pbi_07_good_vs_bad_weather_boxplot_tests.csv", index=False)

    if not metric_map:
        return

    n_metrics = len(metric_map)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.7 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (label, col) in zip(axes, metric_map.items()):
        subset = daily[daily["good_bad_weather_type"].isin([BAD_WEATHER_LABEL, GOOD_WEATHER_LABEL])]
        rainy = subset.loc[subset["good_bad_weather_type"] == BAD_WEATHER_LABEL, col].dropna()
        dry = subset.loc[subset["good_bad_weather_type"] == GOOD_WEATHER_LABEL, col].dropna()
        ax.boxplot([rainy, dry], labels=[f"Regenachtig\n(n={len(rainy)})", f"Goed weer\n(n={len(dry)})"], patch_artist=True)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Vergelijking tussen regenachtig/slecht weer en zonnig/goed weer", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_07_good_vs_bad_weather_boxplots.png", dpi=200)
    plt.close(fig)


# ============================================================
# EXTRA POWERBI CSV'S
# ============================================================

def save_extra_powerbi_files(daily, song_weather):
    daily_out = daily.copy()
    daily_out["date"] = daily_out["date"].dt.strftime("%Y-%m-%d")
    daily_out.to_csv(POWERBI_DIR / "pbi_00_final_daily_dataset.csv", index=False)

    if not song_weather.empty:
        song_out = song_weather.copy()
        song_out["date"] = pd.to_datetime(song_out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        song_out.to_csv(POWERBI_DIR / "pbi_00_song_level_weather.csv", index=False)


# ============================================================
# PAPER SUMMARY
# ============================================================

def write_paper_summary(significance_df):
    path = OUTPUT_DIR / "paper_results_summary.txt"
    sig_count = int(significance_df["significant_005"].sum()) if not significance_df.empty else 0
    total_count = len(significance_df)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Spotify Top 200 x Weather - paper output summary\n")
        f.write("=================================================\n\n")
        f.write(f"Aantal statistische toetsen: {total_count}\n")
        f.write(f"Aantal significant op alpha = 0.05: {sig_count}\n\n")
        f.write("Hoofdwijzigingen in v4:\n")
        f.write("- Hoofdmetric bevat nu ook gemiddelde valence van de Top 200 op regenachtig/slecht weer vs zonnig/goed weer.\n")
        f.write("- Geen neutrale weercategorie in de finale vergelijkingen: tussendagen worden niet als aparte groep getest.\n")
        f.write("- Geen scatterplots.\n")
        f.write("- Genre, artiesten en audiofeatures worden getest tegenover prcp en tsun.\n")
        f.write("- Audiofeature-boxplots vergelijken regenachtig/slecht weer met zonnig/goed weer.\n")
        f.write("- Popularity wordt niet behandeld als audiofeature.\n\n")
        f.write("Interpretatierichtlijn:\n")
        f.write("- conclusion_005 = SIGNIFICANT betekent p < 0.05.\n")
        f.write("- conclusion_005 = NIET SIGNIFICANT betekent p >= 0.05.\n")
        f.write("- Bij genres, artiesten en audiofeatures gebruik je de BH-gecorrigeerde conclusie.\n")
        f.write("- Correlatie bewijst geen causaliteit; formuleer als samenhang.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_output_dirs()

    if not SCIPY_AVAILABLE:
        print("WAARSCHUWING: scipy is niet geïnstalleerd. Statistische p-values worden niet berekend.")
        print("Installeer met: pip install scipy")

    daily = load_daily_data()
    song_weather = load_song_weather(daily)

    significance_rows = []

    save_extra_powerbi_files(daily, song_weather)

    analyse_main_weather(daily, significance_rows)
    analyse_seasons(daily, significance_rows)
    analyse_genres(song_weather, significance_rows)
    analyse_artists(song_weather, significance_rows)
    analyse_audiofeatures_weather(daily, significance_rows)
    analyse_good_bad_weather_boxplots(daily, significance_rows)

    significance_df = pd.DataFrame(significance_rows)
    if not significance_df.empty:
        significance_df = significance_df.sort_values(["analysis", "p_adjusted", "p_value"], na_position="last")
    significance_df.to_csv(STATS_DIR / "stats_08_significance_summary.csv", index=False)
    significance_df.to_csv(POWERBI_DIR / "pbi_08_significance_summary.csv", index=False)

    write_paper_summary(significance_df)

    print("\nKlaar. Output opgeslagen in:")
    print("-", FIG_DIR)
    print("-", TABLE_DIR)
    print("-", STATS_DIR)
    print("-", POWERBI_DIR)
    print("\nBelangrijkste PowerBI-bestanden:")
    print("- pbi_00_final_daily_dataset.csv")
    print("- pbi_00_song_level_weather.csv")
    print("- pbi_01_daily_weather_valence.csv")
    print("- pbi_01_main_weather_correlations.csv")
    print("- pbi_01_main_good_vs_bad_weather_tests.csv")
    print("- pbi_02_season_valence.csv")
    print("- pbi_03_genre_weather_correlations.csv")
    print("- pbi_04_artist_weather_correlations.csv")
    print("- pbi_05_audiofeatures_weather_correlations.csv")
    print("- pbi_07_good_vs_bad_weather_boxplot_data.csv")
    print("- pbi_08_significance_summary.csv")


if __name__ == "__main__":
    main()
