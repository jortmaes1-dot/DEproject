# ============================================================
# FINAL_ANALYSIS.PY
# ============================================================
# DOEL:
# Alle gekozen onderzoeken voor de paper uitvoeren:
#
# HOOFDONDERZOEK
# 1. Valence / aandeel sad songs tegenover weer.
#
# EXTRA ONDERZOEKEN
# 2. Seizoenen vs valence.
# 3. Genre vs weer.
# 4. Artiesten vs weer.
# 5. Audiofeatures vs Top 200-positie.
# 6. Danceability en speechiness vs weer.
# 7. Energy vs weer.
#
# Per onderzoek:
# - Grafieken in output/figures
# - PowerBI-ready CSV in output/powerbi
# - Statistische test met alpha = 0.05 in output/stats
# - Globale significantietabel in pbi_08_significance_summary.csv
#
# Statistische keuzes:
# - Correlaties: Pearson + Spearman, p < 0.05.
# - Twee groepen: Welch t-test + Mann-Whitney als robuustheidscheck.
# - Meer dan twee groepen: one-way ANOVA + Kruskal-Wallis.
# - Veel genres/artiesten/features tegelijk: Benjamini-Hochberg correctie.
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

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
SEASON_LABELS_NL = {
    "winter": "winter",
    "spring": "spring",
    "summer": "summer",
    "autumn": "autumn",
}

STRICT_RAINY_LABEL = "Strict regenachtig"
STRICT_SUNNY_LABEL = "Strict zonnig"


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


def add_weather_types_if_missing(df):
    """Fallback: maak strict weather types als weather_API.py dat nog niet deed."""
    df = df.copy()
    if "prcp" in df.columns:
        df["prcp"] = pd.to_numeric(df["prcp"], errors="coerce")
    if "tsun" in df.columns:
        df["tsun"] = pd.to_numeric(df["tsun"], errors="coerce")

    if "weather_score_sunny" not in df.columns:
        df["weather_score_sunny"] = zscore(df.get("tsun", pd.Series(np.nan, index=df.index))) - zscore(df.get("prcp", pd.Series(np.nan, index=df.index)))
    if "weather_score_dreary" not in df.columns:
        df["weather_score_dreary"] = zscore(df.get("prcp", pd.Series(np.nan, index=df.index))) - zscore(df.get("tsun", pd.Series(np.nan, index=df.index)))

    if "strict_weather_type" not in df.columns:
        tsun_low = df["tsun"].quantile(0.40) if "tsun" in df.columns else np.inf
        tsun_high = df["tsun"].quantile(0.60) if "tsun" in df.columns else -np.inf
        conditions = [
            (df["prcp"] >= 1.0) & (df["tsun"] <= tsun_low),
            (df["prcp"] <= 0.2) & (df["tsun"] >= tsun_high),
        ]
        df["strict_weather_type"] = np.select(conditions, [STRICT_RAINY_LABEL, STRICT_SUNNY_LABEL], default="Gemengd/overig")

    if "rainy_day" not in df.columns and "prcp" in df.columns:
        df["rainy_day"] = df["prcp"] > 0
    if "heavy_rain" not in df.columns and "prcp" in df.columns:
        df["heavy_rain"] = df["prcp"] >= 5
    if "weather_type_key" not in df.columns:
        df["weather_type_key"] = df["strict_weather_type"].map({
            STRICT_RAINY_LABEL: "rainy",
            STRICT_SUNNY_LABEL: "sunny",
            "Gemengd/overig": "mixed",
        })

    return df


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


def welch_and_mannwhitney(data, group_col, value_col, group_a, group_b):
    result = {
        "group_a": group_a,
        "group_b": group_b,
        "value": value_col,
        "n_a": 0,
        "n_b": 0,
        "mean_a": np.nan,
        "mean_b": np.nan,
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


def add_significance_row(rows, analysis, test, variable, comparison, p_value, statistic=np.nan, n=np.nan, p_adjusted=np.nan):
    p_to_use = p_adjusted if not pd.isna(p_adjusted) else p_value
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
        "significant_005": bool(p_to_use < ALPHA) if not pd.isna(p_to_use) else False,
    })


def save_table(df, filename):
    path_table = TABLE_DIR / filename
    path_powerbi = POWERBI_DIR / filename.replace("table_", "pbi_")
    df.to_csv(path_table, index=False)
    df.to_csv(path_powerbi, index=False)
    return path_table, path_powerbi


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
        "weighted_avg_liveness", "weighted_avg_loudness", "weighted_avg_popularity",
    ]
    daily = to_numeric_if_exists(daily, numeric_cols)

    # Zorg dat oudere tussendatasets niet crashen als sommige nieuwe kolommen ontbreken.
    fallback_numeric_cols = [
        "avg_valence", "weighted_avg_valence", "weighted_avg_energy",
        "weighted_avg_danceability", "weighted_avg_speechiness",
        "weighted_share_low_energy", "weighted_share_depressive",
        "main_sad_share_metric"
    ]
    for col in fallback_numeric_cols:
        if col not in daily.columns:
            daily[col] = np.nan

    daily = add_weather_types_if_missing(daily)

    # Hoofdmetric bepalen
    if "main_sad_share_metric" not in daily.columns or daily["main_sad_share_metric"].notna().sum() == 0:
        if "share_sad_streams" in daily.columns and daily["share_sad_streams"].notna().sum() > 0:
            daily["main_sad_share_metric"] = daily["share_sad_streams"]
            daily["main_sad_share_label"] = "streamgewogen aandeel sad songs"
        elif "weighted_share_sad" in daily.columns:
            daily["main_sad_share_metric"] = daily["weighted_share_sad"]
            daily["main_sad_share_label"] = "rankgewogen aandeel sad songs"
        else:
            daily["main_sad_share_metric"] = daily["share_sad_songs"]
            daily["main_sad_share_label"] = "aandeel sad songs"

    if "main_sad_share_label" not in daily.columns:
        daily["main_sad_share_label"] = "aandeel sad songs"

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["month_name"] = daily["date"].dt.month_name()
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
        "rank", "streams", "rank_weight", "analysis_weight", "valence", "energy",
        "danceability", "speechiness", "tempo", "acousticness", "instrumentalness",
        "liveness", "loudness", "popularity",
    ]
    song = to_numeric_if_exists(song, numeric_cols)

    if "rank_weight" not in song.columns:
        song["rank_weight"] = (201 - pd.to_numeric(song["rank"], errors="coerce")).clip(lower=1)

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
        "date", "prcp", "tsun", "weather_score_sunny", "weather_score_dreary",
        "strict_weather_type", "weather_type_key", "season", "year", "month",
    ]
    weather_cols = [c for c in weather_cols if c in daily.columns]

    song_weather = song.merge(daily[weather_cols], on="date", how="left")
    return song_weather


# ============================================================
# ANALYSE 1: HOOFDONDERZOEK VALENCE/SAD SHARE VS WEER
# ============================================================

def analyse_main_valence_weather(daily, significance_rows):
    analysis_name = "01_main_valence_weather"
    metric = "main_sad_share_metric"

    pbi_cols = [
        "date", "year", "month", "season", "prcp", "tsun", "weather_score_sunny",
        "weather_score_dreary", "strict_weather_type", "weather_type_key",
        "avg_valence", "weighted_avg_valence", "share_sad_songs",
        "weighted_share_sad", "share_sad_streams", "main_sad_share_metric",
        "main_sad_share_label",
    ]
    pbi_cols = [c for c in pbi_cols if c in daily.columns]
    pbi = daily[pbi_cols].copy()
    pbi["date"] = pbi["date"].dt.strftime("%Y-%m-%d")
    pbi.to_csv(POWERBI_DIR / "pbi_01_daily_weather_valence.csv", index=False)

    # Correlaties met weer
    weather_vars = ["prcp", "tsun", "weather_score_sunny", "weather_score_dreary"]
    corr_rows = []
    for weather_var in weather_vars:
        res = safe_pearson_spearman(daily, weather_var, metric, min_n=10)
        corr_rows.append(res)
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Pearson correlation",
            variable=metric,
            comparison=f"{metric} vs {weather_var}",
            p_value=res["pearson_p"],
            statistic=res["pearson_r"],
            n=res["n"],
        )
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Spearman correlation",
            variable=metric,
            comparison=f"{metric} vs {weather_var}",
            p_value=res["spearman_p"],
            statistic=res["spearman_rho"],
            n=res["n"],
        )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(STATS_DIR / "stats_01_main_correlations.csv", index=False)
    corr_df.to_csv(POWERBI_DIR / "pbi_01_main_correlations.csv", index=False)

    # Strict regenachtig vs strict zonnig
    t_res = welch_and_mannwhitney(daily, "strict_weather_type", metric, STRICT_RAINY_LABEL, STRICT_SUNNY_LABEL)
    t_df = pd.DataFrame([t_res])
    t_df.to_csv(STATS_DIR / "stats_01_main_strict_weather_ttest.csv", index=False)
    t_df.to_csv(POWERBI_DIR / "pbi_01_main_strict_weather_ttest.csv", index=False)
    add_significance_row(
        significance_rows,
        analysis=analysis_name,
        test="Welch t-test",
        variable=metric,
        comparison=f"{STRICT_RAINY_LABEL} vs {STRICT_SUNNY_LABEL}",
        p_value=t_res["welch_p"],
        statistic=t_res["welch_t"],
        n=t_res["n_a"] + t_res["n_b"],
    )

    # Grafiek: prcp vs sad share
    d = daily[["prcp", metric]].dropna()
    if len(d) > 0:
        plt.figure(figsize=(9, 5))
        plt.scatter(d["prcp"], d[metric], alpha=0.6)
        plt.title("Neerslag vs aandeel sad songs")
        plt.xlabel("Neerslag (prcp)")
        plt.ylabel(daily["main_sad_share_label"].dropna().iloc[0] if daily["main_sad_share_label"].notna().any() else "aandeel sad songs")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_01a_prcp_vs_sad_share.png", dpi=200)
        plt.close()

    # Grafiek: tsun vs sad share
    d = daily[["tsun", metric]].dropna()
    if len(d) > 0:
        plt.figure(figsize=(9, 5))
        plt.scatter(d["tsun"], d[metric], alpha=0.6)
        plt.title("Zonneschijnduur vs aandeel sad songs")
        plt.xlabel("Zonneschijnduur (tsun)")
        plt.ylabel(daily["main_sad_share_label"].dropna().iloc[0] if daily["main_sad_share_label"].notna().any() else "aandeel sad songs")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_01b_tsun_vs_sad_share.png", dpi=200)
        plt.close()


# ============================================================
# ANALYSE 2: SEIZOENEN VS VALENCE
# ============================================================

def analyse_seasons(daily, significance_rows):
    analysis_name = "02_seasons_vs_valence"
    metric = "main_sad_share_metric"

    season_summary = daily.groupby("season", as_index=False).agg(
        n_days=("date", "count"),
        mean_sad_share=(metric, "mean"),
        median_sad_share=(metric, "median"),
        sd_sad_share=(metric, "std"),
        mean_avg_valence=("avg_valence", "mean"),
        mean_weighted_valence=("weighted_avg_valence", "mean"),
        mean_prcp=("prcp", "mean"),
        mean_tsun=("tsun", "mean"),
    )
    season_summary["season_order"] = season_summary["season"].map({s: i for i, s in enumerate(SEASON_ORDER, start=1)})
    season_summary = season_summary.sort_values("season_order")
    season_summary.to_csv(TABLE_DIR / "table_02_season_valence.csv", index=False)
    season_summary.to_csv(POWERBI_DIR / "pbi_02_season_valence.csv", index=False)

    # ANOVA/Kruskal
    res = anova_and_kruskal(daily, "season", metric)
    pd.DataFrame([res]).to_csv(STATS_DIR / "stats_02_season_anova_kruskal.csv", index=False)
    add_significance_row(
        significance_rows,
        analysis=analysis_name,
        test="One-way ANOVA",
        variable=metric,
        comparison="season groups",
        p_value=res["anova_p"],
        statistic=res["anova_f"],
        n=res["n_total"],
    )
    add_significance_row(
        significance_rows,
        analysis=analysis_name,
        test="Kruskal-Wallis",
        variable=metric,
        comparison="season groups",
        p_value=res["kruskal_p"],
        statistic=res["kruskal_h"],
        n=res["n_total"],
    )

    # Grafiek zoals voorbeeld: gemiddeld aandeel sad songs per seizoen
    plt.figure(figsize=(10, 5))
    plt.bar(season_summary["season"], season_summary["mean_sad_share"])
    plt.title("Gemiddeld aandeel sad songs per seizoen")
    plt.xlabel("Seizoen")
    plt.ylabel(daily["main_sad_share_label"].dropna().iloc[0] if daily["main_sad_share_label"].notna().any() else "aandeel sad songs")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_02_season_sad_share.png", dpi=200)
    plt.close()


# ============================================================
# ENTITY DAILY SHARES: GENRES EN ARTIESTEN
# ============================================================

def create_entity_daily_shares(song_weather, entity_col):
    required_cols = ["date", entity_col, "analysis_weight", "prcp", "tsun", "weather_score_sunny", "weather_score_dreary", "strict_weather_type"]
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

    weather_cols = ["date", "prcp", "tsun", "weather_score_sunny", "weather_score_dreary", "strict_weather_type", "season", "year", "month"]
    weather_cols = [c for c in weather_cols if c in song_weather.columns]
    weather_day = song_weather[weather_cols].drop_duplicates(subset=["date"])
    entity_day = entity_day.merge(weather_day, on="date", how="left")

    return entity_day


def entity_weather_correlations(entity_day, entity_col, weather_var, analysis_name, significance_rows):
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

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["pearson_p_adj_bh"] = p_adjust_bh(result["pearson_p"])
    result["spearman_p_adj_bh"] = p_adjust_bh(result["spearman_p"])
    result["significant_pearson_005"] = result["pearson_p_adj_bh"] < ALPHA
    result["significant_spearman_005"] = result["spearman_p_adj_bh"] < ALPHA

    for _, row in result.iterrows():
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

    return result


def strict_weather_entity_ttests(entity_day, entity_col, analysis_name, significance_rows):
    rows = []
    for entity, group in entity_day.groupby(entity_col):
        if len(group) < MIN_DAYS_ENTITY:
            continue
        res = welch_and_mannwhitney(group, "strict_weather_type", "entity_share", STRICT_RAINY_LABEL, STRICT_SUNNY_LABEL)
        res[entity_col] = entity
        rows.append(res)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["welch_p_adj_bh"] = p_adjust_bh(result["welch_p"])
    result["significant_welch_005"] = result["welch_p_adj_bh"] < ALPHA

    for _, row in result.iterrows():
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Welch t-test + BH correction",
            variable=str(row[entity_col]),
            comparison=f"entity_share: {STRICT_RAINY_LABEL} vs {STRICT_SUNNY_LABEL}",
            p_value=row["welch_p"],
            p_adjusted=row["welch_p_adj_bh"],
            statistic=row["welch_t"],
            n=row["n_a"] + row["n_b"],
        )

    return result


def plot_signed_correlations(df, label_col, corr_col, title, xlabel, filename, top_n_each_side=5):
    if df.empty or corr_col not in df.columns:
        return

    plot_df = df.dropna(subset=[corr_col]).copy()
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
    analysis_name = "03_genre_vs_weather"
    if song_weather.empty or "track_genre" not in song_weather.columns:
        return

    genre_day = create_entity_daily_shares(song_weather, "track_genre")
    if genre_day.empty:
        return

    genre_day["date"] = pd.to_datetime(genre_day["date"]).dt.strftime("%Y-%m-%d")
    genre_day.to_csv(POWERBI_DIR / "pbi_03_genre_daily_weather_share.csv", index=False)
    genre_day["date"] = pd.to_datetime(genre_day["date"])

    corr_prcp = entity_weather_correlations(genre_day, "track_genre", "prcp", analysis_name, significance_rows)
    corr_tsun = entity_weather_correlations(genre_day, "track_genre", "tsun", analysis_name, significance_rows)
    corr_score = entity_weather_correlations(genre_day, "track_genre", "weather_score_sunny", analysis_name, significance_rows)

    all_corr = pd.concat([corr_prcp, corr_tsun, corr_score], ignore_index=True)
    all_corr.to_csv(STATS_DIR / "stats_03_genre_weather_correlations.csv", index=False)
    all_corr.to_csv(POWERBI_DIR / "pbi_03_genre_weather_correlations.csv", index=False)

    ttests = strict_weather_entity_ttests(genre_day, "track_genre", analysis_name, significance_rows)
    ttests.to_csv(STATS_DIR / "stats_03_genre_strict_weather_ttests.csv", index=False)
    ttests.to_csv(POWERBI_DIR / "pbi_03_genre_strict_weather_ttests.csv", index=False)

    # Grafiek zoals voorbeeld: genres met sterkste correlatie met neerslag
    if not corr_prcp.empty:
        plot_signed_correlations(
            corr_prcp,
            label_col="track_genre",
            corr_col="pearson_r",
            title="Genres met sterkste correlatie met neerslag",
            xlabel="Correlatie met prcp",
            filename="fig_03_genres_correlation_prcp.png",
        )


# ============================================================
# ANALYSE 4: ARTIESTEN VS WEER
# ============================================================

def analyse_artists(song_weather, significance_rows):
    analysis_name = "04_artists_vs_weather"
    if song_weather.empty or "artist_first" not in song_weather.columns:
        return

    artist_day = create_entity_daily_shares(song_weather, "artist_first")
    if artist_day.empty:
        return

    artist_day["date"] = pd.to_datetime(artist_day["date"]).dt.strftime("%Y-%m-%d")
    artist_day.to_csv(POWERBI_DIR / "pbi_04_artist_daily_weather_share.csv", index=False)
    artist_day["date"] = pd.to_datetime(artist_day["date"])

    corr_tsun = entity_weather_correlations(artist_day, "artist_first", "tsun", analysis_name, significance_rows)
    corr_prcp = entity_weather_correlations(artist_day, "artist_first", "prcp", analysis_name, significance_rows)
    corr_score = entity_weather_correlations(artist_day, "artist_first", "weather_score_sunny", analysis_name, significance_rows)

    all_corr = pd.concat([corr_tsun, corr_prcp, corr_score], ignore_index=True)
    all_corr.to_csv(STATS_DIR / "stats_04_artist_weather_correlations.csv", index=False)
    all_corr.to_csv(POWERBI_DIR / "pbi_04_artist_weather_correlations.csv", index=False)

    ttests = strict_weather_entity_ttests(artist_day, "artist_first", analysis_name, significance_rows)
    ttests.to_csv(STATS_DIR / "stats_04_artist_strict_weather_ttests.csv", index=False)
    ttests.to_csv(POWERBI_DIR / "pbi_04_artist_strict_weather_ttests.csv", index=False)

    # Grafiek zoals voorbeeld: artiesten met sterkste correlatie met zonneschijnduur
    if not corr_tsun.empty:
        plot_signed_correlations(
            corr_tsun,
            label_col="artist_first",
            corr_col="pearson_r",
            title="Artiesten met sterkste correlatie met zonneschijnduur",
            xlabel="Correlatie met tsun",
            filename="fig_04_artists_correlation_tsun.png",
        )


# ============================================================
# ANALYSE 5: AUDIOFEATURES VS TOP 200
# ============================================================

def analyse_audiofeatures_rank(song_weather, significance_rows):
    analysis_name = "05_audiofeatures_vs_top200"
    if song_weather.empty:
        return

    feature_cols = [
        "popularity", "loudness", "danceability", "speechiness", "valence",
        "tempo", "instrumentalness", "acousticness", "liveness", "energy",
    ]
    feature_cols = [c for c in feature_cols if c in song_weather.columns]

    rows = []
    for feature in feature_cols:
        # Correlatie met rank: lagere rank = hoger in Top 200.
        rank_res = safe_pearson_spearman(song_weather, feature, "rank", min_n=MIN_ROWS_FEATURE)

        row = {
            "feature": feature,
            "n_rank": rank_res["n"],
            "pearson_r_with_rank": rank_res["pearson_r"],
            "pearson_p_with_rank": rank_res["pearson_p"],
            "spearman_rho_with_rank": rank_res["spearman_rho"],
            "spearman_p_with_rank": rank_res["spearman_p"],
        }

        # Correlatie met streams indien beschikbaar, anders met analysis_weight.
        if "streams" in song_weather.columns and pd.to_numeric(song_weather["streams"], errors="coerce").fillna(0).sum() > 0:
            weight_var = "streams"
        else:
            weight_var = "analysis_weight"
        weight_res = safe_pearson_spearman(song_weather, feature, weight_var, min_n=MIN_ROWS_FEATURE)
        row.update({
            "weight_var": weight_var,
            "n_weight": weight_res["n"],
            "pearson_r_with_weight": weight_res["pearson_r"],
            "pearson_p_with_weight": weight_res["pearson_p"],
            "spearman_rho_with_weight": weight_res["spearman_rho"],
            "spearman_p_with_weight": weight_res["spearman_p"],
            "abs_best_corr_rank_or_weight": np.nanmax([
                abs(rank_res["spearman_rho"]) if not pd.isna(rank_res["spearman_rho"]) else np.nan,
                abs(weight_res["spearman_rho"]) if not pd.isna(weight_res["spearman_rho"]) else np.nan,
            ]),
        })

        # Top 50 vs rest
        t_res = welch_and_mannwhitney(song_weather, "top_50" if "top_50" in song_weather.columns else "__missing__", feature, True, False)
        row.update({
            "top50_vs_rest_welch_t": t_res["welch_t"],
            "top50_vs_rest_welch_p": t_res["welch_p"],
            "top50_mean": t_res["mean_a"],
            "rest_mean": t_res["mean_b"],
        })

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return

    for p_col in ["spearman_p_with_rank", "spearman_p_with_weight", "top50_vs_rest_welch_p"]:
        result[p_col + "_adj_bh"] = p_adjust_bh(result[p_col])

    result["significant_rank_005"] = result["spearman_p_with_rank_adj_bh"] < ALPHA
    result["significant_weight_005"] = result["spearman_p_with_weight_adj_bh"] < ALPHA
    result["significant_top50_005"] = result["top50_vs_rest_welch_p_adj_bh"] < ALPHA

    result = result.sort_values("abs_best_corr_rank_or_weight", ascending=False)
    result.to_csv(STATS_DIR / "stats_05_audiofeatures_rank.csv", index=False)
    result.to_csv(POWERBI_DIR / "pbi_05_audiofeatures_rank_correlations.csv", index=False)

    for _, row in result.iterrows():
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Spearman rank correlation + BH correction",
            variable=row["feature"],
            comparison="audiofeature vs Top 200 rank",
            p_value=row["spearman_p_with_rank"],
            p_adjusted=row["spearman_p_with_rank_adj_bh"],
            statistic=row["spearman_rho_with_rank"],
            n=row["n_rank"],
        )
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Welch t-test + BH correction",
            variable=row["feature"],
            comparison="Top 50 vs rank 51-200",
            p_value=row["top50_vs_rest_welch_p"],
            p_adjusted=row["top50_vs_rest_welch_p_adj_bh"],
            statistic=row["top50_vs_rest_welch_t"],
            n=np.nan,
        )

    # Grafiek zoals voorbeeld: sterkte verband per audiofeature
    plot_df = result.dropna(subset=["abs_best_corr_rank_or_weight"]).sort_values("abs_best_corr_rank_or_weight")
    if not plot_df.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["feature"], plot_df["abs_best_corr_rank_or_weight"])
        plt.title("Sterkte van verband per audiofeature")
        plt.xlabel("Grootste absolute correlatie met ranking of gewicht")
        plt.ylabel("Spotify-audiofeature")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_05_audiofeature_rank_correlation.png", dpi=200)
        plt.close()


# ============================================================
# ANALYSE 6 EN 7: AUDIOFEATURES VS WEER
# ============================================================

def analyse_audiofeatures_weather(daily, significance_rows):
    analysis_name = "06_07_audiofeatures_vs_weather"

    daily_feature_map = {
        "valence": "weighted_avg_valence",
        "energy": "weighted_avg_energy",
        "danceability": "weighted_avg_danceability",
        "speechiness": "weighted_avg_speechiness",
        "tempo": "weighted_avg_tempo",
        "acousticness": "weighted_avg_acousticness",
        "instrumentalness": "weighted_avg_instrumentalness",
        "liveness": "weighted_avg_liveness",
        "loudness": "weighted_avg_loudness",
        "popularity": "weighted_avg_popularity",
    }
    daily_feature_map = {k: v for k, v in daily_feature_map.items() if v in daily.columns}

    rows = []
    for feature_name, feature_col in daily_feature_map.items():
        for weather_var in ["prcp", "tsun", "weather_score_sunny", "weather_score_dreary"]:
            res = safe_pearson_spearman(daily, weather_var, feature_col, min_n=10)
            rows.append({
                "feature": feature_name,
                "feature_col": feature_col,
                "weather_var": weather_var,
                "n": res["n"],
                "pearson_r": res["pearson_r"],
                "pearson_p": res["pearson_p"],
                "spearman_rho": res["spearman_rho"],
                "spearman_p": res["spearman_p"],
            })

        t_res = welch_and_mannwhitney(daily, "strict_weather_type", feature_col, STRICT_RAINY_LABEL, STRICT_SUNNY_LABEL)
        rows.append({
            "feature": feature_name,
            "feature_col": feature_col,
            "weather_var": "strict_weather_type_rainy_vs_sunny",
            "n": t_res["n_a"] + t_res["n_b"],
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": t_res["welch_t"],
            "spearman_p": t_res["welch_p"],
            "mean_rainy": t_res["mean_a"],
            "mean_sunny": t_res["mean_b"],
            "difference_rainy_minus_sunny": t_res["difference_a_minus_b"],
            "cohens_d": t_res["cohens_d"],
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return

    result["spearman_or_welch_p_adj_bh"] = p_adjust_bh(result["spearman_p"])
    result["significant_005"] = result["spearman_or_welch_p_adj_bh"] < ALPHA
    result.to_csv(STATS_DIR / "stats_06_07_audiofeatures_weather.csv", index=False)
    result.to_csv(POWERBI_DIR / "pbi_06_07_audiofeatures_weather.csv", index=False)

    for _, row in result.iterrows():
        test_name = "Welch t-test + BH correction" if row["weather_var"] == "strict_weather_type_rainy_vs_sunny" else "Spearman correlation + BH correction"
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test=test_name,
            variable=row["feature"],
            comparison=f"{row['feature']} vs {row['weather_var']}",
            p_value=row["spearman_p"],
            p_adjusted=row["spearman_or_welch_p_adj_bh"],
            statistic=row["spearman_rho"],
            n=row["n"],
        )


# ============================================================
# GRAFIEK: STRICT REGENACHTIG VS STRICT ZONNIG BOXPLOTS
# ============================================================

def analyse_weather_type_boxplots(daily, significance_rows):
    analysis_name = "08_strict_weather_type_boxplots"

    metric_map = {
        "Valence\n(positiviteit van songs)": "weighted_avg_valence",
        "Energy\n(intensiteit van songs)": "weighted_avg_energy",
        "Aandeel lage valence\n(valence <= 0.40)": "main_sad_share_metric",
        "Aandeel lage energy\n(energy <= 0.50)": "weighted_share_low_energy",
        "Aandeel depressieve songs\n(valence <= 0.40 en energy <= 0.50)": "weighted_share_depressive",
    }
    metric_map = {label: col for label, col in metric_map.items() if col in daily.columns}

    long_rows = []
    for label, col in metric_map.items():
        for _, row in daily[["date", "strict_weather_type", col]].dropna().iterrows():
            long_rows.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "strict_weather_type": row["strict_weather_type"],
                "metric_label": label.replace("\n", " "),
                "metric_col": col,
                "value": row[col],
            })

        t_res = welch_and_mannwhitney(daily, "strict_weather_type", col, STRICT_RAINY_LABEL, STRICT_SUNNY_LABEL)
        add_significance_row(
            significance_rows,
            analysis=analysis_name,
            test="Welch t-test",
            variable=col,
            comparison=f"{STRICT_RAINY_LABEL} vs {STRICT_SUNNY_LABEL}",
            p_value=t_res["welch_p"],
            statistic=t_res["welch_t"],
            n=t_res["n_a"] + t_res["n_b"],
        )

    boxplot_data = pd.DataFrame(long_rows)
    boxplot_data.to_csv(POWERBI_DIR / "pbi_07_weather_type_boxplot_data.csv", index=False)
    boxplot_data.to_csv(TABLE_DIR / "table_07_weather_type_boxplot_data.csv", index=False)

    if not metric_map:
        return

    # Eén figuur met meerdere boxplots zoals jullie voorbeeld
    n_metrics = len(metric_map)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.7 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (label, col) in zip(axes, metric_map.items()):
        subset = daily[daily["strict_weather_type"].isin([STRICT_RAINY_LABEL, STRICT_SUNNY_LABEL])]
        rainy = subset.loc[subset["strict_weather_type"] == STRICT_RAINY_LABEL, col].dropna()
        sunny = subset.loc[subset["strict_weather_type"] == STRICT_SUNNY_LABEL, col].dropna()

        ax.boxplot([rainy, sunny], labels=[f"Regenachtig\n(n={len(rainy)})", f"Zonnig\n(n={len(sunny)})"], patch_artist=True)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Line2D([0], [0], color="black", lw=6), plt.Line2D([0], [0], color="black", lw=6)]
    fig.suptitle("Vergelijking tussen strict regenachtige en strict zonnige dagen", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_07_weather_type_boxplots.png", dpi=200)
    plt.close(fig)


# ============================================================
# EXTRA POWERBI CSV'S
# ============================================================

def save_extra_powerbi_files(daily, song_weather):
    daily_out = daily.copy()
    daily_out["date"] = daily_out["date"].dt.strftime("%Y-%m-%d")
    daily_out.to_csv(POWERBI_DIR / "pbi_00_final_daily_dataset.csv", index=False)

    if not song_weather.empty:
        # Song-level file kan groot zijn, maar is nuttig voor PowerBI drilldowns.
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
        f.write("Belangrijkste outputmappen:\n")
        f.write("- output/figures: grafieken voor paper\n")
        f.write("- output/stats: statistische testen\n")
        f.write("- output/powerbi: PowerBI-ready CSV's\n")
        f.write("- output/tables: paper-tabellen\n\n")
        f.write("Interpretatierichtlijn:\n")
        f.write("- significant_005 = True betekent p < 0.05.\n")
        f.write("- Bij genres, artiesten en audiofeatures gebruik je best p_adjusted, omdat er veel vergelijkingen tegelijk gebeuren.\n")
        f.write("- Correlatie bewijst geen causaliteit; formuleer als samenhang, niet als oorzaak-gevolg.\n")


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

    analyse_main_valence_weather(daily, significance_rows)
    analyse_seasons(daily, significance_rows)
    analyse_genres(song_weather, significance_rows)
    analyse_artists(song_weather, significance_rows)
    analyse_audiofeatures_rank(song_weather, significance_rows)
    analyse_audiofeatures_weather(daily, significance_rows)
    analyse_weather_type_boxplots(daily, significance_rows)

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
    print("- pbi_01_daily_weather_valence.csv")
    print("- pbi_02_season_valence.csv")
    print("- pbi_03_genre_weather_correlations.csv")
    print("- pbi_04_artist_weather_correlations.csv")
    print("- pbi_05_audiofeatures_rank_correlations.csv")
    print("- pbi_06_07_audiofeatures_weather.csv")
    print("- pbi_07_weather_type_boxplot_data.csv")
    print("- pbi_08_significance_summary.csv")


if __name__ == "__main__":
    main()
