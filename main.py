import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

# ============================================================
# MAIN.PY
# ============================================================
# DOEL:
# - Database to calculate popularity.csv inladen
# - Belgische Spotify Top 200 selecteren
# - Final database.csv inladen
# - Valence koppelen aan elk nummer
# - Per dag een globale Top 200 valence-samenvatting maken
#
# INPUT:
# - Database to calculate popularity.csv
# - Final database.csv
#
# OUTPUT:
# - spotify_belgium_top200_with_valence.csv
# - daily_top200_valence_summary.csv
#
# BELANGRIJK:
# - Dit script houdt rekening met rare datumformaten.
# - Eerst wordt gematcht op track_id/url indien mogelijk.
# - Daarna op title + artist.
# - Daarna op title + first artist.
# ============================================================

CHARTS_FILE = "Database to calculate popularity.csv"
FEATURES_FILE = "Final database.csv"

OUTPUT_SONG_LEVEL = "spotify_belgium_top200_with_valence.csv"
OUTPUT_DAILY_SUMMARY = "daily_top200_valence_summary.csv"

COUNTRY_TARGETS = ["belgium", "be", "belgie", "belgië", "belgique"]

SAD_THRESHOLD = 0.40


# ============================================================
# HULPFUNCTIES
# ============================================================

def normalize_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def strip_accents(text):
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c))


def clean_text(value):
    if pd.isna(value):
        return ""

    value = strip_accents(str(value).lower().strip())
    value = value.replace("&", " and ")

    value = re.sub(r"\(.*?\)", " ", value)
    value = re.sub(r"\[.*?\]", " ", value)
    value = re.sub(r"\bfeat\.?\b.*", " ", value)
    value = re.sub(r"\bfeaturing\b.*", " ", value)
    value = re.sub(r"\bft\.?\b.*", " ", value)

    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", "", value)

    return value.strip()


def first_artist(value):
    text = clean_text(value)

    if not text:
        return ""

    parts = re.split(r"\s*(?:,|;|/| x | and )\s*", text)

    if parts:
        return parts[0].strip()

    return text


def pick_column(columns, candidates, required=True):
    cols = list(columns)
    candidates = [normalize_column_name(c) for c in candidates]

    for candidate in candidates:
        if candidate in cols:
            return candidate

    for col in cols:
        for candidate in candidates:
            if candidate in col:
                return col

    if required:
        raise ValueError(
            f"Kon geen kolom vinden voor: {candidates}\n"
            f"Beschikbare kolommen:\n{cols}"
        )

    return None


def extract_track_id(value):
    if pd.isna(value):
        return None

    value = str(value)

    patterns = [
        r"open\.spotify\.com/track/([A-Za-z0-9]{22})",
        r"spotify:track:([A-Za-z0-9]{22})",
        r"\b([A-Za-z0-9]{22})\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    return None


def clean_raw_date_string(value):
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s == "":
        return None

    if "T" in s:
        s = s.split("T")[0].strip()

    if " " in s:
        s = s.split(" ")[0].strip()

    return s if s != "" else None


def detect_date_order(values):
    dayfirst_score = 0
    monthfirst_score = 0

    for value in values.dropna().astype(str).head(5000):
        s = clean_raw_date_string(value)

        if not s:
            continue

        match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)

        if not match:
            continue

        first = int(match.group(1))
        second = int(match.group(2))

        if first > 12 and second <= 12:
            dayfirst_score += 1
        elif second > 12 and first <= 12:
            monthfirst_score += 1

    if dayfirst_score >= monthfirst_score:
        return "dayfirst"

    return "monthfirst"


def normalize_date_string(value, slash_order="dayfirst"):
    s = clean_raw_date_string(value)

    if not s:
        return None

    # YYYY-MM-DD of YYYY/MM/DD
    match = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))

        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

        return None

    # DD-MM-YYYY of MM-DD-YYYY
    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)

    if match:
        first = int(match.group(1))
        second = int(match.group(2))
        year = int(match.group(3))

        if slash_order == "dayfirst":
            day = first
            month = second
        else:
            month = first
            day = second

        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

        return None

    return None


def clean_numeric_series(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


def weighted_average(values, weights):
    values = pd.Series(values)
    weights = pd.Series(weights)

    mask = values.notna() & weights.notna()
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    weight_sum = weights.sum()

    if weight_sum == 0:
        return np.nan

    return (values * weights).sum() / weight_sum


def weighted_share(boolean_series, weights):
    boolean_series = pd.Series(boolean_series)
    weights = pd.Series(weights)

    mask = boolean_series.notna() & weights.notna()
    boolean_series = boolean_series[mask]
    weights = weights[mask]

    if len(boolean_series) == 0:
        return np.nan

    weight_sum = weights.sum()

    if weight_sum == 0:
        return np.nan

    return weights[boolean_series.astype(bool)].sum() / weight_sum


def has_usable_streams(df, stream_col="streams"):
    if stream_col not in df.columns:
        return False

    s = pd.to_numeric(df[stream_col], errors="coerce")

    if s.notna().sum() == 0:
        return False

    if s.fillna(0).sum() <= 0:
        return False

    return True


# ============================================================
# STAP 1: BESTANDEN CONTROLEREN
# ============================================================

if not Path(CHARTS_FILE).exists():
    raise FileNotFoundError(
        f"Bestand niet gevonden: {CHARTS_FILE}\n"
        "Zorg dat dit bestand in dezelfde map staat als main.py."
    )

if not Path(FEATURES_FILE).exists():
    raise FileNotFoundError(
        f"Bestand niet gevonden: {FEATURES_FILE}\n"
        "Zorg dat dit bestand in dezelfde map staat als main.py."
    )

print("Bestanden gevonden:")
print("-", CHARTS_FILE)
print("-", FEATURES_FILE)


# ============================================================
# STAP 2: FINAL DATABASE / VALENCE INLADEN
# ============================================================

features = pd.read_csv(FEATURES_FILE, low_memory=False)
features.columns = [normalize_column_name(c) for c in features.columns]

print("\nKolommen in Final database:")
print(features.columns.tolist())

valence_col = pick_column(features.columns, ["valence"])

feature_title_col = pick_column(
    features.columns,
    ["track_name", "title", "name", "song_name"],
    required=False
)

feature_artist_col = pick_column(
    features.columns,
    ["artists", "artist", "artist_name"],
    required=False
)

feature_id_col = pick_column(
    features.columns,
    ["track_id", "spotify_id", "id", "url", "spotify_url", "track_url", "uri", "track_uri"],
    required=False
)

features["valence"] = clean_numeric_series(features[valence_col])

if feature_id_col is not None:
    features["track_id"] = features[feature_id_col].apply(extract_track_id)
else:
    features["track_id"] = None

if feature_title_col is not None:
    features["title_clean"] = features[feature_title_col].apply(clean_text)
else:
    features["title_clean"] = ""

if feature_artist_col is not None:
    features["artist_clean"] = features[feature_artist_col].apply(clean_text)
    features["artist_first_clean"] = features[feature_artist_col].apply(first_artist)
else:
    features["artist_clean"] = ""
    features["artist_first_clean"] = ""

features_small = features[
    ["track_id", "title_clean", "artist_clean", "artist_first_clean", "valence"]
].copy()

features_small = features_small.dropna(subset=["valence"]).copy()

features_by_id = (
    features_small
    .dropna(subset=["track_id"])
    .drop_duplicates(subset=["track_id"])
    .copy()
)

features_by_text = (
    features_small[
        (features_small["title_clean"] != "") &
        (features_small["artist_clean"] != "")
    ]
    .drop_duplicates(subset=["title_clean", "artist_clean"])
    .copy()
)

features_by_first_artist = (
    features_small[
        (features_small["title_clean"] != "") &
        (features_small["artist_first_clean"] != "")
    ]
    .drop_duplicates(subset=["title_clean", "artist_first_clean"])
    .copy()
)

print("\nFeatures klaar:")
print("Rijen met valence:", len(features_small))
print("Rijen met track_id:", len(features_by_id))
print("Rijen met title + artist:", len(features_by_text))
print("Rijen met title + first artist:", len(features_by_first_artist))


# ============================================================
# STAP 3: DATABASE TO CALCULATE POPULARITY INLADEN
# ============================================================

charts = pd.read_csv(CHARTS_FILE, low_memory=False)
charts.columns = [normalize_column_name(c) for c in charts.columns]

print("\nKolommen in Database to calculate popularity:")
print(charts.columns.tolist())

date_col = pick_column(charts.columns, ["date", "day"])
rank_col = pick_column(charts.columns, ["rank", "position"])
country_col = pick_column(charts.columns, ["country", "region"])
title_col = pick_column(charts.columns, ["title", "track_name", "name"])
artist_col = pick_column(charts.columns, ["artist", "artists"])

chart_type_col = pick_column(
    charts.columns,
    ["chart", "chart_type", "type"],
    required=False
)

stream_col = pick_column(
    charts.columns,
    ["streams", "stream", "num_streams", "stream_count"],
    required=False
)

chart_id_col = pick_column(
    charts.columns,
    ["track_id", "spotify_id", "id", "url", "spotify_url", "track_url", "uri", "track_uri"],
    required=False
)

print("\nGebruikte chartkolommen:")
print("date:", date_col)
print("rank:", rank_col)
print("country:", country_col)
print("title:", title_col)
print("artist:", artist_col)
print("chart/type:", chart_type_col)
print("streams:", stream_col)
print("track_id/url:", chart_id_col)

print("\nEerste 20 ruwe datumwaarden:")
print(charts[date_col].head(20).to_string(index=False))


# ============================================================
# STAP 4: DATUM CLEANEN
# ============================================================

slash_order = detect_date_order(charts[date_col])
print("\nGedetecteerde slash-volgorde:", slash_order)

charts["date"] = charts[date_col].apply(
    lambda x: normalize_date_string(x, slash_order=slash_order)
)

charts["date_dt"] = pd.to_datetime(charts["date"], errors="coerce")

charts[rank_col] = clean_numeric_series(charts[rank_col])
charts["country_clean"] = charts[country_col].astype(str).apply(clean_text)

charts = charts.dropna(subset=["date", "date_dt", rank_col]).copy()

print("\nEerste 20 genormaliseerde datumwaarden:")
print(charts["date"].head(20).to_string(index=False))


# ============================================================
# STAP 5: FILTEREN OP BELGIË + TOP 200
# ============================================================

charts = charts[charts["country_clean"].isin(COUNTRY_TARGETS)].copy()

if chart_type_col is not None:
    chart_type_clean = charts[chart_type_col].astype(str).str.lower().str.strip()

    top_mask = (
        chart_type_clean.str.contains("top", na=False) &
        ~chart_type_clean.str.contains("viral", na=False)
    )

    if top_mask.sum() > 0:
        charts = charts[top_mask].copy()
        print("\nChart-type filter toegepast: top chart, geen viral chart.")
    else:
        print("\nGeen bruikbare chart-type filter gevonden. Er wordt enkel op rank 1-200 gefilterd.")

charts = charts[charts[rank_col].between(1, 200)].copy()

charts["rank"] = charts[rank_col].astype(int)
charts["title_clean"] = charts[title_col].apply(clean_text)
charts["artist_clean"] = charts[artist_col].apply(clean_text)
charts["artist_first_clean"] = charts[artist_col].apply(first_artist)

if chart_id_col is not None:
    charts["track_id"] = charts[chart_id_col].apply(extract_track_id)
else:
    charts["track_id"] = None

if stream_col is not None:
    charts["streams"] = clean_numeric_series(charts[stream_col])
else:
    charts["streams"] = np.nan

charts["weight_rank_linear"] = 201 - charts["rank"]

print("\nNa filtering op België + Top 200:")
print("Aantal rijen:", len(charts))
print("Aantal unieke dagen:", charts["date"].nunique())
print("Minimum datum:", charts["date"].min())
print("Maximum datum:", charts["date"].max())

if charts.empty:
    raise ValueError(
        "Geen rijen over na filtering op België + Top 200.\n"
        "Controleer country/region, chart-type en rankkolommen."
    )


# ============================================================
# STAP 6: VALENCE KOPPELEN
# ============================================================

song_level = charts.copy().reset_index(drop=True)
song_level["valence"] = np.nan
song_level["merge_method"] = pd.NA

# 1. Match via track_id
if song_level["track_id"].notna().any() and len(features_by_id) > 0:
    id_map = features_by_id.set_index("track_id")["valence"].to_dict()
    matched_values = song_level["track_id"].map(id_map)
    matched_mask = matched_values.notna()

    song_level.loc[matched_mask, "valence"] = matched_values[matched_mask]
    song_level.loc[matched_mask, "merge_method"] = "track_id"

# 2. Match via title + artist
if len(features_by_text) > 0:
    text_map = features_by_text.set_index(["title_clean", "artist_clean"])["valence"].to_dict()

    unmatched_mask = song_level["valence"].isna()

    for idx, row in song_level.loc[unmatched_mask].iterrows():
        key = (row["title_clean"], row["artist_clean"])
        value = text_map.get(key, np.nan)

        if pd.notna(value):
            song_level.at[idx, "valence"] = value
            song_level.at[idx, "merge_method"] = "title_artist"

# 3. Match via title + first artist
if len(features_by_first_artist) > 0:
    first_artist_map = features_by_first_artist.set_index(
        ["title_clean", "artist_first_clean"]
    )["valence"].to_dict()

    unmatched_mask = song_level["valence"].isna()

    for idx, row in song_level.loc[unmatched_mask].iterrows():
        key = (row["title_clean"], row["artist_first_clean"])
        value = first_artist_map.get(key, np.nan)

        if pd.notna(value):
            song_level.at[idx, "valence"] = value
            song_level.at[idx, "merge_method"] = "title_first_artist"

song_level["valence"] = pd.to_numeric(song_level["valence"], errors="coerce")
song_level["sad_song"] = song_level["valence"] <= SAD_THRESHOLD

song_level = song_level.sort_values(
    ["date_dt", "rank"],
    kind="stable"
).reset_index(drop=True)

streams_available = has_usable_streams(song_level, "streams")

print("\nMergecontrole valence:")
print("Totaal aantal rijen:", len(song_level))
print("Aantal rijen met valence:", song_level["valence"].notna().sum())
print("Match rate:", round(song_level["valence"].notna().mean() * 100, 2), "%")

print("\nMatch methodes:")
print(song_level["merge_method"].value_counts(dropna=False))

print("\nStreams beschikbaar:", streams_available)


# ============================================================
# STAP 7: SONG-LEVEL OPSLAAN
# ============================================================

song_level.to_csv(OUTPUT_SONG_LEVEL, index=False)

print("\nSong-level bestand opgeslagen:")
print("-", OUTPUT_SONG_LEVEL)


# ============================================================
# STAP 8: DAGELIJKSE TOP 200 SAMENVATTING
# ============================================================

daily_coverage = (
    song_level
    .groupby("date")
    .agg(
        top200_rows=("rank", "count"),
        tracks_with_valence=("valence", lambda x: x.notna().sum()),
        total_rank_weight_all=("weight_rank_linear", "sum"),
        total_streams_all=("streams", "sum")
    )
    .reset_index()
)

daily_coverage["valence_coverage"] = (
    daily_coverage["tracks_with_valence"] /
    daily_coverage["top200_rows"]
)

usable = song_level.dropna(subset=["date", "valence"]).copy()

daily_rows = []

for date_value, group in usable.groupby("date"):
    row = {
        "date": date_value,
        "matched_rows": len(group),
        "avg_valence": group["valence"].mean(),
        "median_valence": group["valence"].median(),
        "std_valence": group["valence"].std(),
        "min_valence": group["valence"].min(),
        "max_valence": group["valence"].max(),
        "sad_songs_count": int(group["sad_song"].sum()),
        "share_sad_songs": group["sad_song"].mean(),
        "weighted_avg_valence_rank": weighted_average(group["valence"], group["weight_rank_linear"]),
        "weighted_sad_share_rank": weighted_share(group["sad_song"], group["weight_rank_linear"])
    }

    if streams_available:
        row["weighted_avg_valence_streams"] = weighted_average(group["valence"], group["streams"])
        row["weighted_sad_share_streams"] = weighted_share(group["sad_song"], group["streams"])
    else:
        row["weighted_avg_valence_streams"] = np.nan
        row["weighted_sad_share_streams"] = np.nan

    daily_rows.append(row)

daily_metrics = pd.DataFrame(daily_rows)

daily_summary = pd.merge(
    daily_coverage,
    daily_metrics,
    on="date",
    how="left"
)

if streams_available:
    daily_summary["primary_valence_metric"] = "weighted_avg_valence_streams"
    daily_summary["primary_sad_metric"] = "weighted_sad_share_streams"
else:
    daily_summary["primary_valence_metric"] = "weighted_avg_valence_rank"
    daily_summary["primary_sad_metric"] = "weighted_sad_share_rank"

daily_summary = daily_summary.sort_values("date").reset_index(drop=True)

daily_summary.to_csv(OUTPUT_DAILY_SUMMARY, index=False)

print("\nDaily summary opgeslagen:")
print("-", OUTPUT_DAILY_SUMMARY)

print("\nEerste 10 rijen:")
print(daily_summary.head(10).to_string(index=False))

print("\nDatumbereik:")
print(daily_summary["date"].min(), "tot", daily_summary["date"].max())

print("\nAantal unieke dagen:")
print(daily_summary["date"].nunique())

print("\nKlaar.")