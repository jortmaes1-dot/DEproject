import pandas as pd
import re
import unicodedata
from pathlib import Path

# ============================================================
# MAIN.PY
# ============================================================
# DOEL:
# 1. Belgische Spotify Top 200 selecteren
# 2. Valence koppelen
# 3. Song-level output bewaren
# 4. Dagelijkse valence-samenvatting maken
#
# OUTPUT:
# - spotify_belgium_top200_with_valence.csv
# - daily_top200_overview.csv
# - daily_valence_summary.csv
#
# BELANGRIJK:
# - datum wordt veilig genormaliseerd naar YYYY-MM-DD
# - zo vermijden we de "eerste 12 dagen van de maand"-bug
# ============================================================

CHARTS_FILE = "Database to calculate popularity.csv"
FEATURES_FILE = "Final database.csv"

COUNTRY_TARGET = "belgium"

OUTPUT_SONG_LEVEL = "spotify_belgium_top200_with_valence.csv"
OUTPUT_DAILY_OVERVIEW = "daily_top200_overview.csv"
OUTPUT_DAILY_VALENCE = "daily_valence_summary.csv"

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
    return parts[0].strip() if parts else text


def pick_column(columns, candidates, required=True):
    cols = list(columns)
    candidates = [normalize_column_name(x) for x in candidates]

    for cand in candidates:
        if cand in cols:
            return cand

    for col in cols:
        for cand in candidates:
            if cand in col:
                return col

    if required:
        raise ValueError(
            f"Kon geen kolom vinden voor {candidates}.\n"
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
    """
    Probeert te bepalen of data zoals 03/04/2019
    day-first of month-first zijn.
    """
    dayfirst_score = 0
    monthfirst_score = 0

    for value in values.dropna().astype(str).head(5000):
        s = clean_raw_date_string(value)
        if not s:
            continue

        m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
        if not m:
            continue

        a = int(m.group(1))
        b = int(m.group(2))

        if a > 12 and b <= 12:
            dayfirst_score += 1
        elif b > 12 and a <= 12:
            monthfirst_score += 1

    if dayfirst_score >= monthfirst_score:
        return "dayfirst"
    return "monthfirst"


def normalize_date_string(value, slash_order="dayfirst"):
    """
    Zet datum veilig om naar exact YYYY-MM-DD.
    Ondersteunt:
    - YYYY-MM-DD
    - YYYY/MM/DD
    - DD/MM/YYYY
    - DD-MM-YYYY
    - MM/DD/YYYY
    - MM-DD-YYYY
    """
    s = clean_raw_date_string(value)

    if not s:
        return None

    # YYYY-MM-DD of YYYY/MM/DD
    m = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))

        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None

    # DD/MM/YYYY of MM/DD/YYYY
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m:
        first = int(m.group(1))
        second = int(m.group(2))
        year = int(m.group(3))

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


def safe_sort_dates_as_text(df, date_col, rank_col):
    return df.sort_values(by=[date_col, rank_col], kind="stable").reset_index(drop=True)


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
# STAP 2: FEATURES / VALENCE INLADEN
# ============================================================

features = pd.read_csv(FEATURES_FILE, low_memory=False)
features.columns = [normalize_column_name(c) for c in features.columns]

print("\nKolommen in valencebestand:")
print(features.columns.tolist())

valence_col = pick_column(features.columns, ["valence"])
feature_title_col = pick_column(features.columns, ["track_name", "title", "name"], required=False)
feature_artist_col = pick_column(features.columns, ["artists", "artist"], required=False)
feature_id_col = pick_column(
    features.columns,
    ["track_id", "spotify_id", "id", "url", "spotify_url", "track_url", "uri", "track_uri"],
    required=False
)

features["valence"] = pd.to_numeric(features[valence_col], errors="coerce")

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

extra_audio_cols = []
for col in ["danceability", "energy", "tempo", "acousticness", "instrumentalness", "liveness", "speechiness"]:
    if col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        extra_audio_cols.append(col)

keep_cols = ["track_id", "title_clean", "artist_clean", "artist_first_clean", "valence"] + extra_audio_cols
features_small = features[keep_cols].copy()

features_by_id = features_small.dropna(subset=["track_id"]).drop_duplicates(subset=["track_id"])
features_by_text = features_small[
    (features_small["title_clean"] != "") & (features_small["artist_clean"] != "")
].drop_duplicates(subset=["title_clean", "artist_clean"])
features_by_first_artist = features_small[
    (features_small["title_clean"] != "") & (features_small["artist_first_clean"] != "")
].drop_duplicates(subset=["title_clean", "artist_first_clean"])

print("\nFeatures klaar:")
print("Rijen met track_id:", len(features_by_id))
print("Rijen met title + artist:", len(features_by_text))
print("Rijen met title + first artist:", len(features_by_first_artist))
print("Rijen met valence:", features_small["valence"].notna().sum())


# ============================================================
# STAP 3: CHARTBESTAND INLADEN
# ============================================================

charts = pd.read_csv(CHARTS_FILE, low_memory=False)
charts.columns = [normalize_column_name(c) for c in charts.columns]

print("\nKolommen in chartbestand:")
print(charts.columns.tolist())

date_col = pick_column(charts.columns, ["date", "day"])
rank_col = pick_column(charts.columns, ["rank", "position"])
country_col = pick_column(charts.columns, ["country", "region"])
title_col = pick_column(charts.columns, ["title", "track_name", "name"])
artist_col = pick_column(charts.columns, ["artist", "artists"])

chart_id_col = pick_column(
    charts.columns,
    ["track_id", "spotify_id", "id", "url", "spotify_url", "track_url", "uri", "track_uri"],
    required=False
)

streams_col = pick_column(charts.columns, ["streams", "stream"], required=False)

print("\nGebruikte chartkolommen:")
print("date:", date_col)
print("rank:", rank_col)
print("country:", country_col)
print("title:", title_col)
print("artist:", artist_col)
print("track_id/url:", chart_id_col)
print("streams:", streams_col)

print("\nEerste 20 ruwe datumwaarden:")
print(charts[date_col].head(20).to_string(index=False))


# ============================================================
# STAP 4: DATUM VEILIG NORMALISEREN
# ============================================================

slash_order = detect_date_order(charts[date_col])
print("\nGedetecteerde slash-volgorde:", slash_order)

charts["date"] = charts[date_col].apply(lambda x: normalize_date_string(x, slash_order=slash_order))
charts[rank_col] = pd.to_numeric(charts[rank_col], errors="coerce")
charts[country_col] = charts[country_col].astype(str).str.strip().str.lower()

if streams_col is not None:
    charts["streams"] = pd.to_numeric(charts[streams_col], errors="coerce")

print("\nEerste 20 genormaliseerde datumwaarden:")
print(charts["date"].head(20).to_string(index=False))

charts = charts.dropna(subset=["date", rank_col]).copy()

# Alleen België
charts = charts[charts[country_col] == COUNTRY_TARGET].copy()

# Alleen Top 200
charts = charts[charts[rank_col].between(1, 200)].copy()

charts["title_clean"] = charts[title_col].apply(clean_text)
charts["artist_clean"] = charts[artist_col].apply(clean_text)
charts["artist_first_clean"] = charts[artist_col].apply(first_artist)

if chart_id_col is not None:
    charts["track_id"] = charts[chart_id_col].apply(extract_track_id)
else:
    charts["track_id"] = None

charts["rank"] = charts[rank_col]

print("\nNa filtering:")
print("Aantal rijen:", len(charts))
print("Aantal unieke datumstrings:", charts["date"].nunique())
print("Eerste 20 datumstrings:")
print(charts["date"].drop_duplicates().sort_values().head(20).to_string(index=False))
print("Laatste 20 datumstrings:")
print(charts["date"].drop_duplicates().sort_values().tail(20).to_string(index=False))

daily_counts = charts.groupby("date").size().sort_index()

print("\nRijen per dag:")
print(daily_counts.describe())

print("\nDagen met niet exact 200 rijen:")
print(daily_counts[daily_counts != 200].head(20))
print("Aantal zulke dagen:", (daily_counts != 200).sum())


# ============================================================
# STAP 5: MERGEN MET VALENCE
# ============================================================

song_level = charts.copy()
song_level["merge_method"] = None

audio_cols = ["valence"] + extra_audio_cols
for col in audio_cols:
    song_level[col] = pd.NA

# 1. Eerst via track_id
if song_level["track_id"].notna().any() and len(features_by_id) > 0:
    merged_id = song_level.merge(
        features_by_id[["track_id"] + audio_cols],
        on="track_id",
        how="left",
        suffixes=("", "_new")
    )

    matched = merged_id["valence_new"].notna()

    for col in audio_cols:
        merged_id.loc[matched, col] = merged_id.loc[matched, f"{col}_new"]
        merged_id.drop(columns=[f"{col}_new"], inplace=True)

    merged_id.loc[matched, "merge_method"] = "track_id"
    song_level = merged_id

# 2. Dan via title + artist
unmatched_mask = song_level["valence"].isna()
if unmatched_mask.any():
    to_match = song_level.loc[unmatched_mask].merge(
        features_by_text[["title_clean", "artist_clean"] + audio_cols],
        on=["title_clean", "artist_clean"],
        how="left",
        suffixes=("", "_new")
    )

    matched_rows = to_match["valence_new"].notna().values

    if matched_rows.any():
        idx = song_level.loc[unmatched_mask].index[matched_rows]

        for col in audio_cols:
            song_level.loc[idx, col] = to_match.loc[matched_rows, f"{col}_new"].values

        song_level.loc[idx, "merge_method"] = "title_artist"

# 3. Dan via title + first artist
unmatched_mask = song_level["valence"].isna()
if unmatched_mask.any():
    to_match = song_level.loc[unmatched_mask].merge(
        features_by_first_artist[["title_clean", "artist_first_clean"] + audio_cols],
        on=["title_clean", "artist_first_clean"],
        how="left",
        suffixes=("", "_new")
    )

    matched_rows = to_match["valence_new"].notna().values

    if matched_rows.any():
        idx = song_level.loc[unmatched_mask].index[matched_rows]

        for col in audio_cols:
            song_level.loc[idx, col] = to_match.loc[matched_rows, f"{col}_new"].values

        song_level.loc[idx, "merge_method"] = "title_first_artist"

for col in audio_cols:
    song_level[col] = pd.to_numeric(song_level[col], errors="coerce")


# ============================================================
# STAP 6: CONTROLE OP MERGE
# ============================================================

song_level = safe_sort_dates_as_text(song_level, "date", "rank")

total_rows = len(song_level)
matched_rows = song_level["valence"].notna().sum()
match_rate = matched_rows / total_rows * 100 if total_rows else 0

print("\nMergecontrole:")
print("Totaal aantal rijen:", total_rows)
print("Aantal rijen met valence:", matched_rows)
print("Match rate:", round(match_rate, 2), "%")

print("\nMatch methodes:")
print(song_level["merge_method"].value_counts(dropna=False))

daily_coverage = song_level.groupby("date").agg(
    top200_rows=("rank", "count"),
    tracks_with_valence=("valence", lambda x: x.notna().sum())
).reset_index()

daily_coverage["valence_coverage"] = (
    daily_coverage["tracks_with_valence"] / daily_coverage["top200_rows"]
)

daily_coverage = daily_coverage.sort_values("date", kind="stable").reset_index(drop=True)

print("\nDaily overview:")
print(daily_coverage.head(20).to_string(index=False))


# ============================================================
# STAP 7: SONG-LEVEL EN OVERVIEW OPSLAAN
# ============================================================

song_level.to_csv(OUTPUT_SONG_LEVEL, index=False)
daily_coverage.to_csv(OUTPUT_DAILY_OVERVIEW, index=False)

print("\nBestanden opgeslagen:")
print("-", OUTPUT_SONG_LEVEL)
print("-", OUTPUT_DAILY_OVERVIEW)


# ============================================================
# STAP 8: DAGELIJKSE VALENCE-SAMENVATTING
# ============================================================

usable = song_level.dropna(subset=["date", "valence"]).copy()
usable["sad_song"] = usable["valence"] <= SAD_THRESHOLD

daily_metrics = usable.groupby("date").agg(
    avg_valence=("valence", "mean"),
    median_valence=("valence", "median"),
    std_valence=("valence", "std"),
    min_valence=("valence", "min"),
    max_valence=("valence", "max"),
    sad_songs_count=("sad_song", "sum")
).reset_index()

daily_metrics["sad_songs_count"] = daily_metrics["sad_songs_count"].astype(int)

daily_valence_summary = pd.merge(
    daily_coverage,
    daily_metrics,
    on="date",
    how="left"
)

daily_valence_summary["share_sad_songs"] = (
    daily_valence_summary["sad_songs_count"] / daily_valence_summary["tracks_with_valence"]
)

# Weighted metrics als streams bestaat
if "streams" in song_level.columns:
    stream_usable = song_level.dropna(subset=["date", "valence", "streams"]).copy()
    stream_usable = stream_usable[stream_usable["streams"] >= 0].copy()

    if not stream_usable.empty:
        weighted_rows = []

        for date_value, group in stream_usable.groupby("date"):
            total_streams = group["streams"].sum()

            if total_streams > 0:
                weighted_avg_valence = (group["valence"] * group["streams"]).sum() / total_streams
                sad_streams = group.loc[group["valence"] <= SAD_THRESHOLD, "streams"].sum()
                share_sad_streams = sad_streams / total_streams
            else:
                weighted_avg_valence = pd.NA
                sad_streams = pd.NA
                share_sad_streams = pd.NA

            weighted_rows.append({
                "date": date_value,
                "total_streams": total_streams,
                "sad_streams": sad_streams,
                "weighted_avg_valence": weighted_avg_valence,
                "share_sad_streams": share_sad_streams
            })

        weighted_df = pd.DataFrame(weighted_rows)

        daily_valence_summary = pd.merge(
            daily_valence_summary,
            weighted_df,
            on="date",
            how="left"
        )

daily_valence_summary = daily_valence_summary.sort_values("date").reset_index(drop=True)
daily_valence_summary.to_csv(OUTPUT_DAILY_VALENCE, index=False)

print("\nBestand opgeslagen:")
print("-", OUTPUT_DAILY_VALENCE)

print("\nDatumbereik daily valence summary:")
print(daily_valence_summary["date"].min(), "tot", daily_valence_summary["date"].max())

print("\nAantal unieke dagen daily valence summary:")
print(daily_valence_summary["date"].nunique())

print("\nGebruikte sad-definitie:")
print(f"sad_song = valence <= {SAD_THRESHOLD}")

print("\nKlaar.")