# ============================================================
# MAIN.PY
# ============================================================
# DOEL:
# 1. Belgische Spotify Top 200 selecteren uit het chartbestand.
# 2. Spotify-audiofeatures koppelen: valence, energy, danceability,
#    speechiness, tempo, acousticness, liveness, loudness, genre en artist.
#    Popularity wordt wel bewaard als metadata, maar NIET behandeld als audiofeature.
# 3. Song-level dataset bewaren.
# 4. Dagelijkse analysevariabelen maken:
#    - gemiddelde valence
#    - aandeel sad songs
#    - streamgewogen of rankgewogen aandeel sad songs
#    - weighted audiofeatures per dag
#
# INPUT:
# - Database to calculate popularity.csv
#   of Database to calculate popularity.csv (2).zip
# - Final database.csv
#   of Final database.csv (2).zip
#
# OUTPUT:
# - spotify_belgium_top200_with_features.csv
# - daily_top200_overview.csv
# - daily_valence_summary.csv
# - data_quality_summary.csv
#
# BELANGRIJK:
# - Als streams bestaan, gebruikt de code streams als gewicht.
# - Als streams ontbreken, gebruikt de code rank_weight = 201 - rank.
#   Dan is de hoofdmaat geen echte stream share, maar een rankgewogen share.
# ============================================================

import re
import zipfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# INSTELLINGEN
# ============================================================

CHARTS_FILE = "Database to calculate popularity.csv"
FEATURES_FILE = "Final database.csv"
COUNTRY_TARGET = "belgium"

OUTPUT_SONG_LEVEL = "spotify_belgium_top200_with_features.csv"
OUTPUT_DAILY_OVERVIEW = "daily_top200_overview.csv"
OUTPUT_DAILY_VALENCE = "daily_valence_summary.csv"
OUTPUT_QUALITY = "data_quality_summary.csv"

SAD_THRESHOLD = 0.40
LOW_ENERGY_THRESHOLD = 0.50
CHUNKSIZE = 300_000


# ============================================================
# HULPFUNCTIES: BESTANDEN EN KOLOMMEN
# ============================================================

def find_file(preferred_name: str) -> Path:
    """Zoek eerst exact CSV-bestand, daarna zipbestand dat dit CSV-bestand bevat."""
    preferred = Path(preferred_name)
    if preferred.exists():
        return preferred

    # Zoek naar zipbestanden in dezelfde map, bv. "Database ... (2).zip"
    candidates = list(Path(".").glob(f"{preferred.stem}*.zip")) + list(Path(".").glob("*.zip"))
    for zip_path in candidates:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                if preferred_name in names or any(Path(name).name == preferred_name for name in names):
                    return zip_path
        except zipfile.BadZipFile:
            continue

    raise FileNotFoundError(
        f"Bestand niet gevonden: {preferred_name}\n"
        f"Zet {preferred_name} of een zipbestand met dit CSV-bestand in dezelfde map."
    )


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    """Lees CSV of ZIP met CSV in."""
    if path.suffix.lower() == ".zip":
        return pd.read_csv(path, compression="zip", **kwargs)
    return pd.read_csv(path, **kwargs)


def normalize_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def strip_accents(text):
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c))


def clean_text(value):
    """Maak titels en artiesten vergelijkbaar voor matching."""
    if pd.isna(value):
        return ""

    value = strip_accents(str(value).lower().strip())
    value = value.replace("&", " and ")

    # Verwijder meestal storende extra informatie
    value = re.sub(r"\(.*?\)", " ", value)
    value = re.sub(r"\[.*?\]", " ", value)
    value = re.sub(r"\bfeat\.?\b.*", " ", value)
    value = re.sub(r"\bfeaturing\b.*", " ", value)
    value = re.sub(r"\bft\.?\b.*", " ", value)

    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def first_artist(value):
    text = clean_text(value)
    if not text:
        return ""
    parts = re.split(r"\s*(?:,|;|/| x | and )\s*", text)
    return parts[0].strip() if parts else text


def pick_column(columns, candidates, required=True):
    """Kies kolom op basis van mogelijke namen, case-insensitive en flexibel."""
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
    """Haal Spotify track_id uit URL, URI of tekst."""
    if pd.isna(value):
        return None
    value = str(value)
    patterns = [
        r"open\.spotify\.com/track/([A-Za-z0-9]{22})",
        r"spotify:track:([A-Za-z0-9]{22})",
        r"\b([A-Za-z0-9]{22})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


# ============================================================
# HULPFUNCTIES: DATUM
# ============================================================

def clean_raw_date_string(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if "T" in s:
        s = s.split("T")[0].strip()
    if " " in s:
        s = s.split(" ")[0].strip()
    return s if s else None


def detect_date_order(values):
    """Bepaal of slash-data eerder dayfirst of monthfirst zijn."""
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

    return "dayfirst" if dayfirst_score >= monthfirst_score else "monthfirst"


def normalize_date_string(value, slash_order="dayfirst"):
    """Zet verschillende datumformats veilig om naar YYYY-MM-DD."""
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


# ============================================================
# STAP 1: FEATURES INLADEN EN VOORBEREIDEN
# ============================================================

def load_features():
    features_path = find_file(FEATURES_FILE)
    print("Featuresbestand gevonden:", features_path)

    features = read_csv_flexible(features_path, low_memory=False)
    features.columns = [normalize_column_name(c) for c in features.columns]

    print("\nKolommen in featuresbestand:")
    print(features.columns.tolist())

    valence_col = pick_column(features.columns, ["valence"])
    title_col = pick_column(features.columns, ["track_name", "track", "title", "name"], required=False)
    artist_col = pick_column(features.columns, ["artists", "artist", "artist_name"], required=False)
    id_col = pick_column(
        features.columns,
        ["track_id", "spotify_id", "id", "url", "spotify_url", "track_url", "uri", "track_uri"],
        required=False,
    )
    genre_col = pick_column(features.columns, ["track_genre", "genre", "genres"], required=False)

    # Audiofeatures die nuttig zijn voor jullie paper
    candidate_audio_cols = [
        "valence",
        "energy",
        "danceability",
        "speechiness",
        "tempo",
        "acousticness",
        "instrumentalness",
        "liveness",
        "loudness",
        "popularity",
    ]

    features["valence"] = pd.to_numeric(features[valence_col], errors="coerce")

    if id_col is not None:
        features["track_id"] = features[id_col].apply(extract_track_id)
    else:
        features["track_id"] = None

    if title_col is not None:
        features["feature_title"] = features[title_col].astype(str)
        features["title_clean"] = features[title_col].apply(clean_text)
    else:
        features["feature_title"] = ""
        features["title_clean"] = ""

    if artist_col is not None:
        features["feature_artists"] = features[artist_col].astype(str)
        features["artist_clean"] = features[artist_col].apply(clean_text)
        features["artist_first_clean"] = features[artist_col].apply(first_artist)
    else:
        features["feature_artists"] = ""
        features["artist_clean"] = ""
        features["artist_first_clean"] = ""

    if genre_col is not None:
        features["track_genre"] = features[genre_col].astype(str).str.strip().str.lower()
    else:
        features["track_genre"] = "unknown"

    for col in candidate_audio_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
        elif col != "valence":
            features[col] = np.nan

    keep_cols = [
        "track_id",
        "title_clean",
        "artist_clean",
        "artist_first_clean",
        "feature_title",
        "feature_artists",
        "track_genre",
    ] + candidate_audio_cols

    features_small = features[keep_cols].copy()

    features_by_id = (
        features_small.dropna(subset=["track_id"])
        .drop_duplicates(subset=["track_id"])
        .reset_index(drop=True)
    )
    features_by_text = (
        features_small[(features_small["title_clean"] != "") & (features_small["artist_clean"] != "")]
        .drop_duplicates(subset=["title_clean", "artist_clean"])
        .reset_index(drop=True)
    )
    features_by_first_artist = (
        features_small[(features_small["title_clean"] != "") & (features_small["artist_first_clean"] != "")]
        .drop_duplicates(subset=["title_clean", "artist_first_clean"])
        .reset_index(drop=True)
    )

    print("\nFeatures klaar:")
    print("Rijen totaal:", len(features_small))
    print("Rijen met track_id:", len(features_by_id))
    print("Rijen met title + artist:", len(features_by_text))
    print("Rijen met title + first artist:", len(features_by_first_artist))
    print("Rijen met valence:", features_small["valence"].notna().sum())
    print("Aantal genres:", features_small["track_genre"].nunique(dropna=True))

    return features_by_id, features_by_text, features_by_first_artist, candidate_audio_cols


# ============================================================
# STAP 2: CHARTS IN CHUNKS FILTEREN OP BELGIË TOP 200
# ============================================================

def inspect_charts_columns(charts_path):
    header = read_csv_flexible(charts_path, nrows=0, low_memory=False)
    normalized_cols = [normalize_column_name(c) for c in header.columns]

    date_col = pick_column(normalized_cols, ["date", "day"])
    rank_col = pick_column(normalized_cols, ["rank", "position"])
    country_col = pick_column(normalized_cols, ["country", "region"])
    title_col = pick_column(normalized_cols, ["title", "track_name", "track", "name"])
    artist_col = pick_column(normalized_cols, ["artist", "artists", "artist_name"])
    id_col = pick_column(
        normalized_cols,
        ["track_id", "spotify_id", "id", "url", "spotify_url", "track_url", "uri", "track_uri"],
        required=False,
    )
    streams_col = pick_column(normalized_cols, ["streams", "stream"], required=False)
    chart_col = pick_column(normalized_cols, ["chart", "chart_type", "chart_name"], required=False)

    return {
        "date_col": date_col,
        "rank_col": rank_col,
        "country_col": country_col,
        "title_col": title_col,
        "artist_col": artist_col,
        "id_col": id_col,
        "streams_col": streams_col,
        "chart_col": chart_col,
    }


def load_charts_filtered():
    charts_path = find_file(CHARTS_FILE)
    print("\nChartbestand gevonden:", charts_path)

    cols = inspect_charts_columns(charts_path)
    print("\nGebruikte chartkolommen:")
    for key, value in cols.items():
        print(f"{key}: {value}")

    # Datumvolgorde detecteren op een sample
    sample = read_csv_flexible(charts_path, nrows=5000, low_memory=False)
    sample.columns = [normalize_column_name(c) for c in sample.columns]
    slash_order = detect_date_order(sample[cols["date_col"]])
    print("\nGedetecteerde datumvolgorde:", slash_order)

    chunks = []
    total_read = 0
    total_kept = 0

    for i, chunk in enumerate(read_csv_flexible(charts_path, chunksize=CHUNKSIZE, low_memory=False), start=1):
        chunk.columns = [normalize_column_name(c) for c in chunk.columns]
        total_read += len(chunk)

        # Landfilter
        chunk[cols["country_col"]] = chunk[cols["country_col"]].astype(str).str.strip().str.lower()
        chunk = chunk[chunk[cols["country_col"]] == COUNTRY_TARGET].copy()

        if chunk.empty:
            print(f"Chunk {i}: gelezen {total_read:,}, behouden totaal {total_kept:,}")
            continue

        # Alleen Top 200, geen Viral 50 indien chartkolom bestaat
        if cols["chart_col"] is not None:
            chart_text = chunk[cols["chart_col"]].astype(str).str.lower()
            is_top200 = chart_text.str.contains("top", na=False) & chart_text.str.contains("200", na=False)
            is_viral = chart_text.str.contains("viral", na=False)
            chunk = chunk[is_top200 & ~is_viral].copy()

        chunk[cols["rank_col"]] = pd.to_numeric(chunk[cols["rank_col"]], errors="coerce")
        chunk = chunk[chunk[cols["rank_col"]].between(1, 200)].copy()

        if chunk.empty:
            print(f"Chunk {i}: gelezen {total_read:,}, behouden totaal {total_kept:,}")
            continue

        # Datum veilig normaliseren
        chunk["date"] = chunk[cols["date_col"]].apply(lambda x: normalize_date_string(x, slash_order=slash_order))
        chunk = chunk.dropna(subset=["date"]).copy()

        # Basisvelden
        chunk["rank"] = chunk[cols["rank_col"]]
        chunk["title"] = chunk[cols["title_col"]].astype(str)
        chunk["artist"] = chunk[cols["artist_col"]].astype(str)
        chunk["title_clean"] = chunk[cols["title_col"]].apply(clean_text)
        chunk["artist_clean"] = chunk[cols["artist_col"]].apply(clean_text)
        chunk["artist_first_clean"] = chunk[cols["artist_col"]].apply(first_artist)

        if cols["id_col"] is not None:
            chunk["track_id"] = chunk[cols["id_col"]].apply(extract_track_id)
        else:
            chunk["track_id"] = None

        if cols["streams_col"] is not None:
            chunk["streams"] = pd.to_numeric(chunk[cols["streams_col"]], errors="coerce")
        else:
            chunk["streams"] = np.nan

        keep_cols = [
            "date",
            "rank",
            "title",
            "artist",
            "title_clean",
            "artist_clean",
            "artist_first_clean",
            "track_id",
            "streams",
        ]
        chunks.append(chunk[keep_cols])
        total_kept += len(chunk)

        print(f"Chunk {i}: gelezen {total_read:,}, behouden totaal {total_kept:,}")

    if not chunks:
        raise RuntimeError("Geen Belgische Top 200-rijen gevonden. Controleer landnaam, chartkolom en rankkolom.")

    charts = pd.concat(chunks, ignore_index=True)
    charts = charts.sort_values(["date", "rank"], kind="stable").reset_index(drop=True)

    # Duplicaten verwijderen: exact dezelfde song/rank/dag maar één keer houden
    before = len(charts)
    charts = charts.drop_duplicates(subset=["date", "rank", "title_clean", "artist_clean"])
    after = len(charts)

    print("\nCharts gefilterd:")
    print("Aantal rijen:", len(charts))
    print("Duplicaten verwijderd:", before - after)
    print("Aantal dagen:", charts["date"].nunique())
    print("Datumbereik:", charts["date"].min(), "tot", charts["date"].max())

    daily_counts = charts.groupby("date").size()
    print("\nRijen per dag:")
    print(daily_counts.describe())
    print("Aantal dagen met niet exact 200 rijen:", int((daily_counts != 200).sum()))

    return charts


# ============================================================
# STAP 3: MERGEN MET FEATURES
# ============================================================

def fill_features(song_level, right_df, keys, method_name, feature_cols):
    """Vul ontbrekende audiofeatures op basis van een bepaalde merge key."""
    unmatched = song_level["valence"].isna()
    if not unmatched.any() or right_df.empty:
        return song_level

    subset = song_level.loc[unmatched, ["row_id"] + keys].copy()
    subset = subset.merge(right_df[keys + feature_cols], on=keys, how="left")
    subset = subset.drop_duplicates(subset=["row_id"])

    matched = subset["valence"].notna()
    matched_rows = subset.loc[matched, "row_id"]

    if len(matched_rows) == 0:
        print(f"Merge {method_name}: 0 extra matches")
        return song_level

    for col in feature_cols:
        values = subset.loc[matched, ["row_id", col]].set_index("row_id")[col]
        song_level.loc[values.index, col] = values

    song_level.loc[matched_rows, "merge_method"] = method_name
    print(f"Merge {method_name}: {len(matched_rows)} extra matches")
    return song_level


def merge_charts_features(charts, features_by_id, features_by_text, features_by_first_artist, audio_cols):
    feature_cols = [
        "feature_title",
        "feature_artists",
        "track_genre",
    ] + audio_cols

    song_level = charts.copy()
    song_level["row_id"] = np.arange(len(song_level))
    song_level["merge_method"] = pd.NA

    for col in feature_cols:
        song_level[col] = pd.NA

    song_level = fill_features(
        song_level=song_level,
        right_df=features_by_id,
        keys=["track_id"],
        method_name="track_id",
        feature_cols=feature_cols,
    )
    song_level = fill_features(
        song_level=song_level,
        right_df=features_by_text,
        keys=["title_clean", "artist_clean"],
        method_name="title_artist",
        feature_cols=feature_cols,
    )
    song_level = fill_features(
        song_level=song_level,
        right_df=features_by_first_artist,
        keys=["title_clean", "artist_first_clean"],
        method_name="title_first_artist",
        feature_cols=feature_cols,
    )

    for col in audio_cols:
        song_level[col] = pd.to_numeric(song_level[col], errors="coerce")

    song_level["track_genre"] = song_level["track_genre"].fillna("unknown").astype(str).str.lower().str.strip()
    song_level["artist_first"] = song_level["artist"].apply(first_artist)

    song_level = song_level.sort_values(["date", "rank"], kind="stable").reset_index(drop=True)

    total_rows = len(song_level)
    matched_rows = int(song_level["valence"].notna().sum())
    match_rate = matched_rows / total_rows * 100 if total_rows else 0

    print("\nMergecontrole:")
    print("Totaal aantal chart-rijen:", total_rows)
    print("Aantal rijen met valence:", matched_rows)
    print("Match rate:", round(match_rate, 2), "%")
    print("\nMatchmethodes:")
    print(song_level["merge_method"].value_counts(dropna=False))

    return song_level


# ============================================================
# STAP 4: DAGELIJKSE SAMENVATTING
# ============================================================

def weighted_average(group, value_col, weight_col):
    x = pd.to_numeric(group[value_col], errors="coerce")
    w = pd.to_numeric(group[weight_col], errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def weighted_share(group, indicator_col, weight_col):
    indicator = group[indicator_col]
    w = pd.to_numeric(group[weight_col], errors="coerce")
    mask = indicator.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(w[mask & indicator.astype(bool)].sum() / w[mask].sum())


def create_daily_summary(song_level, audio_cols):
    song_level = song_level.copy()
    song_level["rank"] = pd.to_numeric(song_level["rank"], errors="coerce")
    song_level["rank_weight"] = (201 - song_level["rank"]).clip(lower=1)

    has_streams = "streams" in song_level.columns and pd.to_numeric(song_level["streams"], errors="coerce").fillna(0).sum() > 0
    if has_streams:
        song_level["streams"] = pd.to_numeric(song_level["streams"], errors="coerce")
        song_level.loc[song_level["streams"] < 0, "streams"] = np.nan
        song_level["analysis_weight"] = song_level["streams"]
        weight_source = "streams"
    else:
        song_level["analysis_weight"] = song_level["rank_weight"]
        weight_source = "rank_weight"

    song_level["sad_song"] = song_level["valence"] <= SAD_THRESHOLD
    song_level["low_energy_song"] = song_level["energy"] <= LOW_ENERGY_THRESHOLD
    song_level["depressive_song"] = song_level["sad_song"] & song_level["low_energy_song"]
    song_level["top_50"] = song_level["rank"] <= 50
    song_level["top_20"] = song_level["rank"] <= 20

    # Alleen rijen met valence zijn bruikbaar voor valence-analyse
    usable = song_level.dropna(subset=["date", "valence"]).copy()

    coverage = song_level.groupby("date").agg(
        top200_rows=("rank", "count"),
        tracks_with_valence=("valence", lambda x: x.notna().sum()),
    ).reset_index()
    coverage["valence_coverage"] = coverage["tracks_with_valence"] / coverage["top200_rows"]

    rows = []
    for date_value, group in usable.groupby("date"):
        row = {
            "date": date_value,
            "avg_valence": group["valence"].mean(),
            "median_valence": group["valence"].median(),
            "std_valence": group["valence"].std(),
            "min_valence": group["valence"].min(),
            "max_valence": group["valence"].max(),
            "sad_songs_count": int(group["sad_song"].sum()),
            "share_sad_songs": group["sad_song"].mean(),
            "weighted_avg_valence": weighted_average(group, "valence", "analysis_weight"),
            "weighted_share_sad": weighted_share(group, "sad_song", "analysis_weight"),
            "weighted_share_low_energy": weighted_share(group, "low_energy_song", "analysis_weight"),
            "weighted_share_depressive": weighted_share(group, "depressive_song", "analysis_weight"),
            "analysis_weight_total": pd.to_numeric(group["analysis_weight"], errors="coerce").sum(),
            "weight_source": weight_source,
        }

        for col in ["energy", "danceability", "speechiness", "tempo", "acousticness", "instrumentalness", "liveness", "loudness", "popularity"]:
            if col in group.columns:
                row[f"avg_{col}"] = group[col].mean()
                row[f"weighted_avg_{col}"] = weighted_average(group, col, "analysis_weight")

        if has_streams:
            total_streams = pd.to_numeric(group["streams"], errors="coerce").sum()
            sad_streams = pd.to_numeric(group.loc[group["sad_song"], "streams"], errors="coerce").sum()
            row["total_streams"] = total_streams
            row["sad_streams"] = sad_streams
            row["share_sad_streams"] = sad_streams / total_streams if total_streams > 0 else np.nan
            row["main_sad_share_metric"] = row["share_sad_streams"]
            row["main_sad_share_label"] = "streamgewogen aandeel sad songs"
        else:
            row["total_streams"] = np.nan
            row["sad_streams"] = np.nan
            row["share_sad_streams"] = np.nan
            row["main_sad_share_metric"] = row["weighted_share_sad"]
            row["main_sad_share_label"] = "rankgewogen aandeel sad songs"

        rows.append(row)

    daily_metrics = pd.DataFrame(rows)
    daily = coverage.merge(daily_metrics, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)

    print("\nDaily summary klaar:")
    print("Aantal dagen:", daily["date"].nunique())
    print("Datumbereik:", daily["date"].min(), "tot", daily["date"].max())
    print("Gebruikte hoofdgewicht:", weight_source)
    print("Gebruikte sad-definitie:", f"valence <= {SAD_THRESHOLD}")

    return song_level, coverage, daily, weight_source


# ============================================================
# STAP 5: UITVOEREN
# ============================================================

def main():
    features_by_id, features_by_text, features_by_first_artist, audio_cols = load_features()
    charts = load_charts_filtered()
    song_level = merge_charts_features(charts, features_by_id, features_by_text, features_by_first_artist, audio_cols)
    song_level, daily_coverage, daily_valence, weight_source = create_daily_summary(song_level, audio_cols)

    song_level.to_csv(OUTPUT_SONG_LEVEL, index=False)
    daily_coverage.to_csv(OUTPUT_DAILY_OVERVIEW, index=False)
    daily_valence.to_csv(OUTPUT_DAILY_VALENCE, index=False)

    quality = pd.DataFrame([
        {"metric": "chart_rows_after_filter", "value": len(song_level)},
        {"metric": "unique_days", "value": song_level["date"].nunique()},
        {"metric": "rows_with_valence", "value": song_level["valence"].notna().sum()},
        {"metric": "valence_match_rate", "value": song_level["valence"].notna().mean()},
        {"metric": "weight_source", "value": weight_source},
        {"metric": "sad_threshold", "value": SAD_THRESHOLD},
        {"metric": "low_energy_threshold", "value": LOW_ENERGY_THRESHOLD},
    ])
    quality.to_csv(OUTPUT_QUALITY, index=False)

    print("\nBestanden opgeslagen:")
    print("-", OUTPUT_SONG_LEVEL)
    print("-", OUTPUT_DAILY_OVERVIEW)
    print("-", OUTPUT_DAILY_VALENCE)
    print("-", OUTPUT_QUALITY)
    print("\nKlaar. Run nu weather_API.py.")


if __name__ == "__main__":
    main()
