import pandas as pd
import re
from pathlib import Path

# ============================================================
# ============================================================
# MAIN.PY
# ============================================================
# DOEL:
# We maken opnieuw de Spotify-dataset, maar nu VERPLICHT met streams.
#
# Input:
# 1. Database to calculate popularity.csv
#    = dagelijkse chartdata met België, ranking, url en streams
#
# 2. Final database.csv
#    = Spotify audiofeatures met valence
#
# Output:
# 1. spotify_belgium_top200_with_valence.csv
#    = song-level dataset met date, rank, streams, track_id, valence
#
# 2. daily_valence_summary.csv
#    = dagdataset met streamgewogen valence en sad-stream-share
# ============================================================


# ============================================================
# INSTELLINGEN
# ============================================================

CHARTS_FILE = "Database to calculate popularity.csv"
FEATURES_FILE = "Final database.csv"

COUNTRY_VALUE = "Belgium"

OUTPUT_SONG_LEVEL = "spotify_belgium_top200_with_valence.csv"
OUTPUT_DAILY_LEVEL = "daily_valence_summary.csv"

CHUNKSIZE = 500_000
SAD_THRESHOLD = 0.35


# ============================================================
# HULPFUNCTIES
# ============================================================

def normalize_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col


def pick_column(columns, possible_names, required=True):
    columns = list(columns)
    possible_names = [normalize_column_name(x) for x in possible_names]

    # Exacte match
    for name in possible_names:
        if name in columns:
            return name

    # Gedeeltelijke match
    for col in columns:
        for name in possible_names:
            if name in col:
                return col

    if required:
        raise ValueError(
            f"Kon geen passende kolom vinden voor: {possible_names}\n\n"
            f"Beschikbare kolommen:\n{columns}"
        )

    return None


def extract_spotify_track_id(value):
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


def clean_streams(series):
    """
    Maakt streams numeriek.
    Werkt ook als streams geschreven zijn als:
    - 123456
    - 123,456
    - "123.456"
    """
    return (
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace(["nan", "None", ""], pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )


def read_sample(file_path, nrows=20):
    sample = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    sample.columns = [normalize_column_name(c) for c in sample.columns]
    return sample


def weighted_average(values, weights):
    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return None

    return (values[valid] * weights[valid]).sum() / weights[valid].sum()


# ============================================================
# STAP 1: BESTANDEN CONTROLEREN
# ============================================================

if not Path(CHARTS_FILE).exists():
    raise FileNotFoundError(
        f"Bestand niet gevonden: {CHARTS_FILE}\n"
        f"Zorg dat dit bestand in dezelfde map staat als main.py."
    )

if not Path(FEATURES_FILE).exists():
    raise FileNotFoundError(
        f"Bestand niet gevonden: {FEATURES_FILE}\n"
        f"Zorg dat dit bestand in dezelfde map staat als main.py."
    )

print("Bestanden gevonden:")
print(f"- {CHARTS_FILE}")
print(f"- {FEATURES_FILE}")


# ============================================================
# STAP 2: AUDIOFEATURES INLADEN
# ============================================================

print("\nAudiofeatures-bestand wordt ingeladen...")

features = pd.read_csv(FEATURES_FILE, low_memory=False)
features.columns = [normalize_column_name(c) for c in features.columns]

print("\nKolommen in features-bestand:")
print(features.columns.tolist())

valence_col = pick_column(features.columns, ["valence"])

feature_id_col = pick_column(
    features.columns,
    [
        "url",
        "track_url",
        "spotify_url",
        "song_url",
        "uri",
        "track_uri",
        "track_id",
        "id",
        "spotify_id"
    ]
)

features["track_id"] = features[feature_id_col].apply(extract_spotify_track_id)

possible_audio_cols = [
    "valence",
    "danceability",
    "energy",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness"
]

available_audio_cols = [col for col in possible_audio_cols if col in features.columns]

features_small = features[["track_id"] + available_audio_cols].copy()
features_small = features_small.dropna(subset=["track_id"]).copy()
features_small = features_small.drop_duplicates(subset=["track_id"])

for col in available_audio_cols:
    features_small[col] = pd.to_numeric(features_small[col], errors="coerce")

print("\nFeatures klaar.")
print("Aantal unieke tracks met track_id:", len(features_small))
print("Aantal tracks met valence:", features_small["valence"].notna().sum())


# ============================================================
# STAP 3: CHARTBESTAND INSPECTEREN
# ============================================================

print("\nChartbestand wordt geïnspecteerd...")

charts_sample = read_sample(CHARTS_FILE, nrows=50)

print("\nKolommen in chartbestand:")
print(charts_sample.columns.tolist())

date_col = pick_column(charts_sample.columns, ["date", "day"])
rank_col = pick_column(charts_sample.columns, ["rank", "position"])
country_col = pick_column(charts_sample.columns, ["country", "region"])

chart_col = pick_column(charts_sample.columns, ["chart"], required=False)
title_col = pick_column(charts_sample.columns, ["title", "track_name", "name"], required=False)
artist_col = pick_column(charts_sample.columns, ["artist", "artists"], required=False)
trend_col = pick_column(charts_sample.columns, ["trend"], required=False)

# BELANGRIJK:
# streams is nu required=True.
# Als je chartbestand geen streams bevat, stopt de code hier.
streams_col = pick_column(
    charts_sample.columns,
    [
        "streams",
        "stream",
        "number_of_streams",
        "stream_count",
        "daily_streams",
        "total_streams"
    ],
    required=True
)

charts_id_col = pick_column(
    charts_sample.columns,
    [
        "url",
        "track_url",
        "spotify_url",
        "song_url",
        "uri",
        "track_uri",
        "track_id",
        "id",
        "spotify_id"
    ]
)

print("\nGevonden kolommen in chartbestand:")
print("Datumkolom:", date_col)
print("Rankkolom:", rank_col)
print("Land/regio-kolom:", country_col)
print("Chartkolom:", chart_col)
print("Streams-kolom:", streams_col)
print("Titelkolom:", title_col)
print("Artiestkolom:", artist_col)
print("Trendkolom:", trend_col)
print("Track-ID/URL kolom:", charts_id_col)


# ============================================================
# STAP 4: CHARTBESTAND FILTEREN
# ============================================================

filtered_chunks = []

print("\nStart met verwerken van chartdata...")

for i, chunk in enumerate(pd.read_csv(CHARTS_FILE, chunksize=CHUNKSIZE, low_memory=False), start=1):
    chunk.columns = [normalize_column_name(c) for c in chunk.columns]

    original_rows = len(chunk)

    # België filteren
    chunk[country_col] = chunk[country_col].astype(str).str.strip()
    chunk = chunk[chunk[country_col].str.lower() == COUNTRY_VALUE.lower()].copy()

    if chunk.empty:
        print(f"Chunk {i}: 0 Belgische rijen.")
        continue

    belgium_rows = len(chunk)

    # Top 200 filteren
    chunk[rank_col] = pd.to_numeric(chunk[rank_col], errors="coerce")
    chunk = chunk[chunk[rank_col].between(1, 200)].copy()

    if chunk.empty:
        print(f"Chunk {i}: Belgische rijen gevonden, maar geen rank 1-200.")
        continue

    top200_rows = len(chunk)

    # Viral 50 verwijderen als chartkolom bestaat
    if chart_col is not None:
        chart_text = chunk[chart_col].astype(str).str.lower()

        chunk = chunk[
            chart_text.str.contains("top", na=False)
            & ~chart_text.str.contains("viral", na=False)
        ].copy()

    if chunk.empty:
        print(f"Chunk {i}: na verwijderen van Viral 50 geen rijen meer.")
        continue

    # Datum
    chunk["date"] = pd.to_datetime(chunk[date_col], errors="coerce")
    chunk = chunk.dropna(subset=["date"]).copy()
    chunk["date"] = chunk["date"].dt.normalize()

    # Track ID
    chunk["track_id"] = chunk[charts_id_col].apply(extract_spotify_track_id)

    # Streams numeriek maken
    chunk["streams"] = clean_streams(chunk[streams_col])

    # Rijen zonder streams verwijderen
    before_stream_filter = len(chunk)
    chunk = chunk.dropna(subset=["streams"]).copy()
    chunk = chunk[chunk["streams"] >= 0].copy()
    after_stream_filter = len(chunk)

    if chunk.empty:
        print(f"Chunk {i}: geen rijen met geldige streams.")
        continue

    # Outputchunk
    out = pd.DataFrame()
    out["date"] = chunk["date"]
    out["country"] = chunk[country_col]
    out["rank"] = chunk[rank_col]
    out["streams"] = chunk["streams"]
    out["track_id"] = chunk["track_id"]

    if chart_col is not None:
        out["chart"] = chunk[chart_col]

    if title_col is not None:
        out["title"] = chunk[title_col]

    if artist_col is not None:
        out["artist"] = chunk[artist_col]

    if trend_col is not None:
        out["trend"] = chunk[trend_col]

    filtered_chunks.append(out)

    print(
        f"Chunk {i}: "
        f"{original_rows} origineel | "
        f"{belgium_rows} België | "
        f"{top200_rows} rank 1-200 | "
        f"{before_stream_filter} vóór streamfilter | "
        f"{after_stream_filter} met geldige streams | "
        f"{len(out)} behouden"
    )


# ============================================================
# STAP 5: CHUNKS SAMENVOEGEN
# ============================================================

if not filtered_chunks:
    raise ValueError(
        "Er zijn geen Belgische Top 200 rijen met streams gevonden.\n"
        "Controleer of je het juiste chartbestand gebruikt."
    )

charts_be = pd.concat(filtered_chunks, ignore_index=True)

print("\nChartdata gefilterd.")
print("Aantal Belgische Top 200 rijen met streams:", len(charts_be))


# ============================================================
# STAP 6: CONTROLES VOOR MERGE
# ============================================================

print("\nDatumbereik chartdata:")
print(charts_be["date"].min(), "tot", charts_be["date"].max())

print("\nAantal unieke dagen:")
print(charts_be["date"].nunique())

print("\nAantal rijen zonder track_id:")
print(charts_be["track_id"].isna().sum())

print("\nStreamscontrole:")
print(charts_be["streams"].describe())

daily_row_counts = charts_be.groupby("date").size()

print("\nAantal rijen per dag:")
print(daily_row_counts.describe())

print("\nEerste dagen met minder dan 200 rijen:")
print(daily_row_counts[daily_row_counts < 200].head(20))

print("\nEerste dagen met meer dan 200 rijen:")
print(daily_row_counts[daily_row_counts > 200].head(20))


# ============================================================
# STAP 7: MERGEN MET AUDIOFEATURES
# ============================================================

print("\nStart merge met valence/audiofeatures...")

merged = charts_be.merge(
    features_small,
    on="track_id",
    how="left"
)

print("Merge klaar.")

total_rows = len(merged)
rows_with_valence = merged["valence"].notna().sum()
match_percentage = rows_with_valence / total_rows * 100

print("\nMergecontrole:")
print("Totaal aantal rijen:", total_rows)
print("Aantal rijen met valence:", rows_with_valence)
print("Matchpercentage:", round(match_percentage, 2), "%")


# ============================================================
# STAP 8: EXTRA CONTROLES
# ============================================================

daily_coverage = merged.groupby("date").agg(
    top200_rows=("track_id", "count"),
    unique_tracks=("track_id", "nunique"),
    total_streams=("streams", "sum"),
    tracks_with_valence=("valence", lambda x: x.notna().sum())
).reset_index()

daily_coverage["valence_coverage"] = (
    daily_coverage["tracks_with_valence"] / daily_coverage["top200_rows"]
)

print("\nValence coverage per dag:")
print(daily_coverage["valence_coverage"].describe())

print("\nGemiddelde valence coverage:")
print(round(daily_coverage["valence_coverage"].mean() * 100, 2), "%")

print("\nDagelijkse total_streams controle:")
print(daily_coverage["total_streams"].describe())


# ============================================================
# STAP 9: DAGELIJKSE STREAMGEWOGEN SAMENVATTING
# ============================================================

print("\nDagelijkse samenvatting wordt gemaakt...")

merged["sad_song"] = merged["valence"] < SAD_THRESHOLD

def make_daily_summary(group):
    valid = group[group["valence"].notna()].copy()

    top200_rows = len(group)
    tracks_with_valence = len(valid)
    valence_coverage = tracks_with_valence / top200_rows if top200_rows > 0 else None

    total_streams_all = group["streams"].sum(skipna=True)
    total_streams_matched = valid["streams"].sum(skipna=True)

    sad_streams = valid.loc[valid["sad_song"], "streams"].sum(skipna=True)

    if total_streams_matched > 0:
        share_sad_streams_matched = sad_streams / total_streams_matched
        weighted_avg_valence = weighted_average(valid["valence"], valid["streams"])
    else:
        share_sad_streams_matched = None
        weighted_avg_valence = None

    if total_streams_all > 0:
        share_sad_streams_all = sad_streams / total_streams_all
    else:
        share_sad_streams_all = None

    return pd.Series({
        "top200_rows": top200_rows,
        "unique_tracks": group["track_id"].nunique(),
        "tracks_with_valence": tracks_with_valence,
        "valence_coverage": valence_coverage,

        "total_streams_all": total_streams_all,
        "total_streams_matched": total_streams_matched,

        "avg_valence": valid["valence"].mean(),
        "weighted_avg_valence": weighted_avg_valence,

        "sad_songs_count": valid["sad_song"].sum(),
        "share_sad_songs_matched": valid["sad_song"].mean(),

        "sad_streams": sad_streams,
        "share_sad_streams_matched": share_sad_streams_matched,
        "share_sad_streams_all": share_sad_streams_all
    })

daily_summary = merged.groupby("date").apply(make_daily_summary).reset_index()

print("\nEerste 10 rijen daily summary:")
print(daily_summary.head(10).to_string(index=False))


# ============================================================
# STAP 10: OPSLAAN
# ============================================================

merged.to_csv(OUTPUT_SONG_LEVEL, index=False)
daily_summary.to_csv(OUTPUT_DAILY_LEVEL, index=False)

print("\nBestanden opgeslagen:")
print(f"- {OUTPUT_SONG_LEVEL}")
print(f"- {OUTPUT_DAILY_LEVEL}")

print("\nControle: kolommen in song-level output:")
print(merged.columns.tolist())

print("\nControle: zit streams erin?")
print("streams" in merged.columns)

print("\nKlaar.")