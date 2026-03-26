
import pandas as pd

# bestanden inladen
charts = pd.read_csv("charts.csv")
tracks = pd.read_csv("spotify_tracks.csv")

# kolommen proper maken voor koppeling
charts["title_clean"] = charts["title"].str.lower().str.strip()
charts["artist_clean"] = charts["artist"].str.lower().str.strip()

tracks["title_clean"] = tracks["track_name"].str.lower().str.strip()
tracks["artist_clean"] = tracks["artists"].str.lower().str.strip()

# alleen nuttige kolommen uit tracks houden
tracks_small = tracks[
    [
        "title_clean",
        "artist_clean",
        "track_name",
        "artists",
        "popularity",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "track_genre"
    ]
].copy()

# dubbele combinaties verwijderen
tracks_small = tracks_small.drop_duplicates(subset=["title_clean", "artist_clean"])

# datasets koppelen
merged = pd.merge(
    charts,
    tracks_small,
    on=["title_clean", "artist_clean"],
    how="left"
)

# resultaat tonen
print(merged.head())
print(merged.shape)

# hoeveel matches?
matched = merged["track_name"].notna().sum()
total = len(merged)

print("Aantal gematchte rijen:", matched)
print("Totaal aantal rijen:", total)
print("Match percentage:", round((matched / total) * 100, 2), "%")