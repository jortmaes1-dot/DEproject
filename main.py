import pandas as pd

# bestanden inladen
charts = pd.read_csv("charts.csv")
tracks = pd.read_csv("spotify_tracks.csv")

# chat zegt: deze datasets hebben geen gemeenschappelijke ID (zoals track_id),
# dus koppelen via artist en titel

# kolommen proper maken voor koppeling
# alles lowercase + spaties weg zodat matchen beter lukt
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
# anders kan 1 song meerdere keren gematcht worden en krijg je foute resultaten
tracks_small = tracks_small.drop_duplicates(subset=["title_clean", "artist_clean"])

# datasets koppelen
# how="left" betekent: alle rijen uit charts behouden
# en alleen info uit tracks toevoegen als er een match is
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

# alleen rijen houden waarvoor er echt een match is
# valence is enkel ingevuld als de song in tracks gevonden is
merged_clean = merged.dropna(subset=["valence"]).copy()

# dataset kleiner maken: alleen België houden
# zo wordt de analyse veel haalbaarder
merged_clean = merged_clean[merged_clean["region"] == "Belgium"].copy()

# nieuwe grootte tonen
print("Aantal rijen na filtering op matches + Belgium:", len(merged_clean))
print(merged_clean.head())

# opslaan als nieuwe csv zodat je hier later verder mee kan werken
merged_clean.to_csv("merged_clean.csv", index=False)
print("Bestand opgeslagen als merged_clean.csv")