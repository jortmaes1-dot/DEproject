import pandas as pd

# Doel: In dit script combineren we:
# 1. de Spotify dagdataset (daily_spotify_analysis.csv)
# 2. de weather dataset (weather.csv)
# We doen dit zodat we per dag in één bestand hebben:
# - total_streams
# - sad_streams
# - share_sad_streams
# - prcp (regen)
# - tsun (zonuren)
# Waarom is dat nodig?
# Omdat we uiteindelijk willen onderzoeken of regenachtige dagen
# samenhangen met meer streams van sad songs.
# Belangrijk:
# Onze Spotify dataset loopt van 2017 tot 2021,
# maar de weather dataset die jullie al hebben loopt maar over
# een kortere periode.

# Daarom moeten we de Spotify-data eerst beperken tot de dagen
# die ook echt in de weatherdata zitten.
# Anders krijg je heel veel missende waarden (NaN) bij regen en zon.
# STAP 1: BEIDE DATASETS INLADEN
# We laden:
# - de dagdataset die we in analysis.py hebben gemaakt
# - de weather dataset die al gevonden/opgeslagen is
spotify = pd.read_csv("daily_spotify_analysis.csv")
weather = pd.read_csv("weather.csv")
print("Grootte Spotify dataset:", spotify.shape)
print("Grootte weather dataset:", weather.shape)
print("\nEerste 5 rijen van Spotify:")
print(spotify.head())
print("\nEerste 5 rijen van weather:")
print(weather.head())
# STAP 2: DATUMKOLOMMEN OMZETTEN
# Om te kunnen mergen op datum, moeten beide datumkolommen
# exact hetzelfde formaat hebben.
# Daarom zetten we beide kolommen om naar datetime.
# .dt.normalize() zorgt ervoor dat de tijd op 00:00:00 wordt gezet
# en dat enkel de datum nog belangrijk is.
spotify["date"] = pd.to_datetime(spotify["date"], errors="coerce").dt.normalize()
weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
print("\nSpotify datumbereik VOOR filtering:")
print(spotify["date"].min(), "tot", spotify["date"].max())
print("\nWeather datumbereik:")
print(weather["date"].min(), "tot", weather["date"].max())
# STAP 3: SPOTIFY BEPERKEN TOT DE WEATHERPERIODE
# Dit is een heel belangrijke stap.
# Waarom?
# Onze Spotify dagdataset bevat veel meer dagen dan onze weatherdata.
# Als we nu meteen zouden mergen, dan vinden de meeste Spotify-dagen
# geen match in weather, en krijgen we overal NaN bij prcp en tsun.
# Daarom nemen we:
# - de eerste datum uit weather
# - de laatste datum uit weather
# En we houden in Spotify enkel de dagen binnen dat interval.
start_date = weather["date"].min()
end_date = weather["date"].max()
spotify = spotify[
    (spotify["date"] >= start_date) &
    (spotify["date"] <= end_date)
].copy()
print("\nSpotify datumbereik NA filtering:")
print(spotify["date"].min(), "tot", spotify["date"].max())
print("\nNieuwe grootte van Spotify na filtering:", spotify.shape)
# STAP 4: ALLEEN RELEVANTE WEATHERKOLOMMEN HOUDEN
# We hebben uit de weather dataset niet per se alle kolommen nodig.
# Voor onze onderzoeksvraag zijn vooral belangrijk:
# - prcp = precipitation = hoeveelheid regen
# - tsun = sunshine = zonuren / zonneschijn
# Daarom maken we een kleinere dataset met enkel:
# - date
# - prcp
# - tsun
weather_small = weather[["date", "prcp", "tsun"]].copy()
print("\nEerste 5 rijen van weather_small:")
print(weather_small.head())
# STAP 5: DE ECHTE MERGE DOEN
# We koppelen nu beide datasets op basis van de datum.
# We gebruiken how='left'.
# Dat betekent:
# - alle rijen uit Spotify blijven behouden
# - weather info wordt toegevoegd als de datum gevonden wordt
# Omdat we Spotify al gefilterd hebben op de weatherperiode,
# verwachten we nu veel minder of geen missende waarden.
final = pd.merge(
    spotify,
    weather_small,
    on="date",
    how="left"
)
print("\nResultaat na merge:")
print(final.head())
print("\nGrootte van de finale dataset:")
print(final.shape)
# STAP 6: CONTROLEREN OF DE MERGE GOED GELUKT IS
# We kijken hoeveel missende weather waarden er nog zijn.
# Als alles goed gaat, zou dat heel laag moeten zijn.
# Als hier toch veel missende waarden staan, dan kan dat betekenen:
# - dat sommige datums ontbreken in weather
# - of dat de datumkolom toch niet hetzelfde formaat had
print("\nAantal missende weather waarden:")
print(final[["prcp", "tsun"]].isna().sum())
# STAP 7: OPSLAAN ALS NIEUWE CSV
# We slaan het resultaat op als final_dataset.csv
# Waarom?
# Omdat dit nu onze einddataset is voor de analyse:
# per dag hebben we nu zowel Spotify-info als weather-info.
# Zo hoeven we later niet telkens opnieuw te mergen.
final.to_csv("final_dataset.csv", index=False)
print("\nBestand opgeslagen als final_dataset.csv")
# STAP 8: KORTE UITLEG OVER HET RESULTAAT
# Wat zit er nu in final_dataset.csv?
# Per dag:
# - total_streams
# - sad_streams
# - total_songs
# - sad_songs_count
# - share_sad_streams
# - share_sad_songs
# - prcp
# - tsun

# Dit is exact de dataset die we nodig hebben om te testen
# of regen samenhangt met meer sad song streams.
print("\nKlaar. final_dataset.csv is nu de dataset waarmee je de echte analyse kan doen.")