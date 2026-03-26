import pandas as pd
# Doel van dit script:
# We vertrekken van de al opgeschoonde en gemergde dataset:
#   merged_clean.csv, zie uitleg messenger
# In die dataset zit per rij een liedje dat op een bepaalde dag
# in de Belgische Spotify charts stond, plus extra kenmerken
# van dat liedje zoals:
# - valence
# - energy
# - danceability
# - tempo
# Voor onze onderzoeksvraag willen we NIET op songniveau werken, maar op DAGNIVEAU.
# Omdat ons weerbestand ook per dag is opgebouwd.
# Bijvoorbeeld:
# - 2020-01-01 -> hoeveel regen viel er?
# - 2020-01-02 -> hoeveel regen viel er?
# Daarom zetten we de songdata eerst om naar een dagdataset.
# We willen per dag weten:
# - hoeveel totale streams er waren
# - hoeveel streams van "sad songs" er waren
# - welk aandeel van alle streams naar sad songs ging
# Daarna kunnen we die dagdataset later koppelen aan weather data.
# STAP 1: DATA INLADEN

# We lezen de dataset in die we in het vorige script hebben gemaakt.
# Deze file bevat al: (deze noemt merged_clean.csv)
# - alleen bruikbare matches
# - alleen België
# - audiofeatures zoals valence
df = pd.read_csv("merged_clean.csv")


# We tonen hoeveel rijen en kolommen de dataset heeft.
print("Grootte van de ingeladen dataset:", df.shape)

# We tonen ook de eerste 5 rijen om te zien of alles logisch oogt.
print("\nEerste 5 rijen van merged_clean.csv:")
print(df.head())


# ============================================================
# STAP 2: KOLOMMEN CONTROLEREN EN TYPE GOED ZETTEN
# ============================================================
# Voor de analyse hebben we zeker deze kolommen nodig:
# - date
# - streams
# - valence
#
# Soms worden kolommen ingelezen als tekst, terwijl we ze als
# getal of datum nodig hebben. Daarom zetten we ze expliciet om.
# ============================================================

# datumkolom omzetten naar echte datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# streams omzetten naar numerieke waarden
# errors="coerce" betekent:
# als een waarde niet kan omgezet worden, wordt ze NaN
df["streams"] = pd.to_numeric(df["streams"], errors="coerce")

# valence ook zeker numeriek maken
df["valence"] = pd.to_numeric(df["valence"], errors="coerce")

# Controle: hoeveel missende waarden zijn er?
print("\nAantal missende waarden per belangrijke kolom:")
print(df[["date", "streams", "valence"]].isna().sum())


# ============================================================
# STAP 3: RIJEN MET ONBRUIKBARE WAARDEN VERWIJDEREN
# ============================================================
# Voor onze analyse hebben we per rij zeker nodig:
# - een datum
# - een streamsaantal
# - een valence-score
#
# Als één van die drie ontbreekt, kunnen we die rij niet gebruiken.
# ============================================================

df = df.dropna(subset=["date", "streams", "valence"]).copy()

print("\nGrootte na verwijderen van rijen met missende date/streams/valence:")
print(df.shape)


# ============================================================
# STAP 4: DEFINIËREN WAT EEN 'SAD SONG' IS
# ============================================================
# We moeten ons concept "sad songs" meetbaar maken.
#
# In Spotify-audiofeatures is valence een maat voor hoe positief
# of vrolijk een song klinkt:
# - lage valence = eerder droevig / donker / negatief
# - hoge valence = eerder vrolijk / positief
#
# We kiezen hier:
#   sad_song = True als valence < 0.35
#
# Dat is een duidelijke en verdedigbare grens.
# ============================================================

df["sad_song"] = df["valence"] < 0.35

# Even controleren hoeveel songs als sad geclassificeerd worden
print("\nAantal songs die als sad zijn geclassificeerd:")
print(df["sad_song"].value_counts())


# ============================================================
# STAP 5: CONTROLEREN OF STREAMS BRUIKBAAR ZIJN
# ============================================================
# Streams zijn de basis van onze analyse.
# We willen weten of er geen negatieve of vreemde waarden in zitten.
# ============================================================

print("\nSamenvatting van de streams-kolom:")
print(df["streams"].describe())

# Eventueel kunnen we negatieve waarden eruit halen, als die er zouden zijn.
df = df[df["streams"] >= 0].copy()

print("\nGrootte na eventuele verwijdering van negatieve streams:")
print(df.shape)


# ============================================================
# STAP 6: VAN SONGNIVEAU NAAR DAGNIVEAU
# ============================================================
# Dit is de belangrijkste stap van het script.
#
# Momenteel is elke rij:
#   1 song op 1 dag
#
# Maar wij willen uiteindelijk:
#   1 rij per dag
#
# Daarom groeperen we op 'date'.
#
# Per dag berekenen we:
# - total_streams:
#     som van alle streams op die dag
# - sad_streams:
#     som van streams van songs die als sad zijn geclassificeerd
# - total_songs:
#     hoeveel songs er die dag in onze dataset zitten
# - sad_songs_count:
#     hoeveel daarvan sad songs zijn
# ============================================================

daily = df.groupby("date").apply(
    lambda x: pd.Series({
        "total_streams": x["streams"].sum(),
        "sad_streams": x.loc[x["sad_song"], "streams"].sum(),
        "total_songs": len(x),
        "sad_songs_count": x["sad_song"].sum()
    })
).reset_index()

# Opmerking:
# x["sad_song"].sum() werkt hier omdat True telt als 1 en False als 0.


# ============================================================
# STAP 7: EXTRA VARIABELEN MAKEN OP DAGNIVEAU
# ============================================================
# Met alleen total_streams en sad_streams zijn we er nog niet.
# We willen ook relatieve maten, omdat die beter vergelijkbaar zijn.
#
# Daarom maken we:
# - share_sad_streams:
#     welk aandeel van alle streams naar sad songs ging
#
# - share_sad_songs:
#     welk aandeel van de songs in de charts sad songs waren
#
# Deze verhoudingen zijn vaak interessanter dan absolute aantallen.
# ============================================================

daily["share_sad_streams"] = daily["sad_streams"] / daily["total_streams"]
daily["share_sad_songs"] = daily["sad_songs_count"] / daily["total_songs"]


# ============================================================
# STAP 8: RESULTAAT CONTROLEREN
# ============================================================
# Nu hebben we onze eerste echte analysedataset.
# Elke rij stelt nu 1 dag voor.
# ============================================================

print("\nEerste 10 rijen van de dagdataset:")
print(daily.head(10))

print("\nGrootte van de dagdataset:")
print(daily.shape)

print("\nSamenvatting van de belangrijkste variabelen:")
print(daily[[
    "total_streams",
    "sad_streams",
    "share_sad_streams",
    "total_songs",
    "sad_songs_count",
    "share_sad_songs"
]].describe())


# ============================================================
# STAP 9: OPSLAAN ALS NIEUWE CSV
# ============================================================
# We slaan de dagdataset op zodat we in een volgende stap
# heel makkelijk weather data kunnen toevoegen.
#
# Belangrijk:
# We slaan dus NIET de songdataset opnieuw op,
# maar de geaggregeerde dagdataset.
# ============================================================

daily.to_csv("daily_spotify_analysis.csv", index=False)

print("\nBestand opgeslagen als daily_spotify_analysis.csv")


# ============================================================
# STAP 10: UITLEG OVER DE VOLGENDE STAP
# ============================================================
# In een volgend script kunnen we weather data toevoegen.
#
# Dan hebben we per dag bijvoorbeeld:
# - date
# - total_streams
# - sad_streams
# - share_sad_streams
# - prcp (neerslag)
# - tsun (zonuren)
#
# Vanaf dan kunnen we echt testen:
# "Gaan regenachtige dagen samen met meer streams van sad songs?"
# ============================================================

print("\nKlaar. De volgende stap is weather data koppelen aan daily_spotify_analysis.csv")