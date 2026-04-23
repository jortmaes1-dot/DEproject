import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FINAL_ANALYSIS.PY
# ============================================================
# DOEL VAN DIT SCRIPT
# ------------------------------------------------------------
# Dit script is een eerste verkennende analyse van onze finale dataset.
#
# We werken met final_dataset.csv.
# In dat bestand hebben we per dag:
# - Spotify variabelen:
#   total_streams
#   sad_streams
#   share_sad_streams
#   total_songs
#   sad_songs_count
#   share_sad_songs
#
# - Weather variabelen:
#   prcp  = neerslag / regen
#   tsun  = zonuren
#
# Wat willen we hier doen?
# 1. Kijken of de data goed is ingeladen
# 2. Eenvoudige variabelen maken zoals:
#    - rainy_day = dag met regen of niet
# 3. Vergelijken:
#    - is het gemiddelde aandeel sad streams hoger op regendagen?
# 4. Correlaties bekijken
# 5. Een paar eerste grafieken maken
#
# Dit is dus een eerste analyse, geen definitieve conclusie.
# ============================================================


# ============================================================
# STAP 1: DATA INLADEN
# ============================================================

df = pd.read_csv("final_dataset.csv")

print("Grootte van de dataset:", df.shape)

print("\nEerste 5 rijen:")
print(df.head())


# ============================================================
# STAP 2: KOLOMMEN OMZETTEN NAAR JUISTE TYPES
# ============================================================
# We zorgen dat:
# - date een datum is
# - numerieke kolommen echt numeriek zijn
# ============================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

numeric_cols = [
    "total_streams",
    "sad_streams",
    "total_songs",
    "sad_songs_count",
    "share_sad_streams",
    "share_sad_songs",
    "prcp",
    "tsun"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nAantal missende waarden per belangrijke kolom:")
print(df[numeric_cols + ["date"]].isna().sum())


# ============================================================
# STAP 3: RIJEN MET MISSENDE BELANGRIJKE WAARDEN VERWIJDEREN
# ============================================================
# Voor deze eerste analyse hebben we minstens nodig:
# - date
# - share_sad_streams
# - prcp
#
# Zonder die kolommen kunnen we regen en sad streams niet vergelijken.
# ============================================================

df = df.dropna(subset=["date", "share_sad_streams", "prcp"]).copy()

print("\nGrootte na verwijderen van missende waarden:")
print(df.shape)


# ============================================================
# STAP 4: REGENVARIABELEN MAKEN
# ============================================================
# We maken eenvoudige indicatoren om regen makkelijker te analyseren.
#
# rainy_day:
# - True als er regen viel
# - False als er geen regen viel
#
# heavy_rain:
# - True als prcp > 5 mm
# - False anders
#
# Waarom?
# Omdat we zowel een eenvoudige binaire vergelijking willen doen
# (regen vs geen regen)
# als eventueel een zwaardere vorm van regen willen bekijken.
# ============================================================

df["rainy_day"] = df["prcp"] > 0
df["heavy_rain"] = df["prcp"] > 5

print("\nAantal regendagen vs droge dagen:")
print(df["rainy_day"].value_counts())

print("\nAantal zware regendagen vs andere dagen:")
print(df["heavy_rain"].value_counts())


# ============================================================
# STAP 5: BESCHRIJVENDE STATISTIEKEN
# ============================================================
# Eerst bekijken we de algemene verdeling van de belangrijkste variabelen.
# ============================================================

print("\nBeschrijvende statistieken:")
print(df[[
    "share_sad_streams",
    "share_sad_songs",
    "prcp",
    "tsun",
    "sad_streams",
    "total_streams"
]].describe())


# ============================================================
# STAP 6: GEMIDDELDEN VERGELIJKEN
# ============================================================
# We vergelijken het gemiddelde aandeel sad streams op:
# - regendagen
# - droge dagen
#
# Dit is een heel goede eerste stap voor de onderzoeksvraag.
# ============================================================

group_rain = df.groupby("rainy_day")[[
    "share_sad_streams",
    "share_sad_songs",
    "sad_streams",
    "total_streams",
    "prcp",
    "tsun"
]].mean()

print("\nGemiddelden op regendagen vs droge dagen:")
print(group_rain)

# Ook het aantal dagen per groep is nuttig
print("\nAantal dagen per groep:")
print(df.groupby("rainy_day").size())


# ============================================================
# STAP 7: CORRELATIES BEKIJKEN
# ============================================================
# Correlatie geeft een eerste idee van samenhang tussen variabelen.
#
# Belangrijk:
# Correlatie bewijst geen causaliteit.
# Het is enkel een eerste verkenning.
# ============================================================

corr_cols = ["share_sad_streams", "share_sad_songs", "prcp", "tsun"]
corr_matrix = df[corr_cols].corr()

print("\nCorrelatiematrix:")
print(corr_matrix)


# ============================================================
# STAP 8: EENVOUDIGE GRAFIEKEN MAKEN
# ============================================================
# We maken enkele grafieken om patronen visueel te bekijken.
# ============================================================

# Grafiek 1: tijdsreeks van aandeel sad streams
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["share_sad_streams"])
plt.title("Aandeel sad streams doorheen de tijd")
plt.xlabel("Datum")
plt.ylabel("Share sad streams")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafiek_share_sad_streams_tijd.png")
plt.show()

# Grafiek 2: scatterplot van regen vs aandeel sad streams
plt.figure(figsize=(8, 5))
plt.scatter(df["prcp"], df["share_sad_streams"])
plt.title("Regen (prcp) vs aandeel sad streams")
plt.xlabel("Neerslag (mm)")
plt.ylabel("Share sad streams")
plt.tight_layout()
plt.savefig("grafiek_regen_vs_sad.png")
plt.show()

# Grafiek 3: boxplot-achtige vergelijking via pandas
plt.figure(figsize=(8, 5))
df.boxplot(column="share_sad_streams", by="rainy_day")
plt.title("Aandeel sad streams op regendagen vs droge dagen")
plt.suptitle("")  # standaard subtitel van pandas weghalen
plt.xlabel("Rainy day")
plt.ylabel("Share sad streams")
plt.tight_layout()
plt.savefig("grafiek_boxplot_rainy_day.png")
plt.show()


# ============================================================
# STAP 9: EENVOUDIGE INTERPRETATIE PRINTEN
# ============================================================
# We berekenen het verschil in gemiddelde share_sad_streams
# tussen regendagen en droge dagen.
# ============================================================

mean_rainy = df.loc[df["rainy_day"], "share_sad_streams"].mean()
mean_dry = df.loc[~df["rainy_day"], "share_sad_streams"].mean()
difference = mean_rainy - mean_dry

print("\nGemiddelde share_sad_streams op regendagen:", round(mean_rainy, 4))
print("Gemiddelde share_sad_streams op droge dagen:", round(mean_dry, 4))
print("Verschil (regen - droog):", round(difference, 4))

if difference > 0:
    print("\nEerste indicatie: het aandeel sad streams ligt gemiddeld HOGER op regendagen.")
elif difference < 0:
    print("\nEerste indicatie: het aandeel sad streams ligt gemiddeld LAGER op regendagen.")
else:
    print("\nEerste indicatie: er lijkt geen verschil te zijn tussen regendagen en droge dagen.")


# ============================================================
# STAP 10: RESULTATEN OPSLAAN
# ============================================================
# We slaan ook een kleine samenvatting op als csv.
# Dat is handig voor rapportering of groepswerk.
# ============================================================

summary_table = df.groupby("rainy_day").agg(
    aantal_dagen=("date", "count"),
    gemiddelde_share_sad_streams=("share_sad_streams", "mean"),
    gemiddelde_share_sad_songs=("share_sad_songs", "mean"),
    gemiddelde_prcp=("prcp", "mean"),
    gemiddelde_tsun=("tsun", "mean")
).reset_index()

summary_table.to_csv("summary_rain_vs_no_rain.csv", index=False)

print("\nSamenvatting opgeslagen als summary_rain_vs_no_rain.csv")
print("\nKlaar. Dit is je eerste verkennende analyse.")