import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATIE – pas paden aan indien nodig
# ─────────────────────────────────────────────
FINAL_DB_PATH      = "Final database.csv"
POPULARITY_DB_PATH = "Database to calculate popularity.csv"
OUTPUT_PATH        = "merged_spotify.csv"

CHUNK_SIZE = 200_000   # rijen per chunk (verlaag naar 100_000 bij weinig RAM)

# ─────────────────────────────────────────────
# HULPFUNCTIE: extraheer Spotify track-ID
# ─────────────────────────────────────────────
def extract_track_id(uri):
    if pd.isna(uri):
        return None
    return str(uri).strip().rstrip("/").split("/")[-1].split(":")[-1]

# ─────────────────────────────────────────────
# STAP 1: Final Database laden
# ─────────────────────────────────────────────
print("📂 Final Database laden...")
final_df = pd.read_csv(FINAL_DB_PATH, low_memory=False)
final_df["track_id"] = final_df["Uri"].apply(extract_track_id)

# Detecteer de 'country' kolom in final_df
country_col_final = next(
    (c for c in final_df.columns if c.lower() == "country"), None
)
if country_col_final:
    final_df["_country_key"] = final_df[country_col_final].str.strip().str.lower()
    join_on = ["track_id", "_country_key"]
    print(f"   → Mergen op track_id + country ('{country_col_final}')")
else:
    join_on = ["track_id"]
    print("   → Geen country-kolom gevonden, mergen op track_id alleen")

print(f"   {len(final_df):,} rijen geladen.")

# ─────────────────────────────────────────────
# STAP 2: Popularity Database in chunks lezen
#          en direct aggregeren
# ─────────────────────────────────────────────
print(f"\n📂 Popularity Database lezen & aggregeren (chunks van {CHUNK_SIZE:,})...")

first_chunk = next(pd.read_csv(POPULARITY_DB_PATH, chunksize=1, low_memory=False))
all_cols = list(first_chunk.columns)

uri_col = next((c for c in all_cols if "uri" in c.lower()), None)
if uri_col is None:
    raise ValueError(f"Geen URI-kolom gevonden. Kolommen: {all_cols}")
print(f"   → URI-kolom: '{uri_col}'")

country_col_pop = next((c for c in all_cols if c.lower() == "country"), None)
if country_col_pop and country_col_final:
    group_cols = [uri_col, country_col_pop]
    print(f"   → Country-kolom: '{country_col_pop}'")
else:
    group_cols = [uri_col]
    print("   → Geen country-kolom, aggregeren op URI alleen")

skip = set(c.lower() for c in group_cols)
num_cols = [
    c for c in all_cols
    if c not in group_cols
    and c.lower() not in skip
    and first_chunk[c].dtype in ["float64", "int64", "float32", "int32"]
]
print(f"   → {len(num_cols)} numerieke kolommen worden geaggregeerd (gemiddelde)")

agg_parts = []
chunk_num = 0
for chunk in pd.read_csv(POPULARITY_DB_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_num += 1
    print(f"   chunk {chunk_num} ({len(chunk):,} rijen)...", end="\r")

    if country_col_pop:
        chunk["_country_key"] = chunk[country_col_pop].astype(str).str.strip().str.lower()
        agg_group = [uri_col, "_country_key"]
    else:
        agg_group = [uri_col]

    chunk_agg = (
        chunk[agg_group + num_cols]
        .groupby(agg_group, as_index=False)
        .mean()
    )
    agg_parts.append(chunk_agg)

print(f"\n   {chunk_num} chunks verwerkt. Deelresultaten samenvoegen...")

pop_agg = pd.concat(agg_parts, ignore_index=True)
if country_col_pop:
    pop_agg = pop_agg.groupby([uri_col, "_country_key"], as_index=False).mean()
else:
    pop_agg = pop_agg.groupby([uri_col], as_index=False).mean()

pop_agg["track_id"] = pop_agg[uri_col].apply(extract_track_id)
pop_agg.drop(columns=[uri_col], inplace=True)

print(f"   Geaggregeerde popularity: {len(pop_agg):,} unieke (song, country)-combinaties")

# ─────────────────────────────────────────────
# STAP 3: Mergen
# ─────────────────────────────────────────────
print("\n🔗 Datasets samenvoegen...")

overlap = (set(final_df.columns) & set(pop_agg.columns)) - set(join_on) - {"track_id", "_country_key"}
if overlap:
    print(f"   ⚠️  Overlappende kolommen: {overlap} → suffix '_pop' toegevoegd")

merged_df = final_df.merge(
    pop_agg,
    on=join_on,
    how="left",
    suffixes=("", "_pop")
)

for col in ["track_id", "_country_key"]:
    if col in merged_df.columns:
        merged_df.drop(columns=[col], inplace=True)

# ─────────────────────────────────────────────
# STAP 4: Opslaan
# ─────────────────────────────────────────────
print(f"💾 Opslaan naar '{OUTPUT_PATH}'...")
merged_df.to_csv(OUTPUT_PATH, index=False)

# ─────────────────────────────────────────────
# STAP 5: Samenvatting
# ─────────────────────────────────────────────
print("\n✅ Klaar!")
print(f"   Rijen Final Database:          {len(final_df):>10,}")
print(f"   Unieke (song,country) in pop.: {len(pop_agg):>10,}")
print(f"   Rijen in merged bestand:       {len(merged_df):>10,}")
print(f"   Kolommen in resultaat:         {merged_df.shape[1]:>10}")