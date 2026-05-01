from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


SONG_FILE = "merged_clean.csv"
WEATHER_FILE = "weather.csv"

OUTPUT_DAILY_FILE = "spark_daily_spotify_analysis.csv"
OUTPUT_FINAL_FILE = "spark_final_dataset.csv"
OUTPUT_RAIN_SUMMARY_FILE = "spark_rain_group_summary.csv"
OUTPUT_CORRELATION_FILE = "spark_correlation_summary.csv"

SAD_THRESHOLD = 0.40
HEAVY_RAIN_THRESHOLD = 5.0


def require_file(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {path_str}")


def build_spark_session():
    return (
        SparkSession.builder
        .appName("spotify-weather-correlation")
        .master("local[*]")
        .getOrCreate()
    )


def parse_date(column_name):
    return F.to_date(F.substring(F.col(column_name).cast("string"), 1, 10), "yyyy-MM-dd")


def build_daily_spotify_metrics(spark):
    songs = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(SONG_FILE)
        .select(
            parse_date("date").alias("date"),
            F.col("streams").cast("double").alias("streams"),
            F.col("valence").cast("double").alias("valence")
        )
        .filter(F.col("date").isNotNull())
        .filter(F.col("streams").isNotNull())
        .filter(F.col("streams") >= 0)
    )

    daily = (
        songs.groupBy("date")
        .agg(
            F.count("*").alias("song_rows"),
            F.sum("streams").alias("total_streams"),
            F.sum(F.when(F.col("valence").isNotNull(), 1).otherwise(0)).alias("tracks_with_valence"),
            F.sum(
                F.when(F.col("valence").isNotNull(), F.col("streams")).otherwise(F.lit(0.0))
            ).alias("streams_with_valence"),
            F.sum(
                F.when(F.col("valence") < SAD_THRESHOLD, F.col("streams")).otherwise(F.lit(0.0))
            ).alias("sad_streams"),
            F.sum(
                F.when(F.col("valence") < SAD_THRESHOLD, 1).otherwise(0)
            ).alias("sad_songs_count"),
            F.avg("valence").alias("avg_valence"),
            F.expr("percentile_approx(valence, 0.5)").alias("median_valence"),
            F.stddev("valence").alias("std_valence"),
            F.min("valence").alias("min_valence"),
            F.max("valence").alias("max_valence"),
            F.sum(
                F.when(
                    F.col("valence").isNotNull(),
                    F.col("valence") * F.col("streams")
                ).otherwise(F.lit(0.0))
            ).alias("weighted_valence_numerator")
        )
        .withColumn(
            "valence_coverage",
            F.when(F.col("song_rows") > 0, F.col("tracks_with_valence") / F.col("song_rows"))
        )
        .withColumn(
            "stream_coverage",
            F.when(F.col("total_streams") > 0, F.col("streams_with_valence") / F.col("total_streams"))
        )
        .withColumn(
            "share_sad_streams",
            F.when(F.col("streams_with_valence") > 0, F.col("sad_streams") / F.col("streams_with_valence"))
        )
        .withColumn(
            "share_sad_songs",
            F.when(F.col("tracks_with_valence") > 0, F.col("sad_songs_count") / F.col("tracks_with_valence"))
        )
        .withColumn(
            "weighted_avg_valence",
            F.when(
                F.col("streams_with_valence") > 0,
                F.col("weighted_valence_numerator") / F.col("streams_with_valence")
            )
        )
        .drop("weighted_valence_numerator")
        .orderBy("date")
    )

    return daily


def load_weather(spark):
    weather = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(WEATHER_FILE)
        .select(
            parse_date("date").alias("date"),
            F.col("prcp").cast("double").alias("prcp"),
            F.col("tsun").cast("double").alias("tsun")
        )
        .filter(F.col("date").isNotNull())
        .dropDuplicates(["date"])
    )

    return weather


def build_final_dataset(daily, weather):
    return (
        daily.join(weather, on="date", how="left")
        .withColumn("rainy_day", F.when(F.col("prcp").isNull(), F.lit(None)).otherwise(F.col("prcp") > 0))
        .withColumn(
            "heavy_rain",
            F.when(F.col("prcp").isNull(), F.lit(None)).otherwise(F.col("prcp") > HEAVY_RAIN_THRESHOLD)
        )
        .orderBy("date")
    )


def build_correlation_summary(df):
    correlation_pairs = [
        ("share_sad_streams", "prcp"),
        ("share_sad_streams", "tsun"),
        ("share_sad_songs", "prcp"),
        ("share_sad_songs", "tsun"),
        ("weighted_avg_valence", "prcp"),
        ("weighted_avg_valence", "tsun"),
        ("sad_streams", "prcp"),
        ("sad_streams", "tsun")
    ]

    rows = []

    for left_col, right_col in correlation_pairs:
        subset = df.select(left_col, right_col).dropna()
        row_count = subset.count()
        corr_value = subset.stat.corr(left_col, right_col) if row_count >= 2 else None
        rows.append(
            {
                "left_column": left_col,
                "right_column": right_col,
                "row_count": row_count,
                "pearson_correlation": corr_value
            }
        )

    return pd.DataFrame(rows)


def build_rain_group_summary(df):
    rain_summary = (
        df.filter(F.col("prcp").isNotNull())
        .groupBy("rainy_day")
        .agg(
            F.count("*").alias("days"),
            F.avg("prcp").alias("avg_prcp"),
            F.avg("tsun").alias("avg_tsun"),
            F.avg("sad_streams").alias("avg_sad_streams"),
            F.avg("share_sad_streams").alias("avg_share_sad_streams"),
            F.avg("share_sad_songs").alias("avg_share_sad_songs"),
            F.avg("weighted_avg_valence").alias("avg_weighted_valence")
        )
        .orderBy("rainy_day")
    )

    return rain_summary.toPandas()


def save_dataframe(df, output_file):
    pdf = df.toPandas()
    pdf.to_csv(output_file, index=False)
    return pdf


def main():
    require_file(SONG_FILE)
    require_file(WEATHER_FILE)

    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        daily = build_daily_spotify_metrics(spark)
        weather = load_weather(spark)
        final_df = build_final_dataset(daily, weather)

        daily_pd = save_dataframe(daily, OUTPUT_DAILY_FILE)
        final_pd = save_dataframe(final_df, OUTPUT_FINAL_FILE)

        correlation_pd = build_correlation_summary(final_df)
        correlation_pd.to_csv(OUTPUT_CORRELATION_FILE, index=False)

        rain_group_pd = build_rain_group_summary(final_df)
        rain_group_pd.to_csv(OUTPUT_RAIN_SUMMARY_FILE, index=False)

        print("Spark pipeline klaar.")
        print(f"Sad threshold: valence < {SAD_THRESHOLD}")
        print(f"Aantal dagen in dagdataset: {len(daily_pd)}")
        print(f"Aantal dagen na weather join: {len(final_pd)}")
        print("\nGemiddelden op regendagen vs droge dagen:")
        print(rain_group_pd.to_string(index=False))
        print("\nCorrelaties:")
        print(correlation_pd.to_string(index=False))
        print("\nBestanden opgeslagen:")
        print("-", OUTPUT_DAILY_FILE)
        print("-", OUTPUT_FINAL_FILE)
        print("-", OUTPUT_RAIN_SUMMARY_FILE)
        print("-", OUTPUT_CORRELATION_FILE)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
