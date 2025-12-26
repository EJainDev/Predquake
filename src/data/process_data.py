import polars as pl
import os
import math

from src.config import RAW_DATA_FILE_PATH, PROCESSED_DATA_DIR, PROCESSED_DATA_FILE_PATH

if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

df = (
    pl.scan_csv(RAW_DATA_FILE_PATH)
    .drop(
        [
            "tz",
            "url",
            "felt",
            "cdi",
            "detail",
            "mmi",
            "alert",
            "id",
            "nst",
            "dmin",
            "gap",
            "types",
            "title",
            "sources",
            "ids",
            "code",
            "net",
            "updated",
            "place",
            "magType",
        ]
    )
    .filter(
        (pl.col("status") == "reviewed") & (pl.col("type_property") == "earthquake")
    )
    .drop(["status", "type_property"])
    .unique()
    .drop_nulls()
    .with_columns(pl.col("time").str.to_datetime().sort().dt.epoch("d").alias("time"))
    .with_columns(
        pl.col("latitude").radians().alias("latitude"),
        pl.col("longitude").radians().alias("longitude"),
    )
    .with_columns(
        (pl.col("latitude").cos() * pl.col("longitude").cos()).alias("x"),
        (pl.col("latitude").cos() * pl.col("longitude").sin()).alias("y"),
        pl.col("latitude").sin().alias("z"),
    )
    .filter((pl.col("time") > 0))
    .drop(["latitude", "longitude", "time"])
    .collect()
)

df.write_csv(PROCESSED_DATA_FILE_PATH)
