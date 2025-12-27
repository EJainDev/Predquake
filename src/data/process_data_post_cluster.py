from src.models.k_means_clustering import assign_clusters
import polars as pl
from src.config import *
import jax
import jax.numpy as jnp


def update(df: pl.DataFrame) -> pl.DataFrame:
    df = df.slice(1).with_columns(
        (
            df.slice(1)["time"].str.to_datetime()
            - df.slice(0, df.shape[0] - 1)["time"].str.to_datetime()
        ).alias("time_since_last")
    )
    return df


if __name__ == "__main__":
    df: pl.DataFrame = pl.read_csv(PROCESSED_DATA_FILE_PATH)

    centroids = jnp.load(CLUSTERING_MODEL_PATH)

    indices = assign_clusters(jnp.array(df.select("x", "y", "z").to_numpy()), centroids)

    updated_df = pl.DataFrame()
    for cluster in range(centroids.shape[0]):
        cluster_df = df.filter(indices == cluster)
        cluster_df = update(cluster_df)
        updated_df = pl.concat([updated_df, cluster_df], how="vertical")

    updated_df.write_csv(PROCESSED_DATA_POST_CLUSTER_FILE_PATH)
