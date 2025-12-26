"""
This module provides the methods for a k-means clustering algorithm. When ran, it trains the algorithm on the earthquake dataset
and saves the resulting centroids.
"""

import os
import random

import jax
import jax.numpy as jnp
import joblib
import pandas as pd
import polars as pl

from sklearn.preprocessing import StandardScaler

from src.config import *

NUM_CLUSTERS = 7
EPOCHS = 100


def initialize_centroids(X, k):
    """
        Initializes k centroids by randomly selecting k datapoints from the dataset.

    Args:
        X: A ND dataset of shape (num_samples, ...).
        k: The number of clusters.

    Returns:
        A ND array of shape (k, ...) representing the initialized centroids.
    """
    random_indices = random.sample(range(X.shape[0]), k)
    centroids = X[jnp.array(random_indices)]
    return jnp.array(centroids)


@jax.jit
def assign_clusters(X, centroids):
    """
        Assigns a cluster to each datapoint based on the nearest centroid.

    Args:
        X: A 2D array of shape (num_samples, num_features).
        centroids: A 2D array of the centroids of shape (num_clusters, num_features).

    Returns:
        A 1D array of shape (num_samples) containing the index of the assigned cluster for each datapoint.
    """
    # ord=2
    result = jnp.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
    return jnp.argmin(result, axis=1)


def update_centroids(X, cluster_indexes):
    """
        Updates the centroids by calculating the mean of all datapoints assigned to each cluster.

    Args:
        X: A 2D array of shape (num_samples, num_features).
        cluster_indexes: A 1D array of shape (num_samples) containing the index of the assigned cluster for each datapoint.

    Returns:
        A 2D array of the updated centroids of shape (num_clusters, num_features).
    """

    def compute_centroid(i):
        points_in_cluster = X[cluster_indexes == i]
        return jnp.mean(points_in_cluster, axis=0)

    return jnp.array([compute_centroid(i) for i in range(NUM_CLUSTERS)])


if __name__ == "__main__":
    random.seed(42)
    print(f"Training on {jax.default_backend()}")

    df = pl.read_csv(
        PROCESSED_DATA_FILE_PATH,
        schema={
            "mag": pl.Float64,
            "tsunami": pl.Float64,
            "sig": pl.Float64,
            "rms": pl.Float64,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
            "depth": pl.Float64,
        },
    )

    df = df.slice(0, df.shape[0] - sum(VAL_TEST_SPLITS))
    pd_df = df.to_pandas()

    if not os.path.exists(SCALER_PATH):
        scaler = StandardScaler()
        scaler.fit(df.to_numpy())
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)

    dataset = jnp.asarray(
        pl.from_pandas(
            pd.DataFrame(scaler.transform(pd_df.to_numpy()), columns=pd_df.columns)
        )
        .select("x", "y", "z")
        .to_numpy()
    )

    previous_centroids = jnp.zeros((NUM_CLUSTERS, dataset.shape[1]))
    current_centroids = initialize_centroids(dataset, NUM_CLUSTERS)
    cluster_indexes = None
    for i in range(EPOCHS):
        previous_centroids = current_centroids
        cluster_indexes = assign_clusters(dataset, current_centroids)
        current_centroids = update_centroids(dataset, cluster_indexes)
        if jnp.array_equal(current_centroids, previous_centroids):
            print(f"Converged at epoch {i}")
            break

    if not os.path.exists(CLUSTERING_MODEL_PATH.parent):
        os.makedirs(CLUSTERING_MODEL_PATH.parent)
    jnp.save(CLUSTERING_MODEL_PATH, current_centroids)
    jnp.save(CLUSTER_INDEX_PATH, cluster_indexes)
