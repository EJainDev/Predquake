import jax
import jax.numpy as jnp
import joblib
import polars as pl

from functools import partial
import numpy as np

from sklearn.preprocessing import StandardScaler

from src.config import *
from src.models.k_means_clustering import assign_clusters

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

rng_key = jax.random.PRNGKey(0)


def _fourier(batch_X: jax.Array) -> jax.Array:
    FOURIER_FEATURES_SCALE = 1.0
    output_dim = 32
    input_dim = batch_X.shape[-1]
    B = jax.random.normal(rng_key, (input_dim, output_dim)) * FOURIER_FEATURES_SCALE
    batch_X_proj = 2 * jnp.pi * batch_X @ B
    fourier_features = jnp.concatenate(
        [jnp.sin(batch_X_proj), jnp.cos(batch_X_proj)], axis=-1
    )
    return fourier_features


def _sub_create_batch(
    indices: jax.Array, data_X: jax.Array, time_steps: int, batch_size: int
) -> jax.Array:
    indices = jnp.arange(time_steps)[None, :] + indices
    batch_X: jax.Array = jnp.take(data_X, indices.ravel(), axis=0).reshape(
        batch_size, time_steps, -1
    )
    return _fourier(batch_X)


def create_batch(
    data_X: jax.Array,
    data_y: jax.Array,
    start_idx: int,
    time_steps: int,
    batch_size: int,
) -> tuple:
    indices: jax.Array = jnp.arange(batch_size)[:, None] + start_idx
    batch_y = data_y[start_idx : start_idx + batch_size].block_until_ready()
    return _sub_create_batch(indices, data_X, time_steps, batch_size), batch_y


def preload_batches(
    train_X: jax.Array, train_y: jax.Array, time_steps: int, batch_size: int
):
    num_train_datapoints: int = train_X.shape[0] - time_steps
    num_train_batches: int = num_train_datapoints // batch_size
    batches: list[tuple] = [
        create_batch(train_X, train_y, i * batch_size, time_steps, batch_size)
        for i in range(num_train_batches)
    ]
    return batches


def prepare_and_save_batches(
    time_steps: int = TIME_STEPS, batch_size: int = BATCH_SIZE
):
    """Prepare all batches and save them to disk for later use."""
    print("Loading data...")
    index: jax.Array = jnp.load(CLUSTER_INDEX_PATH)
    centroids: jax.Array = jnp.load(CLUSTERING_MODEL_PATH)
    df: pl.DataFrame = pl.read_csv(
        PROCESSED_DATA_POST_CLUSTER_FILE_PATH,
    )

    scaler: StandardScaler = joblib.load(SCALER_PATH)

    print("Preprocessing data...")
    # Pre-compute commonly used values
    total_val_test = sum(VAL_TEST_SPLITS)
    train_end = df.shape[0] - total_val_test
    val_start = train_end

    # Slice dataframes once
    train_df = df.slice(0, train_end)
    val_df = df.slice(val_start, VAL_TEST_SPLITS[0])

    # Transform data once
    train_scaled = jnp.array(scaler.transform(train_df.to_numpy()))
    val_scaled = jnp.array(scaler.transform(val_df.to_numpy()))
    train_xyz = jnp.array(train_df.select("x", "y", "z").to_numpy())
    val_xyz = jnp.array(val_df.select("x", "y", "z").to_numpy())

    # Pre-slice index arrays
    train_index = assign_clusters(
        train_scaled[
            :,
            [
                train_df.columns.index("x"),
                train_df.columns.index("y"),
                train_df.columns.index("z"),
            ],
        ],
        centroids,
    )
    val_index = assign_clusters(
        val_scaled[
            :,
            [
                val_df.columns.index("x"),
                val_df.columns.index("y"),
                val_df.columns.index("z"),
            ],
        ],
        centroids,
    )

    print("Creating batches for all clusters...")
    train_batches = []
    val_batches = []
    for cluster in range(centroids.shape[0]):
        print(f"  Processing cluster {cluster + 1}/{centroids.shape[0]}")
        # Use pre-computed data with boolean indexing
        train_mask = train_index == cluster
        train_X = train_scaled[train_mask][:-1]
        train_y = train_xyz[train_mask][1 + time_steps :]

        val_mask = val_index == cluster
        val_X = val_scaled[val_mask][:-1]
        val_y = val_xyz[val_mask][1 + time_steps :]

        train_batches.extend(preload_batches(train_X, train_y, time_steps, batch_size))
        val_batches.extend(preload_batches(val_X, val_y, time_steps, batch_size))

    train_scaled = None
    val_scaled = None
    train_xyz = None
    val_xyz = None
    train_df = None
    val_df = None
    df = None
    index = None
    centroids = None

    print(f"Total train batches: {len(train_batches)}")
    print(
        f"Train batches dimensions: {train_batches[0][0].shape} -> {train_batches[0][1].shape}"
    )
    print(f"Total validation batches: {len(val_batches)}")

    # Save batches
    print("Saving batches...")
    batches_path = PROCESSED_DATA_DIR / "preloaded_batches.pkl"
    joblib.dump(
        {"train_batches": train_batches, "val_batches": val_batches},
        batches_path,
    )
    print(f"Batches saved to {batches_path}")
    return batches_path


if __name__ == "__main__":
    prepare_and_save_batches()
