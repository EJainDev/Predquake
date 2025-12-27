from src.config import *
from flax import nnx
import orbax.checkpoint as ocp
from flax.nnx import LSTMCell, Linear, Sequential, leaky_relu
import jax
import jax.numpy as jnp
import polars as pl
import joblib
from sklearn.preprocessing import StandardScaler
from src.data.preload_batches import preload_batches
from src.models.k_means_clustering import assign_clusters

VERSION = "v11"
BATCH_SIZE = 1
ckpt_dir = CHECKPOINT_DIR


class ModelConfig:
    LSTM_HIDDEN_SIZE = 32
    LSTM_NUM_LAYERS = 1
    HIDDEN_SIZES = [32, 16]
    INPUT_FEATURES = 0
    OUTPUT_FEATURES = 0

    def __init__(self, input_features, output_features):
        ModelConfig.INPUT_FEATURES = input_features
        ModelConfig.OUTPUT_FEATURES = output_features


class Model(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.lstm_cell = LSTMCell(
            config.INPUT_FEATURES, config.LSTM_HIDDEN_SIZE, rngs=rngs
        )
        hidden_layers: list = []
        for i, hs in enumerate(config.HIDDEN_SIZES):
            in_features = (
                config.LSTM_HIDDEN_SIZE if i == 0 else config.HIDDEN_SIZES[i - 1]
            )
            hidden_layers.append(Linear(in_features, hs, rngs=rngs))
            hidden_layers.append(leaky_relu)
        if len(hidden_layers) == 0:
            hidden_layers.append(lambda x: x)
        self.hidden_layers = Sequential(*hidden_layers)
        self.output_layer = Linear(
            (
                config.HIDDEN_SIZES[-1]
                if len(config.HIDDEN_SIZES) > 0
                else config.LSTM_HIDDEN_SIZE
            ),
            config.OUTPUT_FEATURES,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_len, num_features = x.shape
        carry: tuple[jax.Array, jax.Array] = self.lstm_cell.initialize_carry(
            (batch_size, num_features), rngs=nnx.Rngs(0)
        )

        scan_fn = nnx.scan(
            self.lstm_cell,
            in_axes=(nnx.Carry, 1),
            out_axes=(nnx.Carry, 1),
        )
        x = scan_fn(carry, x)
        x = x[1][:, -1, :]
        x = self.hidden_layers(x)
        y: jax.Array = self.output_layer(x)
        return y


if __name__ == "__main__":
    model = Model(ModelConfig(7, 3), rngs=nnx.Rngs(0))
    checkpointer = ocp.StandardCheckpointer()

    # Get model config from saved state
    input_features = model.lstm_cell.in_features
    output_features = model.output_layer.out_features
    abstract_model = nnx.eval_shape(
        lambda: Model(ModelConfig(input_features, output_features), rngs=nnx.Rngs(0))
    )
    graphdef, abstract_state = nnx.split(abstract_model)
    state_restored = checkpointer.restore(ckpt_dir / f"state_{VERSION}", abstract_state)
    model = nnx.merge(graphdef, state_restored)
    print("Loaded model from disk")

    centroids: jax.Array = jnp.load(CLUSTERING_MODEL_PATH)
    df: pl.DataFrame = (
        pl.scan_csv(PROCESSED_DATA_FILE_PATH)
        .slice(-VAL_TEST_SPLITS[1], VAL_TEST_SPLITS[1])
        .collect()
    )

    scaler: StandardScaler = joblib.load(SCALER_PATH)
    test_X_full = jnp.array(scaler.transform(df.to_numpy()))
    test_y_full = jnp.array(df[["x", "y", "z"]].to_numpy())
    test_index = assign_clusters(
        test_X_full[
            :,
            [
                df.columns.index("x"),
                df.columns.index("y"),
                df.columns.index("z"),
            ],
        ],
        centroids,
    )

    test_batches = []

    for cluster in range(centroids.shape[0]):
        print(f"  Processing cluster {cluster + 1}/{centroids.shape[0]}")
        # Use pre-computed data with boolean indexing
        mask = test_index == cluster
        cluster_X = test_X_full[mask]
        cluster_y = test_y_full[mask]

        test_batches.extend(
            preload_batches(cluster_X, cluster_y, TIME_STEPS, BATCH_SIZE)
        )

    print("Evaluating on test data...")
    total_loss = 0.0
    total_samples = 0
    for batch_X, batch_y in test_batches:
        preds = model(batch_X)
        loss = jnp.mean((preds - batch_y) ** 2)
        total_loss += loss * batch_X.shape[0]
    avg_loss = total_loss / len(test_batches)
