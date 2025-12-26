import os

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import orbax.checkpoint as ocp
import polars as pl

from flax import nnx
from flax.nnx import LSTMCell, Linear, Sequential, leaky_relu

from sklearn.preprocessing import StandardScaler

from src.config import *

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

VERSION = "v7"
LR = 0.001
B1 = 0.9
B2 = 0.999
TIME_STEPS = 32
BATCH_SIZE = 32
EPOCHS = 50
CLUSTER_INDEX = 0
ckpt_dir = CHECKPOINT_DIR


class ModelConfig:
    LSTM_HIDDEN_SIZE = 16
    LSTM_NUM_LAYERS = 1
    HIDDEN_SIZES = []
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

    @nnx.jit
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


def create_batch(
    data_X: jax.Array,
    data_y: jax.Array,
    start_idx: int,
    time_steps: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    # Use vectorized indexing instead of list comprehension
    indices: jax.Array = jnp.arange(batch_size)[:, None] + start_idx
    indices = jnp.arange(time_steps)[None, :] + indices
    batch_X: jax.Array = jnp.take(data_X, indices.ravel(), axis=0).reshape(
        batch_size, time_steps, -1
    )
    batch_y: jax.Array = data_y[start_idx : start_idx + batch_size]
    return batch_X, batch_y


def preload_batches(
    train_X: jax.Array, train_y: jax.Array, time_steps: int, batch_size: int
):
    num_train_datapoints: int = train_X.shape[0] - time_steps
    num_train_batches: int = num_train_datapoints // batch_size
    batches: list[tuple[jax.Array, jax.Array]] = []
    for i in range(num_train_batches):
        batches.append(
            create_batch(train_X, train_y, i * batch_size, time_steps, batch_size)
        )
    return batches


@nnx.jit(donate_argnames=("model"))
def loss_fn(model: Model, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean((model(inputs) - targets) ** 2)


@nnx.jit(donate_argnames=("optimizer"))
def train_step(
    optimizer: nnx.ModelAndOptimizer, batch_X: jax.Array, batch_y: jax.Array
) -> float:
    def loss_and_grads(model: Model) -> jax.Array:
        return loss_fn(model, batch_X, batch_y)

    loss, grads = nnx.value_and_grad(loss_and_grads)(optimizer.model)
    optimizer.update(grads)
    return loss


@nnx.jit(donate_argnames=("model"))
def _validation_step(model: Model, batch_X: jax.Array, batch_y: jax.Array) -> jax.Array:
    preds: jax.Array = model(batch_X)
    loss: jax.Array = jnp.mean((preds - batch_y) ** 2)
    return loss


def validate(model: Model, val_batches: list[tuple[jax.Array, jax.Array]]) -> float:
    val_loss: float = 0.0
    model.eval()
    for batch_X, batch_y in val_batches:
        val_loss += _validation_step(model, batch_X, batch_y).item(0)
    val_loss /= len(val_batches)
    return val_loss


def train(
    model: Model,
    tr: optax.GradientTransformation,
    train_batches: list[tuple[jax.Array, jax.Array]],
    val_batches: list[tuple[jax.Array, jax.Array]],
    num_epochs: int,
):
    optimizer = nnx.ModelAndOptimizer(model, tr, wrt=nnx.Param)
    checkpointer = ocp.StandardCheckpointer()

    if (
        f"state_{VERSION}" in os.listdir(ckpt_dir)
        and os.listdir(ckpt_dir / f"state_{VERSION}") != []
    ):
        abstract_model = nnx.eval_shape(
            lambda: Model(ModelConfig(df.shape[1], 2), rngs=nnx.Rngs(0))
        )
        graphdef, abstract_state = nnx.split(abstract_model)
        state_restored = checkpointer.restore(
            ckpt_dir / f"state_{VERSION}", abstract_state
        )
        jax.tree.map(np.testing.assert_array_equal, optimizer, state_restored)
        model = nnx.merge(graphdef, state_restored)

    for epoch in range(num_epochs):
        train_loss: float = 0.0
        model.train()
        for batch_X, batch_y in train_batches:
            loss = train_step(optimizer, batch_X, batch_y)
            train_loss += loss
        train_loss /= len(train_batches)

        val_loss: float = validate(model, val_batches)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        _, state = nnx.split(model)
        checkpointer.save(ckpt_dir / f"state_{VERSION}", state, force=True)


if __name__ == "__main__":
    index: jax.Array = jnp.load(CLUSTER_INDEX_PATH)
    df: pl.DataFrame = pl.read_csv(
        PROCESSED_DATA_FILE_PATH,
        schema={
            "mag": pl.Float64,
            "time": pl.Float64,
            "tsunami": pl.Float64,
            "sig": pl.Float64,
            "rms": pl.Float64,
            "longitude": pl.Float64,
            "latitude": pl.Float64,
            "depth": pl.Float64,
        },
    )

    scaler: StandardScaler = joblib.load(SCALER_PATH)

    temp_df = df.slice(0, df.shape[0] - sum(VAL_TEST_SPLITS))
    train_X = jnp.array(scaler.transform(temp_df.to_numpy()))[
        index[0 : df.shape[0] - sum(VAL_TEST_SPLITS)] == CLUSTER_INDEX
    ][:-1]
    train_y = jnp.array(temp_df.select("x", "y", "z").to_numpy())[
        index[0 : df.shape[0] - sum(VAL_TEST_SPLITS)] == CLUSTER_INDEX
    ][1 + TIME_STEPS :]

    temp_df = df.slice(-sum(VAL_TEST_SPLITS), VAL_TEST_SPLITS[0])
    val_X = jnp.array(scaler.transform(temp_df.to_numpy()))[
        index[-sum(VAL_TEST_SPLITS) : -VAL_TEST_SPLITS[1]] == CLUSTER_INDEX
    ][:-1]
    val_y = jnp.array(temp_df.select("x", "y", "z").to_numpy())[
        index[-sum(VAL_TEST_SPLITS) : -VAL_TEST_SPLITS[1]] == CLUSTER_INDEX
    ][1 + TIME_STEPS :]

    model = Model(ModelConfig(train_X.shape[1], train_y.shape[1]), rngs=nnx.Rngs(0))
    tx = optax.adam(learning_rate=LR, b1=B1, b2=B2)

    train(
        model,
        tx,
        preload_batches(train_X, train_y, TIME_STEPS, BATCH_SIZE),
        preload_batches(val_X, val_y, TIME_STEPS, BATCH_SIZE),
        num_epochs=EPOCHS,
    )
