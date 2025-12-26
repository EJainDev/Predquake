import os
from time import sleep

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import orbax.checkpoint as ocp

from flax import nnx
from flax.nnx import LSTMCell, Linear, Sequential, leaky_relu

from src.config import *

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

VERSION = "v12"
LR = 0.001
B1 = 0.9
B2 = 0.999
EPOCHS = 50
CLUSTER_INDEX = 0
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
    optimizer: nnx.ModelAndOptimizer,
    checkpointer: ocp.StandardCheckpointer,
    train_batches: list[tuple[jax.Array, jax.Array]],
    val_batches: list[tuple[jax.Array, jax.Array]],
    num_epochs: int,
):
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
    batches_path = PROCESSED_DATA_DIR / "preloaded_batches.pkl"

    if not batches_path.exists():
        print(f"Precomputed batches not found at {batches_path}")
        print("Please run: python -m src.data.preload_batches")
        exit(1)

    model = Model(ModelConfig(7, 3), rngs=nnx.Rngs(0))
    tx = optax.adam(learning_rate=LR, b1=B1, b2=B2)

    optimizer = nnx.ModelAndOptimizer(model, tx)
    checkpointer = ocp.StandardCheckpointer()

    if (
        f"state_{VERSION}" in os.listdir(ckpt_dir)
        and os.listdir(ckpt_dir / f"state_{VERSION}") != []
    ):
        # Get model config from saved state
        input_features = model.lstm_cell.in_features
        output_features = model.output_layer.out_features
        abstract_model = nnx.eval_shape(
            lambda: Model(
                ModelConfig(input_features, output_features), rngs=nnx.Rngs(0)
            )
        )
        graphdef, abstract_state = nnx.split(abstract_model)
        state_restored = checkpointer.restore(
            ckpt_dir / f"state_{VERSION}", abstract_state
        )
        model = nnx.merge(graphdef, state_restored)
        optimizer = nnx.ModelAndOptimizer(model, tx)
        print("Loaded model from disk")

    batch_data = joblib.load(batches_path)
    train_batches = batch_data["train_batches"]
    val_batches = batch_data["val_batches"]
    input_features = batch_data["input_features"]
    output_features = batch_data["output_features"]

    print(
        f"Loaded {len(train_batches)} train batches and {len(val_batches)} validation batches"
    )

    train(
        model,
        optimizer,
        checkpointer,
        train_batches,
        val_batches,
        num_epochs=EPOCHS,
    )

    print("Waiting for 2 minutes before exiting to ensure checkpoint is saved...")
    sleep(120)  # To ensure that the checkpoint is saved before the program exits
