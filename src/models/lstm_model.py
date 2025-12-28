import os
from time import sleep

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import orbax.checkpoint as ocp
import polars as pl

from flax import nnx
from flax.nnx import LSTMCell, Linear, Sequential, leaky_relu, tanh

from src.config import *
from src.data.dataset import Dataset, PrefetchedDataset

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

VERSION = "v24"
LR = 0.001
B1 = 0.9
B2 = 0.999
EPOCHS = 50
CLUSTER_INDEX = 0
ckpt_dir = CHECKPOINT_DIR


class ModelConfig:
    LSTM_HIDDEN_SIZE = 32
    LSTM_NUM_LAYERS = 1
    HIDDEN_SIZES = [64, 32, 16]
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
        hidden_layers: list = [leaky_relu]
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
        self.output_activation = tanh

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
        x = self.output_layer(x)
        y = self.output_activation(x)
        return y


@nnx.jit(donate_argnames=("model"))
def loss_fn(model: Model, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean((model(inputs) - targets) ** 2)


@nnx.jit(donate_argnames=("optimizer"))
def train_step(
    optimizer: nnx.ModelAndOptimizer, batch_X: jax.Array, batch_y: jax.Array
) -> jax.Array:
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


def validate(model: Model, val_dataset) -> float:
    val_loss: float = 0.0
    model.eval()
    for batch_X, batch_y in val_dataset:
        val_loss += _validation_step(model, batch_X, batch_y).item(0)
    val_loss /= len(val_dataset)
    return val_loss


def train(
    model: Model,
    optimizer: nnx.ModelAndOptimizer,
    checkpointer: ocp.StandardCheckpointer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int,
):
    for epoch in range(num_epochs):
        train_loss: float = 0.0
        model.train()
        for i, (batch_X, batch_y) in enumerate(train_dataset):
            loss = train_step(optimizer, batch_X, batch_y)
            train_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch}, Step {i+1}, Batch Loss: {loss.item()}")

        train_loss /= len(train_dataset)

        val_loss: float = validate(model, val_dataset)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        _, state = nnx.split(model)
        checkpointer.save(ckpt_dir / f"state_{VERSION}", state, force=True)


if __name__ == "__main__":
    model = Model(ModelConfig(INPUT_FEATURES, 3), rngs=nnx.Rngs(0))
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

    df = pl.read_csv(PROCESSED_DATA_POST_CLUSTER_FILE_PATH)
    train_X = df.slice(0, df.shape[0] - sum(VAL_TEST_SPLITS) - 1)
    train_y = jnp.array(
        df.slice(1, df.shape[0] - sum(VAL_TEST_SPLITS) - 1)
        .select(["x", "y", "z"])
        .to_numpy()
    )
    val_X = df.slice(
        df.shape[0] - sum(VAL_TEST_SPLITS),
        VAL_TEST_SPLITS[0] - 1,
    )
    val_y = jnp.array(
        df.slice(
            df.shape[0] - sum(VAL_TEST_SPLITS) + 1,
            VAL_TEST_SPLITS[0] - 1,
        )
        .select(["x", "y", "z"])
        .to_numpy()
    )
    PREFETCH_SIZE = 256
    centroids: jax.Array = jnp.load(CLUSTERING_MODEL_PATH)
    train_dataset = Dataset(train_X, train_y, centroids, shuffle=True)
    val_dataset = Dataset(val_X, val_y, centroids, shuffle=False)
    train_dataset = PrefetchedDataset(train_dataset, PREFETCH_SIZE)

    print(
        f"Loaded {len(train_dataset)} train batches and {len(val_dataset)} validation batches"
    )

    train(
        model,
        optimizer,
        checkpointer,
        train_dataset,
        val_dataset,
        num_epochs=EPOCHS,
    )

    print("Waiting for 2 minutes before exiting to ensure checkpoint is saved...")
    sleep(120)  # To ensure that the checkpoint is saved before the program exits
