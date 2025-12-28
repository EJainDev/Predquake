import jax
import jax.numpy as jnp
from src.config import *
import numpy as np
from src.models.k_means_clustering import assign_clusters

from functools import partial

FOURIER_FEATURES_SCALE = 1.0
output_dim = 32
input_dim = 8


@jax.jit
def _fourier(batch_X: jax.Array, B: jax.Array):
    batch_X_proj = 2 * jnp.pi * batch_X @ B
    fourier_features = jnp.concatenate(
        [jnp.sin(batch_X_proj), jnp.cos(batch_X_proj)], axis=-1
    )
    return fourier_features


@partial(jax.jit, static_argnames=("time_steps", "batch_size"))
def _sub_create_batch(
    indices: jax.Array,
    arranged_time_steps: jax.Array,
    data_X: jax.Array,
    B: jax.Array,
    time_steps: int,
    batch_size: int,
):
    indices = arranged_time_steps + indices
    batch_X: jax.Array = jnp.take(data_X, indices.ravel(), axis=0).reshape(
        batch_size, time_steps, -1
    )
    return _fourier(batch_X, B)


def create_batch(
    data_X: jax.Array,
    data_y: jax.Array,
    indices: jax.Array,
    arranged_time_steps: jax.Array,
    start_idx: int,
    time_steps: int,
    batch_size: int,
    B: jax.Array,
):
    indices = indices + start_idx
    batch_y = data_y[start_idx : start_idx + batch_size]
    batch_X = _sub_create_batch(
        indices, arranged_time_steps, data_X, B, time_steps, batch_size
    )
    return batch_X, batch_y


class Dataset:
    def __init__(
        self,
        data_X,
        data_y: jax.Array,
        centroids,
        shuffle: bool = True,
    ):
        self.indices = assign_clusters(
            jnp.array(data_X.select(["x", "y", "z"]).to_numpy()), centroids
        )
        self.batch_indices = jnp.arange(BATCH_SIZE)[:, None]
        self.arranged_time_steps = jnp.arange(TIME_STEPS)[None, :]
        data_X = jnp.array(data_X.to_numpy())
        self.datas_X = []
        self.datas_y = []
        for i in range(centroids.shape[0]):
            mask = self.indices == i
            self.datas_X.append(data_X[mask])
            self.datas_y.append(data_y[mask])
        self.cluster = 0
        self.shuffle = shuffle
        # Seeded NumPy RNG for deterministic shuffling
        self.np_rng = np.random.default_rng(0)
        self.B = (
            jax.random.normal(jax.random.PRNGKey(32), (input_dim, output_dim // 2))
            * FOURIER_FEATURES_SCALE
        )

    def __len__(self) -> int:
        return (sum([x.shape[0] for x in self.datas_X]) - TIME_STEPS) // BATCH_SIZE

    def __getitem__(self, idx: int):
        return create_batch(
            self.datas_X[self.cluster],
            self.datas_y[self.cluster],
            self.batch_indices,
            self.arranged_time_steps,
            idx * BATCH_SIZE,
            TIME_STEPS,
            BATCH_SIZE,
            self.B,
        )

    def __iter__(self):
        self.cluster = 0
        self.sequences = [
            np.arange((x.shape[0] - TIME_STEPS) // BATCH_SIZE) for x in self.datas_X
        ]
        if self.shuffle:
            for sequence in self.sequences:
                np.random.shuffle(sequence)
        self.current_idx = 0
        return self

    def __next__(self) -> tuple[jax.Array, jax.Array]:
        while self.current_idx >= len(self.sequences[self.cluster]):
            self.cluster += 1
            self.current_idx = 0
            if self.cluster >= len(self.datas_X):
                raise StopIteration
        idx = self.sequences[self.cluster][self.current_idx]
        self.current_idx += 1
        return self[idx]


from concurrent.futures import ThreadPoolExecutor
from queue import Queue


class PrefetchedDataset:
    def __init__(self, dataset, prefetch_size: int = 2):
        self.dataset = dataset
        self.prefetch_size = prefetch_size
        self.queue = Queue(maxsize=prefetch_size)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.iterator = None
        self.future = None

    def _loader_worker(self):
        try:
            for batch in self.iterator:
                self.queue.put(batch)
        finally:
            self.queue.put(None)  # Signal end of iteration

    def __iter__(self):
        self.iterator = iter(self.dataset)
        # Submit loading task to thread pool
        self.future = self.executor.submit(self._loader_worker)
        return self

    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        self.executor.shutdown(wait=False)
