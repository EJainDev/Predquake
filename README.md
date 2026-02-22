
# Predquake — Earthquake Location Prediction

Predquake is a research / experimental project that attempts to predict the location of future earthquakes from historical seismic records and contextual features. The project uses JAX and Flax for neural-network modelling alongside classic clustering methods to structure and preprocess the data.

Key ideas
- Use clustering (k-means) to preprocess and discretize spatial structure.
- Train sequence models (LSTM) implemented with JAX + Flax to predict future event properties.
- Evaluate models against held-out data and checkpoints stored under `models/lstm_checkpoints`.

Tech stack
- Python 3.8+
- JAX (for accelerated array and autodiff)
- Flax (neural network library built on JAX)
- NumPy, Pandas for data processing
- scikit-learn (k-means clustering)
- Matplotlib / notebooks for exploration

Repository layout

- `src/` — project source code
  - `src/data/` — data loading and preprocessing scripts
  - `src/models/` — model definitions (k-means, LSTM)
  - `src/eval.py` — evaluation utilities
- `data/raw/` — raw datasets (Earthquakes_USGS.csv)
- `data/processed/` — processed inputs used for training
- `models/` — saved models and checkpoints
- `notebooks/` — exploratory analysis notebooks

Getting started

1. Clone the repository and enter the project directory.

2. Create a Python environment and install dependencies. For a minimal setup (CPU):

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jax flax optax numpy pandas scikit-learn matplotlib
```

Note: Installing JAX with GPU support requires selecting the correct `jaxlib` wheel for your CUDA/cuDNN versions. See the official JAX installation instructions if you need GPU acceleration.

3. Prepare data
- Put raw CSV files in `data/raw/` (the repo already contains `Earthquakes_USGS.csv`).
- Run the included processing scripts or the pipeline helper scripts to build processed datasets:

```
./pipeline.sh
```

4. Train / evaluate
- Training and evaluation utilities live under `src/` and `models/`.
- Checkpoints and saved model artifacts are stored under `models/lstm_checkpoints/`.
- There is a helper `run.sh` that wires common steps; inspect it for environment and invocation details.

Exploration
- Open `notebooks/data_analysis.ipynb` for interactive exploration of processed data and visualizations.

Notes and tips
- This project is experimental — predictive performance on earthquake location is highly challenging; treat models and results as research artifacts.
- If you plan to run experiments on a GPU, follow JAX's GPU installation notes and ensure `jaxlib` is matched to your CUDA version.

Contributing
- If you make improvements, please open issues or pull requests. Include reproducible steps for any experiments.

License & citation
- This repository does not include an explicit license file. If you reuse this code, please credit the author and include appropriate licensing in downstream work.

Ignored files and large artifacts

- This repository's `.gitignore` excludes several large or environment-specific artifacts. Notable entries include:
  - `output.prof`, `prof.py` — profiling outputs
  - `data/*` — processed and raw data directories (raw data is expected in `data/raw/` and processed artifacts in `data/processed/`)
  - `models/*` — trained models and checkpoints (for example `models/lstm_checkpoints/`)
  - `__pycache__/` and other temporary files

- Because `data/` and `models/` are ignored, the repository does not contain the full datasets or large pretrained checkpoints. To reproduce or run experiments locally:
  1. Place raw CSV files in `data/raw/` (the repository may include a small example `Earthquakes_USGS.csv`, but larger datasets should be added by you).
  2. Generate processed inputs by running the processing pipeline: `./pipeline.sh` or the scripts in `src/data/` (for example `src/data/process_data.py`).
  3. Train models locally to create checkpoints, or copy pretrained checkpoints into `models/lstm_checkpoints/` if you have them available.

- If you want to share reproducible experiments or smaller example datasets, consider adding a `data/sample/` folder (and not ignoring it) or documenting download instructions in this README.
