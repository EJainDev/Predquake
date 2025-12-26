from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE_PATH = RAW_DATA_DIR / "Earthquakes_USGS.csv"
PROCESSED_DATA_FILE_PATH = PROCESSED_DATA_DIR / "data.csv"
SCALER_PATH = PROCESSED_DATA_DIR / "scaler.pkl"

VAL_TEST_SPLITS = (10_000, 10_000)  # Number of samples for validation and test sets

CLUSTERING_MODEL_PATH = ROOT_DIR / "models" / "k_means_model.npy"
CLUSTER_INDEX_PATH = ROOT_DIR / "models" / "k_means_cluster_indexes.npy"

CHECKPOINT_DIR = ROOT_DIR / "models" / "lstm_checkpoints"
