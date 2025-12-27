from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE_PATH = RAW_DATA_DIR / "Earthquakes_USGS.csv"
PROCESSED_DATA_FILE_PATH = PROCESSED_DATA_DIR / "data.csv"
LOCATION_SCALER_PATH = PROCESSED_DATA_DIR / "location_scaler.pkl"
PROCESSED_DATA_POST_CLUSTER_FILE_PATH = PROCESSED_DATA_DIR / "data_post_cluster.csv"

VAL_TEST_SPLITS = (20_000, 10_000)  # Number of samples for validation and test sets

CLUSTERING_MODEL_PATH = ROOT_DIR / "models" / "k_means_model.npy"
CLUSTER_INDEX_PATH = ROOT_DIR / "models" / "k_means_cluster_indexes.npy"

CHECKPOINT_DIR = ROOT_DIR / "models" / "lstm_checkpoints"

TIME_STEPS = 64
BATCH_SIZE = 32
