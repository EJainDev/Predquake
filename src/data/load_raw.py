import shutil
import os

import kagglehub

from src.config import RAW_DATA_DIR


def download_data() -> None:
    path = kagglehub.dataset_download(
        "bwandowando/earthquakes-around-the-world-from-1900-2025"
    )

    if len(os.listdir(path)) == 0:
        path = kagglehub.dataset_download(
            "bwandowando/earthquakes-around-the-world-from-1900-2025",
            force_download=True,
        )

    while os.path.isdir(path):
        path = os.path.join(path, os.listdir(path)[0])

    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    shutil.move(path, RAW_DATA_DIR)
    os.rmdir(os.path.dirname(path))


if __name__ == "__main__":
    download_data()
