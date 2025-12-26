source ./.venv/bin/activate
python -m src.data.process_data
rm ./data/processed/scaler.pkl
python -m src.models.k_means_clustering
python -m src.models.lstm_model