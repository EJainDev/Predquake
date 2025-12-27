source ./.venv/bin/activate
echo "Running data processing..."
python -m src.data.process_data
echo "Finished data processing"
rm ./data/processed/location_scaler.pkl
rm ./data/processed/preloaded_batches.pkl
echo "Running k-means clustering..."
python -m src.models.k_means_clustering
echo "Finished running k-means clustering"
echo "Updating processed data with cluster assignments..."
python -m src.data.process_data_post_cluster
echo "Finished updating processed data with cluster assignments"
echo "Preloading batches..."
python -m src.data.preload_batches
echo "Finished preloading batches"
echo "Running LSTM model..."
python -m src.models.lstm_model
echo "Finished running LSTM model"