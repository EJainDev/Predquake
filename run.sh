echo "Preloading batches..."
python -m src.data.preload_batches
echo "Finished preloading batches"
echo "Running LSTM model..."
python -m src.models.lstm_model
echo "Finished running LSTM model"