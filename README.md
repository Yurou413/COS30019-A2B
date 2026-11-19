# COS30019-A2B

## Traffic model training

The LSTM and GRU models train on all SCATS sites in
`data/combined_traffic_timeseries.csv`. The preprocessing step creates
`data/traffic_sequences.npz` and saves a per-site MinMaxScaler mapping to
`models/site_scalers.joblib`.

Use the **Train Selected Model** button in the app to regenerate the sequences
and train the selected LSTM or GRU. To run training from the command line:

```powershell
python preprocessing/create_sequences.py
python ML-models/lstm_model.py
# or
python ML-models/gru_model.py
```

The GRU uses four consecutive 15-minute readings to predict the next interval.
Its input includes the scaled traffic readings and a one-hot SCATS site ID. The
training script saves `models/gru_traffic_model.keras` and writes its evaluation
results to `results/gru_metrics.json`. The metrics include normalized and
original traffic-volume scales, training time, and average inference time.
Training checks 32-unit and 64-unit GRUs using a chronological validation
split, stops early when validation loss stalls, and keeps the candidate with
the lower validation MSE. The held-out test set is used only for the final
evaluation.

The GUI can call `predict_flow(site_id, timestamp)` from
`ML-models/gru_model.py`. It returns the prediction in the dataset's original
traffic-volume units. The requested timestamp must be one 15-minute step after
four consecutive readings available in `data/combined_traffic_timeseries.csv`.
Regenerate the shared sequences and scalers, then retrain the GRU whenever the
SCATS site set or preprocessing features change.

The Regression Tree training option is not implemented. Route travel-time
estimates in `get_routes()` are still placeholder calculations; they do not
yet use LSTM/GRU predictions.
