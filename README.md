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

The Regression Tree training option is not implemented. Route travel-time
estimates in `get_routes()` are still placeholder calculations; they do not
yet use LSTM/GRU predictions.
