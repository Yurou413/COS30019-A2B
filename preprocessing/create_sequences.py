
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load combined dataset
data = pd.read_csv("data/combined_traffic_timeseries.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

# Select SCATS site 2000
site_id = 2000
site_data = data[data["SCATS Number"] == site_id].copy()
site_data = site_data.sort_values("Timestamp").reset_index(drop=True)

# Settings
sequence_length = 4

# Split chronologically: first 80% for training
split_index = int(len(site_data) * 0.8)
cutoff = site_data.iloc[split_index]["Timestamp"]

# Fit scaler using training-period readings only
scaler = MinMaxScaler()
scaler.fit(
    site_data.loc[
        site_data["Timestamp"] < cutoff,
        ["Traffic Volume"]
    ]
)

X_train, y_train = [], []
X_test, y_test = [], []

# Separate continuous time periods
site_data["Group"] = (
    site_data["Timestamp"].diff() != pd.Timedelta(minutes=15)
).cumsum()

for _, segment in site_data.groupby("Group"):

    values = scaler.transform(
        segment[["Traffic Volume"]]
    ).flatten()

    timestamps = segment["Timestamp"].tolist()

    # Create sequences without crossing missing time gaps
    for i in range(sequence_length, len(values)):

        X = values[i-sequence_length:i]
        y = values[i]
        target_time = timestamps[i]

        if target_time < cutoff:
            X_train.append(X)
            y_train.append(y)
        else:
            X_test.append(X)
            y_test.append(y)

# Convert to NumPy arrays
X_train = np.array(X_train).reshape(-1, sequence_length, 1)
X_test = np.array(X_test).reshape(-1, sequence_length, 1)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Save prepared data
np.savez_compressed(
    "data/site2000_sequences.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print("Sequence preparation completed!")
print("Training input:", X_train.shape)
print("Training target:", y_train.shape)
print("Testing input:", X_test.shape)
print("Testing target:", y_test.shape)
print("Training cutoff:", cutoff)
print("Saved to data/site2000_sequences.npz")
