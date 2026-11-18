from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "combined_traffic_timeseries.csv"
DEFAULT_SEQUENCE_PATH = ROOT / "data" / "traffic_sequences.npz"
DEFAULT_SCALER_PATH = ROOT / "models" / "site_scalers.joblib"
SEQUENCE_LENGTH = 4
INTERVAL = pd.Timedelta(minutes=15)


def create_sequences(
    data_path: Path = DEFAULT_DATA_PATH,
    sequence_path: Path = DEFAULT_SEQUENCE_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Create chronological training and test windows from every SCATS site."""
    data = pd.read_csv(data_path)
    required_columns = {"SCATS Number", "Timestamp", "Traffic Volume"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Traffic data is missing columns: {sorted(missing_columns)}")

    data = data.loc[:, ["SCATS Number", "Timestamp", "Traffic Volume"]].copy()
    data["SCATS Number"] = pd.to_numeric(data["SCATS Number"], errors="raise").astype("Int64")
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="raise")
    data["Traffic Volume"] = pd.to_numeric(data["Traffic Volume"], errors="raise")
    data = data.dropna(subset=["SCATS Number", "Timestamp", "Traffic Volume"])
    data["SCATS Number"] = data["SCATS Number"].astype(int)
    data = data.sort_values(["SCATS Number", "Timestamp"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("Traffic data contains no usable readings.")

    site_ids = np.sort(data["SCATS Number"].unique())
    timestamps = np.sort(data["Timestamp"].unique())
    if len(timestamps) < 2:
        raise ValueError("At least two distinct timestamps are required.")
    cutoff = pd.Timestamp(timestamps[int(len(timestamps) * 0.8)])
    if cutoff <= pd.Timestamp(timestamps[0]):
        raise ValueError("Not enough timestamps to create a chronological split.")

    scalers: dict[int, MinMaxScaler] = {}
    site_indices = {int(site_id): index for index, site_id in enumerate(site_ids)}
    for site_id, site_data in data.groupby("SCATS Number", sort=True):
        training_values = site_data.loc[
            site_data["Timestamp"] < cutoff, ["Traffic Volume"]
        ]
        if training_values.empty:
            raise ValueError(f"SCATS site {site_id} has no readings before the split.")
        scaler = MinMaxScaler()
        scaler.fit(training_values)
        scalers[int(site_id)] = scaler

    X_train: list[np.ndarray] = []
    y_train: list[float] = []
    sites_train: list[int] = []
    times_train: list[pd.Timestamp] = []
    X_test: list[np.ndarray] = []
    y_test: list[float] = []
    sites_test: list[int] = []
    times_test: list[pd.Timestamp] = []

    for site_id, site_data in data.groupby("SCATS Number", sort=True):
        site_id = int(site_id)
        scaler = scalers[site_id]
        site_data = site_data.reset_index(drop=True)
        scaled_values = scaler.transform(site_data[["Traffic Volume"]]).ravel()
        site_index = site_indices[site_id]
        site_features = np.zeros(len(site_ids), dtype=np.float32)
        site_features[site_index] = 1.0
        features = np.column_stack(
            (scaled_values, np.tile(site_features, (len(site_data), 1)))
        ).astype(np.float32)

        group_ids = site_data["Timestamp"].diff().ne(INTERVAL).cumsum()
        for _, segment_indices in site_data.groupby(group_ids, sort=False).groups.items():
            indices = list(segment_indices)
            for position in range(SEQUENCE_LENGTH, len(indices)):
                input_indices = indices[position - SEQUENCE_LENGTH : position]
                target_index = indices[position]
                target_time = site_data.loc[target_index, "Timestamp"]
                sequence = features[input_indices]
                target = float(scaled_values[target_index])

                if target_time < cutoff:
                    X_train.append(sequence)
                    y_train.append(target)
                    sites_train.append(site_id)
                    times_train.append(target_time)
                else:
                    X_test.append(sequence)
                    y_test.append(target)
                    sites_test.append(site_id)
                    times_test.append(target_time)

    if not X_train or not X_test:
        raise ValueError(
            "Could not create both training and test sequences; check site coverage and timestamps."
        )

    train_order = np.argsort(times_train)
    test_order = np.argsort(times_test)
    X_train_array = np.stack(X_train)[train_order]
    X_test_array = np.stack(X_test)[test_order]
    y_train_array = np.asarray(y_train, dtype=np.float32)[train_order].reshape(-1, 1)
    y_test_array = np.asarray(y_test, dtype=np.float32)[test_order].reshape(-1, 1)
    sites_train_array = np.asarray(sites_train, dtype=np.int32)[train_order]
    sites_test_array = np.asarray(sites_test, dtype=np.int32)[test_order]

    sequence_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sequence_path,
        X_train=X_train_array,
        y_train=y_train_array,
        site_ids_train=sites_train_array,
        X_test=X_test_array,
        y_test=y_test_array,
        site_ids_test=sites_test_array,
        site_ids=site_ids.astype(np.int32),
        cutoff=np.asarray(cutoff.isoformat()),
    )
    joblib.dump({"scalers": scalers, "site_ids": site_ids.astype(int).tolist()}, scaler_path)

    print(f"Prepared sequences for {len(site_ids)} SCATS sites.")
    print("Training input:", X_train_array.shape)
    print("Training target:", y_train_array.shape)
    print("Testing input:", X_test_array.shape)
    print("Testing target:", y_test_array.shape)
    print("Training cutoff:", cutoff)
    print("Saved sequences to:", sequence_path)
    print("Saved site scalers to:", scaler_path)
    return X_train_array, X_test_array


if __name__ == "__main__":
    create_sequences()
