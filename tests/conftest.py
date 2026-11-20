from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_PATH = ROOT / "data" / "traffic_sequences.npz"
SCALER_PATH = ROOT / "models" / "site_scalers.joblib"
TRAFFIC_PATH = ROOT / "data" / "combined_traffic_timeseries.csv"
MODEL_PATHS = {
    "GRU": ROOT / "models" / "gru_traffic_model.keras",
    "LSTM": ROOT / "models" / "lstm_traffic_model.keras",
}


def _inverse_scale_by_site(
    values: np.ndarray,
    site_ids: np.ndarray,
    scalers: dict,
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for site_id in np.unique(site_ids):
        mask = site_ids == site_id
        scaler = scalers.get(int(site_id))
        if scaler is None:
            pytest.fail(f"No fitted scaler is available for SCATS site {site_id}.")
        result[mask] = scaler.inverse_transform(
            values[mask].reshape(-1, 1)
        ).reshape(-1)
    return result


@pytest.fixture(scope="session")
def model_evaluation():
    required_paths = (SEQUENCE_PATH, SCALER_PATH, TRAFFIC_PATH, *MODEL_PATHS.values())
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        pytest.fail(
            "Cannot evaluate the trained traffic models; required artifacts are "
            f"missing: {', '.join(missing_paths)}"
        )

    with np.load(SEQUENCE_PATH, allow_pickle=False) as data:
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=np.float32).reshape(-1)
        site_ids_test = np.asarray(data["site_ids_test"], dtype=np.int32).reshape(-1)
        cutoff = pd.Timestamp(data["cutoff"].item())

    scalers = joblib.load(SCALER_PATH)["scalers"]
    actual = _inverse_scale_by_site(y_test, site_ids_test, scalers)
    naive_scaled = X_test[:, -1, 0]
    naive = _inverse_scale_by_site(naive_scaled, site_ids_test, scalers)

    predictions = {}
    prediction_shapes = {}
    input_shapes = {}
    for model_name, model_path in MODEL_PATHS.items():
        model = tf.keras.models.load_model(model_path)
        input_shapes[model_name] = tuple(model.input_shape[1:])
        if input_shapes[model_name] != tuple(X_test.shape[1:]):
            pytest.fail(
                f"{model_name} expects {input_shapes[model_name]}, but the shared "
                f"test sequences have shape {tuple(X_test.shape[1:])}."
            )
        scaled_predictions = np.asarray(
            model.predict(X_test, batch_size=512, verbose=0)
        )
        prediction_shapes[model_name] = scaled_predictions.shape
        predictions[model_name] = _inverse_scale_by_site(
            scaled_predictions.reshape(-1), site_ids_test, scalers
        )

    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["Timestamp"])
    traffic = traffic.sort_values(["SCATS Number", "Timestamp"]).reset_index(drop=True)
    train_target_times = []
    test_target_times = []
    for _, site_data in traffic.groupby("SCATS Number", sort=True):
        site_data = site_data.reset_index(drop=True)
        segments = site_data["Timestamp"].diff().ne(pd.Timedelta(minutes=15)).cumsum()
        for _, indices in site_data.groupby(segments, sort=False).groups.items():
            positions = list(indices)
            for position in range(4, len(positions)):
                target_time = site_data.loc[positions[position], "Timestamp"]
                if target_time < cutoff:
                    train_target_times.append(target_time)
                else:
                    test_target_times.append(target_time)

    train_order = np.argsort(train_target_times)
    test_order = np.argsort(test_target_times)
    train_target_times = np.asarray(train_target_times)[train_order]
    test_target_times = np.asarray(test_target_times)[test_order]
    if len(train_target_times) != len(y_train) or len(test_target_times) != len(y_test):
        pytest.fail(
            "Could not align source timestamps with the saved train/test sequences."
        )

    training_traffic = traffic.loc[traffic["Timestamp"] < cutoff].copy()
    training_traffic["hour"] = training_traffic["Timestamp"].dt.hour
    hourly_means = training_traffic.groupby("hour")["Traffic Volume"].mean()
    peak_threshold = hourly_means.quantile(0.75)
    peak_hours = hourly_means.loc[hourly_means >= peak_threshold].index.to_numpy()
    peak_mask = pd.DatetimeIndex(test_target_times).hour.isin(peak_hours)

    return {
        "X_test": X_test,
        "y_test": y_test,
        "site_ids_test": site_ids_test,
        "actual": actual,
        "naive": naive,
        "predictions": predictions,
        "prediction_shapes": prediction_shapes,
        "input_shapes": input_shapes,
        "train_target_times": train_target_times,
        "test_target_times": test_target_times,
        "cutoff": cutoff,
        "peak_hours": peak_hours,
        "peak_mask": np.asarray(peak_mask),
    }
