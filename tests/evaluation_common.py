"""Shared numerical evaluation utilities for 15 traffic-model test cases.

Run an individual test from the project root, for example:
    python tests/tc01_gru_accuracy.py

This module does not retrain models or modify saved artifacts.
"""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "traffic_sequences.npz"
SCALERS = ROOT / "models" / "site_scalers.joblib"
MODELS = {
    "GRU": ROOT / "models" / "gru_traffic_model.keras",
    "LSTM": ROOT / "models" / "lstm_traffic_model.keras",
}


def field(data, *names):
    for name in names:
        if name in data.files:
            return data[name]
    return None


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")
    if len(actual) < 1:
        return None
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Actual/predicted arrays contain NaN or infinite values")
    err = predicted - actual
    mse = mean_squared_error(actual, predicted)
    return dict(
        n=len(actual),
        MAE=mean_absolute_error(actual, predicted),
        MSE=mse,
        RMSE=np.sqrt(mse),
        R2=r2_score(actual, predicted) if len(actual) > 1 and np.var(actual) > 0 else np.nan,
        mean_error=np.mean(err),
        median_abs_error=np.median(np.abs(err)),
        p95_abs_error=np.percentile(np.abs(err), 95),
        max_abs_error=np.max(np.abs(err)),
        over_pct=100 * np.mean(err > 0),
        under_pct=100 * np.mean(err < 0),
        negative_n=int(np.sum(predicted < 0)),
        min_pred=np.min(predicted),
    )


def display(number, title, rows):
    print(f"\nTC{number:02d} — {title}")
    if rows is None:
        print("NOT AVAILABLE — required metadata or comparable data is missing")
    elif isinstance(rows, pd.DataFrame):
        print(rows.round(4).to_string(index=False))
    else:
        for key, value in rows.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"  {key}: {value}")


def _load_scaler_bundle(path=SCALERS):
    """Read the project's {'scalers': dict, 'site_ids': list} format.

    Also accepts a legacy direct mapping of SCATS ID -> scaler.
    """
    bundle = joblib.load(path)  # Only load trusted local project artifacts.
    if not isinstance(bundle, dict):
        raise ValueError("site_scalers.joblib must contain a dictionary")

    if "scalers" in bundle and "site_ids" in bundle:
        raw_scalers = bundle["scalers"]
        saved_site_ids = bundle["site_ids"]
    else:
        raw_scalers = bundle
        saved_site_ids = list(bundle.keys())

    if not isinstance(raw_scalers, dict):
        raise ValueError("The 'scalers' entry must be a dictionary keyed by SCATS ID")
    try:
        scalers = {int(key): scaler for key, scaler in raw_scalers.items()}
        site_order = [int(site) for site in saved_site_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("SCATS scaler keys/site_ids must be convertible to integers") from exc

    if len(site_order) != len(set(site_order)):

        raise ValueError("Duplicate SCATS IDs in saved site_ids")
    if set(site_order) != set(scalers):
        missing = sorted(set(site_order) - set(scalers))
        extra = sorted(set(scalers) - set(site_order))
        raise ValueError(f"Scaler/site_ids mismatch: missing={missing}, extra={extra}")
    return scalers, site_order


def invert(values, sites, scalers):
    """Inverse-transform normalized traffic volume using each site's fitted scaler."""
    values = np.asarray(values, dtype=float).reshape(-1)
    sites = np.asarray(sites).reshape(-1)
    if len(values) != len(sites):
        raise ValueError("Values and SCATS site IDs must have the same length")
    result = np.empty_like(values)
    for site in np.unique(sites):
        key = int(site)
        scaler = scalers.get(key)
        if scaler is None:
            scaler = scalers.get(str(key))
        if scaler is None:
            raise ValueError(f"No fitted scaler for SCATS site {key}")
        if not hasattr(scaler, "inverse_transform"):
            raise ValueError(f"Scaler for SCATS site {key} has no inverse_transform method")
        if getattr(scaler, "n_features_in_", 1) != 1:
            raise ValueError(
                f"Scaler for {key} has multiple features: inspect preprocessing before inverse transform"
            )
        idx = np.flatnonzero(sites == site)
        result[idx] = scaler.inverse_transform(values[idx, None]).reshape(-1)
    return result


def site_ids_from_data(data, X, scalers, site_order=None):
    explicit = field(data, "site_ids_test", "test_site_ids", "scats_ids_test", "site_test")
    if explicit is not None:
        return np.asarray(explicit).astype(int).reshape(-1)

    # README: traffic feature first, then one-hot SCATS site ID.
    # Use the saved encoding order rather than assuming sorted keys.
    keys = list(site_order) if site_order is not None else sorted(int(k) for k in scalers)
    if X.ndim == 3 and X.shape[2] == len(keys) + 1:
        encoded = X[:, -1, 1:]
        if not (
            np.isclose(encoded.sum(axis=1), 1).all()
            and np.isclose(encoded.max(axis=1), 1).all()
            and np.isclose(encoded.min(axis=1), 0).all()
        ):
            raise ValueError("Site features do not appear to be valid one-hot vectors")
        return np.asarray(keys, dtype=int)[np.argmax(encoded, axis=1)]

    raise ValueError(
        "Cannot determine per-sample SCATS IDs. Inspect create_sequences.py "
        "and add explicit site_ids_test metadata."
    )


def load_evaluation():
    for path in [DATA, SCALERS, *MODELS.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Required project artifact missing: {path}")

    scalers, site_order = _load_scaler_bundle()
    with np.load(DATA, allow_pickle=False) as data:
        X = field(data, "X_test")
        y = field(data, "y_test")
        if X is None or y is None:
            raise ValueError("NPZ must contain X_test and y_test")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float).reshape(-1)
        sites = site_ids_from_data(data, X, scalers, site_order)
        timestamps = field(data, "timestamps_test", "test_timestamps", "timestamp_test")
        if timestamps is not None:
            timestamps = pd.to_datetime(np.asarray(timestamps).reshape(-1), errors="coerce")

    if len(X) == 0:
        raise ValueError("X_test is empty")
    if len(sites) != len(y) or len(X) != len(y):
        raise ValueError("X_test, y_test, and site IDs must align")
    if timestamps is not None and len(timestamps) != len(y):
        raise ValueError("Test timestamps and y_test must align")
    if X.ndim != 3 or X.shape[2] < 1:
        raise ValueError(f"Expected 3D model input with traffic feature at index 0, got {X.shape}")

    actual = invert(y, sites, scalers)
    baseline = invert(X[:, -1, 0], sites, scalers)
    predictions = {}
    latency = {}

    for name, path in MODELS.items():
        model = tf.keras.models.load_model(path, compile=False)
        model_shape = model.input_shape
        if isinstance(model_shape, list):
            raise ValueError(f"{name} has multiple inputs; adapt the evaluator to its input signature")
        if tuple(model_shape[1:]) != tuple(X.shape[1:]):
            raise ValueError(f"{name} input mismatch: model {model_shape}, data {X.shape}")

        _ = model.predict(X[:min(32, len(X))], verbose=0)  # Warm-up
        times = []
        output = None
        for _ in range(5):
            start = time.perf_counter()
            output = model.predict(X, batch_size=32, verbose=0).reshape(-1)
            times.append(1000 * (time.perf_counter() - start) / len(X))
        if output is None or len(output) != len(y):
            raise ValueError(f"{name} prediction count does not match y_test")
        predictions[name] = invert(output, sites, scalers)
        latency[name] = np.asarray(times)

    M = {name: metrics(actual, pred) for name, pred in predictions.items()}
    base = metrics(actual, baseline)
    return actual, predictions, sites, timestamps, latency, M, base
