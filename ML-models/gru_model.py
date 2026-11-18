"""Train the shared multi-site GRU model and provide one-step predictions."""

from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GRU, Input
from tensorflow.keras.models import Sequential, load_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "combined_traffic_timeseries.csv"
SEQUENCE_PATH = ROOT / "data" / "traffic_sequences.npz"
SCALER_PATH = ROOT / "models" / "site_scalers.joblib"
MODEL_PATH = ROOT / "models" / "gru_traffic_model.keras"
METRICS_PATH = ROOT / "results" / "gru_metrics.json"

SEQUENCE_LENGTH = 4
INTERVAL = pd.Timedelta(minutes=15)
EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
GRU_UNIT_CANDIDATES = (32, 64)


def _load_sequences(sequence_path: Path) -> dict[str, np.ndarray]:
    """Read and validate the shared chronological train/test arrays."""
    required = {
        "X_train", "y_train", "X_test", "y_test", "site_ids", "site_ids_test"
    }
    with np.load(sequence_path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Sequence file is missing required arrays: {sorted(missing)}. "
                "Regenerate it with preprocessing/create_sequences.py."
            )
        arrays = {name: np.asarray(data[name]) for name in required}

    X_train = np.asarray(arrays["X_train"], dtype=np.float32)
    y_train = np.asarray(arrays["y_train"], dtype=np.float32).reshape(-1)
    X_test = np.asarray(arrays["X_test"], dtype=np.float32)
    y_test = np.asarray(arrays["y_test"], dtype=np.float32).reshape(-1)
    site_ids = np.asarray(arrays["site_ids"], dtype=np.int32).reshape(-1)
    site_ids_test = np.asarray(arrays["site_ids_test"], dtype=np.int32).reshape(-1)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Inputs must have shape (samples, timesteps, features).")
    if X_train.shape[1] != SEQUENCE_LENGTH or X_test.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"Each input must contain {SEQUENCE_LENGTH} time steps.")
    if X_train.shape[2] != len(site_ids) + 1 or X_test.shape[2] != len(site_ids) + 1:
        raise ValueError(
            "Sequence feature width must equal one traffic feature plus one "
            "one-hot feature per SCATS site. Regenerate the shared sequences."
        )
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Each input sequence must have exactly one target.")
    if len(y_test) != len(site_ids_test):
        raise ValueError("Each test target must have a corresponding SCATS site ID.")
    if not len(X_train) or not len(X_test):
        raise ValueError("The shared training and test sets must both be non-empty.")
    if not all(np.isfinite(a).all() for a in (X_train, y_train, X_test, y_test)):
        raise ValueError("The shared sequence data contains NaN or infinite values.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "site_ids": site_ids,
        "site_ids_test": site_ids_test,
    }


def _build_model(input_shape: tuple[int, int], gru_units: int) -> Sequential:
    """Create the GRU architecture using the shared sequence dimensions."""
    model = Sequential(
        [
            Input(shape=input_shape),
            GRU(gru_units, activation="tanh"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ],
        name="traffic_gru",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _raw_scale_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    site_ids: np.ndarray,
    scaler_bundle: dict[str, Any],
) -> dict[str, float]:
    """Calculate MSE/RMSE/MAE after inverse-scaling each site's readings."""
    scalers = scaler_bundle["scalers"]
    y_true = np.empty_like(y_true_scaled, dtype=np.float64)
    y_pred = np.empty_like(y_pred_scaled, dtype=np.float64)

    for site_id in np.unique(site_ids):
        mask = site_ids == site_id
        scaler = scalers.get(int(site_id))
        if scaler is None:
            raise ValueError(f"No fitted scaler was saved for SCATS site {site_id}.")
        y_true[mask] = scaler.inverse_transform(
            y_true_scaled[mask].reshape(-1, 1)
        ).ravel()
        y_pred[mask] = scaler.inverse_transform(
            y_pred_scaled[mask].reshape(-1, 1)
        ).ravel()

    errors = y_pred - y_true
    return {
        "mse": float(np.mean(np.square(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
    }


def train_gru(
    sequence_path: Path = SEQUENCE_PATH,
    scaler_path: Path = SCALER_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> Path:
    """Train on the shared multi-site split, save the GRU, and log metrics."""
    arrays = _load_sequences(sequence_path)
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Fitted site scalers were not found at {scaler_path}. "
            "Regenerate them with preprocessing/create_sequences.py."
        )
    scaler_bundle = joblib.load(scaler_path)

    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]

    candidates: list[tuple[float, int, Sequential, dict[str, Any]]] = []
    tuning_start = time.perf_counter()
    for gru_units in GRU_UNIT_CANDIDATES:
        tf.keras.utils.set_random_seed(42)
        candidate = _build_model(X_train.shape[1:], gru_units)
        history = candidate.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            shuffle=False,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        best_validation_loss = float(min(history.history["val_loss"]))
        candidate_summary = {
            "gru_units": int(gru_units),
            "epochs_trained": int(len(history.history["loss"])),
            "best_validation_mse": best_validation_loss,
        }
        candidates.append(
            (best_validation_loss, gru_units, candidate, candidate_summary)
        )
        print(
            f"GRU tuning candidate: units={gru_units}, "
            f"epochs={candidate_summary['epochs_trained']}, "
            f"best validation MSE={best_validation_loss:.6f}"
        )

    training_seconds = time.perf_counter() - tuning_start
    _, selected_units, model, _ = min(candidates, key=lambda item: item[0])
    print(f"Selected GRU units using validation MSE: {selected_units}")
    model.summary()

    # Warm up the model before measuring prediction latency.
    model.predict(X_test[:1], verbose=0)
    inference_start = time.perf_counter()
    y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    inference_seconds = time.perf_counter() - inference_start

    normalized_errors = y_pred_scaled - y_test
    normalized_mse = float(np.mean(np.square(normalized_errors)))
    normalized_metrics = {
        "mse": normalized_mse,
        "rmse": float(np.sqrt(normalized_mse)),
        "mae": float(np.mean(np.abs(normalized_errors))),
    }
    traffic_metrics = _raw_scale_metrics(
        y_test,
        y_pred_scaled,
        arrays["site_ids_test"],
        scaler_bundle,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    result = {
        "model": "GRU",
        "dataset": str(sequence_path.relative_to(ROOT)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "scats_sites": int(len(arrays["site_ids"])),
        "sequence_length": SEQUENCE_LENGTH,
        "input_features": int(X_train.shape[2]),
        "selected_gru_units": int(selected_units),
        "dropout": 0.2,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "tuning_candidates": [candidate[3] for candidate in candidates],
        "epochs_per_candidate_max": EPOCHS,
        "training_seconds": float(training_seconds),
        "inference_seconds_total": float(inference_seconds),
        "inference_ms_per_sample": float(inference_seconds * 1000 / len(X_test)),
        "normalized_metrics": normalized_metrics,
        "traffic_volume_metrics": traffic_metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nGRU test results (normalized target scale)")
    print(f"MSE:  {normalized_metrics['mse']:.6f}")
    print(f"RMSE: {normalized_metrics['rmse']:.6f}")
    print(f"MAE:  {normalized_metrics['mae']:.6f}")
    print("\nGRU test results (original traffic-volume scale)")
    print(f"MSE:  {traffic_metrics['mse']:.6f}")
    print(f"RMSE: {traffic_metrics['rmse']:.6f}")
    print(f"MAE:  {traffic_metrics['mae']:.6f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Inference: {result['inference_ms_per_sample']:.4f} ms/sample")
    print(f"GRU model saved to: {model_path}")
    print(f"GRU metrics saved to: {metrics_path}")
    return model_path


@lru_cache(maxsize=2)
def _load_gru(model_path: str):
    """Cache the saved GRU so a GUI can make repeated predictions efficiently."""
    return load_model(model_path)


@lru_cache(maxsize=2)
def _load_scaler_bundle(scaler_path: str) -> dict[str, Any]:
    """Cache per-site scalers and the site ordering used for one-hot inputs."""
    bundle = joblib.load(scaler_path)
    bundle["site_ids"] = [int(site_id) for site_id in bundle["site_ids"]]
    bundle["site_index"] = {
        site_id: index for index, site_id in enumerate(bundle["site_ids"])
    }
    bundle["scalers"] = {
        int(site_id): scaler for site_id, scaler in bundle["scalers"].items()
    }
    return bundle


@lru_cache(maxsize=2)
def _load_traffic_data(data_path: str) -> pd.DataFrame:
    """Cache the prepared traffic readings used to build prediction windows."""
    data = pd.read_csv(data_path, parse_dates=["Timestamp"])
    required = {"SCATS Number", "Timestamp", "Traffic Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Traffic data is missing columns: {sorted(missing)}")
    data["SCATS Number"] = pd.to_numeric(data["SCATS Number"], errors="raise").astype(int)
    data["Traffic Volume"] = pd.to_numeric(data["Traffic Volume"], errors="raise")
    return data.sort_values(["SCATS Number", "Timestamp"]).reset_index(drop=True)


def predict_flow(
    site_id: int,
    timestamp: str | pd.Timestamp,
    data_path: Path = DATA_PATH,
    scaler_path: Path = SCALER_PATH,
    model_path: Path = MODEL_PATH,
) -> float:
    """Predict a site's traffic volume for one 15-minute interval.

    Args:
        site_id: SCATS site identifier.
        timestamp: Target interval timestamp. The four preceding 15-minute
            readings must exist in the prepared traffic data.

    Returns:
        Predicted traffic volume in the original dataset's units.
    """
    target_time = pd.Timestamp(timestamp)
    if target_time.tz is not None:
        target_time = target_time.tz_localize(None)

    bundle = _load_scaler_bundle(str(scaler_path.resolve()))
    site_id = int(site_id)
    if site_id not in bundle["site_index"] or site_id not in bundle["scalers"]:
        raise ValueError(f"SCATS site {site_id} is not present in the trained model.")

    data = _load_traffic_data(str(data_path.resolve()))
    site_data = data.loc[data["SCATS Number"] == site_id]
    history = site_data.loc[site_data["Timestamp"] < target_time].tail(SEQUENCE_LENGTH)
    if len(history) != SEQUENCE_LENGTH:
        raise ValueError(
            f"SCATS site {site_id} needs {SEQUENCE_LENGTH} earlier readings "
            f"before {target_time}; found {len(history)}."
        )
    if not history["Timestamp"].diff().iloc[1:].eq(INTERVAL).all():
        raise ValueError("The four input readings must be consecutive 15-minute intervals.")
    if target_time - history["Timestamp"].iloc[-1] != INTERVAL:
        raise ValueError("The requested prediction must be one 15-minute step after history.")

    scaler = bundle["scalers"][site_id]
    scaled_flow = scaler.transform(history[["Traffic Volume"]]).astype(np.float32)
    site_encoding = np.zeros((SEQUENCE_LENGTH, len(bundle["site_ids"])), dtype=np.float32)
    site_encoding[:, bundle["site_index"][site_id]] = 1.0
    features = np.concatenate((scaled_flow, site_encoding), axis=1)[None, ...]

    model = _load_gru(str(model_path.resolve()))
    expected_features = model.input_shape[-1]
    if expected_features != features.shape[-1]:
        raise ValueError(
            "Saved GRU does not match the current shared data/scalers. "
            "Retrain it with preprocessing/create_sequences.py first."
        )
    prediction_scaled = float(model.predict(features, verbose=0).reshape(-1)[0])
    prediction = float(
        scaler.inverse_transform(np.asarray([[prediction_scaled]], dtype=np.float32))[0, 0]
    )
    return max(0.0, prediction)


if __name__ == "__main__":
    train_gru()
