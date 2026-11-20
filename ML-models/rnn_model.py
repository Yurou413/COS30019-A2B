from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    SimpleRNN,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Sequential


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "site2000_sequences.npz"
MODEL_PATH = ROOT_DIR / "models" / "rnn_traffic_model.keras"

EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def load_sequences(path):
    with np.load(path, allow_pickle=False) as data:
        required_arrays = {"X_train", "y_train", "X_test", "y_test"}
        missing_arrays = required_arrays.difference(data.files)
        if missing_arrays:
            raise ValueError(
                f"Sequence file is missing arrays: {sorted(missing_arrays)}"
            )

        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.float32)
        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=np.float32)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "Inputs must have shape (samples, timesteps, features)"
        )

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    if y_train.ndim != 2 or y_train.shape[1] != 1:
        raise ValueError(
            "Training targets must have shape (samples, 1)"
        )

    if y_test.ndim != 2 or y_test.shape[1] != 1:
        raise ValueError(
            "Test targets must have shape (samples, 1)"
        )

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(
            "Each input sequence must have exactly one target"
        )

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Training and test input shapes must match"
        )

    if not len(X_train) or not len(X_test):
        raise ValueError(
            "Training and test sets must both contain samples"
        )

    if not all(
        np.isfinite(array).all()
        for array in (X_train, y_train, X_test, y_test)
    ):
        raise ValueError(
            "Sequence data contains NaN or infinite values"
        )

    return X_train, y_train, X_test, y_test


def build_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),

            SimpleRNN(
                32,
                activation="tanh",
                return_sequences=True,
            ),

            SimpleRNN(
                64,
                activation="tanh",
            ),

            Dense(32, activation="relu"),

            Dropout(0.2),

            Dense(1),
        ],
        name="traffic_rnn",
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


def main():
    tf.keras.utils.set_random_seed(42)

    X_train, y_train, X_test, y_test = load_sequences(DATA_PATH)

    if len(X_train) < 2:
        raise ValueError(
            "At least two training samples are needed for validation"
        )

    model = build_model(X_train.shape[1:])
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1,
    )

    test_loss, test_mae = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print("\nRNN Test Results")
    print("Test Loss (MSE):", test_loss)
    print("Test MAE:", test_mae)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(MODEL_PATH)

    print("\nRNN model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
