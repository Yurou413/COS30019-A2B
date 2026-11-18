from pathlib import Path

import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_PATH = ROOT / "data" / "traffic_sequences.npz"
MODEL_PATH = ROOT / "models" / "lstm_traffic_model.keras"


def train_lstm(
    sequence_path: Path = SEQUENCE_PATH,
    model_path: Path = MODEL_PATH,
) -> Path:
    """Train and save the multi-site LSTM model."""
    with np.load(sequence_path) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

    model = Sequential(
        [
            LSTM(64, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        shuffle=False,
    )

    test_loss, test_mae = model.evaluate(X_test, y_test)
    print("\nLSTM Test Results")
    print("Test Loss (MSE):", test_loss)
    print("Test MAE:", test_mae)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print("\nLSTM model saved to:", model_path)
    return model_path


if __name__ == "__main__":
    train_lstm()
