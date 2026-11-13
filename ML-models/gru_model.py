from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.models import Sequential


# Load the same chronological sequences used by the LSTM model.
data = np.load("data/site2000_sequences.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Build a GRU model with the same layer sizes as the LSTM for comparison.
model = Sequential([
    GRU(
        64,
        activation="tanh",
        input_shape=(X_train.shape[1], X_train.shape[2]),
    ),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
)

test_loss, test_mae = model.evaluate(X_test, y_test)

print("\nGRU Test Results")
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "gru_traffic_model.keras"
model.save(model_path)

print("\nGRU model saved to:", model_path)
