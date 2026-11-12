
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load prepared training data
data = np.load("data/site2000_sequences.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]

# Build LSTM model
model = Sequential([
    LSTM(64, activation="tanh",
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Display model structure
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    shuffle=False
)

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test)

print("\nLSTM Test Results")
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)

# Save trained model
model.save("models/lstm_traffic_model.keras")

print("\nLSTM model saved successfully!")
