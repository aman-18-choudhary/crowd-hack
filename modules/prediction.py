# modules/prediction.py

import os
import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = "models/gru_model.keras"
SCALER_PATH = "models/gru_scaler.pkl"
SEQ_LEN = 10


class CrowdPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None

    # =====================================
    # LOAD MODEL
    # =====================================
    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.model = load_model(MODEL_PATH, compile=False)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            print("✅ GRU model loaded.")
        else:
            print("⚠ No trained model found.")

    # =====================================
    # TRAIN MODEL
    # =====================================
    def train(self, data):
        """
        data format:
        [
            [density, flow],
            [density, flow],
            ...
        ]
        """

        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(np.array(data))

        X, y = [], []

        for i in range(SEQ_LEN, len(scaled)):
            X.append(scaled[i - SEQ_LEN:i])
            y.append(scaled[i])

        X, y = np.array(X), np.array(y)

        model = Sequential([
            Input(shape=(SEQ_LEN, X.shape[2])),
            GRU(64),
            Dense(2)   # 2 outputs → density & flow
        ])

        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=15, batch_size=8, verbose=1)

        os.makedirs("models", exist_ok=True)
        model.save(MODEL_PATH)

        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

        self.model = model
        print("✅ GRU training complete.")

    # =====================================
    # PREDICT
    # =====================================
    def predict(self, recent_data):
        """
        recent_data format:
        [
            [density, flow],
            [density, flow],
            ...
        ]
        Must contain at least SEQ_LEN entries.
        """

        if len(recent_data) < SEQ_LEN:
            return recent_data[-1]  # fallback

        # Take last SEQ_LEN rows
        data = np.array(recent_data[-SEQ_LEN:])

        # Ensure correct 2D shape
        if data.ndim == 1:
            data = data.reshape(-1, 2)

        # Scale
        scaled = self.scaler.transform(data)

        # Reshape to (1, SEQ_LEN, 2)
        scaled = scaled.reshape(1, SEQ_LEN, 2)

        # Predict
        prediction = self.model.predict(scaled, verbose=0)

        # Inverse scale
        prediction = self.scaler.inverse_transform(prediction)

        # Return density & flow separately
        predicted_density = float(prediction[0][0])
        predicted_flow = float(prediction[0][1])

        return predicted_density, predicted_flow