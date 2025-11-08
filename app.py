# app.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="LSTM Sequence Predictor")

# ==== Request Model ====
class SequenceInput(BaseModel):
    sequence: list

# ==== Helper function to train and predict ====
def train_and_predict(user_seq):
    if len(user_seq) < 4:
        raise ValueError("Sequence must have at least 4 numbers.")

    data = np.array(user_seq)
    X, y = [], []
    for i in range(len(data) - 3):
        X.append(data[i:i+3])
        y.append(data[i+3])
    X, y = np.array(X), np.array(y)

    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(64, input_shape=(3, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y_scaled, epochs=200, verbose=0)

    # Predict next number
    test_input = np.array(user_seq[-3:]).reshape(1, 3, 1)
    test_scaled = scaler_X.transform(test_input.reshape(-1,1)).reshape(test_input.shape)
    pred_scaled = model.predict(test_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)

    return float(pred[0][0])

# ==== API Endpoint ====
@app.post("/predict")
def predict_next(data: SequenceInput):
    try:
        result = train_and_predict(data.sequence)
        return {"input_sequence": data.sequence, "predicted_next_number": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==== Run server ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
