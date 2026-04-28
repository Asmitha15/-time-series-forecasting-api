import os
import joblib
import numpy as np

MODEL_DIR = "models"

def load_model_file(state):
    path = os.path.join(MODEL_DIR, f"{state}.pkl")
    if not os.path.exists(path):
        raise ValueError("Model not found")
    return joblib.load(path)

def forecast_arima(model):
    return list(map(float, model.forecast(8)))

def forecast_prophet(model):
    future = model.make_future_dataframe(periods=8, freq="W")
    forecast = model.predict(future)
    return forecast["yhat"].tail(8).tolist()

def forecast_xgboost(model, info):
    import numpy as np

    history = [float(x) for x in info["last_values"]]
    preds = []

    trend = len(history)

    for _ in range(8):
        lag_1 = float(history[-1])
        lag_7 = float(history[-7]) if len(history) >= 7 else lag_1
        lag_30 = float(history[-30]) if len(history) >= 30 else lag_1

        rolling_mean_7 = float(np.mean(history[-7:]))
        rolling_std_7 = float(np.std(history[-7:])) if len(history) >= 7 else 0.0

        features = np.array([[
            lag_1,
            lag_7,
            lag_30,
            rolling_mean_7,
            rolling_std_7,
            0.0,
            0.0,
            0.0,
            float(trend)
        ]], dtype=float)

        pred = float(model.predict(features)[0])

        pred = max(pred, 0.0)

        recent_mean = float(np.mean(history[-10:]))
        pred = min(pred, recent_mean * 1.15)

        preds.append(pred)
        history.append(pred)

        trend += 1

    return preds
def forecast_lstm(info):
    from tensorflow.keras.models import load_model

    model = load_model(info["model_path"])
    scaler = joblib.load(info["scaler_path"])

    window = np.array(info["last_window"]).reshape(1, -1, 1)
    preds = []

    for _ in range(8):
        pred = model.predict(window, verbose=0)[0][0]
        preds.append(pred)
        window = np.append(window[:, 1:, :], [[[pred]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds.tolist()

def forecast_state(state):
    artifact = load_model_file(state)

    if artifact["model_type"] == "prophet":
        base = forecast_prophet(artifact["model"])
        return base

    if artifact["model_type"] == "xgboost":
        return forecast_xgboost(artifact["model"], artifact["info"])

    if artifact["model_type"] == "arima":
        return forecast_arima(artifact["model"])

    if artifact["model_type"] == "lstm":
        return forecast_lstm(artifact["info"])