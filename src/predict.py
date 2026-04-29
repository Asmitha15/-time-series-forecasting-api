import os
import joblib
import numpy as np

MODEL_DIR = "models"


def load_model_file(state):
    return joblib.load(f"{MODEL_DIR}/{state}.pkl")


def forecast_arima(model):
    return list(map(float, model.forecast(8)))


def forecast_prophet(model):
    future = model.make_future_dataframe(periods=8, freq="W")
    forecast = model.predict(future)
    return forecast["yhat"].tail(8).tolist()


def forecast_xgboost(model, info):
    history = info.get("last_values", []).copy()

    if len(history) < 10:
        return [float(np.mean(history))] * 8

    preds = []

    for _ in range(8):
        lag_1 = history[-1]
        lag_7 = history[-7] if len(history) >= 7 else lag_1
        lag_30 = history[-30] if len(history) >= 30 else lag_1

        rolling_mean_7 = np.mean(history[-7:])
        rolling_std_7 = np.std(history[-7:])

        features = [[
            lag_1,
            lag_7,
            lag_30,
            rolling_mean_7,
            rolling_std_7,
            0,
            0,
            0
        ]]

        pred = model.predict(features)[0]

        pred = max(pred, 0)
        pred = min(pred, np.mean(history[-10:]) * 1.2)

        preds.append(float(pred))
        history.append(pred)

    return preds


def forecast_lstm(info):
    from tensorflow.keras.models import load_model

    if not os.path.exists(info.get("model_path", "")):
        return [0.0] * 8

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

    model_type = artifact.get("model_type")
    model = artifact.get("model")
    info = artifact.get("info", {})

    try:
        if model_type == "arima":
            return forecast_arima(model)

        elif model_type == "prophet":
            return forecast_prophet(model)

        elif model_type == "xgboost":
            return forecast_xgboost(model, info)

        elif model_type == "lstm":
            return forecast_lstm(info)

    except Exception as e:
        print(f"Error in {state} ({model_type}):", e)

    return [float(np.mean(info.get("last_values", [0]))) for _ in range(8)]