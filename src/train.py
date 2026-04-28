import os
import json
import joblib
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import create_features

MODEL_DIR = "models"


def ensure_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_safe = np.where(y_true < 1, 1, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def train_arima(df):
    df = df.copy()
    df["sales"] = df["sales"].astype(float)

    split = int(len(df) * 0.8)
    train, val = df.iloc[:split], df.iloc[split:]

    model = SARIMAX(train["sales"], order=(5, 1, 1)).fit(disp=False)
    preds = model.forecast(len(val))

    metrics = evaluate(val["sales"], preds)

    final = SARIMAX(df["sales"], order=(5, 1, 1)).fit(disp=False)

    return final, metrics, {}


def train_prophet(df):
    df = df.copy()
    df["sales"] = df["sales"].astype(float)

    pdf = df.reset_index()
    pdf.columns = ["ds", "y"]

    split = int(len(pdf) * 0.8)
    train, val = pdf.iloc[:split], pdf.iloc[split:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=len(val), freq="W")
    forecast = model.predict(future)

    preds = forecast["yhat"].tail(len(val)).values
    metrics = evaluate(val["y"].values, preds)

    final = Prophet()
    final.fit(pdf)

    return final, metrics, {}


def train_xgboost(df):
    df = df.copy()
    df["sales"] = df["sales"].astype(float)

    df_feat = create_features(df)

    df_feat = df_feat.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df_feat = df_feat.dropna()

    X = df_feat.drop("sales", axis=1)
    y = df_feat["sales"]

    split = int(len(X) * 0.8)

    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    metrics = evaluate(y_val.values, preds)

    model.fit(X, y)

    return model, metrics, {
        "last_values": [float(x) for x in df["sales"].tail(30).values]
    }


def train_lstm(df, state):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    df = df.copy()
    df["sales"] = df["sales"].astype(float)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["sales"]])

    seq_len = 30
    X, y = [], []

    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1)),
        Dense(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, verbose=0)

    preds = model.predict(X_val).flatten()
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    metrics = evaluate(y_true, preds)

    model_path = f"{MODEL_DIR}/{state}_lstm.h5"
    scaler_path = f"{MODEL_DIR}/{state}_scaler.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return None, metrics, {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "last_window": scaled[-seq_len:].tolist()
    }


def score(m):
    return m["RMSE"]


def train_all_models(state_data):
    ensure_dir()
    summary = {}

    for state, df in state_data.items():
        results = {}

        for name in ["arima", "prophet", "xgboost"]:
            try:
                model, metrics, info = globals()[f"train_{name}"](df)
                results[name] = {"model": model, "metrics": metrics, "info": info}
            except:
                continue

        try:
            _, metrics, info = train_lstm(df, state)
            results["lstm"] = {"model": None, "metrics": metrics, "info": info}
        except:
            pass

        if not results:
            continue

        best = min(results, key=lambda x: score(results[x]["metrics"]))
        best_model = results[best]

        path = f"{MODEL_DIR}/{state}.pkl"

        if best == "lstm":
            joblib.dump({
                "model_type": best,
                "info": best_model["info"]
            }, path)
        else:
            joblib.dump({
                "model_type": best,
                "model": best_model["model"],
                "info": best_model["info"]
            }, path)

        summary[state] = {"model": best}

    with open(f"{MODEL_DIR}/summary.json", "w") as f:
        json.dump(summary, f)

    return summary