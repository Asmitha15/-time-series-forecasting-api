import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove NaNs safely
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": 0, "RMSE": 0, "MAPE": 0}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    
    non_zero = y_true > 1
    if non_zero.any():
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    else:
        mape = 0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }



def plot_actual_vs_forecast(
    history: pd.Series,
    forecast: pd.Series,
    title: str = "Actual vs Forecast",
    save_path: str = None,
):
    if history.empty or forecast.empty:
        print("No data to plot")
        return

    plt.figure(figsize=(12, 6))

    plt.plot(history.index, history.values, label="Actual")
    plt.plot(forecast.index, forecast.values, label="Predicted")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()



def plot_future_forecast(
    history: pd.Series,
    future_values: list,
    freq: str = "W",  # IMPORTANT: weekly data
    title: str = "Future Forecast",
    save_path: str = None,
):
    if history.empty or len(future_values) == 0:
        print("No data to plot")
        return

    future_index = pd.date_range(
        start=history.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=len(future_values),
        freq=freq,
    )

    future_series = pd.Series(future_values, index=future_index)

    plt.figure(figsize=(12, 6))

    plt.plot(history.index, history.values, label="Actual")
    plt.plot(future_series.index, future_series.values, label="Forecast")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()
