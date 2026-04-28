import matplotlib.pyplot as plt
import pandas as pd

def plot_actual_vs_predicted(df, y_true, y_pred, title="Actual vs Predicted"):
    df = df.copy()
    val_index = df.index[-len(y_true):]

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["sales"], label="Actual", marker="o")
    plt.plot(val_index, y_pred, label="Predicted", marker="o")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_future_forecast(df, forecast, steps=8, title="Future Forecast"):
    df = df.copy()

    future_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(weeks=1),
        periods=steps,
        freq="W"
    )

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["sales"], label="Historical", marker="o")
    plt.plot(future_index, forecast, label="Forecast", marker="o")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()