import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()

    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_30"] = df["sales"].shift(30)

    df["rolling_mean_7"] = df["sales"].rolling(7).mean()
    df["rolling_std_7"] = df["sales"].rolling(7).std()

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    holidays = pd.to_datetime([
        "2023-01-01",
        "2023-07-04",
        "2023-12-25"
    ])
    df["is_holiday"] = df.index.isin(holidays).astype(int)

    df["trend"] = np.arange(len(df))

    return df.dropna()