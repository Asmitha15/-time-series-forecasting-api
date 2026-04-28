import sys
import os
sys.path.append(os.path.abspath("."))

from src.data_loader import load_and_prepare_data
from src.predict import forecast_state
from src.visualization import plot_future_forecast

data = load_and_prepare_data("data/forecasting_data.xlsx")

state = "Texas"

df = data[state]
forecast = forecast_state(state)

plot_future_forecast(df, forecast)