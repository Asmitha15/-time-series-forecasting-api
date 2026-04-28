from src.data_loader import load_and_prepare_data
from src.train import train_all_models

data = load_and_prepare_data("data/forecasting_data.xlsx")
train_all_models(data)