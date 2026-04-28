# End-to-End Time Series Forecasting System with API

## Overview
This project is a production-ready Time Series Forecasting system that predicts the next 8 weeks of sales for each state using historical data.

The system:
- Trains multiple forecasting models
- Compares performance automatically
- Selects the best model
- Serves predictions through a REST API


## Problem Statement
Forecast the next 8 weeks of sales for each state while:
- Handling missing values and missing dates
- Capturing trend and seasonality
- Avoiding data leakage
- Providing predictions via API


## Project Structure

time_series_api/
│
├── data/
│   └── dataset.xlsx
│
├── models/
│   ├── Texas.pkl
│   └── summary.json
│
├── src/
│   ├── load_data.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│
├── api/
│   └── main.py
│
├── train_runner.py
├── visualize_runner.py
├── requirements.txt
└── README.md


## Data Processing
- Automatically detects and cleans dataset columns
- Converts date column to datetime format
- Sorts data by time
- Handles missing values using interpolation
- Resamples data to weekly frequency
- Ensures continuous time series per state


## Feature Engineering

### Lag Features
- lag_1 (t-1)
- lag_7 (t-7)
- lag_30 (t-30)

### Rolling Features
- rolling_mean_7
- rolling_std_7

### Time Features
- day_of_week
- month
- week_of_year
- is_weekend
- is_holiday

### Trend Feature
- trend index to capture growth patterns


## Train / Validation Split
- Time-based split (no random split)
- Prevents data leakage
- Uses most recent data for validation


## Models Implemented
1. ARIMA
2. Prophet
3. XGBoost
4. LSTM


## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

Best model is selected based on lowest RMSE.


## Model Selection
- All models are trained for each state
- Metrics are compared
- Best model is automatically selected and saved


## Forecasting
- Predicts next 8 weeks
- Uses:
  - Direct forecasting for ARIMA and Prophet
  - Recursive forecasting for XGBoost and LSTM


## API Endpoints

### Health Check
GET /health

Response:
{
  "status": "ok"
}


### Predict
POST /predict

Request:
{
  "state": "Texas"
}

Response:
{
  "state": "Texas",
  "forecast": [ ... ]
}


### Models Info
GET /models

Returns best model for each state


## How to Run

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Train models
python train_runner.py

Step 3: Start API
uvicorn api.main:app --reload

Step 4: Test API
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"state":"Texas"}'


## Visualization
Run:
python visualize_runner.py

This generates:
- Actual vs Predicted plot
- Future Forecast plot


## Error Handling
- Handles missing state input
- Handles model not found
- Handles empty or invalid responses
- Returns appropriate API errors


## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- prophet
- xgboost
- tensorflow
- fastapi
- uvicorn
- joblib


## Key Highlights
- End-to-end machine learning pipeline
- Multiple model comparison
- Automated model selection
- Production-ready API
- Modular and clean code structure


## Future Improvements
- Hyperparameter tuning
- Cloud deployment
- Real-time forecasting
- Dashboard interface


## Author
Asmitha Narla