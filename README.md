# End-to-End Time Series Forecasting System with API

## Overview
This project is a production-ready Time Series Forecasting system that predicts the next 8 weeks of sales for each state using historical data.

The system:
- Trains multiple forecasting models
- Compares performance automatically
- Selects the best model based on evaluation metrics
- Serves predictions through a REST API
- Provides an interactive UI dashboard for visualization

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
│   ├── <state>.pkl
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
├── frontend/
│   └── index.html
│
├── train_runner.py
├── visualize_runner.py
├── requirements.txt
└── README.md

## Data Processing
- Automatically detects and cleans dataset columns
- Converts date column to datetime format
- Sorts data chronologically
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
- trend index to capture long-term patterns

## Train / Validation Split
- Time-based split (no random sampling)
- Prevents data leakage
- Uses recent data for validation

## Models Implemented
1. ARIMA  
2. Prophet  
3. XGBoost  
4. LSTM  

## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

The best model is selected based on the lowest RMSE.

## Model Selection
- All models are trained independently for each state
- Performance metrics are computed
- Best model is automatically selected and saved

## Forecasting
- Predicts next 8 weeks of sales
- Uses:
  - Direct forecasting (ARIMA, Prophet)
  - Recursive forecasting (XGBoost, LSTM)

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
  "model": "arima",
  "forecast": [ ... ]
}

### Models Info
GET /models

Returns best model for each state.

## User Interface (Frontend)
- Search-based input for selecting states
- Dynamic state loading from backend
- Interactive forecast visualization using charts
- Displays selected model and prediction results

## How to Run

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Train models
python train_runner.py

Step 3: Start API
uvicorn api.main:app --reload

Step 4: Open UI
open frontend/index.html

Step 5: Test API manually (optional)
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"state":"Texas"}'

## Visualization
Run:
python visualize_runner.py

Generates:
- Actual vs Predicted plot
- Future Forecast plot

## Error Handling
- Handles missing state input
- Handles model not found
- Handles invalid or empty responses
- Returns appropriate API error messages

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
- Interactive UI dashboard
- Modular and clean code structure

## Future Improvements
- Hyperparameter tuning
- Cloud deployment
- Real-time forecasting
- Advanced dashboard UI

## Author
Asmitha Narla
