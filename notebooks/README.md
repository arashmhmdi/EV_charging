# EV Charging Energy Prediction

## Project Overview
A regression model to predict kWh delivered during EV charging sessions based on session features.

## Dataset
- Source: Synthetic EV Data (50k samples)
- Key Columns: connectionTime_decimal, chargingDuration, kWhDelivered, etc.

## Data Processing
1. Dropped missing values  
2. Engineered features: connection hour/minute, charge rate, weekday

## Modeling
- **Baseline**: Linear Regression (MAE=4.22, R²=0.08)  
- **Advanced**: Random Forest (MAE=3.53, R²=0.26)  
- Hyperparameter tuning with RandomizedSearchCV

## Visualizations
- Histograms and correlation heatmap  
- Residual plot for best Random Forest  
- Feature importances chart

## Usage Example
```python
import joblib

model = joblib.load('best_rf_model.joblib')
pred = model.predict(new_data)
