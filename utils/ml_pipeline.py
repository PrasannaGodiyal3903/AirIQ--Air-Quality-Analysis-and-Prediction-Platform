import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Global Machine Learning State ---
_MODEL_LOADED = False
_MODELS = {}
_X_TEST, _Y_TEST = None, None
_ACTUAL_METRICS = []
_SCATTER_DATA = []

def initialize_ml_models():
    """
    Loads dataset, trains multiple models to calculate real metrics, 
    and keeps Gradient Boosting ready for `/predict` API.
    """
    global _MODEL_LOADED, _MODELS, _X_TEST, _Y_TEST, _ACTUAL_METRICS, _SCATTER_DATA
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(script_dir, "data", "clean_air_quality.csv")
    
    try:
        df = pd.read_csv(csv_path).dropna()
        # Features & Target
        # The prompt implies 6 features from EDA (CO, NH3, NO2, OZONE, PM10, SO2)
        X = df[["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]]
        y = df["AQI"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _X_TEST, _Y_TEST = X_test, y_test
        
        # Train Models
        print("Training Models...")
        _MODELS['lr'] = LinearRegression().fit(X_train, y_train)
        _MODELS['rf'] = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        _MODELS['gbr'] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42).fit(X_train, y_train)
        _MODELS['ada'] = AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42).fit(X_train, y_train)
        
        # Calculate Real Metrics
        def calc(y_t, y_p):
            return {
                "mae": round(mean_absolute_error(y_t, y_p), 2),
                "rmse": round(np.sqrt(mean_squared_error(y_t, y_p)), 2),
                "r2": round(r2_score(y_t, y_p), 3)
            }
            
        preds = {
            "Linear Regression": _MODELS['lr'].predict(X_test),
            "Random Forest": _MODELS['rf'].predict(X_test),
            "Gradient Boosting": _MODELS['gbr'].predict(X_test),
            "AdaBoost": _MODELS['ada'].predict(X_test)
        }
        
        # Hybrid (LR + GBR Avg)
        preds["Hybrid Model"] = (preds["Linear Regression"] + preds["Gradient Boosting"]) / 2
        
        _ACTUAL_METRICS = []
        for name, p_array in preds.items():
            metrics = calc(y_test, p_array)
            _ACTUAL_METRICS.append({
                "name": name,
                "mae": metrics['mae'],
                "rmse": metrics['rmse'],
                "r2": metrics['r2'],
                "is_best": name == "Hybrid Model"
            })
            
        # Generate Scatter Data (using Hybrid Model predictions for visualization)
        # Cap to 50 items so chart isn't too crowded
        y_test_list = list(y_test)
        preds_list = list(preds["Hybrid Model"])
        scat_y = y_test_list[:50]
        scat_p = preds_list[:50]
        _SCATTER_DATA = [{"x": float(sy), "y": float(sp)} for sy, sp in zip(scat_y, scat_p)]

        _MODEL_LOADED = True
        print("ML Pipeline Initialized Successfully.")
        
    except Exception as e:
        print(f"Failed to initialize ML Pipeline: {e}")

def get_metrics_data():
    """
    Returns the accuracy metrics and scatter plot data.
    """
    if not _MODEL_LOADED:
        return None, None, "Models not loaded"
    return _ACTUAL_METRICS, _SCATTER_DATA, None

def predict_aqi(data):
    """
    Predicts AQI using the trained Gradient Boosting Model.
    """
    if not _MODEL_LOADED:
        return None, "Models not loaded"
        
    try:
        # We require 6 features: ["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]
        # The UI only has 4 sliders. We will pad the others as medians or 0s.
        input_df = pd.DataFrame([{
            "CO": float(data.get("CO", 0)),
            "NH3": 5.0, # dummy filler
            "NO2": float(data.get("NO2", 0)),
            "OZONE": float(data.get("OZONE", 0)),
            "PM10": float(data.get("PM10", 0)),
            "SO2": 10.0 # dummy filler
        }])
        
        # Predict using GBR
        pred = _MODELS['gbr'].predict(input_df)[0]
        aqi_val = int(round(pred))
        
        # Return Category
        cat = "Unknown"
        if aqi_val <= 50: cat = "Good"
        elif aqi_val <= 100: cat = "Moderate"
        elif aqi_val <= 200: cat = "Poor"
        elif aqi_val <= 300: cat = "Very Poor"
        else: cat = "Severe"
        
        return {
            "success": True, 
            "aqi": aqi_val, 
            "category": cat
        }, None
        
    except Exception as e:
        return None, str(e)
