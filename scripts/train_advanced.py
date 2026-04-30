import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_all_models():
    # 1. Load the chronologically split and scaled datasets
    train_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\train_data.csv')
    test_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\test_data.csv')
    
    features = ['CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'PM2.5', 'SO2']
    target = 'AQI'
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 2. Initialize Models with fixed random_state for reproducibility
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    results = []
    predictions_df = test_df.copy() # Store predictions for further analysis
    
    # 3. Train and Evaluate
    print("Training Models...")
    for name, model in models.items():
        # Fit
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Store predictions
        predictions_df[f'{name}_Pred'] = y_pred
        
        # Calculate Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Append to results
        results.append({
            'Model': name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2_Score': round(r2, 4)
        })
        print(f"[{name}] Training Complete.")
        
    # 4. Format Results as a Table
    results_df = pd.DataFrame(results)
    
    # Sort by best RMSE (lowest) and R2 (highest)
    results_df = results_df.sort_values(by=['RMSE', 'R2_Score'], ascending=[True, False]).reset_index(drop=True)
    
    print("\n" + "="*50)
    print(" ADVANCED MODEL TRAINING RESULTS")
    print("="*50)
    print(results_df.to_string(index=False))
    print("="*50)
    
    # Save predictions
    preds_path = r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\model_predictions.csv'
    predictions_df.to_csv(preds_path, index=False)
    print(f"\nAll predictions saved for further analysis at: {preds_path}")

if __name__ == '__main__':
    train_all_models()
