import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def tune_models():
    # 1. Load the chronologically split and scaled datasets
    train_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\train_data.csv')
    test_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\test_data.csv')
    
    features = ['CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'PM2.5', 'SO2']
    target = 'AQI'
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 2. Define Parameter Grids
    rf_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    gb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    }
    
    ada_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
    }
    
    models = {
        'Random Forest': (RandomForestRegressor(random_state=42), rf_grid),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=42), gb_grid),
        'AdaBoost': (AdaBoostRegressor(random_state=42), ada_grid)
    }
    
    results = []
    
    print("Starting Hyperparameter Tuning with 5-Fold Cross Validation...\n")
    
    # 3. Tuning Loop
    for name, (model, param_grid) in models.items():
        print(f"Tuning {name}...")
        
        # Grid Search with CV=5
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Predict on Test Data
        y_pred = best_model.predict(X_test)
        
        # Calculate Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Best Parameters': str(best_params),
            'Tuned MAE': round(mae, 2),
            'Tuned RMSE': round(rmse, 2),
            'Tuned R2': round(r2, 4)
        })
        
        print(f"   Done. Best Params: {best_params}\n")
        
    # 4. Format and Print Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['Tuned RMSE', 'Tuned R2'], ascending=[True, False]).reset_index(drop=True)
    
    print("="*80)
    print(" TUNED MODEL PERFORMANCE RESULTS")
    print("="*80)
    # Print each model clearly
    for idx, row in results_df.iterrows():
        print(f"\n--- {row['Model']} ---")
        print(f"Parameters: {row['Best Parameters']}")
        print(f"Metrics: MAE={row['Tuned MAE']} | RMSE={row['Tuned RMSE']} | R2={row['Tuned R2']}")

if __name__ == '__main__':
    tune_models()
