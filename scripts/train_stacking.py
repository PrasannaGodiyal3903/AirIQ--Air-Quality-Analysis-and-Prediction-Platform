import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        'Model': model_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2_Score': round(r2, 4)
    }

def train_advanced_ensembles():
    # 1. Load Data
    train_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\train_data.csv')
    test_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\test_data.csv')
    
    features = ['CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'PM2.5', 'SO2']
    target = 'AQI'
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    results = []
    
    print("Training Base Models...")
    # Base Models
    lr_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42).fit(X_train, y_train)
    gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42).fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_gb = gb_model.predict(X_test)
    
    results.append(evaluate(y_test, y_pred_lr, 'Linear Regression (Base)'))
    results.append(evaluate(y_test, rf_model.predict(X_test), 'Random Forest (Base)'))
    results.append(evaluate(y_test, y_pred_gb, 'Gradient Boosting (Base)'))
    
    # 2. Optimize Weights using Cross-Validation on Training Data
    print("Optimizing Weights for Hybrid Model via Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_w = 0.5
    best_rmse = float('inf')
    
    # Test weights for GBR from 0.0 to 1.0 in 0.05 increments
    weights_to_test = np.arange(0.0, 1.05, 0.05)
    
    for w_gb in weights_to_test:
        w_lr = 1.0 - w_gb
        cv_rmse_list = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            lr_tmp = LinearRegression().fit(X_t, y_t)
            gb_tmp = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).fit(X_t, y_t)
            
            pred_v = (w_lr * lr_tmp.predict(X_v)) + (w_gb * gb_tmp.predict(X_v))
            cv_rmse_list.append(np.sqrt(mean_squared_error(y_v, pred_v)))
            
        avg_rmse = np.mean(cv_rmse_list)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_w = w_gb
            
    print(f"Optimal Weights Found: LR = {1-best_w:.2f}, GBR = {best_w:.2f}")
    
    # Simple Average
    y_pred_simple = (y_pred_lr + y_pred_gb) / 2.0
    results.append(evaluate(y_test, y_pred_simple, 'Hybrid (Simple Average)'))
    
    # Optimized Weighted Average
    y_pred_opt = ((1 - best_w) * y_pred_lr) + (best_w * y_pred_gb)
    results.append(evaluate(y_test, y_pred_opt, f'Hybrid (Optimized {int((1-best_w)*100)}/{int(best_w*100)})'))
    
    # 3. Implement Stacking Regressor
    print("Training Advanced Stacking Regressor...")
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42))
    ]
    
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)
    results.append(evaluate(y_test, y_pred_stack, 'Stacking (RF + GB -> LR)'))
    
    # 4. Format and Print Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['RMSE', 'R2_Score'], ascending=[True, False]).reset_index(drop=True)
    
    print("\n" + "="*80)
    print(" ADVANCED ENSEMBLE PERFORMANCE RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

if __name__ == '__main__':
    train_advanced_ensembles()
