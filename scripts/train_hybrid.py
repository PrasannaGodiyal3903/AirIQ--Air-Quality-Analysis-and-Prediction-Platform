import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def train_hybrid_model():
    # 1. Load Data
    train_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\train_data.csv')
    test_df = pd.read_csv(r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\test_data.csv')
    
    features = ['CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'PM2.5', 'SO2']
    target = 'AQI'
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print("Training Independent Base Models...\n")
    # 2. Train Linear Regression (captures baseline linear trends)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    # Train Gradient Boosting (captures complex non-linear spikes using tuned params)
    gbr_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42)
    gbr_model.fit(X_train, y_train)
    y_pred_gbr = gbr_model.predict(X_test)
    
    results = []
    results.append(evaluate(y_test, y_pred_lr, 'Linear Regression (Base)'))
    results.append(evaluate(y_test, y_pred_gbr, 'Gradient Boosting (Base)'))
    
    # 3. Create Simple Average Hybrid
    print("Calculating Simple Hybrid Average...")
    y_pred_hybrid_simple = (y_pred_lr + y_pred_gbr) / 2.0
    results.append(evaluate(y_test, y_pred_hybrid_simple, 'Hybrid (Simple 50/50 Average)'))
    
    # 4. Create Weighted Average Hybrid
    # Since GBR vastly outperforms LR, we assign it 80% weight, and LR 20% to just stabilize extreme linear trends.
    print("Calculating Weighted Hybrid Average (20% LR + 80% GBR)...")
    y_pred_hybrid_weighted = (0.2 * y_pred_lr) + (0.8 * y_pred_gbr)
    results.append(evaluate(y_test, y_pred_hybrid_weighted, 'Hybrid (Weighted 20/80 Average)'))
    
    # 5. Format and Print
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print(" HYBRID ENSEMBLE MODEL PERFORMANCE")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)

if __name__ == '__main__':
    train_hybrid_model()
