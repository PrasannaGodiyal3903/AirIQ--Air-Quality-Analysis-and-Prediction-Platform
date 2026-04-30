import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_air_quality(input_path):
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    # Pivot from long to wide format
    print("Pivoting data to wide format...")
    wide_df = df.pivot_table(index=['station', 'last_update'], 
                             columns='pollutant_id', 
                             values='pollutant_avg', 
                             aggfunc='mean').reset_index()
    
    # Sort chronologically
    wide_df['last_update'] = pd.to_datetime(wide_df['last_update'])
    wide_df = wide_df.sort_values(by=['station', 'last_update'])
    
    pollutant_cols = [col for col in wide_df.columns if col not in ['station', 'last_update']]
    
    print("Handling missing values via Interpolation and Median Fallback...")
    # 1. Time-Series Interpolation
    wide_df[pollutant_cols] = wide_df.groupby('station')[pollutant_cols].transform(lambda group: group.interpolate(method='linear', limit_direction='both'))
    
    # 2. Median Fallback for remaining NaNs
    for col in pollutant_cols:
        wide_df[col] = wide_df[col].fillna(wide_df[col].median())
        
    print("Calculating AQI...")
    # Dummy AQI calculation for the sake of the script (max of pollutants)
    wide_df['AQI'] = wide_df[pollutant_cols].max(axis=1)
    
    # Outlier Management (Winsorization 3.0 IQR)
    print("Applying Winsorization to cap outliers...")
    for col in pollutant_cols:
        Q1 = wide_df[col].quantile(0.25)
        Q3 = wide_df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3.0 * IQR
        wide_df[col] = np.where(wide_df[col] > upper_bound, upper_bound, wide_df[col])
        
    clean_path = r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\clean_air_quality.csv'
    wide_df.to_csv(clean_path, index=False)
    print(f"Cleaned data saved to {clean_path}")
    
    # Train Test Split (Chronological)
    print("Performing Chronological Train/Test Split...")
    split_index = int(len(wide_df) * 0.8)
    train_df = wide_df.iloc[:split_index].copy()
    test_df = wide_df.iloc[split_index:].copy()
    
    # Standardization
    print("Standardizing numerical features...")
    scaler = StandardScaler()
    train_df[pollutant_cols] = scaler.fit_transform(train_df[pollutant_cols])
    test_df[pollutant_cols] = scaler.transform(test_df[pollutant_cols])
    
    train_path = r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\train_data.csv'
    test_path = r'c:\Users\priya\OneDrive\Desktop\minor project 2\data\test_data.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Training and Testing datasets prepared and saved successfully!")

if __name__ == '__main__':
    preprocess_air_quality(r'c:\Users\priya\OneDrive\Desktop\minor project 2\dataset.csv')
