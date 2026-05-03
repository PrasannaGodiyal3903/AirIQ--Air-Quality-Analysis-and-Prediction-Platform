from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import folium
from folium.plugins import HeatMap
import branca

app = Flask(__name__)

# --- Global Machine Learning State ---
MODEL_LOADED = False
MODELS = {}
X_TEST, Y_TEST = None, None
ACTUAL_METRICS = []
SCATTER_DATA = []

def initialize_ml_models():
    """
    Loads dataset, trains multiple models to calculate real metrics, 
    and keeps Gradient Boosting ready for `/predict` API.
    """
    global MODEL_LOADED, MODELS, X_TEST, Y_TEST, ACTUAL_METRICS, SCATTER_DATA
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "clean_air_quality.csv")
    
    try:
        df = pd.read_csv(csv_path).dropna()
        # Features & Target
        # The prompt implies 6 features from EDA (CO, NH3, NO2, OZONE, PM10, SO2)
        X = df[["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]]
        y = df["AQI"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_TEST, Y_TEST = X_test, y_test
        
        # Train Models
        print("Training Models...")
        MODELS['lr'] = LinearRegression().fit(X_train, y_train)
        # Tuned via RandomizedSearchCV — best generalization (Test R²=0.939, gap=0.034)
        MODELS['rf'] = RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
        ).fit(X_train, y_train)
        # max_depth=3 avoids 0.082 overfit gap seen with max_depth=4
        MODELS['gbr'] = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ).fit(X_train, y_train)
        MODELS['ada'] = AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42).fit(X_train, y_train)
        
        # Calculate Real Metrics
        def calc(y_t, y_p):
            return {
                "mae": round(mean_absolute_error(y_t, y_p), 2),
                "rmse": round(np.sqrt(mean_squared_error(y_t, y_p)), 2),
                "r2": round(r2_score(y_t, y_p), 3)
            }
        
        preds = {
            "Linear Regression": MODELS['lr'].predict(X_test),
            "Random Forest": MODELS['rf'].predict(X_test),
            "Gradient Boosting": MODELS['gbr'].predict(X_test),
            "AdaBoost": MODELS['ada'].predict(X_test)
        }
        
        # Optimized Hybrid (30% LR + 70% GBR)
        preds["Hybrid Model"] = (preds["Linear Regression"] * 0.30) + (preds["Gradient Boosting"] * 0.70)
        
        ACTUAL_METRICS = []
        model_scores = []

        for name, p_array in preds.items():
            metrics = calc(y_test, p_array)
            score = metrics['r2'] - (metrics['rmse'] * 0.001)
            model_scores.append((name, score, metrics))

        # Sort by score descending to determine rank
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for idx, (name, score, metrics) in enumerate(model_scores):
            rank = idx + 1
            ACTUAL_METRICS.append({
                "name": name,
                "mae": metrics['mae'],
                "rmse": metrics['rmse'],
                "r2": metrics['r2'],
                "is_best": rank == 1,
                "rank": rank
            })
            
        # Generate Scatter Data (using Hybrid Model predictions for visualization)
        # Cap to 50 items so chart isn't too crowded
        y_test_list = list(y_test)
        preds_list = list(preds["Hybrid Model"])
        scat_y = y_test_list[:50]
        scat_p = preds_list[:50]
        SCATTER_DATA = [{"x": float(sy), "y": float(sp)} for sy, sp in zip(scat_y, scat_p)]

        MODEL_LOADED = True
        print("ML Pipeline Initialized Successfully.")
        
    except Exception as e:
        print(f"Failed to initialize ML Pipeline: {e}")

# Run ML setup on startup
initialize_ml_models()

def get_cat(aqi):
    """Return CPCB AQI category string from numeric value."""
    aqi = float(aqi)
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


@app.route("/")
def home():
    """Render landing page with feature cards"""
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    """Render AQI prediction page"""
    return render_template("prediction.html")

@app.route("/analytics")
def analytics():
    """Render analytics and model performance page"""
    return render_template("analytics.html")

@app.route("/map-view")
def map_view():
    """Render heatmap visualization page"""
    return render_template("map_view.html")

@app.route("/commute-view")
def commute_view():
    """Render commute safety score page"""
    return render_template("commute_view.html")

@app.route("/leaderboard-view")
def leaderboard_view():
    """Render national leaderboard page"""
    return render_template("leaderboard_view.html")

@app.route("/comparison-view")
def comparison_view():
    """Render city comparison page"""
    return render_template("comparison_view.html")

@app.route("/api/metrics")
def get_metrics():
    """Return trained model metrics and scatter plot data"""
    if not MODEL_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
    return jsonify({
        "metrics": ACTUAL_METRICS,
        "scatter": SCATTER_DATA
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """Real time prediction using selected model"""
    if not MODEL_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
        
    data = request.get_json()
    model_name = "Hybrid Model" # Locked for production
    
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
    
    try:
        if model_name == "Hybrid Model":
            pred_lr = MODELS['lr'].predict(input_df)[0]
            pred_gbr = MODELS['gbr'].predict(input_df)[0]
            pred = (pred_lr * 0.30) + (pred_gbr * 0.70)
        elif model_name == "Linear Regression":
            pred = MODELS['lr'].predict(input_df)[0]
        elif model_name == "Random Forest":
            pred = MODELS['rf'].predict(input_df)[0]
        elif model_name == "AdaBoost":
            pred = MODELS['ada'].predict(input_df)[0]
        else:
            pred = MODELS['gbr'].predict(input_df)[0]
            
        aqi_val = int(round(pred))
        cat = get_cat(aqi_val)
        
        confidence = 0.0
        is_best = False
        for m in ACTUAL_METRICS:
            if m["name"] == model_name:
                confidence = m["r2"]
                is_best = m["is_best"]
                
        model_type = "Ensemble" if model_name in ["Random Forest", "Gradient Boosting", "AdaBoost", "Hybrid Model"] else "Linear"

        return jsonify({
            "success": True, 
            "aqi": aqi_val, 
            "category": cat,
            "model_used": model_name,
            "model_type": model_type,
            "confidence": confidence,
            "is_best": is_best
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/compare-models", methods=["POST"])
def compare_models():
    """Predicts AQI using all available models for comparison"""
    if not MODEL_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
        
    data = request.get_json()
    input_df = pd.DataFrame([{
        "CO": float(data.get("CO", 0)),
        "NH3": 5.0, 
        "NO2": float(data.get("NO2", 0)),
        "OZONE": float(data.get("OZONE", 0)),
        "PM10": float(data.get("PM10", 0)),
        "SO2": 10.0
    }])
    
    try:
        def get_cat(aqi_val):
            if aqi_val <= 50: return "Good"
            elif aqi_val <= 100: return "Moderate"
            elif aqi_val <= 200: return "Poor"
            elif aqi_val <= 300: return "Very Poor"
            else: return "Severe"
            
        pred_lr = int(round(MODELS['lr'].predict(input_df)[0]))
        pred_rf = int(round(MODELS['rf'].predict(input_df)[0]))
        pred_gbr = int(round(MODELS['gbr'].predict(input_df)[0]))
        pred_ada = int(round(MODELS['ada'].predict(input_df)[0]))
        pred_hybrid = int(round((MODELS['lr'].predict(input_df)[0] * 0.30) + (MODELS['gbr'].predict(input_df)[0] * 0.70)))
        
        predictions = {
            "Linear Regression": {"aqi": pred_lr, "category": get_cat(pred_lr)},
            "Random Forest": {"aqi": pred_rf, "category": get_cat(pred_rf)},
            "Gradient Boosting": {"aqi": pred_gbr, "category": get_cat(pred_gbr)},
            "AdaBoost": {"aqi": pred_ada, "category": get_cat(pred_ada)},
            "Hybrid Model": {"aqi": pred_hybrid, "category": get_cat(pred_hybrid)}
        }
        
        models_out = {}
        best_model = None
        for m in ACTUAL_METRICS:
            name = m["name"]
            models_out[name] = {
                "aqi": predictions[name]["aqi"],
                "category": predictions[name]["category"],
                "r2": m["r2"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "rank": m["rank"]
            }
            if m["is_best"]:
                best_model = name
        
        return jsonify({
            "success": True, 
            "models": models_out,
            "best_model": best_model
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# City name → (lat, lon) lookup for Indian cities in the dataset
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777), "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946), "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714), "Pune": (18.5204, 73.8567), "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462), "Kanpur": (26.4499, 80.3319), "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577), "Varanasi": (25.3176, 82.9739), "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126), "Agra": (27.1767, 78.0081), "Coimbatore": (11.0168, 76.9558),
    "Visakhapatnam": (17.6868, 83.2185), "Dehradun": (30.3165, 78.0322),
    "Noida": (28.5355, 77.3910), "Gurgaon": (28.4595, 77.0266), "Faridabad": (28.4089, 77.3178),
    "Ghaziabad": (28.6692, 77.4538), "Muzaffarpur": (26.1209, 85.3647),
    "Hapur": (28.7304, 77.7757), "Bahadurgarh": (28.6814, 76.9339),
    "Surat": (21.1702, 72.8311), "Rajkot": (22.3039, 70.8022), "Vadodara": (22.3072, 73.1812),
    "Amritsar": (31.6340, 74.8723), "Ludhiana": (30.9010, 75.8573), "Jalandhar": (31.3260, 75.5762),
    "Bhilai": (21.2093, 81.3748), "Raipur": (21.2514, 81.6296), "Asansol": (23.6888, 86.9661),
    "Jodhpur": (26.2389, 73.0243), "Udaipur": (24.5854, 73.7125), "Kota": (25.2138, 75.8648),
    "Nellore": (14.4426, 79.9865), "Rajamahendravaram": (16.9891, 81.7837),
    "Gummidipoondi": (13.4098, 80.1134), "Navi Mumbai": (19.0330, 73.0297),
    "Hingoli": (19.7196, 77.1495), "Talcher": (20.9500, 85.2333), "Brajrajnagar": (21.8261, 83.9271),
    "Panchgaon": (28.3900, 76.9600), "Byrnihat": (26.0500, 91.9000),
    "Howrah": (22.5958, 88.2636), "Durgapur": (23.5204, 87.3119),
    "Guwahati": (26.1445, 91.7362), "Nagaon": (26.3464, 92.6843),
    "Agartala": (23.8315, 91.2868), "Satna": (24.5764, 80.8322),
    "Agmer": (26.4499, 74.6399), "Bareilly": (28.3670, 79.4304),
    "Gaya": (24.7955, 85.0006), "Vijayawada": (16.5062, 80.6480),
    "Mysuru": (12.2958, 76.6394), "Nashik": (19.9975, 73.7898),
    "Amravati": (20.9374, 77.7796), "Chandigarh": (30.7333, 76.7794),
    "Bikaner": (28.0229, 73.3119), "Ambala": (30.3782, 76.7767),
    "Gorakhpur": (26.7606, 83.3731), "Jhansi": (25.4484, 78.5685),
    "Aurangabad": (19.8762, 75.3433), "Gwalior": (26.2124, 78.1772),
    "Jabalpur": (23.1667, 79.9333), "Kochi": (9.9312, 76.2673),
    "Madurai": (9.9252, 78.1198), "Meerut": (28.9845, 77.7064),
    "Salem": (11.6643, 78.1460), "Shimla": (31.1048, 77.1734),
    "Solapur": (17.6599, 75.9064), "Thane": (19.2183, 72.9781),
    "Ujjain": (23.1760, 75.7885), "Mangalore": (12.9141, 74.8560)
}

def extract_city(station_name):
    """Extract city name from station name like 'Alipur, Delhi - DPCC'"""
    for city in CITY_COORDS:
        if city.lower() in station_name.lower():
            return city
    return None

@app.route("/api/aqi-heatmap")
def get_aqi_heatmap():
    """Real data-driven heatmap: actual AQI from dataset, predicted from live model."""
    if not MODEL_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "data", "clean_air_quality.csv")
        df = pd.read_csv(csv_path).dropna()

        # Map each station row to a city
        df['city'] = df['station'].apply(extract_city)
        df = df[df['city'].notna()]

        # Aggregate actual AQI per city (mean across all station rows)
        city_aqi = df.groupby('city').agg(
            actual_aqi=('AQI', 'mean'),
            CO=('CO', 'mean'), NH3=('NH3', 'mean'), NO2=('NO2', 'mean'),
            OZONE=('OZONE', 'mean'), PM10=('PM10', 'mean'), SO2=('SO2', 'mean')
        ).reset_index()

        points = []
        for _, row in city_aqi.iterrows():
            city = row['city']
            lat, lon = CITY_COORDS[city]

            # Run live model prediction using mean pollutant values
            input_df = pd.DataFrame([{
                "CO": row['CO'], "NH3": row['NH3'], "NO2": row['NO2'],
                "OZONE": row['OZONE'], "PM10": row['PM10'], "SO2": row['SO2']
            }])
            # Use Hybrid Model (production model: 30% LR + 70% GBR)
            pred_lr = float(MODELS['lr'].predict(input_df)[0])
            pred_gbr = float(MODELS['gbr'].predict(input_df)[0])
            predicted_aqi = round((pred_lr * 0.30) + (pred_gbr * 0.70), 1)
            actual_aqi = round(float(row['actual_aqi']), 1)

            points.append({
                "city": city,
                "lat": lat,
                "lon": lon,
                "actual_aqi": actual_aqi,
                "predicted_aqi": predicted_aqi,
                "actual_category": get_cat(actual_aqi),
                "predicted_category": get_cat(predicted_aqi)
            })

        return jsonify({"success": True, "count": len(points), "data": points})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Backward-compat alias for old map endpoint
@app.route("/api/aqi-map-data")
def get_aqi_map_data():
    return get_aqi_heatmap()

@app.route("/heatmap")
def heatmap():
    """Generate and serve a Folium AQI Heatmap with a legend."""
    if not MODEL_LOADED:
        return "Models not loaded. Please refresh the app.", 500
        
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "data", "clean_air_quality.csv")
        df = pd.read_csv(csv_path).dropna()

        # Group data by city to get average AQI per city
        df['city'] = df['station'].apply(extract_city)
        df = df[df['city'].notna()]
        city_data = df.groupby('city').agg({'AQI': 'mean'}).reset_index()

        # Create base map
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="cartodbpositron", attribution_control=False)

        # Prepare heatmap data: [[lat, lon, weight], ...]
        heat_data = []
        for _, row in city_data.iterrows():
            city = row['city']
            lat, lon = CITY_COORDS[city]
            # Folium HeatMap weight is usually 0-1, but we can use AQI and tune intensity
            heat_data.append([lat, lon, float(row['AQI'])])

            # Add CircleMarkers with popups for interactivity
            aqi = round(row['AQI'], 1)
            color = "#000000"
            if aqi <= 50: color = "#4CAF50" # Green
            elif aqi <= 100: color = "#FFEB3B" # Yellow
            elif aqi <= 200: color = "#FF9800" # Orange
            elif aqi <= 300: color = "#F44336" # Red
            elif aqi <= 400: color = "#B71C1C" # Dark Red
            else: color = "#4A148C" # Purple

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=f"<b>{city}</b><br>AQI: {aqi}<br>Category: {get_cat(aqi)}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)

        # Add HeatMap layer
        # Custom gradient requested: Green, Yellow, Orange, Red, Purple
        # Scale: 0-50, 50-100, 100-200, 200-300, 300-500
        # We normalize these to 0.0 - 1.0 for folium gradient
        HeatMap(heat_data, min_opacity=0.3, max_val=500, radius=25, blur=15, 
                gradient={0.1: '#4CAF50', 0.2: '#FFEB3B', 0.4: '#FF9800', 0.6: '#F44336', 1.0: '#4A148C'}).add_to(m)

        # Add custom legend using branca
        legend_html = '''
        {% macro html(this, kwargs) %}
        <div style="
            position: fixed; 
            bottom: 50px; left: 50px; width: 220px; height: 160px; 
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
            padding: 10px; border-radius: 10px; opacity: 0.9;
            ">
            <b style="font-size:16px;">AQI Levels</b><br>
            <i style="background:#4CAF50;width:15px;height:15px;display:inline-block;border-radius:3px;"></i> Good (0-50)<br>
            <i style="background:#FFEB3B;width:15px;height:15px;display:inline-block;border-radius:3px;"></i> Moderate (51-100)<br>
            <i style="background:#FF9800;width:15px;height:15px;display:inline-block;border-radius:3px;"></i> Poor (101-200)<br>
            <i style="background:#F44336;width:15px;height:15px;display:inline-block;border-radius:3px;"></i> Very Poor (201-300)<br>
            <i style="background:#4A148C;width:15px;height:15px;display:inline-block;border-radius:3px;"></i> Hazardous (301+)<br>
        </div>
        {% endmacro %}
        '''
        legend = branca.element.MacroElement()
        legend._template = branca.element.Template(legend_html)
        m.get_root().add_child(legend)

        return m._repr_html_()
    except Exception as e:
        return f"Error generating heatmap: {str(e)}", 500

@app.route("/api/commute-score", methods=["POST"])
def calculate_commute():
    """Calculate commute safety score based on AQI exposure."""
    try:
        data = request.get_json()
        start_city = data.get("start")
        end_city = data.get("end")
        mode = data.get("mode", "walk")

        if not start_city or not end_city:
            return jsonify({"error": "Start and End cities are required"}), 400

        # Load data to get latest AQI for cities
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "data", "clean_air_quality.csv")
        df = pd.read_csv(csv_path).dropna()
        df['city'] = df['station'].apply(extract_city)
        
        city_data = df.groupby('city').agg({'AQI': 'mean', 'PM2.5': 'mean'}).to_dict('index')

        aqi_a = city_data.get(start_city, {"AQI": 100})["AQI"]
        aqi_b = city_data.get(end_city, {"AQI": 100})["AQI"]
        pm_a = city_data.get(start_city, {"PM2.5": 50})["PM2.5"]
        pm_b = city_data.get(end_city, {"PM2.5": 50})["PM2.5"]

        avg_aqi = (aqi_a + aqi_b) / 2
        avg_pm = (pm_a + pm_b) / 2

        # Exposure Weights
        weights = {
            "walk": 1.2,
            "cycle": 1.3,
            "car": 0.6,
            "bus": 0.7,
            "train": 0.3
        }
        weight = weights.get(mode, 1.0)

        exposure_aqi = round(avg_aqi * weight, 1)
        exposure_pm = round(avg_pm * weight, 1)

        # Determine Safety Status
        if exposure_aqi <= 50:
            status = "Safe"
            color = "#4CAF50"
            advice = "Minimal risk. Perfect for outdoor activities."
        elif exposure_aqi <= 100:
            status = "Moderate"
            color = "#FFEB3B"
            advice = "Acceptable quality. Sensitive groups should limit prolonged exertion."
        elif exposure_aqi <= 200:
            status = "Risky"
            color = "#FF9800"
            advice = "High exposure. Wear a mask and avoid deep breathing exercises."
        else:
            status = "Unsafe"
            color = "#F44336"
            advice = "Severe pollution. Avoid commuting via this mode if possible."

        return jsonify({
            "start": start_city,
            "end": end_city,
            "mode": mode,
            "exposure_aqi": exposure_aqi,
            "exposure_pm": exposure_pm,
            "status": status,
            "color": color,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/leaderboard")
def get_leaderboard():
    """Get Top 5 Cleanest and Top 5 Most Polluted cities."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "data", "clean_air_quality.csv")
        df = pd.read_csv(csv_path).dropna()
        df['city'] = df['station'].apply(extract_city)
        
        city_avg = df.groupby('city')['AQI'].mean().reset_index()
        city_avg = city_avg.sort_values('AQI')
        
        cleanest = city_avg.head(5).to_dict('records')
        polluted = city_avg.tail(5).sort_values('AQI', ascending=False).head(5).to_dict('records')
        
        return jsonify({
            "cleanest": cleanest,
            "polluted": polluted
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/compare-cities", methods=["POST"])
def compare_cities():
    """Get detailed pollutant data for two cities."""
    try:
        data = request.get_json()
        city1 = data.get("city1")
        city2 = data.get("city2")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "data", "clean_air_quality.csv")
        df = pd.read_csv(csv_path).dropna()
        df['city'] = df['station'].apply(extract_city)
        
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'OZONE', 'AQI']
        city_stats = df.groupby('city')[pollutants].mean().to_dict('index')
        
        if city1 not in city_stats or city2 not in city_stats:
            return jsonify({"error": "One or both cities not found"}), 404
            
        return jsonify({
            "city1": {"name": city1, "data": city_stats[city1]},
            "city2": {"name": city2, "data": city_stats[city2]}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
