from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import folium
from folium.plugins import HeatMap
import branca
import requests
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# =========================================================
# GLOBAL VARIABLES
# =========================================================

MODEL_LOADED = False
MODELS = {}
X_TEST, Y_TEST = None, None
ACTUAL_METRICS = []
SCATTER_DATA = []

# =========================================================
# WAQI TOKEN
# =========================================================

WAQI_TOKEN = "demo"

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_cat(aqi):
    aqi = float(aqi)

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# =========================================================
# LOAD CITY COORDINATES
# =========================================================

def load_city_coords():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    try:
        df = pd.read_csv(csv_path)

        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        city_data = df.dropna(subset=['latitude', 'longitude', 'city'])

        city_groups = city_data.groupby('city').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).to_dict('index')

        coords = {
            city: (data['latitude'], data['longitude'])
            for city, data in city_groups.items()
        }

        print(f"Loaded {len(coords)} cities")

        return coords

    except Exception as e:
        print(f"Error loading city coordinates: {e}")

        return {
            "Delhi": (28.6139, 77.2090),
            "Mumbai": (19.0760, 72.8777)
        }

CITY_COORDS = load_city_coords()
CITIES_LIST = sorted(list(CITY_COORDS.keys()))

# =========================================================
# MACHINE LEARNING INITIALIZATION
# =========================================================

def initialize_ml_models():

    global MODEL_LOADED
    global MODELS
    global X_TEST
    global Y_TEST
    global ACTUAL_METRICS
    global SCATTER_DATA

    try:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            script_dir,
            "data",
            "clean_air_quality.csv"
        )

        df = pd.read_csv(csv_path).dropna()

        X = df[["CO", "NH3", "NO2", "OZONE", "PM10", "SO2"]]
        y = df["AQI"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        X_TEST = X_test
        Y_TEST = y_test

        print("Training ML Models...")

        MODELS['lr'] = LinearRegression().fit(X_train, y_train)

        MODELS['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ).fit(X_train, y_train)

        MODELS['et'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ).fit(X_train, y_train)

        MODELS['hgb'] = HistGradientBoostingRegressor(
            max_iter=100,
            random_state=42
        ).fit(X_train, y_train)

        MODELS['gbr'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ).fit(X_train, y_train)

        MODELS['ada'] = AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        ).fit(X_train, y_train)

        def calc_metrics(y_true, y_pred):
            return {
                "mae": round(mean_absolute_error(y_true, y_pred), 2),
                "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
                "r2": round(r2_score(y_true, y_pred), 3)
            }

        predictions = {
            "Linear Regression": MODELS['lr'].predict(X_test),
            "Random Forest": MODELS['rf'].predict(X_test),
            "Extra Trees": MODELS['et'].predict(X_test),
            "Hist GBR": MODELS['hgb'].predict(X_test),
            "Gradient Boosting": MODELS['gbr'].predict(X_test),
            "AdaBoost": MODELS['ada'].predict(X_test)
        }

        predictions["Hybrid Model"] = (
            predictions["Hist GBR"] * 0.4 +
            predictions["Extra Trees"] * 0.3 +
            predictions["Random Forest"] * 0.3
        )

        ACTUAL_METRICS = []

        for name, pred in predictions.items():

            metrics = calc_metrics(y_test, pred)

            ACTUAL_METRICS.append({
                "name": name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"]
            })

        hybrid_preds = predictions["Hybrid Model"]

        SCATTER_DATA = [
            {
                "x": float(actual),
                "y": float(pred)
            }
            for actual, pred in zip(
                list(y_test)[:50],
                list(hybrid_preds)[:50]
            )
        ]

        MODEL_LOADED = True

        print("ML Models Loaded Successfully")

    except Exception as e:
        print(f"ML Initialization Error: {e}")

initialize_ml_models()

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/map-view")
def map_view():
    return render_template("map_view.html")

@app.route("/commute-view")
def commute_view():
    return render_template("commute_view.html")

@app.route("/leaderboard-view")
def leaderboard_view():
    return render_template("leaderboard_view.html")

@app.route("/comparison-view")
def comparison_view():
    return render_template("comparison_view.html")

@app.route("/live-tracker")
def live_tracker():
    return render_template("live_tracker.html")

# =========================================================
# API - CITIES
# =========================================================

@app.route("/api/cities")
def get_cities():

    try:
        cities = sorted(list(CITY_COORDS.keys()))

        return jsonify({
            "success": True,
            "cities": cities
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })

# =========================================================
# API - METRICS
# =========================================================

@app.route("/api/metrics")
def get_metrics():

    if not MODEL_LOADED:
        return jsonify({"error": "Models not loaded"}), 500

    return jsonify({
        "metrics": ACTUAL_METRICS,
        "scatter": SCATTER_DATA
    })

# =========================================================
# API - PREDICT AQI
# =========================================================

@app.route("/predict-aqi", methods=["POST"])
def predict_aqi():

    if not MODEL_LOADED:
        return jsonify({"error": "Models not loaded"}), 500

    try:

        data = request.get_json()

        input_df = pd.DataFrame([{
            "CO": float(data.get("CO", 0)),
            "NH3": float(data.get("NH3", 0)),
            "NO2": float(data.get("NO2", 0)),
            "OZONE": float(data.get("OZONE", 0)),
            "PM10": float(data.get("PM10", 0)),
            "SO2": float(data.get("SO2", 0))
        }])

        pred_rf = MODELS['rf'].predict(input_df)[0]
        pred_et = MODELS['et'].predict(input_df)[0]
        pred_hgb = MODELS['hgb'].predict(input_df)[0]

        aqi = int(round(
            (pred_hgb * 0.4) +
            (pred_et * 0.3) +
            (pred_rf * 0.3)
        ))

        return jsonify({
            "success": True,
            "aqi": aqi,
            "category": get_cat(aqi),
            "model": "Hybrid Ensemble"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })

# =========================================================
# LIVE AQI
# =========================================================

@app.route("/api/live-aqi")
def get_live_aqi():

    try:

        if WAQI_TOKEN == "demo":
            return get_mock_live_data()

        url = (
            f"https://api.waqi.info/map/bounds/"
            f"?latlng=6.0,68.0,38.0,98.0"
            f"&token={WAQI_TOKEN}"
        )

        response = requests.get(url, timeout=10)

        data = response.json()

        live_results = []

        if data.get("status") == "ok":
            live_results = process_bounds_data(data.get("data", []))

        if len(live_results) > 0:

            return jsonify({
                "success": True,
                "count": len(live_results),
                "data": live_results
            })

        return get_mock_live_data(is_fallback=True)

    except Exception as e:

        print(f"Live AQI Error: {e}")

        return get_mock_live_data(is_fallback=True)

def process_bounds_data(all_stations):

    live_results = []

    for station in all_stations:

        try:

            aqi = station.get("aqi")

            if not str(aqi).isdigit():
                continue

            aqi = int(aqi)

            if aqi > 600:
                continue

            station_name = station.get(
                "station",
                {}
            ).get("name", "Unknown")

            live_results.append({
                "city": station_name,
                "aqi": aqi,
                "category": get_cat(aqi),
                "station": station_name,
                "time": "Live"
            })

        except:
            continue

    return live_results

def get_mock_live_data(is_fallback=False):

    data = []

    for city in CITIES_LIST[:100]:

        aqi = np.random.randint(40, 300)

        data.append({
            "city": city,
            "aqi": int(aqi),
            "category": get_cat(aqi),
            "station": f"{city} Station",
            "time": "Simulated"
        })

    return jsonify({
        "success": True,
        "count": len(data),
        "data": data,
        "mode": "demo"
    })

# =========================================================
# HEATMAP
# =========================================================

@app.route("/heatmap")
def heatmap():

    try:

        m = folium.Map(
            location=[22.5, 78.5],
            zoom_start=5,
            tiles="cartodbpositron"
        )

        heat_data = []

        for city, coords in CITY_COORDS.items():

            lat, lon = coords

            aqi = np.random.randint(40, 300)

            heat_data.append([lat, lon, aqi])

            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                popup=f"{city} AQI: {aqi}",
                color="red",
                fill=True
            ).add_to(m)

        HeatMap(
            heat_data,
            radius=25,
            blur=15
        ).add_to(m)

        return m._repr_html_()

    except Exception as e:
        return f"Heatmap Error: {str(e)}"

# =========================================================
# COMMUTE SCORE
# =========================================================

@app.route("/api/commute-score", methods=["POST"])
def commute_score():

    try:

        data = request.get_json()

        start = data.get("start")
        end = data.get("end")
        mode = data.get("mode", "walk")

        score = np.random.randint(50, 95)

        advice = "Safe commute"

        if score < 60:
            advice = "Avoid outdoor exposure"

        return jsonify({
            "start": start,
            "end": end,
            "mode": mode,
            "score": score,
            "advice": advice
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =========================================================
# LEADERBOARD
# =========================================================

@app.route("/api/leaderboard")
def leaderboard():

    data = []

    for city in CITIES_LIST[:20]:

        aqi = np.random.randint(30, 350)

        data.append({
            "city": city,
            "AQI": aqi
        })

    data = sorted(data, key=lambda x: x["AQI"])

    return jsonify({
        "cleanest": data[:5],
        "polluted": data[-5:]
    })

# =========================================================
# COMPARE CITIES
# =========================================================

@app.route("/api/compare-cities", methods=["POST"])
def compare_cities():

    try:

        data = request.get_json()

        city1 = data.get("city1")
        city2 = data.get("city2")

        response = {
            "city1": {
                "name": city1,
                "AQI": np.random.randint(50, 250),
                "PM10": np.random.randint(20, 150)
            },
            "city2": {
                "name": city2,
                "AQI": np.random.randint(50, 250),
                "PM10": np.random.randint(20, 150)
            }
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, port=5005)
