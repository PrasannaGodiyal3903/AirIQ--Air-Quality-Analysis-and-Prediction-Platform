# 🌍 airIQ – Intelligence Air Quality Dashboard

> 🚀 A complete environmental intelligence platform for real-time AQI monitoring, predictive simulation, and comparative analytics.

airIQ is a full-stack environmental intelligence platform engineered to monitor and predict the **Air Quality Index (AQI)** using advanced ensemble machine learning. The system combines a responsive frontend, a Flask-based REST API, and a high-accuracy hybrid ML pipeline to deliver actionable environmental insights.

---

# ✨ Key Features

## 🔮 AI Prediction & What-If Simulation

* Predict AQI using **6 real-time pollutant inputs**
* Interactive **What-If Simulation** using dynamic sliders
* Real-time AQI category prediction
* Live comparison across multiple machine learning algorithms

### Supported Pollutants

* CO (Carbon Monoxide)
* NH3 (Ammonia)
* NO2 (Nitrogen Dioxide)
* OZONE
* PM10
* SO2 (Sulfur Dioxide)

### Supported Models

* Linear Regression
* Random Forest
* Gradient Boosting
* Extra Trees Regressor
* Hist Gradient Boosting
* Hybrid Ensemble Model

---

## 🌍 Live AQI Tracker & National Network

* Integrated with the **WAQI (World Air Quality Index) API**
* Fetches live pollution data from thousands of monitoring stations
* Parallel API requests using `ThreadPoolExecutor`
* Real-time leaderboard:

  * 🟢 Top 5 Cleanest Cities
  * 🔴 Top 5 Most Polluted Cities

---

## 🗺️ Geographical Intelligence

### 📍 Data-Driven Heatmaps

* Interactive Folium-powered AQI heatmaps
* Visualization of actual vs predicted pollution levels
* Coverage across **260+ Indian cities**

### 🏙️ City Comparison

* Side-by-side pollutant comparison between cities
* Comparative environmental analytics dashboard

---

## 🚲 Commute Safety Optimizer

* Calculates commuter exposure risk
* Supports:

  * Walking
  * Cycling
  * Car
  * Public Transport
* Generates:

  * Safety Score
  * Exposure Analysis
  * Health Recommendations

---

# 🧠 Machine Learning Pipeline

## 📊 6-Feature Dataset

The platform uses a chemically sensitive 6-feature dataset:

### Primary Features

* PM10
* OZONE
* NO2
* CO

### Additional Features

* NH3
* SO2

These additions significantly improve industrial and urban pollution analysis.

---

## 🚀 Hybrid Super-Ensemble Model (R² > 0.95)

To achieve high predictive accuracy, airIQ implements a weighted hybrid ensemble model.

### Model Composition

* **40% Hist Gradient Boosting**
* **30% Extra Trees Regressor**
* **30% Random Forest**

### Benefits

* Captures non-linear pollution patterns
* Handles industrial pollution spikes effectively
* Reduces prediction variance
* Improves AQI stability across regions

---

## 📈 Live Analytics

* Real-time performance benchmarking
* Metrics generated at server startup:

  * MAE
  * RMSE
  * R² Score
* Dynamic “Actual vs Predicted” validation plots

---

# 🛠️ Tech Stack

## Backend

* Flask (Python)
* Scikit-Learn
* Pandas
* NumPy
* Joblib

## Frontend

* HTML5
* CSS3
* Vanilla JavaScript

## Visualization

* Chart.js
* Folium
* FontAwesome

## Data Sources

* WAQI REST API
* CPCB Static Dataset

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/airIQ.git
cd airIQ
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure WAQI API Token

Get a free API token from:

https://aqicn.org/data-platform/token/

Update your token inside `app.py`:

```python
WAQI_TOKEN = "YOUR_API_KEY"
```

---

## 5️⃣ Run Application

```bash
python app.py
```

Visit:

```text
http://localhost:5000
```

---

# 📁 Project Structure

```text
airIQ/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── static/
│   ├── js/
│   │   ├── api.js
│   │   ├── ui.js
│   │   ├── charts.js
│   │   ├── map.js
│   │   └── comparison.js
│
├── templates/
│   ├── live_tracker.html
│   ├── analytics.html
│   ├── commute.html
│   └── ...
│
├── data/
│   └── environmental datasets
│
└── VIVA_NOTES.md
```

---

# 📊 Performance

| Metric             | Score  |
| ------------------ | ------ |
| R² Score           | > 0.95 |
| Features Used      | 6      |
| Cities Covered     | 260+   |
| ML Models Compared | 6+     |

---

# 🚀 Future Enhancements

* Deep Learning Integration
* Real-Time Satellite AQI Mapping
* Mobile App Version
* Personalized Health Alerts
* AI-based Pollution Forecasting

---

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
