airIQ - Intelligence Air Quality Dashboard
🚀 Complete Environmental Intelligence: airIQ is no longer just a predictor; it's a comprehensive platform for real-time air quality monitoring, predictive simulation, and comparative analytics.

airIQ is a full-stack environmental intelligence platform engineered to monitor and predict the Air Quality Index (AQI) using advanced ensemble machine learning. The architecture integrates a responsive frontend, a Flask-based REST API, and a 6-feature hybrid ML pipeline to deliver high-precision environmental insights.

🚀 Key Modules & Features
1. 🔮 AI Prediction & What-If Simulation
6-Feature Inference: Predicts AQI using real-time inputs for CO, NH3, NO2, OZONE, PM10, and SO2.
What-If Analysis: Interactive sliders allow users to simulate how varying pollutant levels impact the overall AQI and health categories.
Model Comparison: Live-compare predictions across 6+ different algorithms (Linear, Random Forest, Gradient Boosting, etc.) to see model variance.
2. 🌍 Live Tracker & National Network
Real-Time API Sync: Integrates with the WAQI (World Air Quality Index) API to fetch live pollution data from thousands of stations across India.
Parallel Processing: Uses Python's ThreadPoolExecutor for high-concurrency API requests, ensuring low-latency data fetching.
National Leaderboard: Automatically ranks the Top 5 Cleanest and Top 5 Most Polluted cities in real-time.
3. 🗺️ Geographical Intelligence
Data-Driven Heatmap: A Folium-powered visualization that maps actual vs. predicted pollution levels across 260+ Indian cities.
City Comparison: Side-by-side analytical breakdown of pollutant concentrations between any two selected cities.
4. 🚲 Commute Safety Optimizer
Exposure Calculation: Calculates a safety score for commuters based on their route (Start/End city) and mode of transport (Walking, Cycling, Car, etc.).
Health Advice: Provides actionable health recommendations based on simulated pollutant exposure during transit.
🧠 Machine Learning Pipeline
1. The 6-Feature Dataset
The model has been upgraded from a 4-feature to a 6-feature input vector for superior chemical sensitivity:

Primary Features: PM10, OZONE, NO2, CO.
New Additions: NH3 (Ammonia) and SO2 (Sulfur Dioxide) for better urban industrial monitoring.
2. The 0.95 Milestone Hybrid Model
To achieve industry-leading accuracy (R² > 0.95), we implemented a Super-Ensemble Hybrid Model:

Composition:
40% Hist Gradient Boosting (Excellent for large-scale non-linear data)
30% Extra Trees Regressor (Highly randomized for variance reduction)
30% Random Forest (Robust baseline ensemble)
Impact: This weighted architecture captures both subtle pollutant fluctuations and extreme industrial spikes more effectively than any single model.
3. Live Analytics
Performance Benchmarking: Real-time calculation of MAE, RMSE, and R² scores on server startup.
Visual Validation: Dynamic scatter plots showing "Actual vs. Predicted" values for immediate scientific verification.
🛠️ Tech Stack
Backend: Flask (Python), Scikit-Learn, Pandas, NumPy, Joblib.
Frontend: HTML5, CSS3 (Glassmorphic Design), Vanilla JavaScript.
Visuals: Chart.js (Analytics), Folium (Geospatial), FontAwesome.
Data: WAQI REST API, CPCB Static Dataset.
⚙️ Installation & Setup
Clone & Environment:

git clone https://github.com/GitH-Priyanshu/airIQ.git
cd airIQ
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
Dependencies:

pip install -r requirements.txt
WAQI API Token:

Get a free token from aqicn.org.
Update WAQI_TOKEN in app.py.
Launch:

python app.py
Navigate to http://localhost:5000 (or the port specified in app.py).

📁 Project Structure
app.py: Core REST API & ML Orchestrator.
static/js/: Modular JS (api.js, ui.js, charts.js, map.js, comparison.js).
templates/: Feature-specific views (live_tracker, analytics, commute, etc.).
data/: Cleaned environmental datasets.

👨‍💻 Author
Prasanna, Priyanshu and Rajeev
