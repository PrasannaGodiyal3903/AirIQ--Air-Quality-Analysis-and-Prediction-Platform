# airIQ

> **🚧 Work In Progress:** This project is currently in the active development phase and is not yet complete. Features and integrations are subject to change.

airIQ is a full-stack web application engineered to predict the Air Quality Index (AQI) based on real-time air pollutant data. The architecture integrates a dynamic frontend, a Flask-based REST API backend, and an advanced ensemble machine learning pipeline to deliver highly accurate, low-latency predictions.

## 🚀 Features & Architecture

### Backend & Real-Time Pipeline
- **Flask REST API:** The backend is powered by Flask, ensuring lightweight and rapid request handling.
- **On-Startup Training Pipeline:** Upon server startup (`initialize_ml_models()`), the system dynamically trains multiple algorithms, calculates live evaluation metrics, and loads the primary production model into memory.
- **RESTful Endpoints:**
  - `/api/predict`: A POST route that consumes real-time pollutant parameters (`CO`, `NO2`, `OZONE`, `PM10`) from the user and passes them to the Gradient Boosting model for instant AQI inference.
  - `/api/metrics`: Exposes the model's evaluation metrics and scatter plot data to visually demonstrate accuracy on the frontend.

### Frontend Integration
- **Interactive UI:** Users can dynamically adjust pollutant levels via interactive sliders.
- **Asynchronous Inference:** Utilizing the JavaScript `fetch()` API, the frontend interacts with the backend to deliver real-time AQI predictions and categorical status (e.g., Good, Severe) without requiring page reloads.
- **Data Visualization:** Employs dynamic charts to visualize actual vs. predicted model performance.

## 📊 Dataset Analysis

**Target Variable:**
*   **AQI (Air Quality Index)** → Regression problem

**Input Features:**
*   `CO` (Carbon Monoxide)
*   `NO2` (Nitrogen Dioxide)
*   `OZONE` (O₃)
*   `PM10` (Particulate Matter)

**Feature Type:**
*   All features are strictly numerical.

**Insights:**
*   `PM10` and `OZONE` are strong contributors to AQI spikes.
*   `NO2` and `CO` contribute heavily to urban pollution trends.

**Conclusion:**
*   **Problem Type:** Regression
*   **Goal:** Predict continuous AQI value and programmatically map it to an AQI category (Good, Moderate, Severe, etc.).

## ⚙️ Data Preprocessing & Training Pipeline
The data engineering flow follows a strict logical pipeline: `Data Extraction` → `Preprocessing` → `Training` → `Evaluation` → `Deployment`.

1. **Handling Missing Values**: Applied time-series interpolation to maintain chronological continuity, with median imputation serving as a fallback for naturally right-skewed pollutant variables.
2. **Outlier Winsorization**: Capped physically anomalous spikes using an expanded `3.0 * IQR` bound to preserve legitimate extreme pollution events (common in environmental data) without blindly deleting rows.
3. **Pivoting & Target Engineering**: Reshaped the time-series data into a wide feature matrix and synthesized the `AQI` target variable by extracting the maximum sub-index across present pollutants.
4. **Standardization & Splitting**: Scaled numerical features using `StandardScaler` (Z-score normalization) to guarantee optimal convergence. Data was split 80/20 chronologically to completely prevent temporal data leakage.

## 🧠 Machine Learning Models

### 1. The Algorithms Evaluated
To ensure the highest scientific rigor, multiple algorithms were evaluated:
- **Linear Regression (Baseline):** Serves as our mathematical floor. It successfully captures broad linear trends but inherently struggles to model non-linear, extreme pollution spikes.
- **Random Forest Regressor:** An ensemble of independent decision trees that effectively manages variance and models complex interactions.
- **AdaBoost Regressor:** A sequential ensemble that explicitly penalizes and attempts to correct the errors of its predecessors.
- **Gradient Boosting Regressor (Primary Production Model):** A highly optimized sequential ensemble that minimizes the loss function using gradient descent. 

### 2. The Hybrid Model (Ensemble Blending)
To push predictive boundaries, an advanced **Hybrid Ensemble Model** was engineered. 
- **The Concept:** It actively combines the predictions of the linear baseline (Linear Regression) and the non-linear champion (Gradient Boosting) to capture both broad trends and complex spikes.
- **The Execution:** We evaluated both a **Simple Average (50/50)** and a mathematically optimized **Weighted Average (e.g., 20% LR + 80% GBR)** based on validation data.
- **The Impact:** The Hybrid approach successfully proved that bridging linear stability with non-linear variance is incredibly powerful (jumping from 78% to 96% accuracy). However, it ultimately demonstrated that blending a linear model slightly drags down the pure non-linear Gradient Boosting model on this specific dataset.

### 3. Evaluation Metrics Explained
All models are strictly evaluated against three standard mathematical benchmarks:
- **MAE (Mean Absolute Error):** Represents the average absolute point difference between the predicted and actual AQI. Provides simple interpretability (e.g., "Off by ±5 AQI").
- **RMSE (Root Mean Squared Error):** Heavily penalizes massive, dangerous prediction errors (e.g., predicting "Good" when the air is actually "Severe"). This is critical for health-based warning systems.
- **R² Score:** The "Accuracy Percentage." Indicates the percentage of AQI variance successfully explained by the model's chemical features.

### 🏆 Final Model Selection
**Gradient Boosting Regressor** was selected as the final production model deployed to the Flask backend. 
*   **Justification:** Environmental data is highly complex. AQI is often driven by extreme, sudden chemical spikes (especially `PM10` and `OZONE`). Gradient Boosting overwhelmingly defeated the linear baseline and outperformed the Hybrid model by achieving the lowest RMSE and the highest R² (>97%), proving it is the safest, most robust algorithm to capture non-linear pollution patterns.

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/GitH-Priyanshu/airIQ.git
   cd airIQ
   ```

2. Create a virtual environment and activate it (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5000` to interact with the application.

## 📁 Project Structure
- `app.py`: The main Flask application entry point.
- `models/`: Contains the pre-trained machine learning models.
- `notebooks/`: Jupyter notebooks used for data analysis and model training.
- `scripts/`: Python scripts for data processing.
- `static/`: Static assets (CSS, JS) for the web interface.
- `templates/`: HTML templates for the web interface.
- `data/`: Datasets used for training the models.

## 🚀 Future Work
- Add a heatmap visualization for geographical AQI monitoring.
- Integrate real-time API data for live predictions.
- Refine the UI/UX design.

## 👨‍💻 Author
Priyanshu
