import { fetchMetrics, simulateAQI, compareModels, fetchMapData } from './api.js';
import { renderScatterChart, renderPollutantChart, renderComparisonChart } from './charts.js';
import { initMap, renderMapMarkers } from './map.js';
import { updateDateTime, populateMetricsTable, updateSimulationUI, showSimulationLoading, updateComparisonDashboard, initTheme } from './ui.js';

let debounceTimer;

document.addEventListener("DOMContentLoaded", async () => {
    // 0. Initialize Theme Engine
    try {
        initTheme();
    } catch (e) {
        console.error("Theme Engine failed to initialize, but core features will continue to load.", e);
    }

    // 1. Initialize Date/Time in Navbar (Always run)
    setInterval(updateDateTime, 1000);
    updateDateTime();

    // 2. Conditional Initialization
    
    // Prediction Panel (What-If)
    if (document.getElementById('simPM10')) {
        setupSimulationPanel();
    }

    // Analytics Dashboard
    if (document.getElementById('scatterChart') || document.querySelector('.performance-table')) {
        loadAnalyticsData();
    }

    // Map View
    if (document.getElementById('aqiMap') || document.querySelector('.heatmap-container')) {
        loadMapData();
    }

    // Pollutant Chart (Feature Importance)
    if (document.getElementById('barChart')) {
        renderPollutantChart();
    }
});

async function loadAnalyticsData() {
    try {
        const metricsData = await fetchMetrics();
        if (!metricsData.error) {
            populateMetricsTable(metricsData.metrics);
            renderScatterChart(metricsData.scatter);
            
            // Populate Explainable AI Dashboard with default baseline values
            const defaultPayload = { PM10: 100, NO2: 40, OZONE: 30, CO: 1.0 }; // Use the specific simulation API endpoint for Gradient Boosting
            const predictionData = await simulateAQI(defaultPayload);
            if (!predictionData.error) {
                updateComparisonDashboard(predictionData);
            }
        }
    } catch (err) {
        console.error("Error loading analytics data:", err);
    }
}

async function loadMapData() {
    try {
        initMap();
        const mapData = await fetchMapData();
        if (!mapData.error) {
            renderMapMarkers(mapData.data);
        }
    } catch (err) {
        console.error("Error loading map data:", err);
    }
}


function setupSimulationPanel() {
    const sliders = ['simCO', 'simNO2', 'simNH3', 'simPM10', 'simOzone', 'simSO2'];
    
    // Setup slider value displays
    sliders.forEach(id => {
        const slider = document.getElementById(id);
        if(!slider) return;
        
        slider.addEventListener('input', (e) => {
            const valSpan = id.replace('sim', '').toLowerCase() + 'Val';
            const spanElem = document.getElementById(valSpan);
            if (spanElem) spanElem.textContent = e.target.value;
            
            // Trigger Debounced API call
            triggerSimulation();
        });
    });

    // Initial simulation run
    triggerSimulation();
}

function triggerSimulation() {
    if (!document.getElementById('simPM10')) return;

    clearTimeout(debounceTimer);
    showSimulationLoading();

    debounceTimer = setTimeout(async () => {
        try {
            const payload = {
                CO: parseFloat(document.getElementById('simCO').value),
                NH3: parseFloat(document.getElementById('simNH3').value),
                NO2: parseFloat(document.getElementById('simNO2').value),
                OZONE: parseFloat(document.getElementById('simOzone').value),
                PM10: parseFloat(document.getElementById('simPM10').value),
                SO2: parseFloat(document.getElementById('simSO2').value)
            };

            // Use the specific simulation API endpoint for Gradient Boosting
            const predictionData = await simulateAQI(payload);
            
            // Update the UI with the real model output
            updateSimulationUI(predictionData);
            
        } catch (err) {
            console.error("Simulation error:", err);
            const displayElem = document.getElementById('simResultDisplay');
            if (displayElem) displayElem.innerHTML = `<span class="text-danger">Error connecting to model.</span>`;
        }
    }, 300);
}
